"""Inverse-reachability map and planar trajectory helpers for the bunker cart.

The cached map is independent of the orchard and cart mounting.  Each sample
stores a reachable grasp-centre pose in the arm-base frame and the joint
configuration that produced it.  A query aligns those samples to a desired
world TCP pose using a planar arm-base transform, then converts the arm-base
pose to the cart's ``[x, y, yaw]`` coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


CACHE_VERSION = 1
DEFAULT_CACHE = Path(__file__).resolve().parent / "assets" / "bunker_cart" / "irm_tcp.npz"


@dataclass(frozen=True)
class IRMCandidate:
    cart_qs: np.ndarray
    seed_qs: np.ndarray
    score: float
    orientation_error_rad: float


@dataclass(frozen=True)
class IRMResult:
    cart_qs: np.ndarray
    arm_qs: np.ndarray
    score: float
    checked_candidates: int


def _skew(axis: np.ndarray) -> np.ndarray:
    x, y, z = axis
    return np.array(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)), dtype=np.float64)


def _vectorized_tcp_fk(robot, tcp, qs: np.ndarray, batch_size: int = 50_000):
    """Return TCP positions/rotations in the arm-base frame for full-arm ``qs``."""
    chain = robot.chain("main")
    if qs.shape[1] != len(chain.axes):
        raise ValueError("IRM builder requires one sampled value per main-chain joint")

    old_qs = robot.qs.copy()
    robot.fk(np.zeros_like(old_qs))
    root_tf = robot.runtime_lnks[chain.base_lidx].tf.astype(np.float64).copy()
    tcp_zero = np.linalg.inv(root_tf) @ tcp.tf.astype(np.float64)
    robot.fk(old_qs)

    axes = [np.asarray(axis, dtype=np.float64) for axis in chain.axes]
    origins = [np.asarray(origin, dtype=np.float64) for origin in chain.origins]
    generators = [_skew(axis / np.linalg.norm(axis)) for axis in axes]
    identity = np.eye(3, dtype=np.float64)
    positions = np.empty((len(qs), 3), dtype=np.float32)
    rotations = np.empty((len(qs), 3, 3), dtype=np.float32)

    for start in range(0, len(qs), batch_size):
        stop = min(start + batch_size, len(qs))
        q_batch = np.asarray(qs[start:stop], dtype=np.float64)
        count = len(q_batch)
        transforms = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
        for joint_index, (generator, origin) in enumerate(zip(generators, origins)):
            angle = q_batch[:, joint_index]
            cosine = np.cos(angle)[:, None, None]
            sine = np.sin(angle)[:, None, None]
            rotation = (
                identity[None]
                + sine * generator[None]
                + (1.0 - cosine) * (generator @ generator)[None]
            )
            translation = np.einsum("nij,j->ni", identity[None] - rotation, origin)
            motion = np.tile(np.eye(4, dtype=np.float64), (count, 1, 1))
            motion[:, :3, :3] = rotation
            motion[:, :3, 3] = translation
            transforms = transforms @ motion
        tcp_tf = transforms @ tcp_zero
        positions[start:stop] = tcp_tf[:, :3, 3]
        rotations[start:stop] = tcp_tf[:, :3, :3]
    return positions, rotations


def _sample_density(positions: np.ndarray, rotations: np.ndarray) -> np.ndarray:
    """Estimate local sample density with coarse position/tool-axis voxels."""
    tool_axis = rotations[:, :, 2]
    keys = np.column_stack(
        (
            np.floor(positions / 0.05).astype(np.int16),
            np.floor((tool_axis + 1.0) / 0.25).astype(np.int16),
        )
    )
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    return counts[inverse].astype(np.float32)


def build_irm(
    robot,
    tcp,
    cache_path: Path | str = DEFAULT_CACHE,
    *,
    n_samples: int = 120_000,
    seed: int = 0,
) -> Path:
    """FK-sample the arm and save a reusable inverse-reachability cache."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    cache_path = Path(cache_path)
    chain = robot.chain("main")
    low = np.asarray(chain.lmt_lo, dtype=np.float64)
    high = np.asarray(chain.lmt_up, dtype=np.float64)
    rng = np.random.default_rng(seed)
    qs = rng.uniform(low, high, size=(n_samples, len(low))).astype(np.float32)
    positions, rotations = _vectorized_tcp_fk(robot, tcp, qs)
    density = _sample_density(positions, rotations)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        version=np.int32(CACHE_VERSION),
        qs=qs,
        tcp_positions=positions,
        tcp_rotations=rotations,
        density=density,
    )
    return cache_path


class CartIRM:
    def __init__(self, cache_path: Path | str = DEFAULT_CACHE):
        cache_path = Path(cache_path)
        with np.load(cache_path) as data:
            version = int(data["version"])
            if version != CACHE_VERSION:
                raise ValueError(
                    f"IRM cache version {version} does not match expected {CACHE_VERSION}"
                )
            self.qs = np.asarray(data["qs"], dtype=np.float32)
            self.tcp_positions = np.asarray(data["tcp_positions"], dtype=np.float32)
            self.tcp_rotations = np.asarray(data["tcp_rotations"], dtype=np.float32)
            self.density = np.asarray(data["density"], dtype=np.float32)

    @staticmethod
    def _best_planar_yaw(sample_rotations: np.ndarray, target_rotation: np.ndarray):
        product = sample_rotations @ target_rotation.T
        yaw = np.arctan2(product[:, 0, 1] - product[:, 1, 0],
                         product[:, 0, 0] + product[:, 1, 1])
        cosine = np.cos(yaw)
        sine = np.sin(yaw)
        rz = np.zeros((len(yaw), 3, 3), dtype=np.float64)
        rz[:, 0, 0] = cosine
        rz[:, 0, 1] = -sine
        rz[:, 1, 0] = sine
        rz[:, 1, 1] = cosine
        rz[:, 2, 2] = 1.0
        residual = target_rotation.T[None] @ (rz @ sample_rotations)
        trace = np.trace(residual, axis1=1, axis2=2)
        error = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
        return yaw, rz, error

    def candidates(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        current_cart_qs: np.ndarray,
        arm_loc_tf: np.ndarray,
        *,
        max_candidates: int = 40,
        orientation_tolerance_rad: float = np.deg2rad(12.0),
        base_height_tolerance_m: float = 0.025,
        arm_base_height_m: float | None = None,
    ) -> list[IRMCandidate]:
        """Return diverse, high-density planar cart poses for a world TCP target."""
        target_position = np.asarray(target_position, dtype=np.float64)
        target_rotation = np.asarray(target_rotation, dtype=np.float64)
        current_cart_qs = np.asarray(current_cart_qs, dtype=np.float64)
        arm_loc_tf = np.asarray(arm_loc_tf, dtype=np.float64)
        yaw, rz, orientation_error = self._best_planar_yaw(
            self.tcp_rotations.astype(np.float64), target_rotation
        )
        rotated_tcp = np.einsum("nij,nj->ni", rz, self.tcp_positions)
        arm_positions = target_position[None] - rotated_tcp

        expected_arm_z = (
            float(arm_loc_tf[2, 3])
            if arm_base_height_m is None
            else float(arm_base_height_m)
        )
        valid = (
            (orientation_error <= orientation_tolerance_rad)
            & (np.abs(arm_positions[:, 2] - expected_arm_z) <= base_height_tolerance_m)
        )
        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            return []

        arm_yaw_offset = np.arctan2(arm_loc_tf[1, 0], arm_loc_tf[0, 0])
        cart_yaw = yaw[indices] - arm_yaw_offset
        arm_xy_offset = arm_loc_tf[:2, 3]
        cosine = np.cos(cart_yaw)
        sine = np.sin(cart_yaw)
        offset_world = np.column_stack(
            (cosine * arm_xy_offset[0] - sine * arm_xy_offset[1],
             sine * arm_xy_offset[0] + cosine * arm_xy_offset[1])
        )
        cart_xy = arm_positions[indices, :2] - offset_world
        cart_qs = np.column_stack((cart_xy, _wrap_angle(cart_yaw)))

        distance = np.linalg.norm(cart_xy - current_cart_qs[:2], axis=1)
        yaw_cost = np.abs(_wrap_angle(cart_yaw - current_cart_qs[2]))
        density_bonus = np.log1p(self.density[indices])
        score = distance + 0.12 * yaw_cost + 0.08 * orientation_error[indices] - 0.025 * density_bonus
        order = np.argsort(score)

        # Keep one representative per 4 cm / 4 degree cell so IK checks explore
        # distinct parking poses instead of near-duplicate FK samples.
        selected: list[IRMCandidate] = []
        seen = set()
        for local_index in order:
            q = cart_qs[local_index]
            cell = (
                int(np.round(q[0] / 0.04)),
                int(np.round(q[1] / 0.04)),
                int(np.round(q[2] / np.deg2rad(4.0))),
            )
            if cell in seen:
                continue
            seen.add(cell)
            sample_index = indices[local_index]
            selected.append(
                IRMCandidate(
                    cart_qs=q.astype(np.float64),
                    seed_qs=self.qs[sample_index].astype(np.float64),
                    score=float(score[local_index]),
                    orientation_error_rad=float(orientation_error[sample_index]),
                )
            )
            if len(selected) >= max_candidates:
                break
        return selected

    def query(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        current_cart_qs: np.ndarray,
        arm_loc_tf: np.ndarray,
        validate: Callable[[IRMCandidate], np.ndarray | None],
        **candidate_kwargs,
    ) -> IRMResult:
        candidates = self.candidates(
            target_position,
            target_rotation,
            current_cart_qs,
            arm_loc_tf,
            **candidate_kwargs,
        )
        for checked, candidate in enumerate(candidates, start=1):
            arm_qs = validate(candidate)
            if arm_qs is not None:
                return IRMResult(
                    cart_qs=candidate.cart_qs.copy(),
                    arm_qs=np.asarray(arm_qs, dtype=np.float64),
                    score=candidate.score,
                    checked_candidates=checked,
                )
        raise ValueError(
            "IRM found no IK-reachable, collision-free cart pose "
            f"after checking {len(candidates)} candidates"
        )


def _wrap_angle(angle):
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


def unicycle_segments(
    start_qs: np.ndarray,
    goal_qs: np.ndarray,
    *,
    linear_speed: float,
    angular_speed: float,
    position_tolerance_m: float = 0.002,
    yaw_tolerance_rad: float = np.deg2rad(0.25),
) -> list[tuple[float, float, float]]:
    """Make rotate/translate/rotate velocity segments for a planar cart."""
    if linear_speed <= 0 or angular_speed <= 0:
        raise ValueError("IRM trajectory speeds must be positive")
    start_qs = np.asarray(start_qs, dtype=np.float64)
    goal_qs = np.asarray(goal_qs, dtype=np.float64)
    delta = goal_qs[:2] - start_qs[:2]
    distance = float(np.linalg.norm(delta))
    heading = float(np.arctan2(delta[1], delta[0])) if distance > position_tolerance_m else float(start_qs[2])
    first_turn = float(_wrap_angle(heading - start_qs[2]))
    final_turn = float(_wrap_angle(goal_qs[2] - heading))
    segments: list[tuple[float, float, float]] = []
    if abs(first_turn) > yaw_tolerance_rad:
        segments.append((0.0, np.copysign(angular_speed, first_turn), abs(first_turn) / angular_speed))
    if distance > position_tolerance_m:
        segments.append((linear_speed, 0.0, distance / linear_speed))
    if abs(final_turn) > yaw_tolerance_rad:
        segments.append((0.0, np.copysign(angular_speed, final_turn), abs(final_turn) / angular_speed))
    return segments
