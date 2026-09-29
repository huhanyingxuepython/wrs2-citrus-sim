"""Two-body axial spring specifications and environment-local rupture state.

Positions are body-local metres (body=None means world). MuJoCo supplies all
spring forces; this module only observes tension and switches broken springs off.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class BodyAnchor:
    body: object | None
    local_pos: tuple = (0., 0., 0.)

    def __post_init__(self):
        p = np.asarray(self.local_pos, dtype=float)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('anchor local_pos must be a finite 3-vector')
        object.__setattr__(self, 'local_pos', tuple(p))


@dataclass(frozen=True)
class TensileBreakPolicy:
    break_force: float
    overload_hold_s: float

    def __post_init__(self):
        if (not np.isfinite([self.break_force, self.overload_hold_s]).all()
                or self.break_force <= 0 or self.overload_hold_s < 0):
            raise ValueError('break_force must be positive; overload_hold_s nonnegative')


@dataclass(frozen=True, eq=False)
class LinearSpringConnectionSpec:
    name: str
    anchor_a: BodyAnchor
    anchor_b: BodyAnchor
    rest_length: float
    stiffness: float
    damping: float
    break_policy: TensileBreakPolicy | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name:
            raise ValueError('connection name must be nonempty')
        if not isinstance(self.anchor_a, BodyAnchor) or not isinstance(self.anchor_b, BodyAnchor):
            raise TypeError('connection endpoints must be BodyAnchor')
        if (not np.isfinite([self.rest_length, self.stiffness, self.damping]).all()
                or self.rest_length <= 0 or self.stiffness < 0 or self.damping < 0):
            raise ValueError('invalid linear spring parameters (metres, N/m, N s/m)')
        if self.break_policy is not None and not isinstance(self.break_policy, TensileBreakPolicy):
            raise TypeError('invalid tensile break policy')


@dataclass
class RuntimeConnectionHandle:
    spec: LinearSpringConnectionSpec
    tendon_name: str
    tendon_id: int
    site_ids: tuple
    attached: bool = True
    overload_time_s: float = 0.
    length: float = 0.
    velocity: float = 0.
    tension: float = 0.
    events: list = field(default_factory=list)

    @property
    def extension(self):
        return self.length - self.spec.rest_length

    def observe(self, model, data):
        i = self.tendon_id
        self.length = float(data.ten_length[i])
        self.velocity = float(data.ten_velocity[i])
        self.tension = (max(0., float(model.tendon_stiffness[i]) * self.extension
                            + float(model.tendon_damping[i]) * self.velocity) if self.attached else 0.)

    def advance(self, model, data, dt):
        self.observe(model, data)
        policy = self.spec.break_policy
        if not self.attached or policy is None:
            return False
        self.overload_time_s = self.overload_time_s + dt if self.tension > policy.break_force else 0.
        if self.tension > policy.break_force and self.overload_time_s + 1e-12 >= policy.overload_hold_s:
            self.events.append(dict(time=float(data.time), connection=self.tendon_name,
                name=self.spec.name, extension=self.extension, tension=self.tension,
                relative_velocity=self.velocity, overload_time_s=self.overload_time_s))
            model.tendon_stiffness[self.tendon_id] = 0.
            model.tendon_damping[self.tendon_id] = 0.
            self.attached = False
            return True
        return False

    def report(self):
        return dict(attached=self.attached, length_m=self.length, extension_m=self.extension,
                    relative_velocity_m_s=self.velocity, tension_N=self.tension,
                    overload_time_s=self.overload_time_s, break_events=[dict(e) for e in self.events])
