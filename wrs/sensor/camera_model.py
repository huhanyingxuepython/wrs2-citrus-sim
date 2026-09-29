"""Rectified pinhole calibration: +X right, +Y down, +Z forward, meters."""

from dataclasses import dataclass
from functools import cached_property
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    """Pinhole intrinsics in pixels; integer (u, v) denotes a pixel center.

    Parameters
    ----------
    width, height : int
        Image dimensions in pixels.
    fx, fy, cx, cy : float
        Focal lengths and principal point of the output image.
    distortion_model : str
        'none' or 'brown_conrady'. The latter uses (k1, k2, p1, p2, k3).
        Inverse Brown, modified Brown and fisheye coefficients are not accepted.
    """

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str = 'none'
    distortion_coeffs: tuple = (0, 0, 0, 0, 0)

    def __post_init__(self):
        for value in (self.width, self.height):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError('width and height must be positive integers')
        if not np.all(np.isfinite([self.fx, self.fy, self.cx, self.cy])):
            raise ValueError('intrinsics must be finite')
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError('fx and fy must be positive')
        coefficients = tuple(float(value) for value in self.distortion_coeffs)
        if len(coefficients) != 5 or not np.all(np.isfinite(coefficients)):
            raise ValueError('distortion_coeffs must contain five finite values (k1,k2,p1,p2,k3)')
        object.__setattr__(self, 'distortion_coeffs', coefficients)
        if self.distortion_model not in ('none', 'brown_conrady'):
            raise ValueError('distortion_model must be none or brown_conrady')
        if self.distortion_model == 'none' and any(coefficients):
            raise ValueError('distortion_model="none" requires zero coefficients')

    @property
    def matrix(self):
        """Return a new (3, 3) intrinsic matrix."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
                        dtype=np.float64)

    @classmethod
    def from_fov(cls, width, height, horizontal_deg, vertical_deg):
        """Approximate intrinsics from edge-to-edge FOV angles in degrees.

        This is a nominal model, not a substitute for device calibration.
        """
        angles = np.array([horizontal_deg, vertical_deg], dtype=float)
        if not np.all(np.isfinite(angles) & (angles > 0) & (angles < 180)):
            raise ValueError('FOV angles must lie strictly between 0 and 180 degrees')
        fx, fy = np.array([width, height]) / (2 * np.tan(np.deg2rad(angles) / 2))
        return cls(width, height, fx, fy, (width - 1) / 2, (height - 1) / 2)

    def deproject(self, depth):
        """Convert (H, W) Z depth in meters into (H, W, 3) camera points.

        Nonfinite/nonpositive depths become (0, 0, 0). No range filtering is
        applied here; callers can use a captured frame's valid_mask.
        """
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape != (self.height, self.width):
            raise ValueError('depth shape must match camera height and width')
        z = np.where(np.isfinite(depth) & (depth > 0), depth, 0)
        return self.ray_lut * z[..., None]

    def project(self, points):
        """Project (..., 3) camera points into (..., 2) distorted pixel centers.

        Points must be finite and in front of the camera. Pixel coordinates
        are not clipped to the image bounds.
        """
        points = np.asarray(points, dtype=np.float64)
        if points.shape[-1] != 3 or not np.all(np.isfinite(points)) or np.any(points[..., 2] <= 0):
            raise ValueError('points must be finite (..., 3) with positive Z')
        normalized = points[..., :2] / points[..., 2, None]
        x, y = normalized[..., 0], normalized[..., 1]
        xd, yd = self._distort(x, y)
        return np.stack((xd * self.fx + self.cx, yd * self.fy + self.cy), axis=-1)

    def _distort(self, x, y):
        k1, k2, p1, p2, k3 = self.distortion_coeffs
        r2 = x*x + y*y
        radial = 1 + r2*(k1 + r2*(k2 + r2*k3))
        return (x*radial + 2*p1*x*y + p2*(r2 + 2*x*x),
                y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y)

    @cached_property
    def ray_lut(self):
        """Cached read-only (H, W, 3) optical rays with Z=1.

        Brown-Conrady inversion uses Newton iterations only once. Coefficients
        that fold the image or fail to converge over this FOV are rejected.
        """
        v, u = np.indices((self.height, self.width), dtype=np.float64)
        xd, yd = (u-self.cx)/self.fx, (v-self.cy)/self.fy
        x, y = xd.copy(), yd.copy()
        if any(self.distortion_coeffs):
            k1, k2, p1, p2, k3 = self.distortion_coeffs
            for _ in range(30):
                px, py = self._distort(x, y)
                ex, ey = px-xd, py-yd
                if max(np.max(np.abs(ex)), np.max(np.abs(ey))) < 1e-10:
                    break
                r2 = x*x + y*y
                radial = 1 + r2*(k1+r2*(k2+r2*k3))
                derivative = k1 + 2*k2*r2 + 3*k3*r2*r2
                a = radial + 2*x*x*derivative + 2*p1*y + 6*p2*x
                b = 2*x*y*derivative + 2*p1*x + 2*p2*y
                c = radial + 2*y*y*derivative + 6*p1*y + 2*p2*x
                determinant = a*c-b*b
                if np.any(determinant <= 1e-10) or not np.all(np.isfinite(determinant)):
                    raise ValueError('distortion is not invertible over the requested image')
                x -= (c*ex-b*ey)/determinant
                y -= (a*ey-b*ex)/determinant
            else:
                raise ValueError('distortion inversion did not converge')
        rays = np.stack((x, y, np.ones_like(x)), axis=-1).astype(np.float32)
        rays.setflags(write=False)
        return rays

    @cached_property
    def render_model(self):
        """Undistorted calibration expanded to cover all output rays."""
        if not any(self.distortion_coeffs):
            return CameraModel(self.width, self.height, self.fx, self.fy, self.cx, self.cy)
        rays = self.ray_lut
        u, v = rays[..., 0]*self.fx+self.cx, rays[..., 1]*self.fy+self.cy
        x0, y0 = int(np.floor(u.min())), int(np.floor(v.min()))
        x1, y1 = int(np.ceil(u.max())), int(np.ceil(v.max()))
        return CameraModel(x1-x0+1, y1-y0+1, self.fx, self.fy, self.cx-x0, self.cy-y0)

    @cached_property
    def remap_lut(self):
        """Read-only (H, W, 4): ray X/Y and nearest source pixel U/V.

        Nearest sampling intentionally avoids interpolating depths across
        foreground/background boundaries. The expanded render target avoids
        introducing black borders merely because of lens distortion.
        """
        rays, source = self.ray_lut, self.render_model
        u = np.floor(rays[..., 0]*source.fx + source.cx + .5)
        v = np.floor(rays[..., 1]*source.fy + source.cy + .5)
        lut = np.stack((rays[..., 0], rays[..., 1], u, v), axis=-1)
        lut.setflags(write=False)
        return lut
