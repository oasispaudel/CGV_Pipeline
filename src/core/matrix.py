"""
src/core/matrix.py
==================
All transformation matrices for the CGV Pipeline.

All transforms use 4×4 homogeneous matrices so that 2D and 3D objects
share the same pipeline. 2D objects simply live in the z = 0 plane.

All angles are in DEGREES (converted to radians internally).

Class:
    Transform – Static factory methods that return 4×4 numpy matrices
"""

import numpy as np


class Transform:
    """
    Static factory methods returning 4×4 homogeneous transformation matrices.

    Conventions:
      - Column vectors:  v' = M @ v
      - Angles in degrees
      - compose(A, B, C) applies A first, then B, then C
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @staticmethod
    def identity() -> np.ndarray:
        """4×4 identity matrix."""
        return np.eye(4, dtype=float)

    # ── Translation ───────────────────────────────────────────────────────────

    @staticmethod
    def translate(tx: float = 0, ty: float = 0, tz: float = 0) -> np.ndarray:
        """
        Translation matrix.
        Moves a point by (tx, ty, tz).
        """
        M = np.eye(4, dtype=float)
        M[0, 3] = tx
        M[1, 3] = ty
        M[2, 3] = tz
        return M

    # ── Scaling ───────────────────────────────────────────────────────────────

    @staticmethod
    def scale(sx: float = 1, sy: float = 1, sz: float = 1) -> np.ndarray:
        """
        Uniform or non-uniform scaling about the origin.
        """
        return np.diag([sx, sy, sz, 1.0]).astype(float)

    @staticmethod
    def scale_about_point(sx: float, sy: float, sz: float,
                          px: float, py: float, pz: float = 0) -> np.ndarray:
        """
        Fixed-point scaling: scale about an arbitrary point (px, py, pz).

        Method: translate pivot to origin → scale → translate back.
        """
        T  = Transform.translate( px,  py,  pz)
        S  = Transform.scale(sx, sy, sz)
        Ti = Transform.translate(-px, -py, -pz)
        return T @ S @ Ti

    # ── 2D Rotation (around Z axis) ───────────────────────────────────────────

    @staticmethod
    def rotate_z(angle_deg: float) -> np.ndarray:
        """
        Rotation in the XY plane (2D rotation, or 3D rotation about Z-axis).
        Positive angle = counter-clockwise.
        """
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([
            [ c, -s,  0,  0],
            [ s,  c,  0,  0],
            [ 0,  0,  1,  0],
            [ 0,  0,  0,  1],
        ], dtype=float)

    @staticmethod
    def rotate_about_point_2d(angle_deg: float, px: float, py: float) -> np.ndarray:
        """
        Pivot-point 2D rotation: rotate around an arbitrary point (px, py).

        Method: translate pivot to origin → rotate → translate back.
        """
        T  = Transform.translate( px,  py)
        R  = Transform.rotate_z(angle_deg)
        Ti = Transform.translate(-px, -py)
        return T @ R @ Ti

    # ── 3D Rotations ──────────────────────────────────────────────────────────

    @staticmethod
    def rotate_x(angle_deg: float) -> np.ndarray:
        """Rotation about the X-axis."""
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([
            [1,  0,  0,  0],
            [0,  c, -s,  0],
            [0,  s,  c,  0],
            [0,  0,  0,  1],
        ], dtype=float)

    @staticmethod
    def rotate_y(angle_deg: float) -> np.ndarray:
        """Rotation about the Y-axis."""
        a = np.radians(angle_deg)
        c, s = np.cos(a), np.sin(a)
        return np.array([
            [ c,  0,  s,  0],
            [ 0,  1,  0,  0],
            [-s,  0,  c,  0],
            [ 0,  0,  0,  1],
        ], dtype=float)

    # ── Reflection ────────────────────────────────────────────────────────────

    @staticmethod
    def reflect_x() -> np.ndarray:
        """Reflect across the X-axis (negate Y)."""
        return Transform.scale(1, -1, 1)

    @staticmethod
    def reflect_y() -> np.ndarray:
        """Reflect across the Y-axis (negate X)."""
        return Transform.scale(-1, 1, 1)

    @staticmethod
    def reflect_z() -> np.ndarray:
        """Reflect across the Z-axis (negate Z)."""
        return Transform.scale(1, 1, -1)

    @staticmethod
    def reflect_origin() -> np.ndarray:
        """Reflect through the origin (negate X and Y)."""
        return Transform.scale(-1, -1, 1)

    @staticmethod
    def reflect_arbitrary_line_2d(angle_deg: float) -> np.ndarray:
        """
        Reflect across a line through the origin at angle_deg from the X-axis.

        Method: rotate to align line with X-axis → reflect about X → rotate back.
        """
        R  = Transform.rotate_z( angle_deg)
        Ri = Transform.rotate_z(-angle_deg)
        Rx = Transform.reflect_x()
        return R @ Rx @ Ri

    # ── Shear ─────────────────────────────────────────────────────────────────

    @staticmethod
    def shear_xy(shx: float = 0, shy: float = 0) -> np.ndarray:
        """
        2D shear:
          x' = x + shx * y
          y' = y + shy * x
        """
        M = np.eye(4, dtype=float)
        M[0, 1] = shx
        M[1, 0] = shy
        return M

    @staticmethod
    def shear_3d(sxy: float = 0, sxz: float = 0,
                 syx: float = 0, syz: float = 0,
                 szx: float = 0, szy: float = 0) -> np.ndarray:
        """
        General 3D shear matrix.
        sxy = how much x shifts per unit y, etc.
        """
        M = np.eye(4, dtype=float)
        M[0, 1] = sxy;  M[0, 2] = sxz
        M[1, 0] = syx;  M[1, 2] = syz
        M[2, 0] = szx;  M[2, 1] = szy
        return M

    # ── Composite ─────────────────────────────────────────────────────────────

    @staticmethod
    def compose(*matrices: np.ndarray) -> np.ndarray:
        """
        Compose (multiply) a sequence of matrices.
        Transforms are applied LEFT TO RIGHT:
          compose(A, B, C)  applies A first, then B, then C.

        Internally this is:  C @ B @ A
        """
        result = np.eye(4, dtype=float)
        for M in matrices:
            result = M @ result
        return result

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def is_valid(M: np.ndarray) -> bool:
        """Check that M is a 4×4 float matrix."""
        return isinstance(M, np.ndarray) and M.shape == (4, 4)

    @staticmethod
    def print_matrix(M: np.ndarray, label: str = "Matrix"):
        """Pretty-print a transformation matrix."""
        print(f"\n{label}:")
        for row in M:
            print("  " + "  ".join(f"{v:8.4f}" for v in row))
