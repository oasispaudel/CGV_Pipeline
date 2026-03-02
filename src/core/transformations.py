"""
src/core/transformations.py
============================
Camera (view transformation) and pipeline stage runner functions.

Classes:
    Camera          – Simulates a virtual camera using the look-at model

Functions:
    apply_model_to_world()  – Stage 1: Model space → World space
    apply_world_to_view()   – Stage 2: World space → Camera/View space
"""

import numpy as np
from .data_structures import Mesh, PipelineState


class Camera:
    """
    Simulates a virtual camera using the classic look-at model.

    The view matrix transforms coordinates from world space into
    camera (view) space, where:
      - The camera is at the origin
      - Looking down the -Z axis
      - Y axis points up

    Parameters
    ----------
    eye : list or array  – Camera position in world space
    at  : list or array  – The point the camera looks toward
    up  : list or array  – World up vector (usually [0, 1, 0])

    Example
    -------
    >>> cam = Camera(eye=[4, 3, 5], at=[0, 0, 0], up=[0, 1, 0])
    >>> V = cam.view_matrix()   # 4×4 matrix, pass to Person 2
    """

    def __init__(self,
                 eye: list = [0, 0,  5],
                 at:  list = [0, 0,  0],
                 up:  list = [0, 1,  0]):
        self.eye = np.array(eye, dtype=float)
        self.at  = np.array(at,  dtype=float)
        self.up  = np.array(up,  dtype=float)

    def view_matrix(self) -> np.ndarray:
        """
        Build the 4×4 view (world → camera) matrix using the look-at algorithm.

        Steps:
          1. n = normalize(eye − at)      →  camera's Z axis (points toward viewer)
          2. u = normalize(up × n)        →  camera's X axis (right)
          3. v = n × u                    →  camera's Y axis (true up)
          4. Combine rotation + translation into 4×4 matrix

        Returns
        -------
        np.ndarray  shape (4, 4)
        """
        n = self._normalize(self.eye - self.at)
        u = self._normalize(np.cross(self.up, n))
        v = np.cross(n, u)

        M = np.array([
            [u[0], u[1], u[2], -np.dot(u, self.eye)],
            [v[0], v[1], v[2], -np.dot(v, self.eye)],
            [n[0], n[1], n[2], -np.dot(n, self.eye)],
            [0,    0,    0,     1                   ],
        ], dtype=float)
        return M

    def set_position(self, eye: list):
        """Move the camera to a new position."""
        self.eye = np.array(eye, dtype=float)

    def set_target(self, at: list):
        """Change what the camera looks at."""
        self.at = np.array(at, dtype=float)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            raise ValueError(
                f"Cannot normalize a near-zero vector: {v}\n"
                f"  Hint: 'eye' and 'at' must not be the same point."
            )
        return v / norm

    def __repr__(self):
        return (f"Camera(\n"
                f"  eye={self.eye.tolist()},\n"
                f"  at ={self.at.tolist()},\n"
                f"  up ={self.up.tolist()}\n)")



def apply_model_to_world(state: PipelineState,
                         model_matrix: np.ndarray) -> PipelineState:
    """
    Pipeline Stage 1: Model Space → World Space.

    Applies the model matrix (translation + rotation + scale composed together)
    to place the object in the world at the desired position and orientation.

    Parameters
    ----------
    state        : PipelineState  – current pipeline state
    model_matrix : np.ndarray     – 4×4 composite transform (from Transform class)

    Returns
    -------
    PipelineState with stage '1_model_to_world' logged.

    Example
    -------
    >>> from src.core.matrix import Transform
    >>> M = Transform.compose(
    ...     Transform.scale(2, 2, 2),
    ...     Transform.rotate_y(30),
    ...     Transform.translate(1, 0, 0)
    ... )
    >>> state = apply_model_to_world(state, M)
    """
    transformed = state.latest().copy_with_matrix(model_matrix, name="world_space")
    state.add_stage("1_model_to_world", transformed)
    return state


def apply_world_to_view(state: PipelineState,
                        camera: Camera) -> PipelineState:
    """
    Pipeline Stage 2: World Space → View (Camera) Space.

    Uses the camera's view matrix to transform all world coordinates
    into the camera's local coordinate system.

    Parameters
    ----------
    state  : PipelineState – current pipeline state (after model_to_world)
    camera : Camera        – the virtual camera

    Returns
    -------
    PipelineState with stage '2_world_to_view' logged.

    The resulting mesh in view space is what Person 2 receives
    for projection and clipping.

    Example
    -------
    >>> cam = Camera(eye=[0, 3, 8], at=[0, 0, 0])
    >>> state = apply_world_to_view(state, cam)
    """
    V = camera.view_matrix()
    transformed = state.latest().copy_with_matrix(V, name="view_space")
    state.add_stage("2_world_to_view", transformed)
    return state
