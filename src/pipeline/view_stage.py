"""
src/pipeline/view_stage.py
============================
Pipeline Stage 2 – World Space → View (Camera) Space.

Thin stage wrapper that delegates to
``src.core.transformations.apply_world_to_view``.

Functions
---------
apply_view_stage(state, camera) -> PipelineState
"""

from __future__ import annotations

from ..core.data_structures  import PipelineState
from ..core.transformations  import Camera, apply_world_to_view


def apply_view_stage(
    state:  PipelineState,
    camera: Camera,
) -> PipelineState:
    """
    Pipeline Stage 2: World Space → View (Camera) Space.

    Uses the camera's look-at view matrix to transform all world-space
    coordinates into the camera's local frame.  Records the result as
    stage ``"2_world_to_view"``.

    Parameters
    ----------
    state  : PipelineState  – state after Stage 1 (model_to_world)
    camera : Camera         – virtual camera (eye, at, up)

    Returns
    -------
    PipelineState with stage ``"2_world_to_view"`` added.

    Example
    -------
    >>> from src.core.transformations import Camera
    >>> cam   = Camera(eye=[0, 3, 8], at=[0, 0, 0], up=[0, 1, 0])
    >>> state = apply_view_stage(state, cam)
    """
    return apply_world_to_view(state, camera)
