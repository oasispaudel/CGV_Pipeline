"""
src/pipeline/projection_stage.py
==================================
Pipeline Stage 3 – View Space → Clip Space → NDC.

This module performs the full 3-D projection step:

    view-space mesh
        │  apply projection matrix  (perspective or orthographic)
        ▼
    clip-space mesh
        │  perspective divide  (x/w, y/w, z/w)
        ▼
    NDC mesh  (all coordinates in [-1, 1])

Two projection types are supported; choose via the ``mode`` parameter:

    "perspective"   – realistic foreshortening (fov / aspect / near / far)
    "orthographic"  – parallel projection (no foreshortening)

Functions
---------
apply_projection(state, mode, **kwargs) -> PipelineState
    Main entry point.  Reads the latest mesh from *state*, projects it,
    and logs the result under stage key ``"3_projection"``.

apply_perspective_projection(state, fov_deg, aspect, near, far) -> PipelineState
    Convenience wrapper for perspective-only calls.

apply_orthographic_projection(state, left, right, bottom, top, near, far) -> PipelineState
    Convenience wrapper for orthographic-only calls.
"""

from __future__ import annotations

from ..core.data_structures import PipelineState
from .projection import (
    PerspectiveProjection,
    OrthographicProjection,
    perspective_divide,
)

STAGE_KEY = "3_projection"


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_projection(
    state:  PipelineState,
    mode:   str   = "perspective",
    # ── perspective kwargs ──────────────────────
    fov_deg: float = 60.0,
    aspect:  float = 16.0 / 9.0,
    # ── shared near / far ───────────────────────
    near:    float = 0.1,
    far:     float = 100.0,
    # ── orthographic kwargs ─────────────────────
    left:    float = -1.0,
    right:   float =  1.0,
    bottom:  float = -1.0,
    top:     float =  1.0,
) -> PipelineState:
    """
    Pipeline Stage 3: Project the view-space mesh to NDC.

    Reads ``state.latest()`` (expected to be in view/camera space after
    Stage 2), applies the selected projection matrix, performs the
    perspective divide, and stores the resulting NDC mesh under
    ``state.stages["3_projection"]``.

    Parameters
    ----------
    state   : PipelineState  – pipeline state after Stage 2 (world → view)
    mode    : str            – ``"perspective"`` (default) or ``"orthographic"``

    Perspective parameters (used when mode == "perspective"):
    fov_deg : float  – vertical field of view in degrees  (default 60)
    aspect  : float  – viewport width / height             (default 16/9)

    Shared parameters:
    near    : float  – near clip plane distance, > 0       (default 0.1)
    far     : float  – far  clip plane distance, > near    (default 100)

    Orthographic parameters (used when mode == "orthographic"):
    left, right   : float  – horizontal view volume extent (default -1, 1)
    bottom, top   : float  – vertical   view volume extent (default -1, 1)

    Returns
    -------
    PipelineState  – with stage ``"3_projection"`` added.

    Raises
    ------
    ValueError  – if *mode* is not ``"perspective"`` or ``"orthographic"``.

    Examples
    --------
    Perspective (most common):

    >>> state = apply_projection(state, mode="perspective",
    ...                          fov_deg=60, aspect=16/9, near=0.1, far=100)

    Orthographic:

    >>> state = apply_projection(state, mode="orthographic",
    ...                          left=-5, right=5, bottom=-5, top=5,
    ...                          near=0.1, far=50)
    """
    mode = mode.lower().strip()

    if mode == "perspective":
        return apply_perspective_projection(
            state, fov_deg=fov_deg, aspect=aspect, near=near, far=far
        )
    elif mode == "orthographic":
        return apply_orthographic_projection(
            state, left=left, right=right,
            bottom=bottom, top=top, near=near, far=far
        )
    else:
        raise ValueError(
            f"Unknown projection mode '{mode}'. "
            f"Choose 'perspective' or 'orthographic'."
        )


# ── Perspective convenience wrapper ───────────────────────────────────────────

def apply_perspective_projection(
    state:   PipelineState,
    fov_deg: float = 60.0,
    aspect:  float = 16.0 / 9.0,
    near:    float = 0.1,
    far:     float = 100.0,
) -> PipelineState:
    """
    Apply a perspective projection to the current pipeline mesh.

    Parameters
    ----------
    state   : PipelineState
    fov_deg : float  – vertical FOV in degrees
    aspect  : float  – width / height ratio
    near    : float  – near plane (> 0)
    far     : float  – far  plane (> near)

    Returns
    -------
    PipelineState with stage ``"3_projection"`` logged.
    """
    proj        = PerspectiveProjection(fov_deg=fov_deg, aspect=aspect,
                                        near=near, far=far)
    P           = proj.matrix()
    view_mesh   = state.latest()
    clip_mesh   = view_mesh.copy_with_matrix(P, name="clip_space")
    ndc_mesh    = perspective_divide(clip_mesh, name="ndc_perspective")

    state.add_stage(STAGE_KEY, ndc_mesh)
    return state


# ── Orthographic convenience wrapper ──────────────────────────────────────────

def apply_orthographic_projection(
    state:  PipelineState,
    left:   float = -1.0,
    right:  float =  1.0,
    bottom: float = -1.0,
    top:    float =  1.0,
    near:   float =  0.1,
    far:    float = 100.0,
) -> PipelineState:
    """
    Apply an orthographic projection to the current pipeline mesh.

    Parameters
    ----------
    state              : PipelineState
    left, right        : float  – horizontal extent
    bottom, top        : float  – vertical   extent
    near, far          : float  – depth       extent

    Returns
    -------
    PipelineState with stage ``"3_projection"`` logged.
    """
    proj      = OrthographicProjection(left=left, right=right,
                                       bottom=bottom, top=top,
                                       near=near, far=far)
    P         = proj.matrix()
    view_mesh = state.latest()
    ndc_mesh  = view_mesh.copy_with_matrix(P, name="ndc_orthographic")

    # w stays 1 for ortho, but run perspective_divide anyway so the
    # output mesh is always normalised (w=1) regardless of mode.
    ndc_mesh  = perspective_divide(ndc_mesh, name="ndc_orthographic")

    state.add_stage(STAGE_KEY, ndc_mesh)
    return state

