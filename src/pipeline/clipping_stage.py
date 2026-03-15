"""
src/pipeline/clipping_stage.py
================================
Pipeline Stage 4 – NDC Clipping.

Clips all edges of the NDC mesh against the unit clip window  [-1, 1] × [-1, 1]
using either the Cohen-Sutherland or Liang-Barsky algorithm (selectable).

The clip window in NDC space is always the unit square, so no parameters
are needed beyond the algorithm choice.

Functions
---------
apply_clipping(state, algorithm) -> PipelineState
    Main stage function.  Clips ``state.latest()`` and logs the result
    under stage key ``"4_clipping"``.
"""

from __future__ import annotations

from ..core.data_structures import PipelineState
from ..algorithms.cohen_sutherland import CohenSutherlandClipper, ClipWindow
from ..algorithms.liang_barsky     import LiangBarskyClipper

STAGE_KEY   = "4_clipping"
NDC_WINDOW  = ClipWindow(x_min=-1.0, y_min=-1.0, x_max=1.0, y_max=1.0)


def apply_clipping(
    state:     PipelineState,
    algorithm: str = "cohen_sutherland",
) -> PipelineState:
    """
    Pipeline Stage 4: Clip the NDC mesh to the unit square.

    Reads ``state.latest()`` (NDC mesh from Stage 3 / 3b), clips every edge
    against the  [-1, 1] × [-1, 1]  window, and stores the result under
    ``"4_clipping"``.

    Parameters
    ----------
    state     : PipelineState
    algorithm : str  – ``"cohen_sutherland"`` (default) or ``"liang_barsky"``

    Returns
    -------
    PipelineState with stage ``"4_clipping"`` added.

    Raises
    ------
    ValueError  – if *algorithm* is unrecognised.

    Examples
    --------
    >>> state = apply_clipping(state)                          # Cohen-Sutherland
    >>> state = apply_clipping(state, algorithm="liang_barsky")
    """
    algorithm = algorithm.lower().strip()

    if algorithm == "cohen_sutherland":
        clipper = CohenSutherlandClipper(NDC_WINDOW)
    elif algorithm == "liang_barsky":
        clipper = LiangBarskyClipper(NDC_WINDOW)
    else:
        raise ValueError(
            f"Unknown clipping algorithm '{algorithm}'. "
            f"Choose 'cohen_sutherland' or 'liang_barsky'."
        )

    ndc_mesh     = state.latest()
    clipped_mesh = clipper.clip_mesh(ndc_mesh)

    # Rename to a clean stage label
    clipped_mesh.name = "clipped_ndc"

    state.add_stage(STAGE_KEY, clipped_mesh)
    return state
