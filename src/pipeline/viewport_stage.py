"""
src/pipeline/viewport_stage.py
================================
Pipeline Stage 5 – Viewport Transformation (NDC → Screen / Pixel Coordinates).

The viewport transform is the final geometric step of the pipeline.
It maps the NDC square  [-1, 1] × [-1, 1]  to an integer pixel rectangle
on screen  [x0, x0+width] × [y0, y0+height].

Mapping formula (for a viewport anchored at (x0, y0)):

    x_screen = (x_ndc + 1) / 2 * (width  - 1) + x0
    y_screen = (1 - y_ndc) / 2 * (height - 1) + y0   ← Y flipped (Y-down screen)

The Y-axis flip converts from NDC (Y-up, mathematical) to screen (Y-down,
top-left origin), which is standard for raster displays.

Classes
-------
Viewport  – Stores screen dimensions and computes the mapping.

Functions
---------
apply_viewport(state, viewport) -> PipelineState
    Pipeline stage function.  Logs result under ``"5_viewport"``.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List

from ..core.data_structures import PipelineState, Mesh, Vertex, Edge

STAGE_KEY = "5_viewport"


# ── Viewport descriptor ───────────────────────────────────────────────────────

@dataclass
class Viewport:
    """
    Describes the rectangular region of the screen to render into.

    Parameters
    ----------
    width  : int    – viewport width  in pixels  (must be ≥ 1)
    height : int    – viewport height in pixels  (must be ≥ 1)
    x0     : int    – left edge offset in pixels  (default 0)
    y0     : int    – top  edge offset in pixels  (default 0)

    Example
    -------
    >>> vp = Viewport(width=800, height=600)
    >>> vp = Viewport(width=1280, height=720, x0=0, y0=0)
    """
    width:  int = 800
    height: int = 600
    x0:     int = 0
    y0:     int = 0

    def __post_init__(self):
        if self.width < 1:
            raise ValueError(f"Viewport width must be ≥ 1, got {self.width}.")
        if self.height < 1:
            raise ValueError(f"Viewport height must be ≥ 1, got {self.height}.")

    # ── coordinate conversion ─────────────────────────────────────────────────

    def ndc_to_screen(self, x_ndc: float, y_ndc: float):
        """
        Map a single NDC point to screen (pixel) coordinates.

        NDC  (-1, -1) → bottom-left of viewport.
        NDC  (+1, +1) → top-right  of viewport.
        Screen origin is top-left (Y-down convention).

        Parameters
        ----------
        x_ndc, y_ndc : float  – NDC coordinates in [-1, 1]

        Returns
        -------
        (x_screen, y_screen) : (float, float)
            Floating-point pixel position (caller may round for rasterisation).
        """
        x_screen = (x_ndc + 1.0) / 2.0 * (self.width  - 1) + self.x0
        y_screen = (1.0 - y_ndc) / 2.0 * (self.height - 1) + self.y0
        return x_screen, y_screen

    def screen_to_ndc(self, x_screen: float, y_screen: float):
        """
        Inverse mapping: screen pixel → NDC.  Useful for mouse picking.

        Parameters
        ----------
        x_screen, y_screen : float  – pixel coordinates

        Returns
        -------
        (x_ndc, y_ndc) : (float, float)
        """
        x_ndc =  2.0 * (x_screen - self.x0) / (self.width  - 1) - 1.0
        y_ndc = -2.0 * (y_screen - self.y0) / (self.height - 1) + 1.0
        return x_ndc, y_ndc

    def aspect_ratio(self) -> float:
        """Return width / height."""
        return self.width / self.height

    def __repr__(self):
        return (f"Viewport(width={self.width}, height={self.height}, "
                f"origin=({self.x0}, {self.y0}))")


# ── Stage function ────────────────────────────────────────────────────────────

def apply_viewport(
    state:    PipelineState,
    viewport: Viewport,
) -> PipelineState:
    """
    Pipeline Stage 5: Map NDC coordinates to screen pixel coordinates.

    Reads ``state.latest()`` (clipped NDC mesh from Stage 4), applies the
    viewport transform to every vertex, and stores the result under
    ``"5_viewport"``.

    The z-coordinate is preserved as-is (useful for depth buffering).
    The w-coordinate is set to 1.

    Parameters
    ----------
    state    : PipelineState  – pipeline state after clipping (Stage 4)
    viewport : Viewport       – target screen rectangle

    Returns
    -------
    PipelineState with stage ``"5_viewport"`` added.

    Example
    -------
    >>> vp    = Viewport(width=800, height=600)
    >>> state = apply_viewport(state, vp)
    >>> screen_mesh = state.stages["5_viewport"]
    """
    ndc_mesh = state.latest()

    screen_vertices: List[Vertex] = []
    for v in ndc_mesh.vertices:
        xs, ys = viewport.ndc_to_screen(v.x, v.y)
        screen_vertices.append(
            Vertex(coords=np.array([xs, ys, v.z, 1.0], dtype=float))
        )

    screen_mesh = Mesh(
        name="screen_space",
        vertices=screen_vertices,
        edges=[Edge(e.start, e.end) for e in ndc_mesh.edges],
    )

    state.add_stage(STAGE_KEY, screen_mesh)
    return state
