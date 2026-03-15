"""
src/algorithms/cohen_sutherland.py
====================================
Cohen-Sutherland line-clipping algorithm.

The Cohen-Sutherland algorithm clips a line segment against an axis-aligned
rectangular clipping window by assigning 4-bit region codes (outcodes) to each
endpoint, then trivially accepting, trivially rejecting, or subdividing the
segment until it is fully inside or fully outside the window.

Region code bit layout (from MSB to LSB):
    bit 3 (TOP)   : y > y_max
    bit 2 (BOTTOM): y < y_min
    bit 1 (RIGHT) : x > x_max
    bit 0 (LEFT)  : x < x_min

Classes:
    ClipWindow          – Axis-aligned rectangular clipping boundary
    CohenSutherlandClipper – Clips a single line or a whole Mesh

Functions:
    cohen_sutherland_clip(x0, y0, x1, y1, window) -> tuple | None
        Low-level scalar clip returning clipped endpoints or None.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple

from ..core.data_structures import Mesh, Vertex, Edge


# ── Region-code constants ─────────────────────────────────────────────────────

INSIDE = 0b0000   # 0
LEFT   = 0b0001   # 1
RIGHT  = 0b0010   # 2
BOTTOM = 0b0100   # 4
TOP    = 0b1000   # 8


# ── Clipping window ───────────────────────────────────────────────────────────

@dataclass
class ClipWindow:
    """
    Axis-aligned rectangular clipping window.

    Parameters
    ----------
    x_min, y_min : float  – bottom-left corner
    x_max, y_max : float  – top-right corner

    Example
    -------
    >>> w = ClipWindow(-1, -1, 1, 1)   # unit square window
    """
    x_min: float = -1.0
    y_min: float = -1.0
    x_max: float =  1.0
    y_max: float =  1.0

    def __post_init__(self):
        if self.x_min >= self.x_max or self.y_min >= self.y_max:
            raise ValueError(
                f"Invalid clip window: x_min={self.x_min} must be < x_max={self.x_max} "
                f"and y_min={self.y_min} must be < y_max={self.y_max}."
            )

    def __repr__(self):
        return (f"ClipWindow(x=[{self.x_min}, {self.x_max}], "
                f"y=[{self.y_min}, {self.y_max}])")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _compute_outcode(x: float, y: float, w: ClipWindow) -> int:
    """
    Compute the 4-bit outcode for point (x, y) with respect to window *w*.

    Returns
    -------
    int  – bitmask combining LEFT | RIGHT | BOTTOM | TOP flags
    """
    code = INSIDE
    if   x < w.x_min: code |= LEFT
    elif x > w.x_max: code |= RIGHT
    if   y < w.y_min: code |= BOTTOM
    elif y > w.y_max: code |= TOP
    return code


# ── Public scalar function ────────────────────────────────────────────────────

def cohen_sutherland_clip(
    x0: float, y0: float,
    x1: float, y1: float,
    window: ClipWindow,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Clip the line segment (x0, y0)→(x1, y1) against *window*.

    Algorithm
    ---------
    1. Compute outcodes for both endpoints.
    2. Trivial accept  – both codes are INSIDE (bitwise OR == 0).
    3. Trivial reject  – codes share a set bit (bitwise AND != 0).
    4. Otherwise pick an outside endpoint, find its intersection with
       the boundary it violates, replace the endpoint with the intersection
       point, and repeat.

    Parameters
    ----------
    x0, y0 : float  – start of segment
    x1, y1 : float  – end of segment
    window : ClipWindow

    Returns
    -------
    (cx0, cy0, cx1, cy1) : tuple[float, float, float, float]
        Clipped segment endpoints – if the segment (or part of it) is inside.
    None
        If the segment is entirely outside the window.

    Examples
    --------
    >>> w = ClipWindow(0, 0, 1, 1)
    >>> cohen_sutherland_clip(-0.5, 0.5, 1.5, 0.5, w)
    (0.0, 0.5, 1.0, 0.5)
    >>> cohen_sutherland_clip(2, 2, 3, 3, w)   # entirely outside
    None
    """
    code0 = _compute_outcode(x0, y0, window)
    code1 = _compute_outcode(x1, y1, window)

    while True:
        if not (code0 | code1):          # trivial accept
            return (x0, y0, x1, y1)

        if code0 & code1:                # trivial reject
            return None

        # At least one endpoint is outside. Pick it.
        outside_code = code0 if code0 else code1

        # Find the intersection point using the parametric line equation.
        # Slope guard: use a tiny epsilon to avoid division by zero on
        # degenerate (near-zero-length) segments.
        dx = x1 - x0
        dy = y1 - y0

        if outside_code & TOP:
            x = x0 + dx * (window.y_max - y0) / dy if dy else x0
            y = window.y_max
        elif outside_code & BOTTOM:
            x = x0 + dx * (window.y_min - y0) / dy if dy else x0
            y = window.y_min
        elif outside_code & RIGHT:
            y = y0 + dy * (window.x_max - x0) / dx if dx else y0
            x = window.x_max
        else:  # LEFT
            y = y0 + dy * (window.x_min - x0) / dx if dx else y0
            x = window.x_min

        # Update the outside endpoint and recompute its outcode.
        if outside_code is code0:
            x0, y0 = x, y
            code0 = _compute_outcode(x0, y0, window)
        else:
            x1, y1 = x, y
            code1 = _compute_outcode(x1, y1, window)


# ── Mesh-level clipper ────────────────────────────────────────────────────────

class CohenSutherlandClipper:
    """
    Applies Cohen-Sutherland clipping to every edge of a :class:`Mesh`.

    Parameters
    ----------
    window : ClipWindow
        The clipping boundary.

    Example
    -------
    >>> from src.core.data_structures import Mesh
    >>> clipper = CohenSutherlandClipper(ClipWindow(-1, -1, 1, 1))
    >>> clipped = clipper.clip_mesh(mesh)
    """

    def __init__(self, window: ClipWindow):
        self.window = window

    # ── public ────────────────────────────────────────────────────────────────

    def clip_mesh(self, mesh: Mesh) -> Mesh:
        """
        Clip all edges of *mesh* and return a new Mesh containing only the
        visible (inside) line segments.

        Each surviving edge may have new intersection vertices that did not
        exist in the original mesh. The returned mesh is therefore a
        *flat edge-soup* – vertices are stored per-edge rather than shared.

        Parameters
        ----------
        mesh : Mesh  – input wireframe (2-D or 3-D; only XY used for clipping)

        Returns
        -------
        Mesh  – clipped mesh (name prefixed with "clipped_")
        """
        new_vertices: List[Vertex] = []
        new_edges:    List[Edge]   = []

        for edge in mesh.edges:
            v0 = mesh.vertices[edge.start]
            v1 = mesh.vertices[edge.end]

            result = cohen_sutherland_clip(
                v0.x, v0.y,
                v1.x, v1.y,
                self.window,
            )

            if result is None:
                continue  # edge entirely outside – discard

            cx0, cy0, cx1, cy1 = result

            # Preserve z from the original vertices (midpoint blend if needed)
            z0 = v0.z
            z1 = v1.z

            idx = len(new_vertices)
            new_vertices.append(Vertex.from_3d(cx0, cy0, z0))
            new_vertices.append(Vertex.from_3d(cx1, cy1, z1))
            new_edges.append(Edge(idx, idx + 1))

        return Mesh(
            name=f"clipped_{mesh.name}",
            vertices=new_vertices,
            edges=new_edges,
        )

    def clip_segment(
        self,
        p0: Vertex,
        p1: Vertex,
    ) -> Optional[Tuple[Vertex, Vertex]]:
        """
        Convenience wrapper: clip a single segment defined by two Vertex objects.

        Returns
        -------
        (Vertex, Vertex) or None
        """
        result = cohen_sutherland_clip(p0.x, p0.y, p1.x, p1.y, self.window)
        if result is None:
            return None
        cx0, cy0, cx1, cy1 = result
        return Vertex.from_3d(cx0, cy0, p0.z), Vertex.from_3d(cx1, cy1, p1.z)

    def __repr__(self):
        return f"CohenSutherlandClipper(window={self.window})"
