"""
src/algorithms/liang_barsky.py
================================
Liang-Barsky parametric line-clipping algorithm.

The Liang-Barsky algorithm clips a line segment against an axis-aligned
rectangular window using a parametric representation of the line:

    P(t) = P0 + t * (P1 - P0),   t ∈ [0, 1]

Four inequalities (one per window boundary) yield constraints on *t*:

    p_k * t  ≤  q_k    for  k = left, right, bottom, top

where p_k and q_k depend on direction deltas and boundary distances.
The algorithm intersects all constraints to find the valid t-interval [t0, t1].

Advantages over Cohen-Sutherland:
  - Fewer intersection computations (at most one intersection calculation
    per boundary instead of repeated subdivision).
  - Naturally parametric — easy to extend to 3-D.

Classes:
    LiangBarskyClipper  – Clips a single segment or an entire Mesh.

Functions:
    liang_barsky_clip(x0, y0, x1, y1, x_min, y_min, x_max, y_max)
        Low-level scalar clip. Returns (cx0, cy0, cx1, cy1) or None.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

from ..core.data_structures import Mesh, Vertex, Edge
from .cohen_sutherland import ClipWindow   # reuse the same window dataclass


# ── Public scalar function ────────────────────────────────────────────────────

def liang_barsky_clip(
    x0: float, y0: float,
    x1: float, y1: float,
    x_min: float, y_min: float,
    x_max: float, y_max: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Clip the segment (x0, y0)→(x1, y1) against the window
    [x_min, x_max] × [y_min, y_max] using the Liang-Barsky algorithm.

    Algorithm
    ---------
    Define:
        dx = x1 - x0,  dy = y1 - y0

    The four boundary constraints written as  p_k*t ≤ q_k:

        k=0 (left)   : p=-dx,  q= x0 - x_min
        k=1 (right)  : p= dx,  q= x_max - x0
        k=2 (bottom) : p=-dy,  q= y0 - y_min
        k=3 (top)    : p= dy,  q= y_max - y0

    For each constraint:
      - p_k == 0: segment is parallel to boundary k.
        If q_k < 0 → entirely outside, reject.
      - p_k < 0:  segment enters from this boundary;
        update t0 = max(t0, q_k/p_k)
      - p_k > 0:  segment leaves toward this boundary;
        update t1 = min(t1, q_k/p_k)

    Accept if t0 < t1 (with t0 starting at 0 and t1 at 1).

    Parameters
    ----------
    x0, y0, x1, y1        : float  – segment endpoints
    x_min, y_min, x_max, y_max : float  – clip window bounds

    Returns
    -------
    (cx0, cy0, cx1, cy1) : tuple[float, float, float, float]
        Clipped endpoints if the segment (or part of it) is inside.
    None
        If the segment is entirely outside.

    Examples
    --------
    >>> liang_barsky_clip(-2, 0.5, 2, 0.5, 0, 0, 1, 1)
    (0.0, 0.5, 1.0, 0.5)
    >>> liang_barsky_clip(2, 2, 3, 3, 0, 0, 1, 1)
    None
    """
    dx = x1 - x0
    dy = y1 - y0

    # p and q for the four boundaries
    p = [-dx,  dx, -dy,  dy]
    q = [x0 - x_min, x_max - x0, y0 - y_min, y_max - y0]

    t0, t1 = 0.0, 1.0

    for pk, qk in zip(p, q):
        if pk == 0:
            # Parallel to boundary: reject if outside
            if qk < 0:
                return None
            # Otherwise no constraint from this boundary
        elif pk < 0:
            # Segment enters: tighten t0
            r = qk / pk
            if r > t1:
                return None
            if r > t0:
                t0 = r
        else:  # pk > 0
            # Segment exits: tighten t1
            r = qk / pk
            if r < t0:
                return None
            if r < t1:
                t1 = r

    # Compute clipped endpoints
    cx0 = x0 + t0 * dx
    cy0 = y0 + t0 * dy
    cx1 = x0 + t1 * dx
    cy1 = y0 + t1 * dy

    return cx0, cy0, cx1, cy1


# ── Mesh-level clipper ────────────────────────────────────────────────────────

class LiangBarskyClipper:
    """
    Applies Liang-Barsky clipping to every edge of a :class:`Mesh`.

    Parameters
    ----------
    window : ClipWindow
        The axis-aligned clipping rectangle (reused from cohen_sutherland).

    Example
    -------
    >>> clipper = LiangBarskyClipper(ClipWindow(-1, -1, 1, 1))
    >>> clipped_mesh = clipper.clip_mesh(mesh)
    """

    def __init__(self, window: ClipWindow):
        self.window = window

    # ── public ────────────────────────────────────────────────────────────────

    def clip_mesh(self, mesh: Mesh) -> Mesh:
        """
        Clip all edges of *mesh* using Liang-Barsky and return a new Mesh.

        Behaves identically to :meth:`CohenSutherlandClipper.clip_mesh` in its
        output format: the result is a flat edge-soup of surviving segments,
        where intersection vertices are added as needed.

        Parameters
        ----------
        mesh : Mesh  – source wireframe (XY clipping; Z is preserved)

        Returns
        -------
        Mesh  – clipped mesh (name prefixed with "lb_clipped_")
        """
        new_vertices: List[Vertex] = []
        new_edges:    List[Edge]   = []

        w = self.window

        for edge in mesh.edges:
            v0 = mesh.vertices[edge.start]
            v1 = mesh.vertices[edge.end]

            result = liang_barsky_clip(
                v0.x, v0.y,
                v1.x, v1.y,
                w.x_min, w.y_min, w.x_max, w.y_max,
            )

            if result is None:
                continue

            cx0, cy0, cx1, cy1 = result

            # Blend z linearly according to parameter t progress
            # t0/t1 can be derived from the x-displacement (or y if dx==0)
            dx = v1.x - v0.x
            dy = v1.y - v0.y
            length_sq = dx * dx + dy * dy
            if length_sq > 1e-12:
                t0_approx = ((cx0 - v0.x) * dx + (cy0 - v0.y) * dy) / length_sq
                t1_approx = ((cx1 - v0.x) * dx + (cy1 - v0.y) * dy) / length_sq
            else:
                t0_approx, t1_approx = 0.0, 1.0

            z0 = v0.z + t0_approx * (v1.z - v0.z)
            z1 = v0.z + t1_approx * (v1.z - v0.z)

            idx = len(new_vertices)
            new_vertices.append(Vertex.from_3d(cx0, cy0, z0))
            new_vertices.append(Vertex.from_3d(cx1, cy1, z1))
            new_edges.append(Edge(idx, idx + 1))

        return Mesh(
            name=f"lb_clipped_{mesh.name}",
            vertices=new_vertices,
            edges=new_edges,
        )

    def clip_segment(
        self,
        p0: Vertex,
        p1: Vertex,
    ) -> Optional[Tuple[Vertex, Vertex]]:
        """
        Clip a single segment defined by two Vertex objects.

        Returns
        -------
        (Vertex, Vertex) or None
        """
        w = self.window
        result = liang_barsky_clip(
            p0.x, p0.y, p1.x, p1.y,
            w.x_min, w.y_min, w.x_max, w.y_max,
        )
        if result is None:
            return None
        cx0, cy0, cx1, cy1 = result
        return Vertex.from_3d(cx0, cy0, p0.z), Vertex.from_3d(cx1, cy1, p1.z)

    def __repr__(self):
        return f"LiangBarskyClipper(window={self.window})"
