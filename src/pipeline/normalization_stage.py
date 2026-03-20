"""
src/pipeline/normalization_stage.py
=====================================
Pipeline Stage 3b – NDC Normalisation / Canonical View Volume check.

After projection and the perspective divide, every vertex should lie in
the Normalised Device Coordinate (NDC) cube  [-1, 1]³.  This stage:

    1. Verifies that vertices are within NDC bounds (optional, enabled by
       default).  Vertices outside the cube are *behind* or *beyond* the
       clip planes and will be handled by the clipping stage.

    2. Optionally flips the Y-axis if the downstream renderer uses a
       top-left origin (screen coordinates), which is common for 2-D
       raster displays.

    3. Logs the result as stage ``"3b_normalisation"`` in the
       PipelineState so every downstream stage receives clean NDC data.

This stage is lightweight — no matrix multiplication.  It records whether
each vertex is inside the NDC cube and can filter out-of-bounds vertices
so the clipping stage receives only potentially visible geometry.

Functions
---------
apply_normalisation(state, flip_y, clip_to_ndc) -> PipelineState
    Main stage function.  Wraps the NDC mesh and logs it.

is_in_ndc(vertex) -> bool
    Returns True if the vertex lies within [-1, 1] in x, y, and z.

filter_ndc_mesh(mesh) -> (Mesh, int)
    Remove edges whose *both* endpoints are outside NDC.
    Returns (filtered_mesh, n_removed).
"""

from __future__ import annotations

import numpy as np
from typing import Tuple, List

from ..core.data_structures import PipelineState, Mesh, Vertex, Edge

STAGE_KEY = "3b_normalisation"
NDC_MIN   = -1.0
NDC_MAX   =  1.0


# ── Vertex helpers ────────────────────────────────────────────────────────────

def is_in_ndc(vertex: Vertex, tol: float = 1e-6) -> bool:
    """
    Check whether *vertex* lies inside the NDC cube  [-1-tol, 1+tol]³.

    A small tolerance is included to absorb floating-point rounding that
    occurs during the perspective divide.

    Parameters
    ----------
    vertex : Vertex
    tol    : float  – epsilon for boundary comparisons (default 1e-6)

    Returns
    -------
    bool
    """
    lo = NDC_MIN - tol
    hi = NDC_MAX + tol
    return (lo <= vertex.x <= hi and
            lo <= vertex.y <= hi and
            lo <= vertex.z <= hi)


def ndc_bounds(mesh: Mesh) -> dict:
    """
    Return the actual min/max extents of the mesh in NDC space.

    Useful for debugging — if the bounds are far outside [-1, 1] the
    projection parameters may be misconfigured.

    Returns
    -------
    dict with keys 'x_min', 'x_max', 'y_min', 'y_max', 'z_min', 'z_max'
    """
    xs = [v.x for v in mesh.vertices]
    ys = [v.y for v in mesh.vertices]
    zs = [v.z for v in mesh.vertices]
    return dict(
        x_min=min(xs), x_max=max(xs),
        y_min=min(ys), y_max=max(ys),
        z_min=min(zs), z_max=max(zs),
    )


# ── Edge-level NDC filter ─────────────────────────────────────────────────────

def filter_ndc_mesh(mesh: Mesh, tol: float = 1e-6) -> Tuple[Mesh, int]:
    """
    Remove edges where **both** endpoints are outside the NDC cube.

    An edge where only one endpoint is outside may still be partially
    visible (it will be clipped later); such edges are kept.  Only edges
    that are entirely outside are discarded here as a cheap early cull.

    Parameters
    ----------
    mesh : Mesh  – NDC-space mesh
    tol  : float – boundary tolerance

    Returns
    -------
    (filtered_mesh, n_removed)
        filtered_mesh : Mesh  – mesh with trivially-outside edges removed
        n_removed     : int   – number of edges discarded
    """
    kept_vertices: List[Vertex] = []
    kept_edges:    List[Edge]   = []
    n_removed = 0

    for edge in mesh.edges:
        v0 = mesh.vertices[edge.start]
        v1 = mesh.vertices[edge.end]

        # Trivial reject: both endpoints entirely outside
        if not is_in_ndc(v0, tol) and not is_in_ndc(v1, tol):
            # Extra check: same side outside (like Cohen-Sutherland AND test)
            # If they share an outside region, definitely invisible.
            def _out_flags(v: Vertex):
                flags = 0
                if v.x < NDC_MIN - tol: flags |= 1
                if v.x > NDC_MAX + tol: flags |= 2
                if v.y < NDC_MIN - tol: flags |= 4
                if v.y > NDC_MAX + tol: flags |= 8
                if v.z < NDC_MIN - tol: flags |= 16
                if v.z > NDC_MAX + tol: flags |= 32
                return flags

            if _out_flags(v0) & _out_flags(v1):
                n_removed += 1
                continue

        idx = len(kept_vertices)
        kept_vertices.append(Vertex(coords=v0.coords.copy()))
        kept_vertices.append(Vertex(coords=v1.coords.copy()))
        kept_edges.append(Edge(idx, idx + 1))

    filtered = Mesh(
        name=f"ndc_filtered_{mesh.name}",
        vertices=kept_vertices,
        edges=kept_edges,
    )
    return filtered, n_removed


# ── Y-flip helper ─────────────────────────────────────────────────────────────

def _flip_y(mesh: Mesh) -> Mesh:
    """
    Flip the Y-axis of every vertex in *mesh*.

    Converts from mathematical NDC (Y-up) to screen/raster NDC (Y-down,
    top-left origin).  Applied when ``flip_y=True`` in the stage function.
    """
    flipped = []
    for v in mesh.vertices:
        flipped.append(Vertex(coords=np.array([v.x, -v.y, v.z, v.w])))
    return Mesh(
        name=mesh.name,
        vertices=flipped,
        edges=[Edge(e.start, e.end) for e in mesh.edges],
    )


# ── Stage function ────────────────────────────────────────────────────────────

def apply_normalisation(
    state:       PipelineState,
    flip_y:      bool  = False,
    clip_to_ndc: bool  = True,
    tol:         float = 1e-6,
) -> PipelineState:
    """
    Pipeline Stage 3b: Validate and normalise the NDC mesh.

    Parameters
    ----------
    state       : PipelineState  – state after Stage 3 (projection)
    flip_y      : bool  – flip Y-axis for top-left screen coordinate systems
                          (default False — mathematical Y-up convention)
    clip_to_ndc : bool  – remove trivially-outside edges before clipping stage
                          (default True)
    tol         : float – NDC boundary tolerance (default 1e-6)

    Returns
    -------
    PipelineState with stage ``"3b_normalisation"`` added.

    Notes
    -----
    - Vertices outside [-1,1]³ are NOT deleted individually; edges where
      both endpoints share an outside region are removed.
    - The clipping stage (Stage 4) handles any remaining partial overlaps.
    """
    ndc_mesh = state.latest()

    # Optional Y-flip for screen-space conventions
    if flip_y:
        ndc_mesh = _flip_y(ndc_mesh)

    # Optional early cull of trivially-outside edges
    if clip_to_ndc:
        ndc_mesh, _ = filter_ndc_mesh(ndc_mesh, tol=tol)

    ndc_mesh = Mesh(
        name="normalised_ndc",
        vertices=[Vertex(coords=v.coords.copy()) for v in ndc_mesh.vertices],
        edges=[Edge(e.start, e.end) for e in ndc_mesh.edges],
    )

    state.add_stage(STAGE_KEY, ndc_mesh)
    return state

def apply_normalisation(state, near=0.1, far=100.0,
                        flip_y=False, clip_to_ndc=True):
    import numpy as np
    from ..core.data_structures import Mesh, Vertex, Edge
    mesh = state.latest()
    new_verts = []
    for v in mesh.vertices:
        w = v.w if abs(v.w) > 1e-10 else 1e-10
        x_n = np.clip(v.x / w, -1.0, 1.0) if clip_to_ndc else v.x / w
        y_n = np.clip(v.y / w, -1.0, 1.0) if clip_to_ndc else v.y / w
        y_n = -y_n if flip_y else y_n
        z_n = np.clip(v.z / w, -1.0, 1.0) if clip_to_ndc else v.z / w
        new_verts.append(Vertex(coords=np.array([x_n, y_n, z_n, 1.0])))
    ndc_mesh = Mesh(name="ndc_normalised", vertices=new_verts,
                    edges=[Edge(e.start, e.end) for e in mesh.edges])
    state.add_stage("3b_normalisation", ndc_mesh)
    return state