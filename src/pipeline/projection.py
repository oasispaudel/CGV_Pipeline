"""
src/pipeline/projection.py
===========================
3D → 2D Projection matrices and helpers.

Two classical projection types are provided:

1. **Perspective Projection**
   Simulates how a real camera/eye works — objects farther away appear
   smaller.  Built from the vertical field-of-view, the aspect ratio, and
   near/far clip planes.  Result lives in Normalised Device Coordinates (NDC)
   after the perspective divide  (x/w, y/w, z/w).

2. **Orthographic Projection**
   No foreshortening — parallel lines stay parallel.  Useful for technical /
   engineering views.  Maps the view-space box [l,r]×[b,t]×[n,f] linearly
   into the NDC cube [-1,1]³.

Both return 4×4 homogeneous matrices that are applied with
``Mesh.copy_with_matrix()``.  After the matrix multiply the vertices are
still in *clip space*; call ``perspective_divide()`` to obtain NDC.

Classes
-------
PerspectiveProjection   – fov / aspect / near / far constructor
OrthographicProjection  – left / right / bottom / top / near / far constructor

Functions
---------
perspective_divide(mesh)  – w-divide: converts clip-space Mesh → NDC Mesh
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..core.data_structures import Mesh, Vertex, Edge


# ── Perspective Projection ────────────────────────────────────────────────────

@dataclass
class PerspectiveProjection:
    """
    Perspective projection matrix  (view space → clip space).

    Parameters
    ----------
    fov_deg : float
        Vertical field of view in **degrees** (e.g. 60).  Must be in (0, 180).
    aspect  : float
        Viewport width / height  (e.g. 16/9 = 1.777…).
    near    : float
        Distance to the near clip plane.  Must be > 0.
    far     : float
        Distance to the far clip plane.  Must be > near.

    Matrix derivation
    -----------------
    Starting from the symmetric frustum with half-height  h = near·tan(fov/2)
    and half-width  w = h·aspect:

        f = far,  n = near,  t = h,  r = w

        ┌  n/r   0       0          0      ┐
        │   0   n/t      0          0      │
        │   0    0   -(f+n)/(f-n)  -2fn/(f-n) │
        └   0    0      -1          0      ┘

    The -1 in row 3, col 2 copies  -z_view  into w_clip, which is what
    makes closer objects appear larger after the perspective divide.

    Example
    -------
    >>> proj = PerspectiveProjection(fov_deg=60, aspect=16/9, near=0.1, far=100)
    >>> P = proj.matrix()          # 4×4 numpy array
    >>> clip_mesh = view_mesh.copy_with_matrix(P)
    >>> ndc_mesh  = perspective_divide(clip_mesh)
    """
    fov_deg: float = 60.0
    aspect:  float = 16.0 / 9.0
    near:    float = 0.1
    far:     float = 100.0

    def __post_init__(self):
        if not (0 < self.fov_deg < 180):
            raise ValueError(f"fov_deg must be in (0, 180), got {self.fov_deg}.")
        if self.near <= 0:
            raise ValueError(f"near must be > 0, got {self.near}.")
        if self.far <= self.near:
            raise ValueError(f"far ({self.far}) must be > near ({self.near}).")
        if self.aspect <= 0:
            raise ValueError(f"aspect must be > 0, got {self.aspect}.")

    def matrix(self) -> np.ndarray:
        """
        Build and return the 4×4 perspective projection matrix.

        Returns
        -------
        np.ndarray  shape (4, 4), dtype float
        """
        f   = self.far
        n   = self.near
        t   = n * np.tan(np.radians(self.fov_deg) / 2.0)   # half-height at near
        r   = t * self.aspect                                 # half-width  at near

        return np.array([
            [n / r,    0,           0,              0         ],
            [0,        n / t,       0,              0         ],
            [0,        0,  -(f + n) / (f - n),  -2*f*n / (f - n)],
            [0,        0,          -1,              0         ],
        ], dtype=float)

    def frustum_bounds(self):
        """
        Return (left, right, bottom, top, near, far) of the view frustum
        at the near plane — useful for debugging.
        """
        t = self.near * np.tan(np.radians(self.fov_deg) / 2.0)
        r = t * self.aspect
        return -r, r, -t, t, self.near, self.far

    def __repr__(self):
        return (f"PerspectiveProjection("
                f"fov={self.fov_deg}°, aspect={self.aspect:.3f}, "
                f"near={self.near}, far={self.far})")


# ── Orthographic Projection ───────────────────────────────────────────────────

@dataclass
class OrthographicProjection:
    """
    Orthographic projection matrix  (view space → NDC).

    Maps the axis-aligned box  [left, right] × [bottom, top] × [-near, -far]
    uniformly to the NDC cube  [-1, 1]³.

    Parameters
    ----------
    left, right  : float  – horizontal extent of the view volume
    bottom, top  : float  – vertical extent
    near, far    : float  – depth extent (both positive; near < far)

    Matrix derivation
    -----------------
    Scale each axis to [-1,1] then translate to centre:

        sx = 2/(r-l),  tx = -(r+l)/(r-l)
        sy = 2/(t-b),  ty = -(t+b)/(t-b)
        sz = -2/(f-n), tz = -(f+n)/(f-n)

        ┌ sx   0   0   tx ┐
        │  0  sy   0   ty │
        │  0   0  sz   tz │
        └  0   0   0    1 ┘

    Note: sz is negative because OpenGL/standard NDC convention maps
    the near plane to -1 and far to +1, while view space has
    z increasing toward the viewer (camera looks down -Z).

    Example
    -------
    >>> proj = OrthographicProjection(-5, 5, -5, 5, 0.1, 50)
    >>> P = proj.matrix()
    >>> ndc_mesh = view_mesh.copy_with_matrix(P)
    """
    left:   float = -1.0
    right:  float =  1.0
    bottom: float = -1.0
    top:    float =  1.0
    near:   float =  0.1
    far:    float = 100.0

    def __post_init__(self):
        if self.left >= self.right:
            raise ValueError(f"left ({self.left}) must be < right ({self.right}).")
        if self.bottom >= self.top:
            raise ValueError(f"bottom ({self.bottom}) must be < top ({self.top}).")
        if self.near <= 0:
            raise ValueError(f"near must be > 0, got {self.near}.")
        if self.far <= self.near:
            raise ValueError(f"far ({self.far}) must be > near ({self.near}).")

    def matrix(self) -> np.ndarray:
        """
        Build and return the 4×4 orthographic projection matrix.

        Returns
        -------
        np.ndarray  shape (4, 4), dtype float
        """
        l, r = self.left,   self.right
        b, t = self.bottom, self.top
        n, f = self.near,   self.far

        sx = 2.0 / (r - l);   tx = -(r + l) / (r - l)
        sy = 2.0 / (t - b);   ty = -(t + b) / (t - b)
        sz = -2.0 / (f - n);  tz = -(f + n) / (f - n)

        return np.array([
            [sx,  0,   0,  tx],
            [ 0, sy,   0,  ty],
            [ 0,  0,  sz,  tz],
            [ 0,  0,   0,   1],
        ], dtype=float)

    def __repr__(self):
        return (f"OrthographicProjection("
                f"x=[{self.left},{self.right}], "
                f"y=[{self.bottom},{self.top}], "
                f"near={self.near}, far={self.far})")


# ── Perspective divide ────────────────────────────────────────────────────────

def perspective_divide(mesh: Mesh, name: Optional[str] = None) -> Mesh:
    """
    Perform the perspective divide on every vertex of *mesh*.

    After multiplying by a perspective matrix the vertices are in *clip space*
    where  w ≠ 1.  Dividing by w converts them to NDC  (x/w, y/w, z/w, 1).

    For orthographic projections w is already 1, so this is a no-op.

    Parameters
    ----------
    mesh : Mesh  – clip-space mesh (output of ``copy_with_matrix(P)``)
    name : str, optional  – name for the result mesh

    Returns
    -------
    Mesh  – new Mesh in NDC with all w = 1

    Raises
    ------
    ZeroDivisionError  – if any vertex has w == 0 (degenerate geometry)
    """
    ndc_vertices = []
    for v in mesh.vertices:
        w = v.w
        if abs(w) < 1e-12:
            raise ZeroDivisionError(
                f"Perspective divide by zero: vertex {v} has w={w}.\n"
                f"  Hint: the vertex may be at or behind the camera."
            )
        x_ndc = v.x / w
        y_ndc = v.y / w
        z_ndc = v.z / w
        ndc_vertices.append(Vertex(coords=np.array([x_ndc, y_ndc, z_ndc, 1.0])))

    new_edges = [Edge(e.start, e.end) for e in mesh.edges]
    new_name  = name or f"ndc_{mesh.name}"

    return Mesh(name=new_name, vertices=ndc_vertices, edges=new_edges)
