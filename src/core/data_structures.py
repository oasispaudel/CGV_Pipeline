"""
src/core/data_structures.py
============================
Core data structures for the CGV Pipeline.

Classes
-------
Vertex        – A point in homogeneous coordinates [x, y, z, w].
Edge          – An index pair connecting two vertices inside a Mesh.
Mesh          – A wireframe object (list of vertices + edges).
PipelineState – Tracks a Mesh as it passes through each pipeline stage.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Vertex:
    """A single point in homogeneous coordinates [x, y, z, w]."""
    coords: np.ndarray  # shape (4,)

    @classmethod
    def from_2d(cls, x: float, y: float) -> "Vertex":
        """Create a 2D vertex (z=0, w=1)."""
        return cls(coords=np.array([x, y, 0.0, 1.0], dtype=float))

    @classmethod
    def from_3d(cls, x: float, y: float, z: float) -> "Vertex":
        """Create a 3D vertex (w=1)."""
        return cls(coords=np.array([x, y, z, 1.0], dtype=float))

    @property
    def x(self) -> float: return self.coords[0]
    @property
    def y(self) -> float: return self.coords[1]
    @property
    def z(self) -> float: return self.coords[2]
    @property
    def w(self) -> float: return self.coords[3]

    def to_cartesian(self) -> np.ndarray:
        """Convert from homogeneous to Cartesian by dividing by w."""
        return self.coords[:3] / self.w if self.w != 0 else self.coords[:3]

    def __repr__(self):
        return f"Vertex({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"


@dataclass
class Edge:
    """Connects two vertex indices inside a Mesh."""
    start: int   # index into Mesh.vertices
    end: int

    def __repr__(self):
        return f"Edge({self.start} → {self.end})"


@dataclass
class Mesh:
    """
    A geometric object defined by a list of vertices and edges.
    The pipeline operates on Mesh objects at every stage.
    """
    name: str
    vertices: List[Vertex] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)

    # ── Convenience constructors ──────────────────────────────────────────────

    @classmethod
    def from_2d_points(cls, name: str, points: List[tuple], closed: bool = True) -> "Mesh":
        """
        Build a Mesh from a list of (x, y) tuples.
        If closed=True the last point connects back to the first (polygon).
        """
        vertices = [Vertex.from_2d(x, y) for x, y in points]
        n = len(points)
        edges = [Edge(i, (i + 1) % n if closed else i + 1)
                 for i in range(n - (0 if closed else 1))]
        return cls(name=name, vertices=vertices, edges=edges)

    @classmethod
    def from_3d_points(cls, name: str, points: List[tuple], edges: List[tuple]) -> "Mesh":
        """
        Build a Mesh from (x, y, z) tuples and explicit (start, end) edge pairs.
        """
        vertices = [Vertex.from_3d(*p) for p in points]
        edge_objs = [Edge(s, e) for s, e in edges]
        return cls(name=name, vertices=vertices, edges=edge_objs)

    # ── Matrix operations ─────────────────────────────────────────────────────

    def vertex_matrix(self) -> np.ndarray:
        """
        Pack all vertices into a (4, N) homogeneous matrix.

        Each column is one vertex [x, y, z, w]. This allows a transformation
        to be applied to every vertex at once:

            transformed = M @ mesh.vertex_matrix()

        Returns
        -------
        np.ndarray  shape (4, N)  where N = len(self.vertices)
        """
        return np.stack([v.coords for v in self.vertices], axis=1)

    def copy_with_matrix(self, M: np.ndarray, name: Optional[str] = None) -> "Mesh":
        """
        Apply the 4×4 homogeneous matrix *M* to all vertices and return a
        new Mesh. The original Mesh is **not** mutated.

        Parameters
        ----------
        M    : np.ndarray shape (4, 4)  – transformation matrix
        name : str, optional  – name for the new mesh
                                (defaults to self.name + "_transformed")

        Returns
        -------
        Mesh  – a new Mesh with transformed vertices and copies of all edges.

        Example
        -------
        >>> from src.core.matrix import Transform
        >>> moved = mesh.copy_with_matrix(Transform.translate(1, 2))
        """
        if M.shape != (4, 4):
            raise ValueError(f"Expected a 4×4 matrix, got shape {M.shape}.")

        transformed_cols = M @ self.vertex_matrix()   # (4, N)

        new_vertices = [
            Vertex(coords=transformed_cols[:, i].copy())
            for i in range(transformed_cols.shape[1])
        ]
        new_edges = [Edge(e.start, e.end) for e in self.edges]
        new_name = name if name is not None else f"{self.name}_transformed"

        return Mesh(name=new_name, vertices=new_vertices, edges=new_edges)

    def __repr__(self):
        return (f"Mesh('{self.name}', "
                f"{len(self.vertices)} vertices, "
                f"{len(self.edges)} edges)")


# ── Pipeline state ────────────────────────────────────────────────────────────

@dataclass
class PipelineState:
    """
    Tracks a Mesh as it progresses through the graphics pipeline.

    The original input mesh is stored in ``self.original``.
    Each pipeline stage adds a transformed copy under a named key in
    ``self.stages``.  ``latest()`` always returns the most recently
    added stage mesh (or the original if no stages have run yet).

    Parameters
    ----------
    original : Mesh  – the untransformed source mesh

    Example
    -------
    >>> state = PipelineState(original=make_unit_cube())
    >>> state = apply_model_to_world(state, Transform.scale(2, 2, 2))
    >>> world_mesh = state.stages["1_model_to_world"]
    """

    original: Mesh
    stages:   Dict[str, Mesh] = field(default_factory=dict)

    def add_stage(self, name: str, mesh: Mesh) -> None:
        """
        Record a pipeline stage result.

        Parameters
        ----------
        name : str   – stage identifier (e.g. ``"1_model_to_world"``)
        mesh : Mesh  – the mesh after this stage's transform
        """
        self.stages[name] = mesh

    def latest(self) -> Mesh:
        """
        Return the mesh from the most recently completed stage,
        or the original mesh if no stages have been run yet.

        Returns
        -------
        Mesh
        """
        if not self.stages:
            return self.original
        last_key = list(self.stages.keys())[-1]
        return self.stages[last_key]

    def get_stage(self, name: str) -> Mesh:
        """
        Retrieve a specific stage by name.

        Raises
        ------
        KeyError  – if the stage has not been computed yet.
        """
        if name not in self.stages:
            available = list(self.stages.keys())
            raise KeyError(
                f"Stage '{name}' not found. "
                f"Available stages: {available}"
            )
        return self.stages[name]

    def stage_names(self) -> List[str]:
        """Return the ordered list of completed stage names."""
        return list(self.stages.keys())

    def __repr__(self):
        stage_list = list(self.stages.keys()) or ["(none)"]
        return (f"PipelineState(\n"
                f"  original={self.original!r},\n"
                f"  stages={stage_list}\n)")
