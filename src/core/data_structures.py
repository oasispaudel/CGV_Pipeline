import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


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