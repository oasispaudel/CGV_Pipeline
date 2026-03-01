"""
src/input/sample_objects.py
============================
Predefined sample objects for testing and demonstration.
Person 1 – Pipeline Core

These meshes are used by:
  - demo scripts
  - unit tests
  - Person 3's visualization to show default objects
  - Person 2's projection tests
"""

from ..core.data_structures import Mesh


def make_unit_square() -> Mesh:
    """
    A 1×1 square in the XY plane centred at the origin.
    Vertices: (-0.5,-0.5), (0.5,-0.5), (0.5,0.5), (-0.5,0.5)
    """
    return Mesh.from_2d_points("unit_square", [
        (-0.5, -0.5),
        ( 0.5, -0.5),
        ( 0.5,  0.5),
        (-0.5,  0.5),
    ], closed=True)


def make_unit_cube() -> Mesh:
    """
    A 1×1×1 wireframe cube centred at the origin.
    8 vertices, 12 edges (4 back + 4 front + 4 connecting).
    """
    pts = [
        (-0.5, -0.5, -0.5),  # 0
        ( 0.5, -0.5, -0.5),  # 1
        ( 0.5,  0.5, -0.5),  # 2
        (-0.5,  0.5, -0.5),  # 3  ← back face
        (-0.5, -0.5,  0.5),  # 4
        ( 0.5, -0.5,  0.5),  # 5
        ( 0.5,  0.5,  0.5),  # 6
        (-0.5,  0.5,  0.5),  # 7  ← front face
    ]
    edges = [
        # back face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # front face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # connecting edges
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return Mesh.from_3d_points("unit_cube", pts, edges)


def make_triangle_2d() -> Mesh:
    """An equilateral-ish triangle in the XY plane."""
    return Mesh.from_2d_points("triangle_2d", [
        ( 0.0,  0.5),
        (-0.5, -0.5),
        ( 0.5, -0.5),
    ], closed=True)


def make_triangle_3d() -> Mesh:
    """A triangle in 3D space (z=0 plane)."""
    pts   = [(0.0, 1.0, 0.0), (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0)]
    edges = [(0, 1), (1, 2), (2, 0)]
    return Mesh.from_3d_points("triangle_3d", pts, edges)


def make_house_2d() -> Mesh:
    """
    A simple 2D house shape (pentagon).
    Good for demonstrating transformations and clipping.
    """
    return Mesh.from_2d_points("house_2d", [
        ( 0.0,  1.0),   # roof peak
        ( 1.0,  0.0),   # roof right
        ( 1.0, -1.0),   # bottom right
        (-1.0, -1.0),   # bottom left
        (-1.0,  0.0),   # roof left
    ], closed=True)


def make_pyramid_3d() -> Mesh:
    """A square-base pyramid centred at origin."""
    pts = [
        ( 0.0,  1.0,  0.0),  # 0 apex
        (-0.5, -0.5, -0.5),  # 1 base corners
        ( 0.5, -0.5, -0.5),  # 2
        ( 0.5, -0.5,  0.5),  # 3
        (-0.5, -0.5,  0.5),  # 4
    ]
    edges = [
        # base
        (1, 2), (2, 3), (3, 4), (4, 1),
        # sides
        (0, 1), (0, 2), (0, 3), (0, 4),
    ]
    return Mesh.from_3d_points("pyramid_3d", pts, edges)


# Registry: makes it easy for the user_input module to list & load objects
SAMPLE_OBJECTS = {
    "unit_square" : make_unit_square,
    "unit_cube"   : make_unit_cube,
    "triangle_2d" : make_triangle_2d,
    "triangle_3d" : make_triangle_3d,
    "house_2d"    : make_house_2d,
    "pyramid_3d"  : make_pyramid_3d,
}


def load_sample(name: str) -> Mesh:
    """
    Load a sample object by name.

    Parameters
    ----------
    name : str  – one of the keys in SAMPLE_OBJECTS

    Returns
    -------
    Mesh

    Raises
    ------
    KeyError if name not found.
    """
    if name not in SAMPLE_OBJECTS:
        available = ", ".join(SAMPLE_OBJECTS.keys())
        raise KeyError(f"Unknown sample object '{name}'. Available: {available}")
    return SAMPLE_OBJECTS[name]()
