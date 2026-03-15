"""
src/algorithms/__init__.py
===========================
Public API for the CGV Pipeline algorithms package.

This package contains three core 2-D graphics algorithms used in the
clipping and rasterisation stages of the pipeline:

Modules
-------
cohen_sutherland
    Cohen-Sutherland line-clipping algorithm.
    Region-code based; fast for trivial accept/reject cases.

liang_barsky
    Liang-Barsky parametric line-clipping algorithm.
    Fewer intersection computations than Cohen-Sutherland.

line_drawing
    Digital Differential Analyzer (DDA) and Bresenham's line
    rasterisation algorithms.

Typical Usage
-------------
    from src.algorithms import (
        ClipWindow,
        CohenSutherlandClipper,
        LiangBarskyClipper,
        cohen_sutherland_clip,
        liang_barsky_clip,
        bresenham_line,
        dda_line,
        Rasterizer,
    )
"""

from .cohen_sutherland import (
    ClipWindow,
    CohenSutherlandClipper,
    cohen_sutherland_clip,
    INSIDE, LEFT, RIGHT, BOTTOM, TOP,
)

from .liang_barsky import (
    LiangBarskyClipper,
    liang_barsky_clip,
)

from .line_drawing import (
    dda_line,
    bresenham_line,
    Rasterizer,
)

__all__ = [
    # Clipping window (shared by both clipping algorithms)
    "ClipWindow",
    # Cohen-Sutherland
    "CohenSutherlandClipper",
    "cohen_sutherland_clip",
    "INSIDE", "LEFT", "RIGHT", "BOTTOM", "TOP",
    # Liang-Barsky
    "LiangBarskyClipper",
    "liang_barsky_clip",
    # Line drawing / rasterisation
    "dda_line",
    "bresenham_line",
    "Rasterizer",
]
