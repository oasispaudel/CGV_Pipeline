"""
src/algorithms/line_drawing.py
================================
Rasterisation algorithms that convert a continuous line segment into a
sequence of integer pixel coordinates.

Algorithms implemented:
    1. Digital Differential Analyzer (DDA)
       Simple incremental algorithm using floating-point arithmetic.
       Straightforward but accumulates rounding errors over long lines.

    2. Bresenham's Line Algorithm
       Integer-only mid-point decision algorithm.  Faster and more
       accurate than DDA for software rasterisers; the standard choice
       in embedded or fixed-point contexts.

Both algorithms:
  - Accept two endpoints in *any* order (dx or dy can be negative).
  - Handle all eight octants.
  - Return a list of (x, y) integer pixel coordinates ordered from
    the start endpoint to the end endpoint.

Classes:
    Rasterizer  – Convenience wrapper that dispatches to either algorithm.

Functions:
    dda_line(x0, y0, x1, y1)          -> List[Tuple[int, int]]
    bresenham_line(x0, y0, x1, y1)    -> List[Tuple[int, int]]
"""

from __future__ import annotations

import math
from typing import List, Tuple


# ── DDA ───────────────────────────────────────────────────────────────────────

def dda_line(
    x0: int | float,
    y0: int | float,
    x1: int | float,
    y1: int | float,
) -> List[Tuple[int, int]]:
    """
    Digital Differential Analyzer (DDA) line rasterisation.

    Algorithm
    ---------
    1. Compute dx = x1 - x0, dy = y1 - y0.
    2. steps = max(|dx|, |dy|)   – number of increments needed.
    3. x_inc = dx / steps, y_inc = dy / steps  – per-step increment.
    4. Walk from (x0, y0) for *steps* iterations, rounding each position
       to the nearest integer pixel.

    Time complexity : O(max(|dx|, |dy|))
    Space complexity: O(max(|dx|, |dy|)) for the output list.

    Parameters
    ----------
    x0, y0 : int or float  – start pixel
    x1, y1 : int or float  – end pixel

    Returns
    -------
    List[Tuple[int, int]]  – pixel coordinates from (x0,y0) to (x1,y1)

    Examples
    --------
    >>> dda_line(0, 0, 4, 2)
    [(0, 0), (1, 0), (2, 1), (3, 1), (4, 2)]
    >>> dda_line(0, 0, 0, 3)    # vertical line
    [(0, 0), (0, 1), (0, 2), (0, 3)]
    """
    x0, y0 = float(x0), float(y0)
    x1, y1 = float(x1), float(y1)

    dx = x1 - x0
    dy = y1 - y0

    steps = int(max(abs(dx), abs(dy)))

    if steps == 0:
        return [(round(x0), round(y0))]

    x_inc = dx / steps
    y_inc = dy / steps

    pixels: List[Tuple[int, int]] = []
    x, y = x0, y0

    for _ in range(steps + 1):
        pixels.append((round(x), round(y)))
        x += x_inc
        y += y_inc

    return pixels


# ── Bresenham ─────────────────────────────────────────────────────────────────

def bresenham_line(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
) -> List[Tuple[int, int]]:
    """
    Bresenham's line algorithm (integer-only mid-point decision).

    Algorithm
    ---------
    The algorithm works in the first octant (0 ≤ slope ≤ 1, left-to-right)
    and uses axis-swapping + mirroring to handle all eight octants:

    1. Compute dx = |x1-x0|, dy = |y1-y0|.
    2. If dy > dx, swap x and y roles (steep flag = True).
    3. Ensure we always move left → right (swap endpoints if needed).
    4. Initialise decision parameter  D = 2*dy - dx.
    5. For each step in x:
       - Plot (x, y) [or (y, x) if steep].
       - Increment x.
       - If D > 0: increment y, D += 2*(dy - dx).
       - Else:     D += 2*dy.

    Time complexity : O(max(|dx|, |dy|))
    Uses only integer addition — no multiplication or division in the inner loop.

    Parameters
    ----------
    x0, y0 : int  – start pixel (truncated to int if float is passed)
    x1, y1 : int  – end pixel

    Returns
    -------
    List[Tuple[int, int]]  – pixel coordinates ordered from (x0,y0) to (x1,y1)

    Examples
    --------
    >>> bresenham_line(0, 0, 5, 3)
    [(0, 0), (1, 1), (2, 1), (3, 2), (4, 2), (5, 3)]
    >>> bresenham_line(0, 0, 0, 4)    # vertical line
    [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]
    """
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    # Detect steep slope (> 45°) and swap axes so we always step along
    # the longer axis — keeps the line connected.
    steep = dy > dx
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        dx, dy = dy, dx

    # Always draw left → right
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    y_step = 1 if y0 < y1 else -1

    D = 2 * dy - dx
    y = y0

    pixels: List[Tuple[int, int]] = []

    for x in range(x0, x1 + 1):
        pixels.append((y, x) if steep else (x, y))

        if D > 0:
            y += y_step
            D += 2 * (dy - dx)
        else:
            D += 2 * dy

    return pixels


# ── Rasterizer class ──────────────────────────────────────────────────────────

class Rasterizer:
    """
    Convenience wrapper around DDA and Bresenham.

    Provides a unified interface for the rest of the pipeline and
    visualization code to rasterize line segments.

    Parameters
    ----------
    algorithm : str
        ``"bresenham"`` (default) or ``"dda"``.

    Example
    -------
    >>> r = Rasterizer("bresenham")
    >>> pixels = r.rasterize(0, 0, 10, 5)
    """

    ALGORITHMS = {"bresenham", "dda"}

    def __init__(self, algorithm: str = "bresenham"):
        algorithm = algorithm.lower()
        if algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Choose from {self.ALGORITHMS}."
            )
        self.algorithm = algorithm

    # ── public ────────────────────────────────────────────────────────────────

    def rasterize(
        self,
        x0: int | float,
        y0: int | float,
        x1: int | float,
        y1: int | float,
    ) -> List[Tuple[int, int]]:
        """
        Rasterize the segment (x0, y0)→(x1, y1) using the configured algorithm.

        Parameters
        ----------
        x0, y0, x1, y1 : int or float  – segment endpoints (rounded for Bresenham)

        Returns
        -------
        List[Tuple[int, int]]  – ordered pixel coordinates
        """
        if self.algorithm == "dda":
            return dda_line(x0, y0, x1, y1)
        else:
            return bresenham_line(int(round(x0)), int(round(y0)),
                                  int(round(x1)), int(round(y1)))

    def rasterize_all(
        self,
        segments: List[Tuple[float, float, float, float]],
    ) -> List[List[Tuple[int, int]]]:
        """
        Rasterize a list of segments.

        Parameters
        ----------
        segments : list of (x0, y0, x1, y1) tuples

        Returns
        -------
        list of pixel-coordinate lists, one per segment
        """
        return [self.rasterize(x0, y0, x1, y1) for x0, y0, x1, y1 in segments]

    def pixel_count(
        self,
        x0: int | float,
        y0: int | float,
        x1: int | float,
        y1: int | float,
    ) -> int:
        """Return the number of pixels that would be plotted for a segment."""
        return max(abs(int(x1) - int(x0)), abs(int(y1) - int(y0))) + 1

    def __repr__(self):
        return f"Rasterizer(algorithm='{self.algorithm}')"
