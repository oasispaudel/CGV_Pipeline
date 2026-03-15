"""
src/pipeline/model_stage.py
=============================
Pipeline Stage 1 – Model Space → World Space.

Thin stage wrapper that delegates to
``src.core.transformations.apply_model_to_world``.

Importing from here keeps all stage entry-points in the same package.

Functions
---------
apply_model_stage(state, model_matrix) -> PipelineState
"""

from __future__ import annotations

import numpy as np
from ..core.data_structures import PipelineState
from ..core.transformations  import apply_model_to_world


def apply_model_stage(
    state:        PipelineState,
    model_matrix: np.ndarray,
) -> PipelineState:
    """
    Pipeline Stage 1: Model Space → World Space.

    Applies *model_matrix* (a composite translation + rotation + scale built
    with :class:`~src.core.matrix.Transform`) to the original mesh and
    records the result as stage ``"1_model_to_world"``.

    Parameters
    ----------
    state        : PipelineState  – initial state wrapping the source mesh
    model_matrix : np.ndarray     – 4×4 homogeneous model transform

    Returns
    -------
    PipelineState with stage ``"1_model_to_world"`` added.

    Example
    -------
    >>> from src.core.matrix import Transform
    >>> M = Transform.compose(
    ...     Transform.scale(2, 2, 2),
    ...     Transform.rotate_y(30),
    ...     Transform.translate(1, 0, 0),
    ... )
    >>> state = apply_model_stage(state, M)
    """
    return apply_model_to_world(state, model_matrix)
