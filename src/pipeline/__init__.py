"""
src/pipeline/__init__.py
=========================
Public API for the CGV graphics pipeline.

The full pipeline runs 5 ordered stages:

    Stage 1  model_stage        Model Space  → World Space
    Stage 2  view_stage         World Space  → View/Camera Space
    Stage 3  projection_stage   View Space   → NDC  (perspective or orthographic)
    Stage 3b normalization_stage NDC validation + early cull
    Stage 4  clipping_stage     NDC          → Clipped NDC
    Stage 5  viewport_stage     NDC          → Screen Pixel Coordinates

Typical usage
-------------
    from src.pipeline import (
        apply_model_stage,
        apply_view_stage,
        apply_projection,
        apply_normalisation,
        apply_clipping,
        apply_viewport,
        Viewport,
    )
    from src.core.data_structures  import PipelineState
    from src.core.matrix           import Transform
    from src.core.transformations  import Camera

    state  = PipelineState(original=mesh)
    state  = apply_model_stage(state, Transform.identity())
    state  = apply_view_stage(state, Camera(eye=[0,3,8], at=[0,0,0]))
    state  = apply_projection(state, mode="perspective", fov_deg=60, aspect=16/9)
    state  = apply_normalisation(state)
    state  = apply_clipping(state)
    state  = apply_viewport(state, Viewport(width=800, height=600))

    screen_mesh = state.stages["5_viewport"]
"""

from .model_stage         import apply_model_stage
from .view_stage          import apply_view_stage
from .projection_stage    import (
    apply_projection,
    apply_perspective_projection,
    apply_orthographic_projection,
)
from .normalization_stage import apply_normalisation, is_in_ndc, filter_ndc_mesh
from .clipping_stage      import apply_clipping
from .viewport_stage      import apply_viewport, Viewport
from .projection          import (
    PerspectiveProjection,
    OrthographicProjection,
    perspective_divide,
)

__all__ = [
    # Stage functions (in order)
    "apply_model_stage",
    "apply_view_stage",
    "apply_projection",
    "apply_perspective_projection",
    "apply_orthographic_projection",
    "apply_normalisation",
    "apply_clipping",
    "apply_viewport",
    # Supporting classes
    "Viewport",
    "PerspectiveProjection",
    "OrthographicProjection",
    # Helpers
    "perspective_divide",
    "is_in_ndc",
    "filter_ndc_mesh",
]
