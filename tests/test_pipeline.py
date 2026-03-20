import unittest

from src.core.data_structures import PipelineState
from src.core.matrix import Transform
from src.core.transformations import Camera
from src.input.sample_objects import make_unit_square
from src.pipeline.model_stage import apply_model_stage
from src.pipeline.view_stage import apply_view_stage
from src.pipeline.projection_stage import apply_projection
from src.pipeline.normalization_stage import apply_normalisation
from src.pipeline.clipping_stage import apply_clipping
from src.pipeline.viewport_stage import Viewport, apply_viewport


class TestPipelineIntegration(unittest.TestCase):
    def test_full_pipeline_2d_square(self):
        mesh = make_unit_square()
        state = PipelineState(original=mesh)

        model_matrix = Transform.identity()
        camera = Camera(eye=[0, 0, 5], at=[0, 0, 0], up=[0, 1, 0])
        viewport = Viewport(width=100, height=80, x0=0, y0=0)

        state = apply_model_stage(state, model_matrix)
        state = apply_view_stage(state, camera)
        state = apply_projection(
            state,
            mode="orthographic",
            left=-1.0, right=1.0,
            bottom=-1.0, top=1.0,
            near=0.1, far=100.0,
        )
        state = apply_normalisation(state, flip_y=False, clip_to_ndc=True)
        state = apply_clipping(state, algorithm="cohen_sutherland")
        state = apply_viewport(state, viewport)

        self.assertEqual(
            state.stage_names(),
            [
                "1_model_to_world",
                "2_world_to_view",
                "3_projection",
                "3b_normalisation",
                "4_clipping",
                "5_viewport",
            ],
        )

        ndc_mesh = state.get_stage("3b_normalisation")
        for v in ndc_mesh.vertices:
            self.assertTrue(-1.0 <= v.x <= 1.0)
            self.assertTrue(-1.0 <= v.y <= 1.0)
            self.assertTrue(-1.0 <= v.z <= 1.0)

        screen_mesh = state.get_stage("5_viewport")
        for v in screen_mesh.vertices:
            self.assertTrue(0.0 <= v.x <= viewport.width - 1)
            self.assertTrue(0.0 <= v.y <= viewport.height - 1)


if __name__ == "__main__":
    unittest.main()
