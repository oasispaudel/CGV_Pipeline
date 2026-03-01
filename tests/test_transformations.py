"""
tests/test_transformations.py
==============================
Unit tests for Person 1's core modules:
  - src/core/data_structures.py
  - src/core/matrix.py
  - src/core/transformations.py
  - src/input/sample_objects.py

Run with:
  python -m unittest tests.test_transformations -v
  or (if pytest installed):
  pytest tests/test_transformations.py -v
"""

import sys
import os
import unittest
import numpy as np

# Allow running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.core.data_structures import Vertex, Edge, Mesh, PipelineState
from src.core.matrix import Transform
from src.core.transformations import Camera, apply_model_to_world, apply_world_to_view
from src.input.sample_objects import (
    make_unit_square, make_unit_cube, make_triangle_2d,
    make_triangle_3d, make_house_2d, make_pyramid_3d, load_sample
)

TOL = 1e-9


# ──────────────────────────────────────────────────────────────────────────────
# Vertex
# ──────────────────────────────────────────────────────────────────────────────

class TestVertex(unittest.TestCase):

    def test_2d_w_is_one(self):
        self.assertEqual(Vertex.from_2d(3, 4).w, 1.0)

    def test_2d_z_is_zero(self):
        self.assertEqual(Vertex.from_2d(5, 6).z, 0.0)

    def test_3d_xyz(self):
        v = Vertex.from_3d(1, 2, 3)
        self.assertEqual(v.x, 1); self.assertEqual(v.y, 2); self.assertEqual(v.z, 3)

    def test_to_cartesian(self):
        v = Vertex.from_3d(2, 4, 6)
        self.assertTrue(np.allclose(v.to_cartesian(), [2, 4, 6]))


# ──────────────────────────────────────────────────────────────────────────────
# Mesh
# ──────────────────────────────────────────────────────────────────────────────

class TestMesh(unittest.TestCase):

    def test_square_has_4_vertices(self):
        self.assertEqual(len(make_unit_square().vertices), 4)

    def test_square_is_closed_4_edges(self):
        self.assertEqual(len(make_unit_square().edges), 4)

    def test_cube_has_8_vertices(self):
        self.assertEqual(len(make_unit_cube().vertices), 8)

    def test_cube_has_12_edges(self):
        self.assertEqual(len(make_unit_cube().edges), 12)

    def test_vertex_matrix_shape(self):
        # 4 homogeneous rows × N vertices
        self.assertEqual(make_unit_square().vertex_matrix().shape, (4, 4))
        self.assertEqual(make_unit_cube().vertex_matrix().shape, (4, 8))

    def test_copy_with_identity_unchanged(self):
        sq = make_unit_square()
        sq2 = sq.copy_with_matrix(Transform.identity())
        orig = sq.vertex_matrix()
        copy = sq2.vertex_matrix()
        self.assertTrue(np.allclose(orig, copy, atol=TOL))

    def test_copy_preserves_edge_count(self):
        sq = make_unit_square()
        sq2 = sq.copy_with_matrix(Transform.translate(1, 1))
        self.assertEqual(len(sq2.edges), len(sq.edges))

    def test_copy_does_not_mutate_original(self):
        sq = make_unit_square()
        orig_x = sq.vertices[0].x
        sq.copy_with_matrix(Transform.translate(99, 0))
        self.assertEqual(sq.vertices[0].x, orig_x)


# ──────────────────────────────────────────────────────────────────────────────
# PipelineState
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineState(unittest.TestCase):

    def test_latest_returns_original_before_stages(self):
        sq = make_unit_square()
        state = PipelineState(original=sq)
        self.assertIs(state.latest(), sq)

    def test_add_stage_and_retrieve(self):
        sq = make_unit_square()
        state = PipelineState(original=sq)
        new_mesh = sq.copy_with_matrix(Transform.identity())
        state.add_stage("test_stage", new_mesh)
        self.assertIn("test_stage", state.stages)
        self.assertIs(state.latest(), new_mesh)


# ──────────────────────────────────────────────────────────────────────────────
# Transform – Translation
# ──────────────────────────────────────────────────────────────────────────────

class TestTranslation(unittest.TestCase):

    def test_translates_2d_point(self):
        T = Transform.translate(3, 4)
        result = T @ np.array([0, 0, 0, 1], dtype=float)
        self.assertTrue(np.allclose(result[:2], [3, 4], atol=TOL))

    def test_translates_3d_point(self):
        T = Transform.translate(1, 2, 3)
        result = T @ np.array([0, 0, 0, 1], dtype=float)
        self.assertTrue(np.allclose(result[:3], [1, 2, 3], atol=TOL))

    def test_identity_no_movement(self):
        p = np.array([5, 7, 3, 1], dtype=float)
        self.assertTrue(np.allclose(Transform.identity() @ p, p, atol=TOL))

    def test_inverse_translation_cancels(self):
        T  = Transform.translate(3, 4, 5)
        Ti = Transform.translate(-3, -4, -5)
        self.assertTrue(np.allclose(Ti @ T, np.eye(4), atol=TOL))


# ──────────────────────────────────────────────────────────────────────────────
# Transform – Scaling
# ──────────────────────────────────────────────────────────────────────────────

class TestScaling(unittest.TestCase):

    def test_uniform_scale_doubles(self):
        S = Transform.scale(2, 2, 2)
        p = np.array([1, 1, 1, 1], dtype=float)
        self.assertTrue(np.allclose((S @ p)[:3], [2, 2, 2], atol=TOL))

    def test_non_uniform_scale(self):
        S = Transform.scale(3, 0.5, 1)
        p = np.array([2, 4, 1, 1], dtype=float)
        self.assertTrue(np.allclose((S @ p)[:3], [6, 2, 1], atol=TOL))

    def test_fixed_point_pivot_does_not_move(self):
        px, py = 3.0, 3.0
        S = Transform.scale_about_point(2, 2, 1, px, py, 0)
        pivot = np.array([px, py, 0, 1], dtype=float)
        self.assertTrue(np.allclose((S @ pivot)[:2], [px, py], atol=TOL))

    def test_fixed_point_other_point_moves(self):
        S = Transform.scale_about_point(2, 2, 1, 0, 0)
        p = np.array([1, 1, 0, 1], dtype=float)
        self.assertTrue(np.allclose((S @ p)[:2], [2, 2], atol=TOL))


# ──────────────────────────────────────────────────────────────────────────────
# Transform – 2D Rotation
# ──────────────────────────────────────────────────────────────────────────────

class TestRotation2D(unittest.TestCase):

    def test_rotate_90_ccw(self):
        # (1,0) rotated 90° CCW → (0,1)
        R = Transform.rotate_z(90)
        result = R @ np.array([1, 0, 0, 1], dtype=float)
        self.assertTrue(np.allclose(result[:2], [0, 1], atol=TOL))

    def test_rotate_180(self):
        R = Transform.rotate_z(180)
        result = R @ np.array([1, 0, 0, 1], dtype=float)
        self.assertTrue(np.allclose(result[:2], [-1, 0], atol=TOL))

    def test_rotate_360_is_identity(self):
        self.assertTrue(np.allclose(Transform.rotate_z(360), np.eye(4), atol=TOL))

    def test_rotate_negative_is_cw(self):
        # (0,1) rotated -90° → (1,0)
        R = Transform.rotate_z(-90)
        result = R @ np.array([0, 1, 0, 1], dtype=float)
        self.assertTrue(np.allclose(result[:2], [1, 0], atol=TOL))

    def test_pivot_rotation(self):
        # Rotate (2,1) by 90° around pivot (1,1) → (1,2)
        R = Transform.rotate_about_point_2d(90, 1, 1)
        result = R @ np.array([2, 1, 0, 1], dtype=float)
        self.assertTrue(np.allclose(result[:2], [1, 2], atol=TOL))

    def test_pivot_stays_fixed(self):
        # The pivot point itself should not move
        R = Transform.rotate_about_point_2d(45, 2, 3)
        pivot = np.array([2, 3, 0, 1], dtype=float)
        self.assertTrue(np.allclose((R @ pivot)[:2], [2, 3], atol=TOL))


# ──────────────────────────────────────────────────────────────────────────────
# Transform – 3D Rotation
# ──────────────────────────────────────────────────────────────────────────────

class TestRotation3D(unittest.TestCase):

    def test_rotate_x_90(self):
        # (0,1,0) rotated 90° about X → (0,0,1)
        R = Transform.rotate_x(90)
        result = R @ np.array([0, 1, 0, 1], dtype=float)
        self.assertTrue(np.allclose(result[:3], [0, 0, 1], atol=TOL))

    def test_rotate_y_90(self):
        # (1,0,0) rotated 90° about Y → (0,0,-1)
        R = Transform.rotate_y(90)
        result = R @ np.array([1, 0, 0, 1], dtype=float)
        self.assertTrue(np.allclose(result[:3], [0, 0, -1], atol=TOL))

    def test_rotate_z_same_as_2d(self):
        self.assertTrue(np.allclose(
            Transform.rotate_z(45), Transform.rotate_z(45), atol=TOL
        ))

    def test_rotate_x_360_is_identity(self):
        self.assertTrue(np.allclose(Transform.rotate_x(360), np.eye(4), atol=TOL))


# ──────────────────────────────────────────────────────────────────────────────
# Transform – Reflection
# ──────────────────────────────────────────────────────────────────────────────

class TestReflection(unittest.TestCase):

    def test_reflect_x_negates_y(self):
        p = np.array([3, 4, 0, 1], dtype=float)
        self.assertTrue(np.allclose((Transform.reflect_x() @ p)[:2], [3, -4], atol=TOL))

    def test_reflect_y_negates_x(self):
        p = np.array([3, 4, 0, 1], dtype=float)
        self.assertTrue(np.allclose((Transform.reflect_y() @ p)[:2], [-3, 4], atol=TOL))

    def test_reflect_origin(self):
        p = np.array([3, 4, 0, 1], dtype=float)
        self.assertTrue(np.allclose((Transform.reflect_origin() @ p)[:2], [-3, -4], atol=TOL))

    def test_reflect_across_45_deg_line(self):
        # (1, 0) reflected across y=x → (0, 1)
        R = Transform.reflect_arbitrary_line_2d(45)
        p = np.array([1, 0, 0, 1], dtype=float)
        self.assertTrue(np.allclose((R @ p)[:2], [0, 1], atol=TOL))

    def test_reflect_twice_is_identity(self):
        R = Transform.reflect_x()
        self.assertTrue(np.allclose(R @ R, np.eye(4), atol=TOL))


# ──────────────────────────────────────────────────────────────────────────────
# Transform – Shear
# ──────────────────────────────────────────────────────────────────────────────

class TestShear(unittest.TestCase):

    def test_shear_x_by_y(self):
        # shx=2: x' = x + 2*y, y unchanged
        S = Transform.shear_xy(shx=2, shy=0)
        p = np.array([1, 1, 0, 1], dtype=float)
        self.assertTrue(np.allclose((S @ p)[:2], [3, 1], atol=TOL))

    def test_shear_y_by_x(self):
        # shy=3: y' = y + 3*x, x unchanged
        S = Transform.shear_xy(shx=0, shy=3)
        p = np.array([2, 1, 0, 1], dtype=float)
        self.assertTrue(np.allclose((S @ p)[:2], [2, 7], atol=TOL))

    def test_zero_shear_is_identity(self):
        self.assertTrue(np.allclose(Transform.shear_xy(0, 0), np.eye(4), atol=TOL))


# ──────────────────────────────────────────────────────────────────────────────
# Transform – Compose
# ──────────────────────────────────────────────────────────────────────────────

class TestCompose(unittest.TestCase):

    def test_translate_then_scale(self):
        # Translate (0,0)→(1,0), then scale ×2 → (2,0)
        M = Transform.compose(Transform.translate(1, 0), Transform.scale(2, 2))
        p = np.array([0, 0, 0, 1], dtype=float)
        self.assertTrue(np.allclose((M @ p)[:2], [2, 0], atol=TOL))

    def test_order_matters(self):
        T = Transform.translate(1, 0)
        S = Transform.scale(2, 2)
        p = np.array([1, 0, 0, 1], dtype=float)
        r1 = Transform.compose(T, S) @ p  # translate then scale
        r2 = Transform.compose(S, T) @ p  # scale then translate
        self.assertFalse(np.allclose(r1, r2))

    def test_compose_single_is_same(self):
        T = Transform.translate(3, 4)
        self.assertTrue(np.allclose(Transform.compose(T), T, atol=TOL))

    def test_compose_with_identity(self):
        T = Transform.translate(5, 5)
        self.assertTrue(np.allclose(
            Transform.compose(T, Transform.identity()), T, atol=TOL
        ))


# ──────────────────────────────────────────────────────────────────────────────
# Camera
# ──────────────────────────────────────────────────────────────────────────────

class TestCamera(unittest.TestCase):

    def test_view_matrix_shape(self):
        self.assertEqual(Camera().view_matrix().shape, (4, 4))

    def test_last_row_is_0001(self):
        self.assertTrue(np.allclose(Camera().view_matrix()[3], [0, 0, 0, 1], atol=TOL))

    def test_origin_has_negative_z_in_view(self):
        cam = Camera(eye=[0, 0, 5], at=[0, 0, 0], up=[0, 1, 0])
        view_origin = cam.view_matrix() @ np.array([0, 0, 0, 1], dtype=float)
        self.assertLess(view_origin[2], 0)

    def test_degenerate_eye_equals_at_raises(self):
        with self.assertRaises(ValueError):
            Camera(eye=[1, 1, 1], at=[1, 1, 1]).view_matrix()

    def test_different_eye_positions_give_different_matrices(self):
        V1 = Camera(eye=[0, 0, 5]).view_matrix()
        V2 = Camera(eye=[5, 0, 0]).view_matrix()
        self.assertFalse(np.allclose(V1, V2))


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline Stage Functions
# ──────────────────────────────────────────────────────────────────────────────

class TestPipelineStages(unittest.TestCase):

    def setUp(self):
        self.sq = make_unit_square()
        self.cube = make_unit_cube()

    def test_model_to_world_logged(self):
        state = PipelineState(original=self.sq)
        state = apply_model_to_world(state, Transform.translate(1, 2))
        self.assertIn("1_model_to_world", state.stages)

    def test_world_to_view_logged(self):
        state = PipelineState(original=self.sq)
        state = apply_model_to_world(state, Transform.identity())
        state = apply_world_to_view(state, Camera())
        self.assertIn("2_world_to_view", state.stages)

    def test_translation_shifts_vertices(self):
        state = PipelineState(original=self.sq)
        state = apply_model_to_world(state, Transform.translate(10, 0))
        for v in state.stages["1_model_to_world"].vertices:
            self.assertGreater(v.x, 9)

    def test_two_stages_chained(self):
        state = PipelineState(original=self.cube)
        state = apply_model_to_world(state, Transform.scale(2, 2, 2))
        state = apply_world_to_view(state, Camera(eye=[3, 3, 3]))
        self.assertEqual(len(state.stages), 2)

    def test_stage_mesh_has_same_edge_count(self):
        state = PipelineState(original=self.cube)
        state = apply_model_to_world(state, Transform.rotate_y(45))
        world = state.stages["1_model_to_world"]
        self.assertEqual(len(world.edges), len(self.cube.edges))


# ──────────────────────────────────────────────────────────────────────────────
# Sample Objects
# ──────────────────────────────────────────────────────────────────────────────

class TestSampleObjects(unittest.TestCase):

    def test_load_sample_unit_square(self):
        m = load_sample("unit_square")
        self.assertEqual(m.name, "unit_square")

    def test_load_sample_unknown_raises(self):
        with self.assertRaises(KeyError):
            load_sample("does_not_exist")

    def test_pyramid_has_5_vertices(self):
        self.assertEqual(len(make_pyramid_3d().vertices), 5)

    def test_house_has_5_vertices(self):
        self.assertEqual(len(make_house_2d().vertices), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
