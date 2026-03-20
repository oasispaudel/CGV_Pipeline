"""
CGV Pipeline demo entry point.

Run as a module from the project root:
  python -m src.main --mode 3d --object unit_cube
  python -m src.main --run full --mode 3d --object unit_cube
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.core.data_structures import Mesh, PipelineState
from src.core.matrix import Transform
from src.core.transformations import Camera
from src.input.sample_objects import SAMPLE_OBJECTS, load_sample
from src.pipeline.model_stage import apply_model_stage
from src.pipeline.view_stage import apply_view_stage
from src.pipeline.projection_stage import apply_projection
from src.pipeline.normalization_stage import apply_normalisation
from src.pipeline.clipping_stage import apply_clipping
from src.pipeline.viewport_stage import Viewport, apply_viewport


def mesh_to_lines_2d(mesh: Mesh) -> List[List[float]]:
    lines = []
    for e in mesh.edges:
        v1 = mesh.vertices[e.start].to_cartesian()
        v2 = mesh.vertices[e.end].to_cartesian()
        lines.append([v1[0], v1[1], v2[0], v2[1]])
    return lines


def mesh_to_lines_3d(mesh: Mesh) -> List[List[float]]:
    lines = []
    for e in mesh.edges:
        v1 = mesh.vertices[e.start].to_cartesian()
        v2 = mesh.vertices[e.end].to_cartesian()
        lines.append([v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]])
    return lines


def is_2d_mesh(mesh: Mesh, tol: float = 1e-9) -> bool:
    for v in mesh.vertices:
        if abs(v.z) > tol:
            return False
    return True


def _plot_2d(ax, title: str, mesh: Mesh) -> None:
    for x1, y1, x2, y2 in mesh_to_lines_2d(mesh):
        ax.plot([x1, x2], [y1, y2], marker="o")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)
    ax.set_aspect("equal", "box")


def _plot_3d(ax, title: str, mesh: Mesh) -> None:
    for x1, y1, z1, x2, y2, z2 in mesh_to_lines_3d(mesh):
        ax.plot([x1, x2], [y1, y2], [z1, z2], marker="o")
    ax.set_title(title)


def build_model_matrix(mode: str) -> np.ndarray:
    if mode == "2d":
        return Transform.compose(
            Transform.translate(1.0, 0.5, 0.0),
            Transform.rotate_z(30),
            Transform.scale(1.5, 1.0, 1.0),
        )
    return Transform.compose(
        Transform.scale(1.5, 1.2, 1.0),
        Transform.rotate_y(30),
        Transform.rotate_x(20),
        Transform.translate(0.5, 0.2, 0.0),
    )


def run_basic_demo(mode: str, object_name: str, show_gui: bool) -> None:
    mesh = load_sample(object_name)

    if mode == "2d" and not is_2d_mesh(mesh):
        raise ValueError(
            f"Object '{object_name}' is not 2D (z != 0). "
            f"Pick a 2D object or use --mode 3d."
        )

    model_matrix = build_model_matrix(mode)
    camera = Camera(eye=[3, 3, 4], at=[0, 0, 0], up=[0, 1, 0])

    state = PipelineState(original=mesh)
    state = apply_model_stage(state, model_matrix)
    state = apply_view_stage(state, camera)

    original = state.original
    world = state.get_stage("1_model_to_world")
    view = state.get_stage("2_world_to_view")

    print("CGV Pipeline Demo (Basic)")
    print(f"  object: {object_name}")
    print(f"  mode:   {mode}")
    print(f"  stages: {', '.join(state.stage_names())}")

    if show_gui:
        if mode == "2d":
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            _plot_2d(axs[0], "Original (Model Space)", original)
            _plot_2d(axs[1], "After Model -> World", world)
            _plot_2d(axs[2], "After World -> View", view)
            plt.tight_layout()
            plt.show()
        else:
            fig = plt.figure(figsize=(12, 4))
            ax1 = fig.add_subplot(131, projection="3d")
            ax2 = fig.add_subplot(132, projection="3d")
            ax3 = fig.add_subplot(133, projection="3d")
            _plot_3d(ax1, "Original (Model Space)", original)
            _plot_3d(ax2, "After Model -> World", world)
            _plot_3d(ax3, "After World -> View", view)
            plt.tight_layout()
            plt.show()
    else:
        print("GUI disabled. Use --no-gui to suppress plots.")
        print("Original vertices (first 4):")
        for v in original.vertices[:4]:
            print(f"  {v}")


def run_full_pipeline(
    mode: str,
    object_name: str,
    projection: str,
    clipper: str,
    viewport_size: Tuple[int, int],
    show_gui: bool,
) -> None:
    mesh = load_sample(object_name)

    if mode == "2d" and not is_2d_mesh(mesh):
        raise ValueError(
            f"Object '{object_name}' is not 2D (z != 0). "
            f"Pick a 2D object or use --mode 3d."
        )

    model_matrix = build_model_matrix(mode)
    camera = Camera(eye=[3, 3, 4], at=[0, 0, 0], up=[0, 1, 0])
    vp_w, vp_h = viewport_size
    viewport = Viewport(width=vp_w, height=vp_h, x0=0, y0=0)

    # Projection defaults: keep 2D in orthographic unless overridden.
    projection = projection.lower().strip()
    if projection == "auto":
        projection = "orthographic" if mode == "2d" else "perspective"

    state = PipelineState(original=mesh)
    state = apply_model_stage(state, model_matrix)
    state = apply_view_stage(state, camera)
    state = apply_projection(state, mode=projection, aspect=viewport.aspect_ratio())
    state = apply_normalisation(state, flip_y=False, clip_to_ndc=True)
    state = apply_clipping(state, algorithm=clipper)
    state = apply_viewport(state, viewport)

    stages = state.stage_names()
    print("CGV Pipeline Demo (Full)")
    print(f"  object:     {object_name}")
    print(f"  mode:       {mode}")
    print(f"  projection: {projection}")
    print(f"  clipper:    {clipper}")
    print(f"  viewport:   {vp_w}x{vp_h}")
    print(f"  stages:     {', '.join(stages)}")

    # Console sample: first few vertices of key stages
    for key in ["1_model_to_world", "2_world_to_view", "3_projection",
                "3b_normalisation", "4_clipping", "5_viewport"]:
        if key in state.stages:
            mesh_k = state.stages[key]
            print(f"\nStage {key} ({mesh_k.name}) sample vertices:")
            for v in mesh_k.vertices[:4]:
                print(f"  {v}")

    if not show_gui:
        print("\nGUI disabled. Use --no-gui to suppress plots.")
        return

    # GUI: 3D for early stages (if 3D), 2D for NDC/viewport stages.
    if mode == "2d":
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        _plot_2d(axs[0, 0], "Original", state.original)
        _plot_2d(axs[0, 1], "Model -> World", state.get_stage("1_model_to_world"))
        _plot_2d(axs[0, 2], "World -> View", state.get_stage("2_world_to_view"))
        _plot_2d(axs[1, 0], "Projection (NDC)", state.get_stage("3_projection"))
        _plot_2d(axs[1, 1], "Clipped (NDC)", state.get_stage("4_clipping"))
        _plot_2d(axs[1, 2], "Viewport (Screen)", state.get_stage("5_viewport"))
        plt.tight_layout()
        plt.show()
        return

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(231, projection="3d")
    ax2 = fig.add_subplot(232, projection="3d")
    ax3 = fig.add_subplot(233, projection="3d")
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    ax6 = fig.add_subplot(236)

    _plot_3d(ax1, "Original", state.original)
    _plot_3d(ax2, "Model -> World", state.get_stage("1_model_to_world"))
    _plot_3d(ax3, "World -> View", state.get_stage("2_world_to_view"))
    _plot_2d(ax4, "Projection (NDC)", state.get_stage("3_projection"))
    _plot_2d(ax5, "Clipped (NDC)", state.get_stage("4_clipping"))
    _plot_2d(ax6, "Viewport (Screen)", state.get_stage("5_viewport"))

    plt.tight_layout()
    plt.show()


def run_presentation_demo(show_gui: bool) -> None:
    demos = [
        dict(
            title="Full 3D (default perspective)",
            mode="3d",
            object_name="unit_cube",
            projection="auto",
            clipper="cohen_sutherland",
            viewport_size=(800, 600),
        ),
        dict(
            title="3D Orthographic + Liang-Barsky",
            mode="3d",
            object_name="unit_cube",
            projection="orthographic",
            clipper="liang_barsky",
            viewport_size=(800, 600),
        ),
        dict(
            title="3D Custom Viewport 1024x768",
            mode="3d",
            object_name="unit_cube",
            projection="auto",
            clipper="cohen_sutherland",
            viewport_size=(1024, 768),
        ),
    ]

    print("CGV Presentation Demo")
    print("Running a sequence of full-pipeline demos.")
    print("Close each window to move to the next demo.\n")

    for step in demos:
        print(f"Demo: {step['title']}")
        run_full_pipeline(
            mode=step["mode"],
            object_name=step["object_name"],
            projection=step["projection"],
            clipper=step["clipper"],
            viewport_size=step["viewport_size"],
            show_gui=show_gui,
        )


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CGV Pipeline demo runner.")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run the built-in presentation demo sequence.",
    )
    parser.add_argument(
        "--run",
        choices=["basic", "full"],
        default="basic",
        help="Run basic (model->view) or full pipeline.",
    )
    parser.add_argument(
        "--mode",
        choices=["2d", "3d"],
        default="3d",
        help="Render in 2D or 3D (default: 3d).",
    )
    parser.add_argument(
        "--object",
        default=None,
        help="Sample object name (use --list-objects to see options).",
    )
    parser.add_argument(
        "--list-objects",
        action="store_true",
        help="List available sample objects and exit.",
    )
    parser.add_argument(
        "--projection",
        choices=["auto", "perspective", "orthographic"],
        default="auto",
        help="Projection mode for full pipeline (default: auto).",
    )
    parser.add_argument(
        "--clipper",
        choices=["cohen_sutherland", "liang_barsky"],
        default="cohen_sutherland",
        help="Clipping algorithm for full pipeline.",
    )
    parser.add_argument(
        "--viewport",
        default="800x600",
        help="Viewport size as WIDTHxHEIGHT (default: 800x600).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without opening Matplotlib windows.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.demo:
        run_presentation_demo(show_gui=not args.no_gui)
        return 0

    if args.list_objects:
        print("Available sample objects:")
        for name in sorted(SAMPLE_OBJECTS.keys()):
            print(f"  - {name}")
        return 0

    try:
        default_object = "unit_square" if args.mode == "2d" else "unit_cube"
        object_name = args.object or default_object

        if args.run == "basic":
            run_basic_demo(args.mode, object_name, show_gui=not args.no_gui)
        else:
            if "x" not in args.viewport:
                raise ValueError("Viewport must be WIDTHxHEIGHT, e.g. 800x600.")
            w_str, h_str = args.viewport.lower().split("x", 1)
            viewport_size = (int(w_str), int(h_str))

            run_full_pipeline(
                args.mode,
                object_name,
                projection=args.projection,
                clipper=args.clipper,
                viewport_size=viewport_size,
                show_gui=not args.no_gui,
            )
    except Exception as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
