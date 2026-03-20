"""
CGV Pipeline – Main Entry Point
=================================
Run from the project root (CGV_Pipeline/) as a module:

  python -m src.main                                    # asks you everything interactively
  python -m src.main --list-objects                     # see all available shapes
  python -m src.main --mode 3d --object pyramid_3d      # specific shape, basic demo
  python -m src.main --run full --mode 3d --object unit_cube --projection perspective
  python -m src.main --run full --mode 2d --object house_2d
  python -m src.main --demo                             # presentation sequence
  python -m src.main --no-gui --run full --object unit_cube  # headless / no plots
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.core.data_structures import Mesh, PipelineState
from src.core.matrix import Transform
from src.core.transformations import Camera, apply_model_to_world as apply_model_stage
from src.pipeline.view_stage import apply_view_stage
from src.input.sample_objects import SAMPLE_OBJECTS, load_sample
from src.pipeline.projection_stage import apply_projection
from src.pipeline.normalization_stage import apply_normalisation
from src.pipeline.clipping_stage import apply_clipping
from src.pipeline.viewport_stage import Viewport, apply_viewport
# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_2d_mesh(mesh: Mesh, tol: float = 1e-9) -> bool:
    return all(abs(v.z) <= tol for v in mesh.vertices)


def mesh_to_segments_2d(mesh: Mesh) -> List[Tuple]:
    segs = []
    for e in mesh.edges:
        v1 = mesh.vertices[e.start].to_cartesian()
        v2 = mesh.vertices[e.end].to_cartesian()
        segs.append((v1[0], v1[1], v2[0], v2[1]))
    return segs


def mesh_to_segments_3d(mesh: Mesh) -> List[Tuple]:
    segs = []
    for e in mesh.edges:
        v1 = mesh.vertices[e.start].to_cartesian()
        v2 = mesh.vertices[e.end].to_cartesian()
        segs.append((v1[0], v1[1], v1[2], v2[0], v2[1], v2[2]))
    return segs


def _plot_2d(ax, title: str, mesh: Mesh) -> None:
    for x1, y1, x2, y2 in mesh_to_segments_2d(mesh):
        ax.plot([x1, x2], [y1, y2], "b-o", markersize=3, linewidth=1.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.grid(True); ax.set_aspect("equal", "box")


def _plot_3d(ax, title: str, mesh: Mesh) -> None:
    for x1, y1, z1, x2, y2, z2 in mesh_to_segments_3d(mesh):
        ax.plot([x1, x2], [y1, y2], [z1, z2], "b-o", markersize=3, linewidth=1.5)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")


# ─────────────────────────────────────────────────────────────────────────────
# Default transform
# ─────────────────────────────────────────────────────────────────────────────

def build_model_matrix(mode: str) -> np.ndarray:
    if mode == "2d":
        return Transform.compose(
            Transform.scale(1.5, 1.0, 1.0),
            Transform.rotate_z(30),
            Transform.translate(1.0, 0.5, 0.0),
        )
    return Transform.compose(
        Transform.scale(1.5, 1.2, 1.0),
        Transform.rotate_y(30),
        Transform.rotate_x(20),
        Transform.translate(0.5, 0.2, 0.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Interactive shape picker (the key fix)
# ─────────────────────────────────────────────────────────────────────────────

def _ask_object_interactively(mode: str) -> str:
    all_names = sorted(SAMPLE_OBJECTS.keys())

    if mode == "2d":
        options = [n for n in all_names if "2d" in n or "square" in n]
    else:
        options = [n for n in all_names if "3d" in n or "cube" in n or "pyramid" in n]

    if not options:
        options = all_names

    print(f"\nAvailable {mode.upper()} shapes:")
    for i, name in enumerate(options, 1):
        print(f"  [{i}] {name}")
    print(f"  [A] Show ALL {len(all_names)} objects")

    while True:
        choice = input("\nPick a shape number (or A): ").strip()

        if choice.lower() == "a":
            print("\nAll shapes:")
            for i, name in enumerate(all_names, 1):
                obj = SAMPLE_OBJECTS[name]()
                dim = "2D" if is_2d_mesh(obj) else "3D"
                print(f"  [{i}] {name:<20} ({dim})")
            choice = input("Pick a number: ").strip()
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(all_names):
                    return all_names[idx]
            except ValueError:
                pass
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass
        print("  Invalid, try again.")


# ─────────────────────────────────────────────────────────────────────────────
# Basic demo (stages 1 & 2 only)
# ─────────────────────────────────────────────────────────────────────────────

def run_basic_demo(mode: str, object_name: str, show_gui: bool) -> None:
    mesh = load_sample(object_name)
    model_matrix = build_model_matrix(mode)
    camera = Camera(eye=[3, 3, 4], at=[0, 0, 0], up=[0, 1, 0])

    state = PipelineState(original=mesh)
    state = apply_model_stage(state, model_matrix)
    state = apply_view_stage(state, camera)

    print("\n── Basic Pipeline (Stages 1–2) ──────────────────")
    print(f"  Object  : {object_name}")
    print(f"  Mode    : {mode}")
    print(f"  Stages  : {', '.join(state.stage_names())}")
    print(f"  Verts   : {len(mesh.vertices)},  Edges: {len(mesh.edges)}")

    if not show_gui:
        for v in mesh.vertices[:4]:
            print(f"  {v}")
        return

    if mode == "2d":
        fig, axs = plt.subplots(1, 3, figsize=(13, 4))
        fig.suptitle(f"Basic Pipeline – {object_name}", fontsize=12)
        _plot_2d(axs[0], "① Original (Model Space)",      state.original)
        _plot_2d(axs[1], "② After Model → World",          state.get_stage("1_model_to_world"))
        _plot_2d(axs[2], "③ After World → View (Camera)",  state.get_stage("2_world_to_view"))
    else:
        fig = plt.figure(figsize=(13, 4))
        fig.suptitle(f"Basic Pipeline – {object_name}", fontsize=12)
        _plot_3d(fig.add_subplot(131, projection="3d"), "① Original",        state.original)
        _plot_3d(fig.add_subplot(132, projection="3d"), "② Model → World",   state.get_stage("1_model_to_world"))
        _plot_3d(fig.add_subplot(133, projection="3d"), "③ World → View",    state.get_stage("2_world_to_view"))

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline (all 5 stages)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_pipeline(mode: str, object_name: str, projection: str,
                      clipper: str, viewport_size: Tuple[int, int],
                      show_gui: bool) -> None:
    mesh = load_sample(object_name)

    if projection == "auto":
        projection = "orthographic" if mode == "2d" else "perspective"

    model_matrix = build_model_matrix(mode)
    camera   = Camera(eye=[3, 3, 4], at=[0, 0, 0], up=[0, 1, 0])
    vp_w, vp_h = viewport_size
    viewport = Viewport(width=vp_w, height=vp_h, x0=0, y0=0)

    state = PipelineState(original=mesh)
    state = apply_model_stage(state, model_matrix)
    state = apply_view_stage(state, camera)
    state = apply_projection(state, mode=projection,
                                     aspect=viewport.aspect_ratio())
    state = apply_normalisation(state, flip_y=False, clip_to_ndc=True)
    state = apply_clipping(state, algorithm=clipper)
    state = apply_viewport(state, viewport)

    print("\n── Full Pipeline (All Stages) ───────────────────")
    print(f"  Object     : {object_name}")
    print(f"  Mode       : {mode}")
    print(f"  Projection : {projection}")
    print(f"  Clipper    : {clipper}")
    print(f"  Viewport   : {vp_w}×{vp_h}")

    for key, label in [("1_model_to_world", "Model→World"),
                        ("2_world_to_view",  "World→View"),
                        ("3_projection",     "Projection"),
                        ("3b_normalisation", "NDC"),
                        ("4_clipping",       "Clipping"),
                        ("5_viewport",       "Viewport")]:
        if key in state.stages:
            m = state.stages[key]
            print(f"  {label:<14}: {len(m.edges)} edges")

    if not show_gui:
        return

    if mode == "2d":
        fig, axs = plt.subplots(2, 3, figsize=(13, 8))
        fig.suptitle(f"Full Pipeline – {object_name}  ({projection})", fontsize=12)
        _plot_2d(axs[0,0], "① Original",              state.original)
        _plot_2d(axs[0,1], "② Model → World",          state.get_stage("1_model_to_world"))
        _plot_2d(axs[0,2], "③ World → View",           state.get_stage("2_world_to_view"))
        _plot_2d(axs[1,0], "④ Projection / NDC",       state.get_stage("3_projection"))
        _plot_2d(axs[1,1], "⑤ After Clipping",         state.get_stage("4_clipping"))
        _plot_2d(axs[1,2], f"⑥ Viewport {vp_w}×{vp_h}", state.get_stage("5_viewport"))
    else:
        fig = plt.figure(figsize=(13, 8))
        fig.suptitle(f"Full Pipeline – {object_name}  ({projection})", fontsize=12)
        _plot_3d(fig.add_subplot(231, projection="3d"), "① Original",       state.original)
        _plot_3d(fig.add_subplot(232, projection="3d"), "② Model → World",  state.get_stage("1_model_to_world"))
        _plot_3d(fig.add_subplot(233, projection="3d"), "③ World → View",   state.get_stage("2_world_to_view"))
        _plot_2d(fig.add_subplot(234),                  "④ Projection/NDC", state.get_stage("3_projection"))
        _plot_2d(fig.add_subplot(235),                  "⑤ After Clipping", state.get_stage("4_clipping"))
        _plot_2d(fig.add_subplot(236), f"⑥ Viewport {vp_w}×{vp_h}",        state.get_stage("5_viewport"))

    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# Presentation demo
# ─────────────────────────────────────────────────────────────────────────────

def run_presentation_demo(show_gui: bool) -> None:
    demos = [
        dict(mode="3d", object_name="unit_cube",   projection="perspective",
             clipper="cohen_sutherland", viewport_size=(800, 600)),
        dict(mode="3d", object_name="pyramid_3d",  projection="perspective",
             clipper="liang_barsky",    viewport_size=(800, 600)),
        dict(mode="2d", object_name="house_2d",    projection="orthographic",
             clipper="cohen_sutherland", viewport_size=(800, 600)),
        dict(mode="2d", object_name="unit_square", projection="orthographic",
             clipper="liang_barsky",    viewport_size=(800, 600)),
    ]
    print("CGV Presentation Demo – cycling through shapes")
    print("Close each plot window to advance.\n")
    for d in demos:
        print(f"\n→ {d['object_name']}  ({d['mode'].upper()}, {d['projection']})")
        run_full_pipeline(show_gui=show_gui, **d)


# ─────────────────────────────────────────────────────────────────────────────
# Interactive mode
# ─────────────────────────────────────────────────────────────────────────────

def run_interactive() -> None:
    print("\n┌────────────────────────────────────────┐")
    print("│   CGV Pipeline Simulator               │")
    print("│   Computer Graphics & Visualization    │")
    print("└────────────────────────────────────────┘")

    # 1. Mode
    print("\nDimension mode:")
    print("  [1] 2D    [2] 3D")
    while True:
        m = input("Choose (1/2, default 2): ").strip() or "2"
        if m in ("1", "2"):
            mode = "2d" if m == "1" else "3d"
            break

    # 2. Shape – user picks from a list
    object_name = _ask_object_interactively(mode)
    print(f"  ✓ Selected: {object_name}")

    # 3. Pipeline depth
    print("\nPipeline depth:")
    print("  [1] Basic (stages 1–2: transform only)")
    print("  [2] Full  (all 5 stages)")
    depth = input("Choose (1/2, default 2): ").strip() or "2"

    if depth == "1":
        run_basic_demo(mode, object_name, show_gui=True)
        return

    # 4. Projection
    default_proj = "orthographic" if mode == "2d" else "perspective"
    print(f"\nProjection type (default: {default_proj}):")
    print("  [1] Perspective    [2] Orthographic")
    p = input("Choose (Enter = default): ").strip()
    projection = "perspective" if p == "1" else "orthographic" if p == "2" else default_proj

    # 5. Clipper
    print("\nClipping algorithm:")
    print("  [1] Cohen-Sutherland (default)    [2] Liang-Barsky")
    c = input("Choose (Enter = default): ").strip()
    clipper = "liang_barsky" if c == "2" else "cohen_sutherland"

    # 6. Viewport
    vp = input("\nViewport size (default 800x600): ").strip() or "800x600"
    try:
        separator = "x" if "x" in vp.lower() else "*"
        w_str, h_str = vp.lower().split(separator)
        viewport_size = (int(w_str), int(h_str))
    except Exception:
        print("  Bad format, using 800x600.")
        viewport_size = (800, 600)

    run_full_pipeline(mode, object_name, projection, clipper, viewport_size,
                      show_gui=True)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CGV Pipeline Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                                          # interactive
  python -m src.main --list-objects                           # list shapes
  python -m src.main --mode 3d --object pyramid_3d            # basic demo
  python -m src.main --run full --object unit_cube            # full pipeline
  python -m src.main --run full --mode 2d --object house_2d
  python -m src.main --run full --object pyramid_3d --projection orthographic --clipper liang_barsky
  python -m src.main --demo                                   # all shapes
        """
    )
    parser.add_argument("--demo",         action="store_true")
    parser.add_argument("--run",          choices=["basic", "full"], default=None)
    parser.add_argument("--mode",         choices=["2d", "3d"], default="3d")
    parser.add_argument("--object",       default=None,
                        help="Shape name. Run --list-objects to see all.")
    parser.add_argument("--list-objects", action="store_true")
    parser.add_argument("--projection",   choices=["auto", "perspective", "orthographic"],
                        default="auto")
    parser.add_argument("--clipper",      choices=["cohen_sutherland", "liang_barsky"],
                        default="cohen_sutherland")
    parser.add_argument("--viewport",     default="800x600")
    parser.add_argument("--no-gui",       action="store_true")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.list_objects:
        print("Available shapes:")
        for name in sorted(SAMPLE_OBJECTS.keys()):
            obj = SAMPLE_OBJECTS[name]()
            dim = "2D" if is_2d_mesh(obj) else "3D"
            print(f"  {name:<20} ({dim}, {len(obj.vertices)} verts, {len(obj.edges)} edges)")
        return 0

    if args.demo:
        run_presentation_demo(show_gui=not args.no_gui)
        return 0

    # No flags → fully interactive
    if args.run is None and args.object is None:
        run_interactive()
        return 0

    # CLI flags provided
    try:
        default_obj = "unit_square" if args.mode == "2d" else "unit_cube"
        object_name = args.object or default_obj

        if object_name not in SAMPLE_OBJECTS:
            print(f"Unknown shape '{object_name}'.")
            print(f"Run --list-objects to see all options.")
            return 1

        if args.run in ("basic", None):
            run_basic_demo(args.mode, object_name, show_gui=not args.no_gui)
        else:
            try:
                w_str, h_str = args.viewport.lower().split("x")
                viewport_size = (int(w_str), int(h_str))
            except Exception:
                print("Viewport must be WIDTHxHEIGHT e.g. 800x600")
                return 1
            run_full_pipeline(
                mode=args.mode, object_name=object_name,
                projection=args.projection, clipper=args.clipper,
                viewport_size=viewport_size, show_gui=not args.no_gui,
            )
    except Exception as exc:
        import traceback
        print(f"\nError: {exc}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())