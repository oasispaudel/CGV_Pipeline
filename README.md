# CGV Pipeline

Academic project that implements core building blocks of a Computer Graphics and Visualization (CGV) pipeline. The focus is on data structures for meshes, homogeneous transformation matrices, camera (view) transformation, and sample objects used for testing and demonstration. Additional pipeline stages and algorithms are scaffolded as placeholders for future work.

**Project Status**
Several modules in `src/` and `docs/` are currently placeholders (empty files). The implemented core is in `src/core/`, `src/input/`, and `src/visulaization/`, with tests for the transformation and camera logic.

**Features**
- Homogeneous coordinates and mesh data structures (vertices, edges, mesh).
- 2D and 3D transformation matrices: translate, scale, rotate, reflect, shear, compose.
- Look-at camera (view) matrix.
- Sample 2D/3D objects (square, cube, triangles, house, pyramid).
- Simple 2D/3D visualization helpers using Matplotlib.
- Unit tests covering transformation math and camera behavior.

**Requirements**
- Python 3.x
- See `requirements.txt` for Python dependencies.

**Setup**
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

**How To Run**
`src/main.py` includes a demo runner for basic (model->view) and full (model->viewport) pipeline flows. You can still validate the math with unit tests:

```bash
python -m unittest tests.test_transformations -v
```

If you prefer `pytest` (optional):

```bash
pytest tests/test_transformations.py -v
```

**Project Flowchart**
See `docs/flow_diagram.md` for a Mermaid flowchart of the end-to-end project flow (input -> pipeline stages -> visualization + tests).

**Folder Structure**
```text
CGV_Pipeline/
  docs/
    algorithms/
      clipping.md
      projection.md
      transformations.md
      viewport_mapping.md
    flow_diagram.png
    pipeline_overview.md
    report_outline.md
  src/
    algorithms/
      cohen_sutherland.py
      liang_barsky.py
      line_drawing.py
      __init__.py
    core/
      data_structures.py
      matrix.py
      transformations.py
      __init__.py
    input/
      sample_objects.py
      user_input.py
      __init__.py
    pipeline/
      clipping_stage.py
      model_stage.py
      normalization_stage.py
      projection_stage.py
      viewport_stage.py
      view_stage.py
      __init__.py
    utils/
      config.py
      helpers.py
      __init__.py
    visulaization/
      animation.py
      color_scheme.py
      plot_2d.py
      plot_3d.py
      __init__.py
  tests/
    test_clipping.py
    test_pipeline.py
    test_projection.py
    test_transformations.py
  .gitignore
  LICENSE
  README.md
  requirements.txt
```

**File Descriptions**
`README.md`: Project overview, setup, and usage instructions.
`requirements.txt`: Python dependencies.

`docs/flow_diagram.md`: Mermaid flowchart of the full project flow.
`docs/flow_diagram.png`: Placeholder for pipeline flow diagram image (optional).
`docs/pipeline_overview.md`: Short overview of the pipeline and how to run the demo.
`docs/report_outline.md`: Placeholder for report outline (currently empty).
`docs/explanation.md`: Step-by-step explanation of the pipeline stages.
`docs/algorithms/clipping.md`: Placeholder for clipping algorithm notes (currently empty).
`docs/algorithms/projection.md`: Placeholder for projection algorithm notes (currently empty).
`docs/algorithms/transformations.md`: Placeholder for transformations notes (currently empty).
`docs/algorithms/viewport_mapping.md`: Placeholder for viewport mapping notes (currently empty).

`src/main.py`: Intended project entry point (currently empty).

`src/algorithms/cohen_sutherland.py`: Placeholder for Cohen-Sutherland clipping (currently empty).
`src/algorithms/liang_barsky.py`: Placeholder for Liang-Barsky clipping (currently empty).
`src/algorithms/line_drawing.py`: Placeholder for line drawing algorithms (currently empty).
`src/algorithms/__init__.py`: Package init for algorithms (currently empty).

`src/core/data_structures.py`: Defines `Vertex`, `Edge`, and `Mesh` classes for pipeline geometry.
`src/core/matrix.py`: Defines `Transform` factory methods for 4x4 homogeneous matrices.
`src/core/transformations.py`: Defines `Camera` (look-at) and model-to-world / world-to-view stage functions.
`src/core/__init__.py`: Package init for core (currently empty).

`src/input/sample_objects.py`: Factory functions for sample meshes (square, cube, triangles, house, pyramid).
`src/input/user_input.py`: Simple CLI prompts for choosing dimension, projection, and line input.
`src/input/__init__.py`: Package init for input (currently empty).

`src/pipeline/clipping_stage.py`: Placeholder for clipping pipeline stage (currently empty).
`src/pipeline/model_stage.py`: Placeholder for model stage logic (currently empty).
`src/pipeline/normalization_stage.py`: Placeholder for normalization stage (currently empty).
`src/pipeline/projection_stage.py`: Placeholder for projection stage (currently empty).
`src/pipeline/viewport_stage.py`: Placeholder for viewport mapping stage (currently empty).
`src/pipeline/view_stage.py`: Placeholder for view stage logic (currently empty).
`src/pipeline/__init__.py`: Package init for pipeline (currently empty).

`src/utils/config.py`: Placeholder for configuration values (currently empty).
`src/utils/helpers.py`: Placeholder for helper utilities (currently empty).
`src/utils/__init__.py`: Package init for utils (currently empty).

`src/visulaization/animation.py`: Placeholder for animation helpers (currently empty).
`src/visulaization/color_scheme.py`: Placeholder for color palettes (currently empty).
`src/visulaization/plot_2d.py`: 2D Matplotlib visualization utilities for pipeline stages.
`src/visulaization/plot_3d.py`: 3D Matplotlib visualization utilities for pipeline stages.
`src/visulaization/__init__.py`: Package init for visualization (currently empty).

`tests/test_transformations.py`: Unit tests for core data structures, transformations, camera, and sample objects.
`tests/test_clipping.py`: Placeholder for clipping tests (currently empty).
`tests/test_pipeline.py`: Placeholder for pipeline tests (currently empty).
`tests/test_projection.py`: Placeholder for projection tests (currently empty).

**Notes**
- Some core features referenced by tests (for example `PipelineState`) are expected to live in `src/core/data_structures.py` but are not present yet. Add them as you continue the implementation.
