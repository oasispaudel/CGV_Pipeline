CGV Pipeline Overview

This project is a small Computer Graphics and Visualization (CGV) pipeline simulator. It takes a sample object, runs it through standard pipeline stages, and visualizes the result.

Pipeline Stages (High Level)
- Model space (original object)
- Model -> World transform
- World -> View (camera) transform
- Projection to NDC
- Normalise NDC
- Clipping
- Viewport mapping (screen coordinates)
- Visualization

How To Run (Demo)
From the project root:

  python -m src.main --run basic --mode 3d --object unit_cube
  python -m src.main --run full --mode 3d --object unit_cube

You can also run tests:

  python -m unittest tests.test_transformations -v

See docs/flow_diagram.md for a simple flowchart.
