CGV Pipeline Setup and Run

This project includes a demo runner in `src/main.py` so you can visualize the pipeline stages for sample objects.

Prerequisites
- Python 3.x

Setup
1. Open a terminal in `f:\clz proj\CGV_Pipeline`.
2. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
.venv\Scripts\activate
```

If you want to use the existing repo venv instead:

```bash
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

Presentation demo (single command)
```bash
python -m src.main --demo
```

Run without opening Matplotlib windows:
```bash
python -m src.main --demo --no-gui
```

Run the demo (GUI)
```bash
python -m src.main --mode 3d --object unit_cube
```

Try 2D:
```bash
python -m src.main --mode 2d --object unit_square
```

List available objects:
```bash
python -m src.main --list-objects
```

Run without opening Matplotlib windows:
```bash
python -m src.main --mode 3d --no-gui
```

Run full pipeline (GUI)
```bash
python -m src.main --run full --mode 3d --object unit_cube
```

Full pipeline with orthographic projection and Liang-Barsky clipping:
```bash
python -m src.main --run full --mode 3d --projection orthographic --clipper liang_barsky
```

Full pipeline with a custom viewport and console-only output:
```bash
python -m src.main --run full --mode 3d --viewport 1024x768 --no-gui
```

Run tests
```bash
python -m unittest tests.test_transformations -v
```

Notes
- The full pipeline run visualizes stages 1 to 5: model, view, projection, clipping, viewport.
- The academic report is provided at `docs/CGV_Pipeline_Report.docx`.
