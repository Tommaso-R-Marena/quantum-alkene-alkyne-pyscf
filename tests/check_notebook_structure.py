"""
tests/check_notebook_structure.py
-----------------------------------
Standalone script (also used by CI) that validates the JSON
structure of every notebook in notebooks/ without executing it.

Checks:
  1. Valid JSON and nbformat 4
  2. Has at least one code cell and one markdown cell
  3. First code cell contains a pip install command
  4. No cell has outputs that contain ERROR-level tracebacks
     (catches stale broken outputs from previous runs)
  5. All cells have valid cell_type (code | markdown | raw)
  6. No obviously broken import patterns (e.g. bare `import qml`)
"""

import json
import sys
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).parent.parent / "notebooks"
FAILURES = []


def fail(nb_name: str, msg: str):
    FAILURES.append(f"[{nb_name}] {msg}")
    print(f"  FAIL: {msg}")


def check_notebook(path: Path):
    print(f"\nChecking {path.name} ...")
    try:
        with open(path) as f:
            nb = json.load(f)
    except json.JSONDecodeError as e:
        fail(path.name, f"Invalid JSON: {e}")
        return

    # Check 1: nbformat
    if nb.get("nbformat") != 4:
        fail(path.name, f"Expected nbformat=4, got {nb.get('nbformat')}")

    cells = nb.get("cells", [])
    if not cells:
        fail(path.name, "No cells found")
        return

    code_cells = [c for c in cells if c.get("cell_type") == "code"]
    md_cells   = [c for c in cells if c.get("cell_type") == "markdown"]

    # Check 2: has both code and markdown cells
    if not code_cells:
        fail(path.name, "No code cells found")
    if not md_cells:
        fail(path.name, "No markdown cells found")

    # Check 3: first code cell has pip install
    if code_cells:
        first_src = "".join(code_cells[0].get("source", []))
        if "pip install" not in first_src:
            fail(path.name, "First code cell does not contain 'pip install'")

    # Check 4: no stale error tracebacks in outputs
    for i, cell in enumerate(code_cells):
        for out in cell.get("outputs", []):
            if out.get("output_type") == "error":
                ename = out.get("ename", "?")
                fail(path.name, f"Cell {i} has stale error output: {ename}")

    # Check 5: all cell_type values are valid
    valid_types = {"code", "markdown", "raw"}
    for i, cell in enumerate(cells):
        ct = cell.get("cell_type")
        if ct not in valid_types:
            fail(path.name, f"Cell {i} has invalid cell_type: {ct!r}")

    # Check 6: no bare 'import qml' (should be 'import pennylane as qml')
    for i, cell in enumerate(code_cells):
        src = "".join(cell.get("source", []))
        if "import qml" in src and "import pennylane" not in src:
            fail(path.name, f"Cell {i} uses bare 'import qml' without 'import pennylane'")

    # Check 7: Colab badge present in first markdown cell
    if md_cells:
        first_md = "".join(md_cells[0].get("source", []))
        if "colab.research.google.com" not in first_md:
            fail(path.name, "First markdown cell missing Colab badge/link")

    print(f"  OK ({len(code_cells)} code cells, {len(md_cells)} markdown cells)")


if __name__ == "__main__":
    notebooks = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if not notebooks:
        print("No notebooks found — check NOTEBOOKS_DIR path.")
        sys.exit(1)

    for nb_path in notebooks:
        check_notebook(nb_path)

    print()
    if FAILURES:
        print(f"\n{len(FAILURES)} structural check(s) failed:")
        for f in FAILURES:
            print(f"  {f}")
        sys.exit(1)
    else:
        print(f"All {len(notebooks)} notebooks passed structural checks.")
        sys.exit(0)
