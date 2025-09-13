#!/usr/bin/env python3
"""
Reduce the notebook's front matter to title-only, removing per-file PDF styling.
This lets _quarto.yml control all HTML/PDF styles consistently.
"""
from __future__ import annotations
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "main.ipynb"


def main() -> None:
    nb = json.loads(NB.read_text())
    cells = nb.get("cells", [])
    changed = False
    for cell in cells:
        if cell.get("cell_type") == "raw":
            src = "".join(cell.get("source", []))
            if src.lstrip().startswith("---") and ("format:" in src or "include-in-header:" in src):
                cell["source"] = [
                    "---\n",
                    "title: \"AI Pipeline Report\"\n",
                    "---",
                ]
                changed = True
                break
    if changed:
        NB.write_text(json.dumps(nb, indent=1))
        print("Simplified front matter in", NB)
    else:
        print("No complex front matter found; nothing to change.")


if __name__ == "__main__":
    main()

