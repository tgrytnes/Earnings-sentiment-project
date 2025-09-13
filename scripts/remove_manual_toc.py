#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "main.ipynb"


def main() -> None:
    nb = json.loads(NB.read_text())
    cells = nb.get("cells", [])
    new_cells = []
    removed = False
    for c in cells:
        if c.get("id") == "proj-toc":
            removed = True
            continue
        new_cells.append(c)
    if removed:
        nb["cells"] = new_cells
        NB.write_text(json.dumps(nb, indent=1))
        print("Removed manual TOC cell 'proj-toc'.")
    else:
        print("Manual TOC cell not found; nothing changed.")


if __name__ == "__main__":
    main()

