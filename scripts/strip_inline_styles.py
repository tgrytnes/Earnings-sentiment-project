#!/usr/bin/env python3
"""
Strip inline <style>...</style> blocks from notebooks/main.ipynb markdown cells.
Keeps the rest of the cell content unchanged.
"""
from __future__ import annotations
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "main.ipynb"


def strip_styles(lines: list[str]) -> list[str]:
    out: list[str] = []
    skipping = False
    for line in lines:
        if not skipping and line.strip() == "<style>":
            skipping = True
            continue
        if skipping and line.strip() == "</style>":
            skipping = False
            continue
        if not skipping:
            out.append(line)
    return out


def main() -> None:
    nb = json.loads(NB.read_text())
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            src = cell.get("source", [])
            new_src = strip_styles(src)
            if new_src != src:
                cell["source"] = new_src
                changed = True
    if changed:
        NB.write_text(json.dumps(nb, indent=1))
        print(f"Stripped inline styles from {NB}")
    else:
        print("No inline styles found to strip.")


if __name__ == "__main__":
    main()

