#!/usr/bin/env python3
"""
Move the document title/subtitle from the first raw YAML cell in notebooks/main.ipynb
into notebooks/_metadata.yml, and remove that cell from the notebook.

Keeps rendering identical while the notebook itself contains only Markdown/code.
"""
from __future__ import annotations
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "main.ipynb"
META = Path(__file__).resolve().parents[1] / "notebooks" / "_metadata.yml"


def extract_yaml(lines: list[str]) -> dict[str, str]:
    in_block = False
    out: dict[str, str] = {}
    for line in lines:
        s = line.strip()
        if s == "---":
            in_block = not in_block
            continue
        if in_block:
            if s.startswith("title:"):
                out["title"] = s.split(":", 1)[1].strip().strip('"')
            if s.startswith("subtitle:"):
                out["subtitle"] = s.split(":", 1)[1].strip().strip('"')
    return out


def main() -> None:
    nb = json.loads(NB.read_text())
    cells = nb.get("cells", [])
    if not cells:
        print("No cells found; nothing to move.")
        return
    first = cells[0]
    if first.get("cell_type") != "raw":
        print("First cell is not raw YAML; nothing to move.")
        return
    meta = extract_yaml(first.get("source", []))
    if not meta.get("title"):
        print("No title found in first raw cell; nothing to move.")
        return

    # Write _metadata.yml
    lines = ["title: \"%s\"\n" % meta["title"]]
    if meta.get("subtitle"):
        lines.append("subtitle: \"%s\"\n" % meta["subtitle"])
    META.write_text("".join(lines))

    # Remove the first cell
    nb["cells"] = cells[1:]
    NB.write_text(json.dumps(nb, indent=1))
    print("Moved title/subtitle to", META)


if __name__ == "__main__":
    main()

