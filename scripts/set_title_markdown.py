#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "main.ipynb"

TITLE = "Project Template: End-to-End AI/ML Pipeline"
SUBTITLE = (
    "A reusable notebook scaffold for DS projects — ingestion, preprocessing, EDA, modeling, training, evaluation"
)


def main() -> None:
    nb = json.loads(NB.read_text())
    if not nb.get("cells"):
        raise SystemExit("Notebook has no cells")
    first = nb["cells"][0]
    first["cell_type"] = "markdown"
    first["source"] = [
        f"# {TITLE}\n",
        f"<p class=\"banner-subtitle\">{SUBTITLE}</p>\n",
    ]
    NB.write_text(json.dumps(nb, indent=1))
    print("Updated first cell to H1 + banner subtitle in", NB)


if __name__ == "__main__":
    main()

