#!/usr/bin/env python3
"""
Convert custom HTML card headings (<div class="section-card"><h2>..</h2></div>)
and (<div class="subsection-card"><h3>..</h3></div>) into Markdown headings (## / ###).

This makes Quarto's auto-TOC and anchors work while we preserve the look
via CSS (see notebooks/config/styles.css).
"""
from __future__ import annotations
import json
import re
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "notebooks" / "main.ipynb"


# Patterns match across newlines after JSON is parsed (no escaping needed)
SECTION_RE = re.compile(r"<div\s+class=\"section-card\">\s*<h2>(.*?)</h2>\s*</div>", re.DOTALL)
SUBSECTION_RE = re.compile(
    r"<div\s+class=\"subsection-card\">\s*<h3>(?P<title>.*?)</h3>(?:\s*<div\s+class=\"subtitle\">(?P<subtitle>.*?)</div>)?\s*</div>",
    re.DOTALL,
)


def convert_source(src_lines: list[str]) -> list[str]:
    src = "".join(src_lines)
    changed = False

    # Replace section-card blocks with Markdown h2
    def sec_sub(m: re.Match) -> str:
        nonlocal changed
        changed = True
        title = m.group(1).strip()
        return f"## {title}\n"

    # Replace subsection-card blocks with Markdown h3
    def sub_sub(m: re.Match) -> str:
        nonlocal changed
        changed = True
        title = m.group("title").strip()
        subtitle = m.group("subtitle")
        extra = f"\n\n{subtitle.strip()}\n" if subtitle else "\n"
        return f"### {title}\n{extra}"

    src2 = SECTION_RE.sub(sec_sub, src)
    src2 = SUBSECTION_RE.sub(sub_sub, src2)

    # Normalize trailing whitespace and split back into lines list
    if src2 != src:
        # Ensure newline endings and keep as a list
        lines = src2.splitlines()
        return [l + ("\n" if not l.endswith("\n") else "") for l in lines]
    return src_lines


def main() -> None:
    nb = json.loads(NB.read_text())
    changed = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            new_src = convert_source(cell.get("source", []))
            if new_src != cell.get("source", []):
                cell["source"] = new_src
                changed = True
    if changed:
        NB.write_text(json.dumps(nb, indent=1))
        print("Converted HTML card headings to Markdown in", NB)
    else:
        print("No card headings found to convert.")


if __name__ == "__main__":
    main()
