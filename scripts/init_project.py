#!/usr/bin/env python3
"""
Initialize this repo from the template by renaming the package
and updating references in configs, scripts, and notebooks.

Usage examples:

  # Dry run (prints what would change)
  python scripts/init_project.py --package earnings_sentiment \
      --dist-name earnings-sentiment --title "Earnings Sentiment Project" \
      --kernel-name earnings-sentiment-venv --dry-run

  # Apply changes
  python scripts/init_project.py --package earnings_sentiment \
      --dist-name earnings-sentiment --title "Earnings Sentiment Project" \
      --kernel-name earnings-sentiment-venv

What it does:
  - Renames src/yourproj -> src/<package>
  - Updates imports in scripts and notebooks from `yourproj` to <package>
  - Updates pyproject project name to <dist-name>
  - Updates default Jupyter kernel name in bootstrap_env.sh
  - Updates titles in scripts/set_title_markdown.py and notebooks/_quarto.yml

Notes:
  - Notebook updates replace occurrences of "yourproj." in code cells only.
  - This is a one-time init script; re-run is idempotent when already renamed.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def die(msg: str) -> None:
    raise SystemExit(msg)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, data: str, dry_run: bool, changes: List[str]) -> None:
    if dry_run:
        changes.append(f"Would write: {path}")
    else:
        path.write_text(data, encoding="utf-8")
        changes.append(f"Updated: {path}")


def replace_in_file(path: Path, subs: List[Tuple[str, str]], dry_run: bool, changes: List[str]) -> None:
    if not path.exists():
        return
    original = read_text(path)
    new = original
    for old, newval in subs:
        new = new.replace(old, newval)
    if new != original:
        write_text(path, new, dry_run, changes)


def update_pyproject(dist_name: str, dry_run: bool, changes: List[str]) -> None:
    pp = REPO_ROOT / "pyproject.toml"
    if not pp.exists():
        return
    txt = read_text(pp)
    # Replace the [project] name = "..."
    new = re.sub(r'(?m)^name\s*=\s*"[^"]+"', f'name = "{dist_name}"', txt)
    if new != txt:
        write_text(pp, new, dry_run, changes)


def update_bootstrap_kernel(kernel_name: str, dry_run: bool, changes: List[str]) -> None:
    bs = REPO_ROOT / "bootstrap_env.sh"
    if not bs.exists():
        return
    txt = read_text(bs)
    txt2 = re.sub(r'(?m)^KERNEL_NAME=\"[^\"]+\"', f'KERNEL_NAME="{kernel_name}"', txt)
    txt2 = txt2.replace(
        "Jupyter kernel name (default: yourproj-venv)",
        f"Jupyter kernel name (default: {kernel_name})",
    )
    if txt2 != txt:
        write_text(bs, txt2, dry_run, changes)


def update_run_script(pkg: str, dry_run: bool, changes: List[str]) -> None:
    run = REPO_ROOT / "scripts" / "run_train.sh"
    replace_in_file(run, [("python -m yourproj.", f"python -m {pkg}.")], dry_run, changes)


def update_refresh_snippets(pkg: str, dry_run: bool, changes: List[str]) -> None:
    scr = REPO_ROOT / "scripts" / "refresh_snippets.py"
    replace_in_file(scr, [("src/yourproj/", f"src/{pkg}/")], dry_run, changes)


def update_titles(title: str, dry_run: bool, changes: List[str]) -> None:
    # scripts/set_title_markdown.py
    stm = REPO_ROOT / "scripts" / "set_title_markdown.py"
    if stm.exists():
        txt = read_text(stm)
        txt2 = re.sub(r'(?m)^TITLE\s*=\s*\"[^\"]*\"', f'TITLE = "{title}"', txt)
        if txt2 != txt:
            write_text(stm, txt2, dry_run, changes)

    # notebooks/config/_quarto.yml
    qy = REPO_ROOT / "notebooks" / "config" / "_quarto.yml"
    if qy.exists():
        txt = read_text(qy)
        txt2 = re.sub(r'(?m)^\s*title:\s*\"[^\"]*\"', f'  title: "{title}"', txt)
        if txt2 != txt:
            write_text(qy, txt2, dry_run, changes)


def update_readme_header(title: str, dry_run: bool, changes: List[str]) -> None:
    rd = REPO_ROOT / "README.md"
    if not rd.exists():
        return
    txt = read_text(rd)
    # Replace top-level H1 if present
    lines = txt.splitlines()
    if lines and lines[0].startswith("# "):
        lines[0] = f"# {title}"
        new = "\n".join(lines) + ("\n" if txt.endswith("\n") else "")
        if new != txt:
            write_text(rd, new, dry_run, changes)


def update_notebook_imports(pkg: str, dry_run: bool, changes: List[str]) -> None:
    nb = REPO_ROOT / "notebooks" / "main.ipynb"
    if not nb.exists():
        return
    data = json.loads(nb.read_text(encoding="utf-8"))
    touched = False
    for cell in data.get("cells", []):
        if cell.get("cell_type") == "code":
            src = cell.get("source", [])
            new_src = []
            for line in src:
                new_src.append(line.replace("yourproj.", f"{pkg}."))
            if new_src != src:
                cell["source"] = new_src
                touched = True
    if touched:
        payload = json.dumps(data, indent=1)
        if dry_run:
            changes.append(f"Would update imports in {nb}")
        else:
            nb.write_text(payload, encoding="utf-8")
            changes.append(f"Updated imports in {nb}")


def rename_package_dir(from_pkg: str, to_pkg: str, dry_run: bool, changes: List[str]) -> None:
    src = REPO_ROOT / "src" / from_pkg
    dst = REPO_ROOT / "src" / to_pkg
    if src.exists() and src.is_dir():
        if dst.exists():
            changes.append(f"Target package already exists: {dst} (skipping rename)")
            return
        if dry_run:
            changes.append(f"Would rename {src} -> {dst}")
        else:
            shutil.move(str(src), str(dst))
            changes.append(f"Renamed {src} -> {dst}")
    else:
        # If already renamed, skip
        changes.append(f"Source package not found at {src} (already renamed?)")


def update_repo(pkg: str, dist: str, title: str, kernel: str, from_pkg: str, dry_run: bool) -> List[str]:
    changes: List[str] = []
    rename_package_dir(from_pkg, pkg, dry_run, changes)
    update_run_script(pkg, dry_run, changes)
    update_refresh_snippets(pkg, dry_run, changes)
    update_pyproject(dist, dry_run, changes)
    update_bootstrap_kernel(kernel, dry_run, changes)
    update_titles(title, dry_run, changes)
    update_readme_header(title, dry_run, changes)
    update_notebook_imports(pkg, dry_run, changes)
    return changes


def main() -> None:
    ap = argparse.ArgumentParser(description="Initialize project by renaming template package and metadata.")
    ap.add_argument("--package", required=True, help="New Python package name (e.g., earnings_sentiment)")
    ap.add_argument("--dist-name", required=True, help="Distribution name for pyproject (e.g., earnings-sentiment)")
    ap.add_argument("--title", required=True, help="Human-readable project title for docs")
    ap.add_argument("--kernel-name", default=None, help="Jupyter kernel default name (if unset, derived from dist-name)")
    ap.add_argument("--from-package", default="yourproj", help="Current template package name (default: yourproj)")
    ap.add_argument("--dry-run", action="store_true", help="Print actions without modifying files")
    args = ap.parse_args()

    new_pkg = args.package.strip()
    if not re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", new_pkg):
        die("--package must be a valid Python identifier (letters, digits, underscore; not starting with digit)")
    dist = args.dist_name.strip()
    if not re.match(r"^[a-zA-Z0-9._-]+$", dist):
        die("--dist-name contains invalid characters")
    title = args.title.strip()
    kernel = args.kernel_name or f"{dist}-venv"

    changes = update_repo(new_pkg, dist, title, kernel, args.from_package, args.dry_run)
    print("Init summary:")
    for c in changes:
        print(" -", c)
    if args.dry_run:
        print("\nDry run complete. Re-run without --dry-run to apply changes.")
    else:
        print("\nDone.")


if __name__ == "__main__":
    main()

