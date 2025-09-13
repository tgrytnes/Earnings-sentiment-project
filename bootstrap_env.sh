#!/usr/bin/env bash
set -euo pipefail

# Bootstrap local dev environment:
# - Installs Python (optional), Quarto (optional), and TeX (optional)
# - Creates a virtualenv and installs project dependencies (pyproject)
# - Registers a Jupyter kernel for the venv
#
# Usage (from repo root):
#   bash AI_Template/scripts/bootstrap_env.sh [options]
#
# Common examples:
#   bash AI_Template/scripts/bootstrap_env.sh
#   bash AI_Template/scripts/bootstrap_env.sh --venv .venv --tex basic
#   bash AI_Template/scripts/bootstrap_env.sh --python-version 3.11 --tex full
#   bash AI_Template/scripts/bootstrap_env.sh --no-system   # skip brew/apt steps

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Defaults
VENVDIR=".venv"
PYTHON_BIN=""
PY_VER=""
INSTALL_SYSTEM=true
INSTALL_PYTHON=true
INSTALL_TEX=true
TEX_FLAVOR="basic"     # basic | full | none
INSTALL_QUARTO=true
REGISTER_KERNEL=true
KERNEL_NAME="yourproj-venv"

usage() {
  cat <<EOF
Bootstrap local environment for this project.

Options:
  --venv DIR              Virtualenv directory (default: .venv)
  --python BIN            Python binary to use (e.g. python3.11)
  --python-version X.Y    Install/use Python version (e.g. 3.11)
  --no-system             Skip system package installs (brew/apt)
  --no-python             Do not install Python via brew/apt
  --no-tex                Do not install TeX
  --tex {basic|full|none} TeX flavor (default: basic)
  --no-quarto             Do not install Quarto
  --no-kernel             Do not register Jupyter kernel
  --kernel-name NAME      Jupyter kernel name (default: yourproj-venv)
  -h, --help              Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv) VENVDIR="$2"; shift 2;;
    --python) PYTHON_BIN="$2"; shift 2;;
    --python-version) PY_VER="$2"; shift 2;;
    --no-system) INSTALL_SYSTEM=false; shift;;
    --no-python) INSTALL_PYTHON=false; shift;;
    --no-tex) INSTALL_TEX=false; TEX_FLAVOR="none"; shift;;
    --tex) TEX_FLAVOR="$2"; [[ "$TEX_FLAVOR" == none ]] && INSTALL_TEX=false; shift 2;;
    --no-quarto) INSTALL_QUARTO=false; shift;;
    --no-kernel) REGISTER_KERNEL=false; shift;;
    --kernel-name) KERNEL_NAME="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    return 1
  fi
}

detect_os() {
  local uname_s
  uname_s=$(uname -s)
  case "$uname_s" in
    Darwin) echo mac;;
    Linux) echo linux;;
    *) echo unknown;;
  esac
}

install_mac_system_deps() {
  # Homebrew
  if ! command -v brew >/dev/null 2>&1; then
    echo "Homebrew not found. Install from https://brew.sh/ and rerun or run with --no-system." >&2
    return 1
  fi

  if $INSTALL_PYTHON; then
    if [[ -n "$PY_VER" ]]; then
      echo "[mac] Installing Python $PY_VER via Homebrew"
      brew install "python@$PY_VER" || true
      # Determine python binary path
      local prefix
      prefix=$(brew --prefix "python@$PY_VER" 2>/dev/null || true)
      if [[ -n "$prefix" && -x "$prefix/bin/python$PY_VER" ]]; then
        PYTHON_BIN="$prefix/bin/python$PY_VER"
      elif [[ -n "$prefix" && -x "$prefix/bin/python3.$(echo "$PY_VER" | cut -d. -f2)" ]]; then
        PYTHON_BIN="$prefix/bin/python3.$(echo "$PY_VER" | cut -d. -f2)"
      fi
    else
      echo "[mac] Installing latest Python3 via Homebrew"
      brew install python || true
      PYTHON_BIN=$(command -v python3 || true)
    fi
  fi

  if $INSTALL_QUARTO; then
    echo "[mac] Installing Quarto"
    brew install quarto || true
  fi

  echo "[mac] Installing Pandoc"
  brew install pandoc || true

  if $INSTALL_TEX; then
    # Check if TeX is already installed
    if command -v pdflatex >/dev/null 2>&1; then
      echo "[mac] TeX already installed, skipping installation"
      tex_version=$(pdflatex --version | head -n1 || echo "unknown version")
      echo "[mac] Found: $tex_version"
    else
      case "$TEX_FLAVOR" in
        basic)
          echo "[mac] Installing BasicTeX (smaller TeX Live)"
          brew install --cask basictex || true
          echo "[mac] Installing essential TeX packages via tlmgr"
          export PATH="/Library/TeX/texbin:$PATH"
          if command -v tlmgr >/dev/null 2>&1; then
            sudo tlmgr update --self || true
            sudo tlmgr install latexmk collection-latexrecommended collection-fontsrecommended || true
          else
            echo "tlmgr not found at /Library/TeX/texbin; ensure TeX is on PATH and rerun this step." >&2
          fi
          ;;
        full)
          echo "[mac] Installing full MacTeX (large download)"
          brew install --cask mactex-no-gui || true
          ;;
        none)
          ;;
        *) echo "Unknown --tex value: $TEX_FLAVOR" >&2; return 1;;
      esac
    fi
  fi
}

install_linux_system_deps() {
  if command -v apt-get >/dev/null 2>&1; then
    echo "[linux] Using apt-get for system dependencies (sudo required)"
    sudo apt-get update -y
    if $INSTALL_PYTHON; then
      if [[ -n "$PY_VER" ]]; then
        echo "[linux] Installing Python $PY_VER"
        sudo apt-get install -y "python$PY_VER" "python$PY_VER-venv" python3-pip || true
        PYTHON_BIN=$(command -v "python$PY_VER" || true)
      else
        sudo apt-get install -y python3 python3-venv python3-pip || true
        PYTHON_BIN=$(command -v python3 || true)
      fi
    fi

    if $INSTALL_TEX; then
      # Check if TeX is already installed
      if command -v pdflatex >/dev/null 2>&1; then
        echo "[linux] TeX already installed, skipping installation"
        tex_version=$(pdflatex --version | head -n1 || echo "unknown version")
        echo "[linux] Found: $tex_version"
      else
        case "$TEX_FLAVOR" in
          basic)
            echo "[linux] Installing TeX Live (recommended subsets)"
            sudo apt-get install -y texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-xetex latexmk || true
            ;;
          full)
            echo "[linux] Installing full TeX Live (large download)"
            sudo apt-get install -y texlive-full || true
            ;;
          none)
            ;;
        esac
      fi
    fi

    if $INSTALL_QUARTO; then
      echo "[linux] Installing Quarto"
      # Install latest Quarto .deb
      require_cmd curl || { echo "curl required to install Quarto" >&2; return 1; }
      tmpdeb=$(mktemp /tmp/quarto-XXXX.deb)
      curl -fsSL -o "$tmpdeb" https://quarto.org/download/latest/quarto-linux-amd64.deb
      sudo dpkg -i "$tmpdeb" || sudo apt-get install -f -y
      rm -f "$tmpdeb"
    fi
    echo "[linux] Installing Pandoc"
    sudo apt-get install -y pandoc || true
  else
    echo "apt-get not found. Please install dependencies for your distro manually or run with --no-system." >&2
  fi
}

ensure_python_bin() {
  if [[ -n "$PYTHON_BIN" ]]; then
    return 0
  fi
  if [[ -n "$PY_VER" ]]; then
    if command -v "python$PY_VER" >/dev/null 2>&1; then
      PYTHON_BIN=$(command -v "python$PY_VER")
      return 0
    fi
    if command -v "python3.$(echo "$PY_VER" | cut -d. -f2)" >/dev/null 2>&1; then
      PYTHON_BIN=$(command -v "python3.$(echo "$PY_VER" | cut -d. -f2)")
      return 0
    fi
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
  else
    echo "No python interpreter found. Install Python or run with --no-system after installing it manually." >&2
    exit 1
  fi
}

create_venv() {
  echo "Creating virtualenv at $VENVDIR using $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENVDIR"
  # shellcheck disable=SC1090
  source "$VENVDIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel
}

install_project_deps() {
  echo "Installing project (editable) and dependencies from pyproject"
  python -m pip install -e "$PROJECT_ROOT"
}

register_kernel() {
  echo "Registering Jupyter kernel: $KERNEL_NAME"
  python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"
}

post_checks() {
  echo "\nTool versions:"
  "$PYTHON_BIN" --version || true
  if command -v quarto >/dev/null 2>&1; then
    quarto --version || true
  else
    echo "quarto not found (skipped or not on PATH)" || true
  fi
  if command -v pdflatex >/dev/null 2>&1; then
    pdflatex --version | head -n1 || true
  else
    echo "pdflatex not found (TeX skipped or not on PATH)" || true
  fi
}

main() {
  local os
  os=$(detect_os)
  echo "Detected OS: $os"

  if $INSTALL_SYSTEM; then
    case "$os" in
      mac) install_mac_system_deps;;
      linux) install_linux_system_deps;;
      *) echo "Unsupported OS for automatic system setup. Use --no-system and install prerequisites manually.";;
    esac
  else
    echo "Skipping system package installs (--no-system)"
  fi

  ensure_python_bin
  create_venv
  install_project_deps
  if $REGISTER_KERNEL; then
    register_kernel
  else
    echo "Skipping Jupyter kernel registration (--no-kernel)"
  fi
  post_checks

  echo "\nDone. Activate your venv with:"
  echo "  source $VENVDIR/bin/activate"
  echo "Then open notebooks or run scripts using this environment."
}

main "$@"

