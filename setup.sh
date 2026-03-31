#!/bin/bash

# setup.sh
# checks and installs dependencies for the wheres waldo detection pipeline
# only installs packages that are missing or have incompatible versions
#
# usage: bash setup.sh

echo "========================================"
echo "  Wheres Waldo - Dependency Setup"
echo "  COMP 4360 Machine Learning"
echo "========================================"
echo ""

# figure out which python to use
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &> /dev/null; then
        PYTHON="$cmd"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: python3 or python not found. Please install Python 3.8+ first."
    exit 1
fi

PY_VERSION=$($PYTHON --version 2>&1)
echo "Python found: $PY_VERSION ($PYTHON)"
echo ""

# figure out which pip to use
PIP=""
if $PYTHON -m pip --version &> /dev/null 2>&1; then
    PIP="$PYTHON -m pip"
elif command -v pip3 &> /dev/null; then
    PIP="pip3"
elif command -v pip &> /dev/null; then
    PIP="pip"
fi

if [ -z "$PIP" ]; then
    echo "ERROR: pip not found. Please install pip first."
    exit 1
fi

echo "pip found: $($PIP --version)"
echo ""

# read requirements.txt and check each package
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

if [ ! -f "$REQ_FILE" ]; then
    echo "ERROR: requirements.txt not found at $REQ_FILE"
    exit 1
fi

MISSING=()
INSTALLED=()

echo "Checking installed packages..."
echo "----------------------------------------"

while IFS= read -r line || [ -n "$line" ]; do
    # skip empty lines and comments
    line=$(echo "$line" | xargs)
    [[ -z "$line" || "$line" == \#* ]] && continue

    # extract just the package name (before any version specifier)
    pkg_name=$(echo "$line" | sed 's/[><=!].*//' | xargs)

    # check if the package is installed and satisfies the version constraint
    if $PYTHON -c "
import importlib.metadata, sys
from packaging.specifiers import SpecifierSet
from packaging.version import Version

req_line = '''$line'''.strip()
pkg = req_line.split('>')[0].split('<')[0].split('=')[0].split('!')[0].strip()

try:
    installed_ver = importlib.metadata.version(pkg)
except importlib.metadata.PackageNotFoundError:
    sys.exit(1)

# extract version specifier part
spec_str = req_line[len(pkg):].strip()
if spec_str:
    spec = SpecifierSet(spec_str)
    if Version(installed_ver) not in spec:
        sys.exit(2)

print(installed_ver)
sys.exit(0)
" 2>/dev/null; then
        ver=$($PYTHON -c "import importlib.metadata; print(importlib.metadata.version('$(echo $pkg_name | tr '[:upper:]' '[:lower:]')'))" 2>/dev/null || echo "unknown")
        echo "  [OK] $pkg_name ($ver)"
        INSTALLED+=("$pkg_name")
    else
        echo "  [MISSING] $pkg_name"
        MISSING+=("$line")
    fi
done < "$REQ_FILE"

echo "----------------------------------------"
echo ""

if [ ${#MISSING[@]} -eq 0 ]; then
    echo "All dependencies are already installed!"
    echo ""
    echo "You can run the pipeline with: bash run.sh"
    exit 0
fi

echo "${#MISSING[@]} package(s) need to be installed: ${MISSING[*]}"
echo ""
echo "Installing missing dependencies..."
echo ""

$PIP install "${MISSING[@]}"
INSTALL_EXIT=$?

if [ $INSTALL_EXIT -ne 0 ]; then
    echo ""
    echo "ERROR: pip install failed (exit code $INSTALL_EXIT)"
    echo "Try running manually: $PIP install -r $REQ_FILE"
    exit 1
fi

echo ""
echo "========================================"
echo "  Setup complete!"
echo "  Run the pipeline with: bash run.sh"
echo "========================================"
