#!/bin/bash

set -e  # Exit on error
set -o pipefail  # Ensure pipeline errors propagate

echo "Initializing project setup..."

# 1. Check if uv is installed, install if not
if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 2. Installing project dependencies
echo "Installing dependencies..."
uv sync

## aaaaaaaand that's it! 🚀 (long live UV)