#!/bin/bash
set -e

echo "=============================="
echo "Running isort (import sorting)"
echo "=============================="
python3.12 -m isort src tests

echo "=============================="
echo "Running Black (code formatting)"
echo "=============================="
python3.12 -m black src tests

echo "=============================="
echo "Running Flake8 (linting)"
echo "=============================="
python3.12 -m flake8 src tests

echo "=============================="
echo "All formatting and linting complete!"
echo "=============================="
