#!/bin/bash
set -e

# Define paths
PATHS="main.py src tests"

echo "=============================="
echo "Running isort (import sorting)"
echo "=============================="
python3.12 -m isort $PATHS

echo "=============================="
echo "Running Black (code formatting)"
echo "=============================="
python3.12 -m black $PATHS

echo "=============================="
echo "Running autoflake (remove unused imports & variables)"
echo "=============================="
python3.12 -m pip install --quiet autoflake
python3.12 -m autoflake --in-place --remove-unused-variables --remove-all-unused-imports -r $PATHS

echo "=============================="
echo "Running Flake8 (linting)"
echo "=============================="
python3.12 -m flake8 $PATHS

echo "=============================="
echo "All formatting and linting complete!"
echo "=============================="
