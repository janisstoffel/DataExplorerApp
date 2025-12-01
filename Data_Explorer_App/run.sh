#!/bin/bash

# Ensure we are in the script's directory
cd "$(dirname "$0")"

# Define environment directory
VENV_DIR="venv"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv $VENV_DIR
fi

# Activate venv
source $VENV_DIR/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the application
echo "Starting Data Explorer App..."
python src/app.py
