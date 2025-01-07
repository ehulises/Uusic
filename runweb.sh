#!/bin/bash

# Define the virtual environment directory
VENV_DIR="venv"
AUTO_CREATE_VENV=0 # by default do not create virtual environment

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -setup) AUTO_CREATE_VENV=1 ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check if the virtual environment directory exists
if [ ! -d "$VENV_DIR" ]; then
    if [ "$AUTO_CREATE_VENV" -eq 1 ]; then
        echo "Virtual environment not found. Creating one..."
        python3 -m venv "$VENV_DIR"
    else
        echo "Virtual environment not found. Exiting..."
        exit 1
    fi
fi


# Install the required packages, if auto setup is enabled
if [ "$AUTO_CREATE_VENV" -eq 1 ]; then
    # Activate the virtual environment
    source "$VENV_DIR/Scripts/activate"
    echo "Installing required packages..."
    pip install -r requirements.txt
fi

# Run the app.py script, using the current directory as the working directory
PYTHONPATH=$(pwd) 
export PYTHONPATH

python persona_gen/app.py