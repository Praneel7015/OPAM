#!/bin/bash

# OPAM Automation Script

echo "----------------------------------------------------------------"
echo "  OPAM (Online Purchasing-behavior Analysis And Management)"
echo "----------------------------------------------------------------"

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH."
    exit 1
fi

# Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment found."
fi

# Activate Virtual Environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "Error: Could not find activate script."
    exit 1
fi

# Install Dependencies
echo "Installing/Updating dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing dependencies."
    exit 1
fi
echo "Dependencies ready."

# Check Data
if [ ! -f "data/transactions.csv" ]; then
    echo "No data found in data/transactions.csv"
    read -p "Generate sample data? (y/n): " choice
    if [[ "$choice" =~ ^[Yy]$ ]]; then
        echo "Generating sample data..."
        python scripts/generate_sample_data.py
    else
        echo "Please place 'transactions.csv' in the 'data' folder."
        exit 1
    fi
fi

# Run Main System
echo "Starting OPAM..."
cd src
python run_all_systems.py
