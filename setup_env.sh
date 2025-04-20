#!/bin/bash

# Set environment name
ENV_NAME="venv"

echo "🚀 Setting up virtual environment: $ENV_NAME"

# Remove old environment if exists
if [ -d "$ENV_NAME" ]; then
    echo "⚠️ Removing existing environment..."
    rm -rf $ENV_NAME
fi

# Create virtual environment
python3 -m venv $ENV_NAME
echo "✅ Virtual environment created!"

# Activate environment
source $ENV_NAME/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

echo "✅ All dependencies installed!"
echo "🎯 To activate the environment, run: source $ENV_NAME/bin/activate"
