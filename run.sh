#!/bin/bash

# Quick Start Script for Employability ML Project
# This script sets up and runs the Streamlit app

echo "=========================================="
echo "Employability ML Project - Quick Start"
echo "=========================================="
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -d "assignment" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv assignment
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source assignment/bin/activate

# Check if requirements are installed
if [ ! -f "assignment/bin/streamlit" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Check if model files exist
if [ ! -f "model.pkl" ]; then
    echo ""
    echo "⚠️  Model files not found!"
    echo "Training model... (this will take a minute)"
    python classification_model.py
    echo "✅ Model trained and saved"
else
    echo "✅ Model files found"
fi

echo ""
echo "=========================================="
echo "🚀 Starting Streamlit App..."
echo "=========================================="
echo ""
echo "The app will open in your browser at:"
echo "http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""

# Run Streamlit app
streamlit run app.py
