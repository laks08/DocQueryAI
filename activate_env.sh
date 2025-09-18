#!/bin/bash
# Activation script for the local virtual environment
source venv/bin/activate
echo "Virtual environment activated!"
echo "Python path: $(which python)"
echo "Pip path: $(which pip)"
echo ""
echo "To run the Streamlit app:"
echo "streamlit run app.py"
echo ""
echo "To deactivate the environment, run: deactivate"