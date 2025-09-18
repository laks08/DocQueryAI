#!/bin/bash

# DocQueryAI Startup Script
echo "🚀 Starting DocQueryAI Application..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create it first:"
    echo "   python3 -m venv venv"
    echo "   source venv/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if Ollama is running
echo "🔍 Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "⚠️  Ollama is not running. Please start it first:"
    echo "   ollama serve &"
    echo ""
    echo "Also make sure you have the required models:"
    echo "   ollama pull phi3:mini"
    echo "   ollama pull nomic-embed-text"
    exit 1
fi

# Check if required models are available
echo "🤖 Checking required models..."
if ! ollama list | grep -q "phi3:mini"; then
    echo "⚠️  phi3:mini model not found. Installing..."
    ollama pull phi3:mini
fi

if ! ollama list | grep -q "nomic-embed-text"; then
    echo "⚠️  nomic-embed-text model not found. Installing..."
    ollama pull nomic-embed-text
fi

echo "✅ All prerequisites met!"
echo ""
echo "🌐 Starting Streamlit application..."
echo "📄 DocQueryAI will open in your browser at http://localhost:8501"
echo ""
echo "To stop the application, press Ctrl+C"
echo ""

# Start the Streamlit app
streamlit run app.py --server.headless true --server.port 8501 &

echo "🎉 DocQueryAI is now running!"
echo "📱 Access the application at: http://localhost:8501"
echo ""
echo "To stop the application:"
echo "   - Find the process: ps aux | grep streamlit"
echo "   - Kill it: kill <process_id>"
echo "   - Or use: pkill -f streamlit"
echo ""
echo "Application is running in the background..."

# Keep the script running to show the PID
STREAMLIT_PID=$!
echo "Streamlit PID: $STREAMLIT_PID"
echo "Press Ctrl+C to stop monitoring (app will continue running)"

# Wait for user to stop monitoring
trap "echo 'Monitoring stopped. App still running at http://localhost:8501'; exit 0" INT
wait