#!/bin/bash

# DocQueryAI Stop Script
echo "🛑 Stopping DocQueryAI Application..."

# Find and kill Streamlit processes
STREAMLIT_PIDS=$(pgrep -f "streamlit run app.py")

if [ -z "$STREAMLIT_PIDS" ]; then
    echo "ℹ️  No DocQueryAI processes found running"
else
    echo "🔍 Found Streamlit processes: $STREAMLIT_PIDS"
    echo "🛑 Stopping processes..."
    pkill -f "streamlit run app.py"
    sleep 2
    
    # Check if processes are still running
    REMAINING_PIDS=$(pgrep -f "streamlit run app.py")
    if [ -z "$REMAINING_PIDS" ]; then
        echo "✅ DocQueryAI stopped successfully"
    else
        echo "⚠️  Some processes may still be running. Force killing..."
        pkill -9 -f "streamlit run app.py"
        echo "✅ DocQueryAI force stopped"
    fi
fi

echo "🏁 Done!"