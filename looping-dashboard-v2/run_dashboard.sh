#!/bin/bash
# Simple shell script to run the Looping Physics Dashboard

echo "🔄 Looping Physics Dashboard"
echo "============================"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python3 is not installed or not in PATH"
    exit 1
fi

# Check if Streamlit is available
if ! python3 -c "import streamlit" &> /dev/null; then
    echo "❌ Error: Streamlit is not installed"
    echo "💡 Install it with: pip install streamlit"
    exit 1
fi

# Check if the dashboard file exists
if [ ! -f "streamlit_looping_dashboard.py" ]; then
    echo "❌ Error: streamlit_looping_dashboard.py not found"
    exit 1
fi

echo "🚀 Starting dashboard..."
echo "🌐 The dashboard will open in your browser at http://localhost:8501"
echo "⏹️  Press Ctrl+C to stop the dashboard"
echo ""

# Run the dashboard
python3 -m streamlit run streamlit_looping_dashboard.py --server.headless false --server.port 8501
