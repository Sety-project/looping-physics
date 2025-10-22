#!/usr/bin/env python3
"""
Simple launcher for the Looping Physics Dashboard
This script runs the streamlit_looping_dashboard.py without any modifications
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dashboard_file = os.path.join(script_dir, "streamlit_looping_dashboard.py")
    
    # Check if the dashboard file exists
    if not os.path.exists(dashboard_file):
        print(f"❌ Error: Dashboard file not found at {dashboard_file}")
        sys.exit(1)
    
    print("🚀 Starting Looping Physics Dashboard...")
    print(f"📁 Dashboard file: {dashboard_file}")
    print("🌐 The dashboard will open in your browser")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Run streamlit with the dashboard file
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_file,
            "--server.headless", "false",
            "--server.port", "8501"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Dashboard stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running dashboard: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install it with: pip install streamlit")
        sys.exit(1)

if __name__ == "__main__":
    main()
