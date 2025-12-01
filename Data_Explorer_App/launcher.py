import os
import sys
import webbrowser
from threading import Timer
from src.app import app

def open_browser():
    # Wait 1.5 seconds for server to start, then open browser
    webbrowser.open_new("http://127.0.0.1:8050")

def main():
    print("------------------------------------------------")
    print("   Data Explorer App - Starting...")
    print("   Please wait while the interface loads.")
    print("------------------------------------------------")

    # Determine path for data loading context
    if getattr(sys, 'frozen', False):
        base_path = os.path.dirname(sys.executable)
    else:
        base_path = os.path.dirname(os.path.abspath(__file__))
    
    print(f"   Working Directory: {base_path}")
    print("   Looking for .csv files in this folder...")

    # Schedule browser launch
    Timer(1.5, open_browser).start()

    # Run the server
    # debug=False is REQUIRED for PyInstaller
    try:
        app.run_server(debug=False, port=8050)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        input("Press Enter to close...")

if __name__ == '__main__':
    main()
