import sys
import os
from streamlit.web import cli as stcli

def main():
    print("Launching Cleaniphile...")
    
    # Path to the app.py file
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    
    # Set arguments for streamlit
    sys.argv = ["streamlit", "run", app_path]
    
    # Run streamlit
    sys.exit(stcli.main())

if __name__ == "__main__":
    main()
