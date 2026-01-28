# app.py
import streamlit as st
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import dashboard
from interfaces.streamlit_components.dashboard import SecurityDashboard
import streamlit as st

def main():
    """Main application entry point."""

    try:
        # Initialize and run dashboard
        dashboard = SecurityDashboard()
        dashboard.run()

    except Exception as e:
        # Error handling
        st.error(f"Application error: {str(e)}")

        # Show debugging info in development
        if os.getenv("ENVIRONMENT", "development") == "development":
            st.exception(e)


if __name__ == "__main__":
    main()