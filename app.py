# app.py — Render entry point redirect
# Render auto-detects app.py as the default Streamlit entry point.
# This file imports and runs everything from Home.py so both
# `streamlit run app.py` and `streamlit run Home.py` work identically.

import importlib, sys, os

# Add current directory to path so all imports resolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Simply execute Home.py in the current namespace
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "Home.py")).read())
#Home.py — GAGS Resilience Framework v1.0
