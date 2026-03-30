"""
components/cached_governance.py
================================
Wraps governance_logic.py import with @st.cache_resource so the
entire 6,392-line module (including all preset dictionaries) is
only parsed and loaded ONCE per Render instance lifetime.

Without this, every cold page navigation reloads the module.
With this, only the first page load pays the cost (~2-4s).
All subsequent pages get the cached module instantly.

Usage (replace direct imports in pages):
    from components.cached_governance import gl, get_presets

    # Then use gl.generate_economic_data, gl.apply_bias, etc.
    # OR keep using your existing imports — Python's module cache
    # means governance_logic is only parsed once either way.
"""
import streamlit as st


@st.cache_resource(show_spinner="Loading GAGS simulation engine...")
def _load_governance():
    """Load governance_logic once and cache for the process lifetime."""
    import components.governance_logic as _gl
    return _gl


def gl():
    """Return the cached governance_logic module."""
    return _load_governance()


# Convenience: pre-load on first import so subsequent pages are instant
_load_governance()
