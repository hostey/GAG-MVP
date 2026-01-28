# debug_widget.py
import streamlit as st
import uuid

# Global store to track all widget keys used in the session
if "widget_keys_used" not in st.session_state:
    st.session_state["widget_keys_used"] = set()


def get_unique_key(base_key: str) -> str:
    """Generate a truly unique key for Streamlit widgets."""
    unique_key = f"{base_key}_{uuid.uuid4().hex[:8]}"

    # Check duplicates
    if unique_key in st.session_state["widget_keys_used"]:
        st.warning(f"[DUPLICATE DETECTED] {unique_key}")
    st.session_state["widget_keys_used"].add(unique_key)

    return unique_key
def show_used_keys():
    st.sidebar.markdown("### 🔑 Widget Keys Used")
    for key in st.session_state.get("widget_keys_used", []):
        st.sidebar.write(key)
