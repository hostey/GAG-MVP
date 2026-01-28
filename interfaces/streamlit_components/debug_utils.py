import streamlit as st
import uuid
import uuid

def get_unique_key(base_name: str) -> str:
    return f"{base_name}_{uuid.uuid4().hex[:8]}"
