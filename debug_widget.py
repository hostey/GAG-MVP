import streamlit as st
from interfaces.streamlit_components import widgets

st.title("Widget Key Debugger")

# Call all widgets so their keys are registered
widgets.create_model_selector()
widgets.create_strategy_selector()
widgets.create_threshold_sliders()
widgets.create_parameter_slider("surveillance_level", "Surveillance Intensity", "Set intensity")
widgets.create_parameter_preset_selector()
widgets.create_export_buttons(result={})
widgets.create_color_theme_selector()

# Report duplicates
widgets.report_duplicate_keys()
