import streamlit as st
from typing import Optional, Any
from datetime import datetime
from config.constants import DEFAULT_PARAMETERS, COLOR_SCHEMES
from services.export_service import ExportService, ExportFormat

# -------------------------
# Parameter Slider
# -------------------------
def create_parameter_slider(
    param_name: str,
    label: str,
    help_text: str,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    default_value: Optional[float] = None,
    step: Optional[float] = None,
    key: Optional[str] = None
) -> Any:
    """
    Create a slider with stable key to avoid duplicate ID errors.
    """
    if param_name in DEFAULT_PARAMETERS:
        config = DEFAULT_PARAMETERS[param_name]
        min_value = min_value if min_value is not None else config['min']
        max_value = max_value if max_value is not None else config['max']
        default_value = default_value if default_value is not None else config['default']
        step = step if step is not None else config['step']

    if key is None:
        key = f"slider_{param_name}"

    return st.slider(
        label=label,
        min_value=min_value,
        max_value=max_value,
        value=default_value,
        step=step,
        help=help_text,
        key=key
    )

# -------------------------
# Model Selector
# -------------------------
def create_model_selector(key: Optional[str] = None) -> Any:
    models = ["SimpleNN", "DeepNN", "CNN", "Transformer"]
    if key is None:
        key = "model_selector"
    return st.selectbox("Select Model Architecture", models, index=0, key=key)

# -------------------------
# Simulation Strategy Selector
# -------------------------
def create_strategy_selector(key: Optional[str] = None) -> Any:
    strategies = ["quick", "standard", "advanced", "custom"]
    if key is None:
        key = "sim_strategy"
    return st.selectbox("Select Simulation Strategy", strategies, index=0, key=key)

# -------------------------
# Threshold Sliders
# -------------------------
def create_threshold_sliders():
    with st.expander("Threshold Configuration", expanded=False):
        col1, col2 = st.columns(2)

        safety_threshold = col1.slider(
            "Safety Threshold", 0, 100, 70,
            help="Minimum acceptable safety score",
            key="threshold_safety"
        )
        fairness_threshold = col1.slider(
            "Fairness Threshold", 0, 100, 75,
            help="Minimum acceptable fairness score",
            key="threshold_fairness"
        )

        resilience_threshold = col2.slider(
            "Resilience Threshold", 0, 100, 65,
            help="Minimum acceptable resilience score",
            key="threshold_resilience"
        )
        accuracy_threshold = col2.slider(
            "Accuracy Threshold", 0, 100, 80,
            help="Minimum acceptable accuracy",
            key="threshold_accuracy"
        )

        return {
            "safety": safety_threshold,
            "fairness": fairness_threshold,
            "resilience": resilience_threshold,
            "accuracy": accuracy_threshold
        }

# -------------------------
# Parameter Presets
# -------------------------
def create_parameter_preset_selector():
    presets = {
        "Balanced": {"surveillance_level": 50, "liberty_threshold": 70, "poison_rate": 0.1, "noise_level": 1.0},
        "High Security": {"surveillance_level": 80, "liberty_threshold": 40, "poison_rate": 0.05, "noise_level": 0.5},
        "High Liberty": {"surveillance_level": 30, "liberty_threshold": 85, "poison_rate": 0.2, "noise_level": 1.5},
        "Under Attack": {"surveillance_level": 60, "liberty_threshold": 60, "poison_rate": 0.3, "noise_level": 2.0},
    }

    selected = st.selectbox(
        "Quick Presets",
        options=list(presets.keys()),
        help="Select a preset configuration",
        key="preset_selector"
    )

    if st.button("Apply Preset", key="apply_preset"):
        for k, v in presets[selected].items():
            st.session_state[f"slider_{k}"] = v
        st.rerun()

    return selected

# -------------------------
# Export Buttons
# -------------------------
def create_export_buttons(result: Any):
    export_service = ExportService()
    col1, col2, col3, col4 = st.columns(4)

    if col1.button("Export JSON", key="export_json"):
        try:
            filepath = export_service.export_simulation(result, ExportFormat.JSON)
            st.success(f"Exported to {filepath.name}")
        except Exception as e:
            st.error(f"Export failed: {e}")

    if col2.button("Export CSV", key="export_csv"):
        try:
            filepath = export_service.export_simulation(result, ExportFormat.CSV)
            st.success(f"Exported to {filepath.name}")
        except Exception as e:
            st.error(f"Export failed: {e}")

    if col3.button("Copy Summary", key="copy_summary"):
        try:
            summary = f"""
Simulation Summary:
- Safety Score: {getattr(result, 'safety_score', 0):.1f}%
- Liberty Score: {getattr(result, 'liberty_score', 0):.1f}/100
- Resilience Index: {getattr(result, 'resilience_score', 0):.1f}
- Fairness Score: {getattr(result, 'fairness_score', 0):.1f}
- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
            st.code(summary)
        except Exception as e:
            st.error(f"Failed to generate summary: {e}")

# -------------------------
# Color Theme Selector
# -------------------------
def create_color_theme_selector():
    themes = {
        "Security Blue": COLOR_SCHEMES["security"],
        "Healthcare Green": COLOR_SCHEMES["healthcare"],
        "Dark Mode": {"primary": "#2E7D32", "secondary": "#1B5E20", "accent": "#4CAF50"},
        "Corporate": {"primary": "#1565C0", "secondary": "#0D47A1", "accent": "#42A5F5"},
    }

    selected = st.selectbox(
        "Color Theme",
        options=list(themes.keys()),
        help="Select dashboard color theme",
        key="color_theme_selector"
    )

    return themes[selected]
