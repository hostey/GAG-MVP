# pages/3_🌱_Sustainable_Agrotech.py
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

from components.governance_logic import (
    generate_synthetic_data,
    apply_bias,
    simulate_data_poisoning,
    run_simple_simulation
)
from utils.config import simulation_config, settings

# ───────────────────────────────────────────────
# Page Configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="Sustainable Agrotech • GAGS",
    layout="wide",
    page_icon="🌱"
)

# Custom CSS for agricultural-themed styling
st.markdown("""
<style>
    .agro-header {
        background: linear-gradient(135deg, #0b4619 0%, #1e6b30 50%, #2a8c44 100%);
        padding: 2.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        border-left: 6px solid #ffcc00;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .yield-metric {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #2ecc71;
    }
    .equity-metric {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #3498db;
    }
    .climate-metric {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #e74c3c;
    }
    .farm-metric {
        background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border: 1px solid #f39c12;
    }
    .stButton>button {
        background: linear-gradient(135deg, #27ae60 0%, #219653 100%);
        color: white;
        border: none;
        padding: 0.8rem 2.5rem;
        font-weight: bold;
        border-radius: 8px;
        transition: all 0.3s;
        border: 2px solid #2ecc71;
    }
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(46, 204, 113, 0.4);
    }
    .farm-card {
        background: #e8f5e9;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2ecc71;
        margin: 1rem 0;
        color: #1a5276;
    }
    .crop-tag {
        display: inline-block;
        background: #27ae60;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-family: monospace;
    }
    .bias-tag-agro {
        display: inline-block;
        background: #f39c12;
        color: #2f3542;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        font-weight: bold;
    }
    .climate-indicator {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #3498db;
        margin: 1rem 0;
    }
    .water-drop {
        display: inline-block;
        background: #3498db;
        color: white;
        padding: 0.2rem 0.6rem;
        border-radius: 50%;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Header Section
# ───────────────────────────────────────────────
st.markdown("""
<div class="agro-header">
    <h1 style="margin:0; color:white; font-size:2.5rem;">🌱 Sustainable Agrotech Simulation</h1>
    <p style="margin:0; opacity:0.9; font-size:1.2rem; margin-top:0.5rem;">
        Optimize <strong>crop yield predictions</strong> while ensuring <strong>equity</strong> across farmers in AI-driven agriculture
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="background: #e8f5e9; padding: 1rem; border-radius: 8px; margin-bottom: 2rem; border-left: 4px solid #2ecc71;">
    <p style="margin:0; color:#1a5276;">
        <strong>🌍 Food Security Mission:</strong> AI can revolutionize agriculture but risks amplifying existing inequities. 
        This simulation explores how bias, climate stress, and data poisoning affect yield predictions for different farmer groups.
    </p>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Session State Initialization
# ───────────────────────────────────────────────
if "agrotech_run_history" not in st.session_state:
    st.session_state.agrotech_run_history = []
if "yield_distributions" not in st.session_state:
    st.session_state.yield_distributions = {}
if "farm_types" not in st.session_state:
    st.session_state.farm_types = {}

# ───────────────────────────────────────────────
# Sidebar Controls
# ───────────────────────────────────────────────
with st.sidebar:
    # Header with icon
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <h2 style="color:#27ae60; margin:0;">⚙️ Farm Configuration</h2>
        <p style="color:#7f8c8d; font-size:0.9rem;">Configure your agricultural simulation</p>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Crop Selection
    st.subheader("🌾 Crop Selection")
    crop_type = st.selectbox(
        "Select Primary Crop",
        ["Maize/Corn", "Wheat", "Rice", "Soybeans", "Cotton", "Coffee", "Vegetables"],
        help="Different crops have different yield patterns and climate sensitivities"
    )

    # Region Selection
    region = st.selectbox(
        "Select Region",
        ["Sub-Saharan Africa", "South Asia", "Southeast Asia", "Latin America",
         "North America", "Europe", "Oceania"],
        help="Regional differences in farming practices and climate"
    )

    st.divider()

    # Bias Configuration
    st.subheader("🎭 Bias Configuration")

    selected_biases = st.multiselect(
        "**Select Bias Types**",
        options=simulation_config.BIAS_TYPES,
        default=["geographic", "socioeconomic", "representation"],
        help="Agricultural biases affecting predictions",
        format_func=lambda x: f"🌍 {x}" if x == "geographic" else f"💰 {x}" if x == "socioeconomic" else f"👥 {x}"
    )

    # Display selected biases as tags
    if selected_biases:
        tags_html = "".join([f'<span class="bias-tag-agro">{bias}</span>' for bias in selected_biases])
        st.markdown(f"**Active Biases:**<br>{tags_html}", unsafe_allow_html=True)

    bias_intensity = st.slider(
        "**Bias Intensity**",
        0.0, simulation_config.MAX_BIAS_FACTOR, 0.25, 0.05,
        help="Strength of bias in agricultural data"
    )

    st.divider()

    # Attack Configuration
    st.subheader("⚠️ Adversarial Attacks")

    attack_type = st.selectbox(
        "**Attack Type**",
        ["Data Poisoning", "Sensor Tampering", "Falsified Records", "Market Manipulation"],
        help="Type of attack on agricultural data"
    )

    poison_rate = st.slider(
        "**Attack Strength**",
        0.0, 0.5, 0.05, 0.01,
        format="%.2f",
        help="Proportion of agricultural data corrupted"
    )

    st.divider()

    # Climate & Environmental Factors
    st.subheader("🌤️ Environmental Factors")

    climate_stress = st.slider(
        "**Climate Stress Factor**",
        0.0, 1.0, 0.4, 0.05,
        help="0 = Normal conditions, 1 = Severe drought/flood/extreme weather"
    )

    # Climate indicators
    col1, col2 = st.columns(2)
    with col1:
        rainfall_deviation = st.slider(
            "Rainfall Deviation (%)",
            -50, 50, 0,
            help="Deviation from normal rainfall patterns"
        )
    with col2:
        temperature_rise = st.slider(
            "Temperature Rise (°C)",
            0.0, 3.0, 1.2, 0.1,
            help="Average temperature increase"
        )

    st.markdown(f"""
    <div class="climate-indicator">
        <span class="water-drop">💧</span>
        <strong>Climate Impact:</strong> {'Moderate' if climate_stress < 0.5 else 'Severe'}
        <br>Estimated yield reduction: {climate_stress * 30:.0f}%
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Farm Demographics
    st.subheader("👨‍🌾 Farm Demographics")

    smallholder_ratio = st.slider(
        "**Smallholder Representation**",
        0.0, 1.0, 0.6, 0.05,
        help="Proportion of smallholder farms in dataset"
    )

    mechanization_level = st.slider(
        "**Mechanization Level**",
        0.0, 1.0, 0.3,
        help="0 = Manual farming, 1 = Fully mechanized"
    )

    access_to_credit = st.select_slider(
        "**Access to Credit**",
        options=["Limited", "Moderate", "Good", "Excellent"],
        value="Moderate"
    )

    st.divider()

    # Simulation Configuration
    st.subheader("📊 Simulation Parameters")

    sample_size = st.number_input(
        "**Number of Farm Records**",
        1000, 100000, settings.DEFAULT_N_SAMPLES, step=1000
    )

    n_runs = st.slider(
        "**Number of Simulation Runs**",
        1, 10, 3,
        help="More runs provide more reliable statistics"
    )

    include_advanced = st.checkbox(
        "Include Advanced Models",
        value=True,
        help="Use ensemble models for better predictions"
    )

    st.divider()

    # Run Button
    col_run, col_reset = st.columns(2)
    with col_run:
        run_button = st.button(
            "🚜 **Run**",
            type="primary",
            use_container_width=True
        )
    with col_reset:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.agrotech_run_history = []
            st.session_state.yield_distributions = {}
            st.rerun()


# ───────────────────────────────────────────────
# Helper Functions
# ───────────────────────────────────────────────
def generate_agricultural_data(n_samples, n_features=14, smallholder_ratio=0.6):
    """Generate realistic agricultural data with farm types."""
    # Get the base synthetic data (now returns 3 values)
    X, y_base, _ = generate_synthetic_data(
        n_samples=n_samples,
        n_features=n_features,
        feature_range=(-2.0, 2.0),
        decision_boundary=0.0
    )

    # Add realistic agricultural features
    # Feature 0: Soil quality (0-10 scale)
    X[:, 0] = np.random.uniform(3, 9, n_samples)

    # Feature 1: Rainfall (mm)
    X[:, 1] = np.random.uniform(400, 1200, n_samples)

    # Feature 2: Temperature (°C)
    X[:, 2] = np.random.uniform(15, 30, n_samples)

    # Feature 3: Fertilizer use (kg/ha)
    X[:, 3] = np.random.uniform(50, 300, n_samples)

    # Feature 4: Farm size (hectares)
    # Bimodal distribution: smallholders vs large farms
    n_small = int(n_samples * smallholder_ratio)
    farm_sizes = np.zeros(n_samples)
    farm_sizes[:n_small] = np.random.uniform(0.5, 5, n_small)  # Smallholders
    farm_sizes[n_small:] = np.random.uniform(20, 200, n_samples - n_small)  # Large farms

    X[:, 4] = farm_sizes

    # Feature 5: Irrigation access (0-1)
    irrigation = np.zeros(n_samples)
    # Smallholders have less irrigation access
    irrigation[:n_small] = np.random.uniform(0.2, 0.6, n_small)
    irrigation[n_small:] = np.random.uniform(0.7, 1.0, n_samples - n_small)
    X[:, 5] = irrigation

    # Generate realistic yields (tons/ha) based on features
    soil_effect = X[:, 0] * 0.5  # Better soil = higher yield
    rain_effect = np.clip(X[:, 1] / 500, 0.5, 2.0)  # Optimal rainfall around 500mm
    fertilizer_effect = np.log(X[:, 3] + 1) * 0.3
    irrigation_effect = X[:, 5] * 0.4
    farm_size_effect = np.where(farm_sizes < 5, 0.8, 1.2)  # Scale efficiency

    base_yield = (soil_effect + rain_effect + fertilizer_effect + irrigation_effect) * farm_size_effect

    # Add realistic noise and scale to tons/ha range
    y = np.clip(base_yield + np.random.normal(0, 1, n_samples), 1.0, 12.0)

    # Create farm type labels (0 = smallholder, 1 = large farm)
    farm_types = np.zeros(n_samples)
    farm_types[n_small:] = 1

    return X, y, farm_types


# ───────────────────────────────────────────────
# Main Content Area
# ───────────────────────────────────────────────
if run_button or st.session_state.agrotech_run_history:

    if run_button:
        # Clear previous results
        st.session_state.agrotech_run_history = []

        # Run simulations
        progress_bar = st.progress(0)

        for i in range(n_runs):
            with st.spinner(f"Running agrotech simulation {i + 1}/{n_runs}..."):

                # Generate agricultural data
                X, y, farm_types = generate_agricultural_data(
                    n_samples=sample_size,
                    smallholder_ratio=smallholder_ratio
                )

                # Apply climate stress
                if climate_stress > 0:
                    # Reduce yields based on climate stress
                    climate_reduction = 1 - (climate_stress * 0.5)
                    y *= climate_reduction

                    # Add more variance under stress
                    climate_noise = climate_stress * np.random.normal(0, 2, len(y))
                    y += climate_noise

                    # Apply rainfall and temperature effects
                    rainfall_factor = 1 + (rainfall_deviation / 100) * 0.5
                    temperature_factor = 1 - (temperature_rise * 0.1)
                    y *= rainfall_factor * temperature_factor

                # Apply selected biases (now returns 3 values)
                for bias_type in selected_biases:
                    if bias_type == "geographic":
                        # Geographic bias: certain regions underrepresented
                        region_mask = np.random.rand(len(X)) < 0.7  # 70% from "accessible" regions
                        # Downsample "inaccessible" regions
                        keep_mask = region_mask | (np.random.rand(len(X)) < 0.3)
                        X = X[keep_mask]
                        y = y[keep_mask]
                        farm_types = farm_types[keep_mask]

                    elif bias_type == "socioeconomic":
                        # Socioeconomic bias: wealthier farms have better data quality
                        wealth_proxy = X[:, 3] + X[:, 5]  # Fertilizer + irrigation
                        data_quality = 0.5 + 0.5 * (wealth_proxy - wealth_proxy.min()) / (
                                    wealth_proxy.max() - wealth_proxy.min())
                        # Add more noise to poorer farms' data
                        noise_scale = 1 - data_quality
                        X += np.random.normal(0, noise_scale[:, np.newaxis] * 0.5, X.shape)

                # Apply poisoning attack (now returns 3 values)
                X_p, y_p, farm_types = simulate_data_poisoning(
                    X, y, poison_rate,
                    attack_type="feature_noise" if attack_type == "Sensor Tampering" else "label_flipping",
                    demographic_info=farm_types
                )

                # Split data
                X_train, X_test, y_train, y_test, farm_train, farm_test = train_test_split(
                    X_p, y_p, farm_types, test_size=0.3, random_state=42 + i
                )

                # Standardize features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Train models
                if include_advanced:
                    # Use ensemble of models
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        random_state=42 + i
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=100,
                        random_state=42 + i
                    )

                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)

                # Calculate comprehensive metrics
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                r2 = r2_score(y_test, y_pred)

                # Calculate metrics by farm type
                small_mask = (farm_test == 0)
                large_mask = (farm_test == 1)

                if np.any(small_mask):
                    mae_small = mean_absolute_error(y_test[small_mask], y_pred[small_mask])
                    r2_small = r2_score(y_test[small_mask], y_pred[small_mask])
                    yield_small = np.mean(y_test[small_mask])
                else:
                    mae_small = 0
                    r2_small = 0
                    yield_small = 0

                if np.any(large_mask):
                    mae_large = mean_absolute_error(y_test[large_mask], y_pred[large_mask])
                    r2_large = r2_score(y_test[large_mask], y_pred[large_mask])
                    yield_large = np.mean(y_test[large_mask])
                else:
                    mae_large = 0
                    r2_large = 0
                    yield_large = 0

                # Equity metrics
                mae_disparity = abs(mae_small - mae_large)
                r2_disparity = abs(r2_small - r2_large)
                yield_disparity = abs(yield_small - yield_large) / max(yield_small, yield_large) if max(yield_small,
                                                                                                        yield_large) > 0 else 0

                # Overall equity score (0-1, higher is better)
                equity_score = 1.0 - min(1.0, (mae_disparity * 0.5 + r2_disparity * 0.3 + yield_disparity * 0.2))

                # Sustainability score (accounts for resource use and climate impact)
                avg_fertilizer = np.mean(X_test[:, 3])
                avg_water = np.mean(X_test[:, 1]) * np.mean(X_test[:, 5])  # Rainfall × irrigation

                # Lower resource use with similar yields = more sustainable
                resource_efficiency = np.mean(y_pred) / (avg_fertilizer * 0.01 + avg_water * 0.001)
                sustainability_score = min(1.0, resource_efficiency / 50)  # Normalize

                # Store results
                run_result = {
                    "run_id": i + 1,
                    "crop": crop_type,
                    "region": region,
                    "r2_score": r2,
                    "mae": mae,
                    "rmse": rmse,
                    "equity_score": equity_score,
                    "sustainability_score": sustainability_score,
                    "climate_stress": climate_stress,
                    "poison_rate": poison_rate,
                    "bias_intensity": bias_intensity,
                    "attack_type": attack_type,
                    "smallholder_ratio": smallholder_ratio,
                    "mae_smallholder": mae_small,
                    "mae_large": mae_large,
                    "r2_smallholder": r2_small,
                    "r2_large": r2_large,
                    "yield_smallholder": yield_small,
                    "yield_large": yield_large,
                    "biases": ", ".join(selected_biases) if selected_biases else "None"
                }

                st.session_state.agrotech_run_history.append(run_result)
                progress_bar.progress((i + 1) / n_runs)

        progress_bar.empty()

    # ── Display Results ───────────────────────────────
    if st.session_state.agrotech_run_history:
        df = pd.DataFrame(st.session_state.agrotech_run_history)

        # Summary Metrics Section
        st.markdown("## 📊 Agricultural Performance Dashboard")

        # Calculate averages
        avg_r2 = df["r2_score"].mean()
        avg_mae = df["mae"].mean()
        avg_equity = df["equity_score"].mean()
        avg_sustainability = df["sustainability_score"].mean()

        # Display metrics in styled cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown('<div class="yield-metric">', unsafe_allow_html=True)
            st.metric(
                label="🌾 Prediction Accuracy (R²)",
                value=f"{avg_r2:.3f}",
                delta=None,
                help="How well the model explains yield variance (1.0 = perfect)"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="farm-metric">', unsafe_allow_html=True)
            st.metric(
                label="⚖️ Average Error (MAE)",
                value=f"{avg_mae:.2f} tons/ha",
                delta_color="inverse",
                help="Mean Absolute Error in yield predictions"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col3:
            st.markdown('<div class="equity-metric">', unsafe_allow_html=True)
            st.metric(
                label="🤝 Equity Score",
                value=f"{avg_equity:.2f}/1.0",
                delta=None,
                help="Fairness across farmer groups (1.0 = perfect equity)"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="climate-metric">', unsafe_allow_html=True)
            st.metric(
                label="🌍 Sustainability Score",
                value=f"{avg_sustainability:.2f}/1.0",
                delta=None,
                help="Resource efficiency and climate resilience"
            )
            st.markdown('</div>', unsafe_allow_html=True)

        # Performance comparison by farm size
        st.markdown("### 👨‍🌾 Performance by Farm Size")

        col1, col2 = st.columns(2)

        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                name="Smallholder Farms",
                x=df["run_id"],
                y=df["mae_smallholder"],
                marker_color='#3498db'
            ))
            fig1.add_trace(go.Bar(
                name="Large Farms",
                x=df["run_id"],
                y=df["mae_large"],
                marker_color='#2ecc71'
            ))
            fig1.update_layout(
                title="Prediction Error by Farm Size",
                barmode='group',
                yaxis_title="MAE (tons/ha)",
                xaxis_title="Simulation Run"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                name="Smallholder R²",
                x=df["run_id"],
                y=df["r2_smallholder"],
                mode='lines+markers',
                line=dict(color='#3498db', width=3)
            ))
            fig2.add_trace(go.Scatter(
                name="Large Farm R²",
                x=df["run_id"],
                y=df["r2_large"],
                mode='lines+markers',
                line=dict(color='#2ecc71', width=3)
            ))
            fig2.update_layout(
                title="Model Fit (R²) by Farm Size",
                yaxis_title="R² Score",
                xaxis_title="Simulation Run",
                yaxis_range=[-0.1, 1.0]
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        # ── Detailed Analysis Tabs ───────────────────────────────
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Overview", "⚖️ Equity Analysis",
                                          "🌾 Yield Distribution", "📋 Detailed Results"])

        with tab1:
            # Performance gauges
            fig = make_subplots(
                rows=1, cols=3,
                specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]],
                subplot_titles=("Prediction Accuracy", "Fairness", "Sustainability")
            )

            # R² Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_r2 * 100,
                title={'text': "R² Score", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkgreen"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ), row=1, col=1)

            # Equity Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_equity * 100,
                title={'text': "Equity Score", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ]
                }
            ), row=1, col=2)

            # Sustainability Gauge
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=avg_sustainability * 100,
                title={'text': "Sustainability", 'font': {'size': 16}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkorange"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "gray"},
                        {'range': [70, 100], 'color': "darkgray"}
                    ]
                }
            ), row=1, col=3)

            fig.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            # Trade-off scatter plot
            fig2 = px.scatter(
                df,
                x="equity_score",
                y="r2_score",
                size="sustainability_score",
                color="climate_stress",
                hover_data=["biases", "poison_rate", "attack_type"],
                title="The Agrotech Trilemma: Accuracy vs Equity vs Sustainability",
                labels={
                    "equity_score": "Equity Score",
                    "r2_score": "Prediction Accuracy (R²)",
                    "sustainability_score": "Sustainability",
                    "climate_stress": "Climate Stress"
                },
                size_max=30
            )

            # Add optimal zone
            fig2.add_shape(
                type="rect",
                x0=0.7, x1=1.0,
                y0=0.7, y1=1.0,
                line=dict(color="Green", width=2, dash="dash"),
                fillcolor="rgba(0, 255, 0, 0.1)",
                label=dict(
                    text="Optimal Zone",
                    font=dict(size=12, color="green"),
                    xanchor="center",
                    yanchor="middle")
            )

            st.plotly_chart(fig2, use_container_width=True)

        with tab2:
            # Equity analysis
            st.markdown("### ⚖️ Equity Gap Analysis")

            # Calculate equity gaps
            df["mae_gap"] = abs(df["mae_smallholder"] - df["mae_large"])
            df["r2_gap"] = abs(df["r2_smallholder"] - df["r2_large"])
            df["yield_gap"] = abs(df["yield_smallholder"] - df["yield_large"])

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    df,
                    x="run_id",
                    y=["mae_gap", "r2_gap"],
                    barmode="group",
                    title="Performance Gaps Between Farm Types",
                    labels={"value": "Gap Size", "variable": "Metric"},
                    color_discrete_sequence=["#e74c3c", "#f39c12"]
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Equity impact factors
                factors = pd.DataFrame({
                    "Factor": ["Climate Stress", "Bias Intensity", "Poison Rate", "Smallholder Ratio"],
                    "Impact on Equity": [
                        -df["climate_stress"].mean() * 0.8,  # Negative impact
                        -df["bias_intensity"].mean() * 0.9,
                        -df["poison_rate"].mean() * 0.7,
                        df["smallholder_ratio"].mean() * 0.3  # Positive impact (more representation)
                    ]
                })

                fig2 = px.bar(
                    factors,
                    x="Factor",
                    y="Impact on Equity",
                    title="Factors Affecting Equity",
                    color="Impact on Equity",
                    color_continuous_scale="RdYlGn"
                )
                st.plotly_chart(fig2, use_container_width=True)

            # Equity recommendations
            if avg_equity < 0.7:
                st.error("""
                **⚠️ SIGNIFICANT EQUITY GAPS DETECTED**

                **Recommended Interventions:**
                1. **Data Collection:** Improve representation of smallholder farms
                2. **Bias Mitigation:** Apply fairness constraints in models
                3. **Targeted Support:** Develop separate models for different farm types
                4. **Transparency:** Publish performance metrics by farm size
                """)
            else:
                st.success("""
                **✅ GOOD EQUITY ACHIEVED**

                **Maintenance Actions:**
                1. Continue monitoring performance by farm type
                2. Regular bias audits of data collection
                3. Engage with farmer groups for feedback
                4. Update models as farming practices evolve
                """)

        with tab3:
            # Yield distribution analysis
            st.markdown("### 🌾 Yield Distribution Analysis")

            # Simulated yield distributions for visualization
            np.random.seed(42)
            small_yields = np.random.normal(3.5, 1.0, 1000)
            large_yields = np.random.normal(5.0, 1.5, 1000)

            # Apply climate stress effect
            small_yields *= (1 - climate_stress * 0.3)
            large_yields *= (1 - climate_stress * 0.2)  # Large farms less affected

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=small_yields,
                name="Smallholder Farms",
                opacity=0.7,
                marker_color='#3498db'
            ))
            fig.add_trace(go.Histogram(
                x=large_yields,
                name="Large Farms",
                opacity=0.7,
                marker_color='#2ecc71'
            ))

            fig.update_layout(
                title="Simulated Yield Distribution by Farm Size",
                xaxis_title="Yield (tons/ha)",
                yaxis_title="Number of Farms",
                barmode='overlay'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Yield statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Smallholder Yield", f"{np.mean(small_yields):.2f} tons/ha")
            with col2:
                st.metric("Avg Large Farm Yield", f"{np.mean(large_yields):.2f} tons/ha")
            with col3:
                st.metric("Yield Gap", f"{np.mean(large_yields) - np.mean(small_yields):.2f} tons/ha")

        with tab4:
            # Detailed results table
            st.dataframe(
                df.style.format({
                    "r2_score": "{:.3f}",
                    "mae": "{:.2f}",
                    "rmse": "{:.2f}",
                    "equity_score": "{:.2f}",
                    "sustainability_score": "{:.2f}",
                    "mae_smallholder": "{:.2f}",
                    "mae_large": "{:.2f}",
                    "r2_smallholder": "{:.3f}",
                    "r2_large": "{:.3f}",
                    "yield_smallholder": "{:.2f}",
                    "yield_large": "{:.2f}"
                }).background_gradient(subset=["r2_score"], cmap="Greens")
                .background_gradient(subset=["equity_score"], cmap="RdYlGn")
                .background_gradient(subset=["mae"], cmap="Reds_r")
            )

            # Download options
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Download Results (CSV)",
                    csv,
                    f"gags_agrotech_{crop_type.lower().replace(' ', '_')}.csv",
                    "text/csv",
                    use_container_width=True
                )

            with col2:
                # Export configuration
                config_data = {
                    "crop": crop_type,
                    "region": region,
                    "selected_biases": selected_biases,
                    "bias_intensity": bias_intensity,
                    "attack_type": attack_type,
                    "poison_rate": poison_rate,
                    "climate_stress": climate_stress,
                    "rainfall_deviation": rainfall_deviation,
                    "temperature_rise": temperature_rise,
                    "smallholder_ratio": smallholder_ratio,
                    "average_r2": f"{avg_r2:.3f}",
                    "average_equity": f"{avg_equity:.2f}"
                }
                import json

                json_str = json.dumps(config_data, indent=2)
                st.download_button(
                    "📋 Export Configuration (JSON)",
                    json_str,
                    f"agrotech_config_{crop_type.lower().replace(' ', '_')}.json",
                    "application/json",
                    use_container_width=True
                )

        # ── Policy Recommendations ───────────────────────────────
        st.divider()

        # Generate actionable recommendations
        st.markdown("## 💡 Agricultural Policy Recommendations")

        rec_col1, rec_col2 = st.columns(2)

        with rec_col1:
            st.info("""
            **🌱 For Smallholder Farmers:**

            1. **Digital Inclusion:** Improve access to agricultural apps and data
            2. **Credit Access:** Develop AI-powered microcredit scoring
            3. **Extension Services:** AI-assisted advisory services
            4. **Weather Insurance:** Parametric insurance using AI predictions
            5. **Market Access:** Digital platforms connecting to buyers

            **📊 For Data Collection:**

            1. **Participatory Sensing:** Involve farmers in data collection
            2. **Multilingual Interfaces:** Support local languages
            3. **Low-tech Solutions:** SMS/USSD for data entry
            4. **Privacy Protection:** Secure farmer data
            """)

        with rec_col2:
            st.success("""
            **🤖 For AI Development:**

            1. **Fairness by Design:** Build equity into algorithms
            2. **Explainable AI:** Farmers understand predictions
            3. **Federated Learning:** Train on decentralized data
            4. **Continuous Validation:** Monitor for bias drift
            5. **Farmer Feedback Loops:** Incorporate user feedback

            **🌍 For Sustainability:**

            1. **Precision Agriculture:** Optimize resource use
            2. **Climate Adaptation:** Predict climate impacts
            3. **Biodiversity Monitoring:** Track ecosystem health
            4. **Circular Economy:** Optimize waste-to-resource
            """)

        st.caption(
            "⚠️ **Disclaimer:** This simulation is for educational purposes. Real agricultural AI systems require field validation and farmer consultation.")

else:
    # Welcome/Instruction state
    st.markdown("## 🌱 Welcome to Sustainable Agrotech Simulation")

    st.markdown("""
    This module explores how AI can transform agriculture while ensuring equity and sustainability. 
    You'll configure farming scenarios and see how bias, climate change, and data attacks affect predictions for different farmer groups.
    """)

    # Quick start examples
    st.markdown("### 🚜 Quick Start Scenarios")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="farm-card">
            <h4>🌾 High Productivity</h4>
            <p>Maximize yield predictions:</p>
            <ul>
                <li>Low climate stress</li>
                <li>Moderate mechanization</li>
                <li>All farm sizes represented</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="farm-card">
            <h4>⚖️ Equity Focus</h4>
            <p>Prioritize smallholder fairness:</p>
            <ul>
                <li>High smallholder ratio</li>
                <li>Bias mitigation</li>
                <li>Climate adaptation</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="farm-card">
            <h4>🌍 Climate Resilience</h4>
            <p>Focus on sustainability:</p>
            <ul>
                <li>High climate stress</li>
                <li>Water optimization</li>
                <li>Resource efficiency</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # How to use guide
    with st.expander("📖 How to Use This Simulation", expanded=True):
        st.markdown("""
        1. **Select crop and region** for context
        2. **Configure biases** affecting agricultural data
        3. **Set environmental factors** (climate stress, rainfall, temperature)
        4. **Adjust farm demographics** (smallholder ratio, mechanization)
        5. **Configure adversarial attacks** on agricultural data
        6. **Run multiple simulations** to see statistical trends
        7. **Analyze trade-offs** between accuracy, equity, and sustainability
        8. **Explore recommendations** for policy and implementation

        **Key Metrics to Watch:**
        - **R² Score:** Prediction accuracy (0-1, higher is better)
        - **MAE:** Mean Absolute Error in tons/hectare (lower is better)
        - **Equity Score:** Fairness across farm types (0-1, higher is better)
        - **Sustainability Score:** Resource efficiency (0-1, higher is better)
        - **Performance Gaps:** Differences between smallholder and large farms
        """)

    # Real-world context
    st.warning("""
    **Real-World Context:**

    Agricultural AI faces unique challenges:
    - **Data Scarcity:** Limited data from smallholder farms
    - **Climate Variability:** Changing weather patterns
    - **Infrastructure Gaps:** Limited internet and electricity
    - **Digital Literacy:** Varying farmer tech skills
    - **Market Access:** Unequal access to markets

    Successful agrotech must address these challenges while ensuring benefits reach all farmers.
    """)

# Footer
st.divider()
st.caption("🌱 Sustainable Agrotech Simulation • GAGS Framework • v2.0 • Feeding the Future with Fair AI")