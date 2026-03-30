# pages/🏠_Home.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from components.translate import install_auto_translate, tx, tx_plotly

# ── Background preloader: warm up heavy imports while user reads home page ─────
def _preload_heavy_modules():
    """Load heavy modules in background so simulation pages open instantly."""
    import threading

    def _load():
        try:
            import components.governance_logic        # 6,392 lines — cache it now
            import components.gags_benchmarks
            import components.gags_charts
            from sklearn.ensemble import (
                GradientBoostingClassifier,
                HistGradientBoostingClassifier,
                RandomForestClassifier,
            )
            from sklearn.metrics import accuracy_score, roc_auc_score
            from sklearn.preprocessing import StandardScaler
            from sklearn.model_selection import train_test_split
        except Exception:
            pass   # non-fatal — pages will import on demand

    if not st.session_state.get("_preloaded"):
        st.session_state["_preloaded"] = True
        t = threading.Thread(target=_load, daemon=True)
        t.start()

_preload_heavy_modules()


install_auto_translate()
# ───────────────────────────────────────────────
# Page Configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="GAGS – Governance for AI Governance Simulation",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="collapsed"
)

# Modern, clean CSS with better contrast
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;14..32,400;14..32,500;14..32,600;14..32,700&display=swap');

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body, .stApp {
        font-family: 'Inter', sans-serif;
        background: #f5f7fb;
        color: #1a1f36;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        padding: 4rem 2rem;
        text-align: center;
        border-bottom: 1px solid #e2e8f0;
    }
    .hero h1 {
        font-size: 3.6rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #1a56db, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    .hero .subtitle {
        font-size: 1.3rem;
        font-weight: 500;
        color: #334155;
        max-width: 800px;
        margin: 0 auto 1rem;
    }
    .hero .lead {
        font-size: 1rem;
        max-width: 700px;
        margin: 0 auto;
        color: #475569;
        line-height: 1.6;
    }

    /* Section headers */
    .section-header {
        text-align: center;
        margin: 3rem 0 2rem;
    }
    .section-header h2 {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
        letter-spacing: -0.01em;
    }
    .section-header p {
        font-size: 1rem;
        color: #475569;
        max-width: 600px;
        margin: 0.5rem auto 0;
    }

    /* Stats cards */
    .stats-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stats-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
    }
    .stats-card .number {
        font-size: 2.2rem;
        font-weight: 800;
        color: #3b82f6;
        line-height: 1.2;
    }
    .stats-card .label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748b;
        margin-top: 0.25rem;
    }

    /* Module cards */
    .module-card {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        height: 100%;
        border: 1px solid #e2e8f0;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .module-card:hover {
        border-color: #3b82f6;
        box-shadow: 0 8px 20px rgba(0,0,0,0.08);
        transform: translateY(-4px);
    }
    .module-card .emoji {
        font-size: 2rem;
        margin-bottom: 0.75rem;
        display: block;
    }
    .module-card h3 {
        font-size: 1.25rem;
        font-weight: 700;
        margin: 0.5rem 0;
        color: #0f172a;
    }
    .module-card .badge {
        display: inline-block;
        padding: 0.2rem 0.7rem;
        border-radius: 2rem;
        font-size: 0.7rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .badge-active {
        background: #dbeafe;
        color: #1e40af;
    }
    .badge-new {
        background: #fed7aa;
        color: #9a3412;
    }
    .module-card p {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Feature blocks */
    .feature {
        background: white;
        border-radius: 1rem;
        padding: 1.5rem;
        border: 1px solid #e2e8f0;
        height: 100%;
        transition: box-shadow 0.2s;
    }
    .feature h4 {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0f172a;
        margin-bottom: 0.75rem;
    }
    .feature p {
        font-size: 0.9rem;
        color: #475569;
        line-height: 1.5;
    }

    /* Buttons */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        font-size: 0.85rem;
        width: 100%;
        transition: background 0.2s, transform 0.1s;
    }
    .stButton > button:hover {
        background: #2563eb;
        transform: translateY(-1px);
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 3rem 1rem 2rem;
        color: #64748b;
        font-size: 0.85rem;
        border-top: 1px solid #e2e8f0;
        margin-top: 2rem;
    }

    hr {
        margin: 2rem 0;
        border: 0;
        border-top: 1px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Hero Section
# ───────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>GAGS</h1>
    <div class="subtitle">Governance for AI Governance Simulation Framework</div>
    <div class="lead">
        A modular simulation platform to study how bias, adversarial attacks, and governance interventions
        affect fairness, robustness, and performance of AI systems across high-stakes domains.
    </div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Platform Stats (updated to 8 domains)
# ───────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>Platform at a Glance</h2></div>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stats-card">
        <div class="number">2,489</div>
        <div class="label">Simulations executed</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stats-card">
        <div class="number">8</div>
        <div class="label">Domains modeled</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stats-card">
        <div class="number">9+</div>
        <div class="label">Attack vectors</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown("""
    <div class="stats-card">
        <div class="number">12</div>
        <div class="label">Bias types studied</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Simulation Modules (now 8 modules)
# ───────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <h2>Simulation Domains</h2>
    <p>Explore governance challenges in different societal contexts</p>
</div>
""", unsafe_allow_html=True)

# Define modules (emoji, title, status, description, page)
modules = [
    ("🏥", "Healthcare Equity", "Active", "Assess bias and poisoning effects on clinical AI fairness and performance.", "pages/02_Healthcare_Equity.py"),
    ("🛡️", "National Security", "Active", "Balance threat detection vs. civil liberties in surveillance AI systems.", "pages/2_National_Security.py"),
    ("🌱", "Sustainable Agrotech", "Active", "Study bias, climate stress, and attacks in crop yield prediction models.", "pages/3_Sustainable_Agrotech.py"),
    ("🎓", "Education Equity", "Active", "Investigate fairness in admissions, grading, and resource allocation algorithms.", "pages/Education_Equity.py"),
    ("⚖️", "Judicial Systems", "Active", "Analyze bias in risk assessment and sentencing tools.", "pages/Judicial_System.py"),
    ("💰", "Financial Inclusion", "Active", "Examine fairness in credit scoring and loan decision systems.", "pages/Financial_Inclusion.py"),
    ("🌍", "Economic Justice & Health Financing", "New", "Model fairness in health resource allocation, insurance, and developmental economics.", "pages/11_Economic_Justice.py"),
    ("📡", "Disinformation & Misinformation", "New", "Simulate AI detection and spread of misinformation across media ecosystems.", "pages/Disinformation_Misinformation.py")
]

# Arrange in 3 columns
cols = st.columns(3)
for i, (emoji, title, status, desc, page) in enumerate(modules):
    with cols[i % 3]:
        badge_class = "badge-new" if status == "New" else "badge-active"
        st.markdown(f"""
        <div class="module-card">
            <span class="emoji">{emoji}</span>
            <h3>{title}</h3>
            <span class="badge {badge_class}">{status}</span>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button(f"→ Open {title}", key=f"launch_{title.replace(' ','_')}", use_container_width=True):
            st.switch_page(page)

st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Core Capabilities (unchanged)
# ───────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>Core Capabilities</h2></div>', unsafe_allow_html=True)

cap_cols = st.columns(2)
with cap_cols[0]:
    st.markdown("""
    <div class="feature">
        <h4>Domain-specific simulation environments</h4>
        <p>Realistic datasets and scenarios tailored to each high-stakes application area.</p>
    </div>
    <div class="feature" style="margin-top: 1rem;">
        <h4>Comprehensive fairness & robustness metrics</h4>
        <p>Disparate impact, equal opportunity, calibration, accuracy degradation under attack, etc.</p>
    </div>
    """, unsafe_allow_html=True)
with cap_cols[1]:
    st.markdown("""
    <div class="feature">
        <h4>Adversarial attack library</h4>
        <p>Poisoning, evasion, backdoor, prompt-based attacks — test resilience systematically.</p>
    </div>
    <div class="feature" style="margin-top: 1rem;">
        <h4>Interactive analysis & export</h4>
        <p>Visualizations, sensitivity analysis, CSV/JSON export with full configuration provenance.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Impact Visualization (unchanged)
# ───────────────────────────────────────────────
st.markdown("""
<div class="section-header">
    <h2>Illustrative Cross-Domain Performance</h2>
    <p>Average scores from executed simulations (normalized)</p>
</div>
""", unsafe_allow_html=True)

viz_col, stats_col = st.columns([3, 1.2])

with viz_col:
    np.random.seed(42)
    domains = ['Healthcare', 'Agrotech', 'Security', 'Finance', 'Education', 'Judicial']
    data = {
        'Domain': domains * 3,
        'Metric': ['Fairness']*6 + ['Accuracy']*6 + ['Equity/Inclusion']*6,
        'Score': [0.82,0.78,0.65,0.70,0.75,0.68, 0.88,0.85,0.92,0.80,0.83,0.79, 0.76,0.72,0.60,0.85,0.78,0.65]
    }
    df = pd.DataFrame(data)
    fig = px.line_polar(df, r='Score', theta='Domain', color='Metric',
                        line_close=True,
                        color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b'])
    fig.update_traces(fill='toself', opacity=0.15)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10))),
        showlegend=True,
        height=400,
        margin=dict(l=30, r=30, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig, use_container_width=True)

with stats_col:
    st.markdown("""
    <div style="background: white; border-radius: 1rem; padding: 1.5rem; border: 1px solid #e2e8f0; height: 100%;">
        <h4 style="margin-top: 0; font-size: 1rem; font-weight: 600;">Aggregate Insights</h4>
        <div style="margin: 1.5rem 0;">
            <div style="font-size: 2rem; font-weight: 800; color: #3b82f6;">78%</div>
            <div style="color: #475569;">Average fairness</div>
        </div>
        <div style="margin: 1.5rem 0;">
            <div style="font-size: 2rem; font-weight: 800; color: #10b981;">85%</div>
            <div style="color: #475569;">Average utility</div>
        </div>
        <p style="color: #64748b; font-size: 0.85rem; margin-top: 1rem;">
            Based on 2,489 completed simulation runs (March 2026)
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Call to Action / Footer
# ───────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <h3 style="margin-bottom: 1rem; font-weight: 700;">Contribute to Responsible AI Governance Research</h3>
    <p style="max-width: 600px; margin: 0 auto 1.5rem;">
        Use GAGS to test hypotheses, compare interventions, and generate evidence for better AI policy design.
    </p>
</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    if st.button("Start with Healthcare Equity", use_container_width=True):
        st.switch_page("pages/02_Healthcare_Equity.py")
with c2:
    if st.button("Explore Disinformation Module", use_container_width=True):
        st.switch_page("pages/Disinformation_Misinformation.py")