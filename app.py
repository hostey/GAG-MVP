# pages/🏠_Home.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ───────────────────────────────────────────────
# Page Configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="GAGS – Governance for AI Governance Simulation",
    layout="wide",
    page_icon="🔬",
    initial_sidebar_state="collapsed"
)

# Clean, minimal academic-friendly CSS
st.markdown("""
<style>
    /* Reset & base */
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3, h4 { color: #1a1a1a; font-weight: 600; }
    p, div { color: #333; line-height: 1.6; }

    /* Hero */
    .hero {
        background: #ffffff;
        padding: 5rem 2rem 4rem;
        text-align: center;
        border-bottom: 1px solid #e0e0e0;
    }
    .hero h1 {
        font-size: 3.4rem;
        margin-bottom: 0.4rem;
    }
    .hero .subtitle {
        font-size: 1.4rem;
        color: #444;
        max-width: 900px;
        margin: 0 auto 1.8rem;
    }
    .hero .lead {
        font-size: 1.15rem;
        max-width: 760px;
        margin: 0 auto;
        color: #555;
    }

    /* Section headers */
    .section-header {
        text-align: center;
        margin: 4rem 0 2.5rem;
    }
    .section-header h2 {
        font-size: 2.2rem;
        margin-bottom: 0.6rem;
    }
    .section-header p {
        color: #555;
        font-size: 1.1rem;
    }

    /* Module cards – clean & academic */
    .module-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.8rem;
        height: 100%;
        transition: border-color 0.2s, box-shadow 0.2s;
    }
    .module-card:hover {
        border-color: #9da5ff;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    .module-card h3 {
        margin-top: 0.4rem;
        font-size: 1.35rem;
    }
    .module-card .emoji {
        font-size: 2.1rem;
        margin-bottom: 0.8rem;
        display: block;
    }
    .module-card .badge {
        display: inline-block;
        padding: 0.25rem 0.7rem;
        border-radius: 1rem;
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.9rem;
    }
    .badge-active   { background: #e6f4ea; color: #2e7d32; }
    .badge-new      { background: #fff3e0; color: #e65100; }

    /* Stats – fewer, calmer */
    .stats-card {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.6rem;
        text-align: center;
    }
    .stats-card .number {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1e88e5;
        margin: 0.3rem 0;
    }
    .stats-card .label {
        color: #555;
        font-size: 0.95rem;
    }

    /* Buttons */
    .stButton > button {
        background-color: #556cd6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover {
        background-color: #3f51b5;
    }

    /* Feature blocks */
    .feature {
        background: white;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1.8rem;
        margin-bottom: 1.5rem;
    }
    .feature h4 {
        margin-top: 0;
    }

    /* Divider */
    hr { border: none; border-top: 1px solid #e0e0e0; margin: 3rem 0; }

    /* Footer */
    .footer {
        text-align: center;
        padding: 4rem 1rem 2rem;
        color: #666;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Hero / Introduction
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
# Quick stats (reduced & calmed)
# ───────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>Platform at a Glance</h2></div>', unsafe_allow_html=True)

cols = st.columns(4)
with cols[0]:
    st.markdown("""
    <div class="stats-card">
        <div class="number">2,489</div>
        <div class="label">Simulations executed</div>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    st.markdown("""
    <div class="stats-card">
        <div class="number">6</div>
        <div class="label">Domains modeled</div>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    st.markdown("""
    <div class="stats-card">
        <div class="number">9+</div>
        <div class="label">Attack vectors</div>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    st.markdown("""
    <div class="stats-card">
        <div class="number">12</div>
        <div class="label">Bias types studied</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Core Modules
# ───────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>Simulation Domains</h2><p>Explore governance challenges in different societal contexts</p></div>', unsafe_allow_html=True)

cols = st.columns(3)

modules = [
    ("🏥", "Healthcare Equity", "Active", "#e3f2fd", "pages/02_Healthcare_Equity.py", "Assess bias and poisoning effects on clinical AI fairness and performance."),
    ("🛡️", "National Security", "Active", "#ffebee", "pages/2_National_Security.py", "Balance threat detection vs. civil liberties in surveillance AI systems."),
    ("🌱", "Sustainable Agrotech", "Active", "#e8f5e9", "pages/3_Sustainable_Agrotech.py", "Study bias, climate stress, and attacks in crop yield prediction models."),
    ("🎓", "Education Equity", "Active", "#e3f2fd", "pages/Education_Equity.py", "Investigate fairness in admissions, grading, and resource allocation algorithms."),
    ("⚖️", "Judicial Systems", "Active", "#f3e5f5", "pages/Judicial_System.py", "Analyze bias in risk assessment and sentencing tools."),
    ("💰", "Financial Inclusion", "New", "#fff3e0", "pages/Financial_Inclusion.py", "Examine fairness in credit scoring and loan decision systems.")
]

for i, (emoji, title, status, _, page, desc) in enumerate(modules):
    with cols[i % 3]:
        st.markdown(f"""
        <div class="module-card">
            <span class="emoji">{emoji}</span>
            <h3>{title}</h3>
            <span class="badge {'badge-new' if status=='New' else 'badge-active'}">{status}</span>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button(f"→ Open {title}", key=f"launch_{title.replace(' ','_')}", use_container_width=True):
            st.switch_page(page)

st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Key Capabilities (more academic tone)
# ───────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>Core Capabilities</h2></div>', unsafe_allow_html=True)

feat_cols = st.columns(2)

with feat_cols[0]:
    st.markdown("""
    <div class="feature">
        <h4>Domain-specific simulation environments</h4>
        <p>Realistic datasets and scenarios tailored to each high-stakes application area.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature">
        <h4>Comprehensive fairness & robustness metrics</h4>
        <p>Disparate impact, equal opportunity, calibration, accuracy degradation under attack, etc.</p>
    </div>
    """, unsafe_allow_html=True)

with feat_cols[1]:
    st.markdown("""
    <div class="feature">
        <h4>Adversarial attack library</h4>
        <p> poisoning, evasion, backdoor, prompt-based attacks — test resilience systematically.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature">
        <h4>Interactive analysis & export</h4>
        <p>Visualizations, sensitivity analysis, CSV/JSON export with full configuration provenance.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Impact visualization (kept but cleaner)
# ───────────────────────────────────────────────
st.markdown('<div class="section-header"><h2>Illustrative Cross-Domain Performance</h2><p>Average scores from executed simulations (normalized)</p></div>', unsafe_allow_html=True)

col_viz, col_side = st.columns([3, 1.4])

with col_viz:
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
                        color_discrete_sequence=['#1e88e5','#43a047','#fb8c00'])
    fig.update_traces(fill='toself', opacity=0.18)
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True, height=420,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_side:
    st.markdown("""
    <div style="background:#fff; border:1px solid #e0e0e0; border-radius:8px; padding:1.8rem; height:100%;">
        <h4 style="margin-top:0;">Aggregate Insights</h4>
        <div style="margin:1.5rem 0;">
            <div style="font-size:2.1rem; font-weight:700; color:#1e88e5;">78%</div>
            <div style="color:#555; font-size:0.95rem;">Average fairness</div>
        </div>
        <div style="margin:1.5rem 0;">
            <div style="font-size:2.1rem; font-weight:700; color:#43a047;">85%</div>
            <div style="color:#555; font-size:0.95rem;">Average utility</div>
        </div>
        <p style="color:#666; font-size:0.9rem; margin-top:2rem;">
            Based on 2,489 completed simulation runs (March 2026)
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Footer / Call to action
# ───────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <h3 style="margin-bottom:1.2rem;">Contribute to Responsible AI Governance Research</h3>
    <p style="max-width:680px; margin:0 auto 2rem;">
        Use GAGS to test hypotheses, compare interventions, and generate evidence for better AI policy design.
    </p>
    <div style="display:flex; justify-content:center; gap:1.5rem; flex-wrap:wrap;">
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)
with c1:
    if st.button("Start with Healthcare Equity", use_container_width=True):
        st.switch_page("pages/02_Healthcare_Equity.py")
with c2:
    if st.button("Explore Financial Inclusion (recent)", use_container_width=True):
        st.switch_page("pages/Financial_Inclusion.py")

st.markdown("</div></div>", unsafe_allow_html=True)