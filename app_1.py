import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GAGS • AI Governance Modeling",
    layout="wide",
    page_icon="🏠",
    initial_sidebar_state="collapsed"
)

# ──────────────────────────────────────────────────────────────────────────────
# Enhanced CSS (Modern, Glassmorphism, & Soft UI)
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');

    html, body, [class*="viewerBadge"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background and Padding */
    .stApp {
        background-color: #f8fafc;
    }

    /* Modern Header */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 5rem 2rem;
        border-radius: 24px;
        margin-bottom: 3.5rem;
        color: white;
        text-align: center;
        box-shadow: 0 20px 50px rgba(0,0,0,0.1);
    }

    /* Glassmorphism Stat Cards */
    .stats-card {
        background: white;
        padding: 1.5rem;
        border-radius: 20px;
        text-align: center;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .stats-card:hover {
        transform: translateY(-5px);
    }

    /* Module Cards */
    .module-card {
        background: white;
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid #f1f5f9;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.05);
        height: 100%;
    }

    .badge {
        padding: 4px 12px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-active { background: #dcfce7; color: #166534; }
    .badge-new { background: #fef9c3; color: #854d0e; }

    /* Custom Button Styling to override Streamlit defaults */
    div.stButton > button {
        border-radius: 12px;
        border: none;
        padding: 0.6rem 1rem;
        background: #4f46e5;
        color: white;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: #4338ca;
        box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.4);
        transform: scale(1.02);
    }

    .stat-number {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e293b;
        margin: 0.5rem 0;
    }

    .section-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="font-size:4rem; font-weight:800; margin-bottom:1rem; letter-spacing:-0.02em;">GAGS</h1>
    <p style="font-size:1.4rem; opacity:0.8; font-weight:400; max-width:700px; margin:0 auto;">
        Advancing AI Transparency through Governance Modeling and Stress-Testing
    </p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Stats Row
# ──────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
stats = [
    ("Simulations", "2,489", "↑ 12%", "#6366f1"),
    ("Bias Types", "12", "Active", "#10b981"),
    ("Security", "9", "Hardened", "#f43f5e"),
    ("Active Users", "842", "Global", "#f59e0b")
]

for i, col in enumerate([col1, col2, col3, col4]):
    label, val, delta, color = stats[i]
    col.markdown(f"""
    <div class="stats-card">
        <p style="color:#64748b; font-size:0.85rem; font-weight:600; margin:0;">{label}</p>
        <div class="stat-number" style="color:{color};">{val}</div>
        <p style="color:#94a3b8; font-size:0.75rem; margin:0;">{delta}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Modules Grid
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">🚀 Simulation Modules</p>', unsafe_allow_html=True)
st.markdown('<p style="color:#64748b; margin-bottom:2rem;">Select a domain to begin stress-testing AI governance policies.</p>', unsafe_allow_html=True)

def module_card(icon, title, badge_text, badge_type, description, tags, key, page):
    st.markdown(f"""
    <div class="module-card">
        <div style="font-size:2rem; margin-bottom:1rem;">{icon}</div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem;">
            <h3 style="margin:0; font-size:1.25rem; font-weight:700; color:#1e293b;">{title}</h3>
            <span class="badge badge-{badge_type}">{badge_text}</span>
        </div>
        <p style="color:#475569; font-size:0.9rem; line-height:1.6; min-height:80px;">{description}</p>
        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:1.5rem;">
            {" ".join([f'<span style="background:#f1f5f9; color:#475569; padding:4px 10px; border-radius:8px; font-size:0.75rem;">{t}</span>' for t in tags])}
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button(f"Launch {title} →", key=key):
        st.switch_page(page)

# Grid Layout
m_col1, m_col2, m_col3 = st.columns(3)

with m_col1:
    module_card("🏥", "Healthcare", "Active", "active",
                "Evaluate clinical bias and data poisoning impact on patient diagnostic equity.",
                ["Fairness", "HIPAA", "Poisoning"], "btn_health", "pages/02_Healthcare_Equity.py")

with m_col2:
    module_card("🛡️", "Security", "Active", "active",
                "Balance threat detection with civil liberties in national surveillance simulations.",
                ["Ethics", "Surveillance", "Recall"], "btn_sec", "pages/2_National_Security.py")

with m_col3:
    module_card("🌱", "Agrotech", "New", "new",
                "Analyze climate-driven bias in crop yield predictions and resource distribution.",
                ["Sustainability", "Agri-AI", "Equity"], "btn_agro", "pages/3_Sustainable_Agrotech.py")

st.markdown("<br>", unsafe_allow_html=True)

m_col4, m_col5, m_col6 = st.columns(3)

with m_col4:
    module_card("🎓", "Education", "Active", "active",
                "Simulate admissions fairness and algorithmic grading across demographic splits.",
                ["Access", "ED-Tech", "Grading"], "btn_edu", "pages/Education_Equity.py")

with m_col5:
    module_card("⚖️", "Judicial", "Active", "active",
                "Test recidivism algorithms for racial and socioeconomic disparate impact.",
                ["Recidivism", "Law", "Bias"], "btn_jud", "pages/Judicial_System.py")

with m_col6:
    module_card("💰", "Finance", "Active", "active",
                "Explore algorithmic credit scoring and inclusive lending for marginalized groups.",
                ["Credit", "FICO", "Lending"], "btn_fin", "pages/Financial_Inclusion.py")

# ──────────────────────────────────────────────────────────────────────────────
# Analytics Preview
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">📊 Aggregate Performance</p>', unsafe_allow_html=True)

c1, c2 = st.columns([2, 1])

with c1:
    df = pd.DataFrame({
        'Domain': ['Health', 'Agro', 'Security', 'Finance', 'Edu', 'Law'],
        'Fairness': [0.82, 0.78, 0.65, 0.70, 0.75, 0.68],
        'Accuracy': [0.88, 0.85, 0.92, 0.80, 0.83, 0.79]
    })
    fig = px.bar(df, x='Domain', y=['Fairness', 'Accuracy'], barmode='group',
                 color_discrete_sequence=['#4f46e5', '#10b981'])
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend_title=None,
        margin=dict(t=10, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("""
    <div style="background:white; padding:2rem; border-radius:24px; border:1px solid #f1f5f9;">
        <h4 style="margin-top:0;">Insights Summary</h4>
        <p style="font-size:0.9rem; color:#64748b;">The current aggregate fairness score across all active simulations is <b>0.74</b>.</p>
        <hr style="border:0.5px solid #f1f5f9;">
        <ul style="font-size:0.85rem; color:#475569; padding-left:1.2rem;">
            <li>Security modules show highest accuracy.</li>
            <li>Healthcare modules currently have the highest bias variance.</li>
            <li>Agrotech equity scores improved by 5% this week.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)