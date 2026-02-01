# pages/🏠_Home.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ───────────────────────────────────────────────
# Page Configuration
# ───────────────────────────────────────────────
st.set_page_config(
    page_title="GAGS • Governance for AI Governance Simulation",
    layout="wide",
    page_icon="🏠",
    initial_sidebar_state="collapsed"
)

# Custom CSS with modern gradient designs
st.markdown("""
<style>
    /* Main gradient background */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 4rem 2rem;
        border-radius: 0 0 20px 20px;
        margin-bottom: 3rem;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }

    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 1px, transparent 1px);
        background-size: 50px 50px;
        animation: float 20s linear infinite;
        opacity: 0.3;
    }

    @keyframes float {
        0% { transform: translate(0, 0) rotate(0deg); }
        100% { transform: translate(-50px, -50px) rotate(360deg); }
    }

    .feature-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
        border: 1px solid rgba(0,0,0,0.05);
        height: 100%;
        position: relative;
        overflow: hidden;
    }

    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, var(--card-color), transparent);
    }

    .feature-card:hover {
        transform: translateY(-10px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.15);
    }

    .stats-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border-left: 5px solid var(--card-color);
    }

    .module-card {
        background: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 5px 20px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        border: 2px solid transparent;
        height: 100%;
        text-decoration: none !important;
        display: block;
        color: inherit;
    }

    .module-card:hover {
        transform: scale(1.02);
        box-shadow: 0 15px 30px rgba(0,0,0,0.1);
        border-color: var(--module-color);
        text-decoration: none;
        color: inherit;
    }

    .quick-start-card {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        padding: 2rem;
        border-radius: 15px;
        color: #2c3e50;
    }

    .news-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #3498db;
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.1);
    }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        border: none;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        text-align: center;
        margin: 0.5rem;
        width: 100%;
    }

    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        color: white;
    }

    .pulse {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .floating-icon {
        animation: float-icon 3s ease-in-out infinite;
    }

    @keyframes float-icon {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
    }

    .badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: bold;
        margin: 0.2rem;
    }

    .badge-new {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
    }

    .badge-update {
        background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);
        color: white;
    }

    .badge-active {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        color: white;
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .highlight-text {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: bold;
    }

    .divider {
        height: 3px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
        border-radius: 3px;
    }

    .module-link {
        text-decoration: none;
        color: inherit;
    }

    .module-link:hover {
        text-decoration: none;
        color: inherit;
    }

    .launch-button {
        background: linear-gradient(135deg, var(--btn-color-start), var(--btn-color-end));
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 50px;
        border: none;
        font-weight: bold;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        text-align: center;
        width: 100%;
        margin-top: 1rem;
    }

    .launch-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        color: white;
        text-decoration: none;
    }
</style>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Header Section
# ───────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div style="position: relative; z-index: 2;">
        <h1 style="margin:0; font-size:3.5rem; font-weight:800;">GAGS</h1>
        <p style="font-size:1.2rem; opacity:0.9; margin-top:0.5rem;">
            Governance for AI Governance Simulation Framework
        </p>
        <p style="font-size:1.1rem; max-width:800px; margin:1.5rem auto; opacity:0.9;">
            Explore, simulate, and understand how bias, attacks, and governance policies 
            impact AI systems across different domains
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Quick Stats Section
# ───────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-bottom:3rem;">
    <h2 style="color:#2c3e50;">Global AI Governance Insights</h2>
    <p style="color:#7f8c8d; max-width:800px; margin:0 auto;">
        Real-time simulation metrics and impact analysis across domains
    </p>
</div>
""", unsafe_allow_html=True)

# Stats cards
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="stats-card" style="--card-color:#667eea;">
        <div style="font-size:0.9rem; color:#7f8c8d;">Simulations Run</div>
        <div class="stat-number">2,489</div>
        <div style="font-size:0.8rem; color:#27ae60;">↑ 38% this month</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stats-card" style="--card-color:#4ecdc4;">
        <div style="font-size:0.9rem; color:#7f8c8d;">Bias Types Analyzed</div>
        <div class="stat-number">12</div>
        <div style="font-size:0.8rem; color:#3498db;">+4 new types</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stats-card" style="--card-color:#ff6b6b;">
        <div style="font-size:0.9rem; color:#7f8c8d;">Attack Vectors</div>
        <div class="stat-number">9</div>
        <div style="font-size:0.8rem; color:#e74c3c;">Critical: 3</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stats-card" style="--card-color:#f39c12;">
        <div style="font-size:0.9rem; color:#7f8c8d;">Active Users</div>
        <div class="stat-number">842</div>
        <div style="font-size:0.8rem; color:#2ecc71;">↑ 32% this week</div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ───────────────────────────────────────────────
# All Simulation Modules Section
# ───────────────────────────────────────────────
st.markdown('<a name="modules"></a>', unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom:3rem;">
    <h2 style="color:#2c3e50;">🚀 All Simulation Modules</h2>
    <p style="color:#7f8c8d;">
        Dive into real-world AI governance challenges across different domains
    </p>
</div>
""", unsafe_allow_html=True)

# First row of modules
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
        <div class="module-card" style="--module-color:#667eea;">
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding:0.8rem; border-radius:12px; margin-right:1rem;">
                    <span style="font-size:1.5rem;">🏥</span>
                </div>
                <div>
                    <h3 style="margin:0; color:#2c3e50;">Healthcare Equity</h3>
                    <span class="badge badge-active">Active</span>
                </div>
            </div>
            <p style="color:#7f8c8d; margin-bottom:1.5rem;">
                Test how different types of bias and data poisoning affect 
                model fairness and accuracy in healthcare AI systems.
            </p>
            <div style="margin-bottom:1rem;">
                <span style="font-size:0.8rem; color:#667eea;">🔬 Key Metrics:</span>
                <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.5rem;">
                    <span style="background:#eef2ff; color:#667eea; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Fairness Score
                    </span>
                    <span style="background:#eef2ff; color:#667eea; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Accuracy Result
                    </span>
                    <span style="background:#eef2ff; color:#667eea; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Bias Impact
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button(
        "🚀 Launch Simulation →",
        key="launch_healthcare_equity",
        use_container_width=True
    ):
        st.switch_page("pages/02_Healthcare_Equity.py")

with col2:
    st.markdown(f"""
        <div class="module-card" style="--module-color:#e74c3c;">
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <div style="background:linear-gradient(135deg, #e74c3c 0%, #c0392b 100%); 
                            padding:0.8rem; border-radius:12px; margin-right:1rem;">
                    <span style="font-size:1.5rem;">🛡️</span>
                </div>
                <div>
                    <h3 style="margin:0; color:#2c3e50;">National Security</h3>
                    <span class="badge badge-active">Active</span>
                </div>
            </div>
            <p style="color:#7f8c8d; margin-bottom:1.5rem;">
                Balance threat detection with civil liberties in AI-powered 
                surveillance systems. Test bias and attack resilience.
            </p>
            <div style="margin-bottom:1rem;">
                <span style="font-size:0.8rem; color:#e74c3c;">🎯 Key Metrics:</span>
                <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.5rem;">
                    <span style="background:#fdedec; color:#e74c3c; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Detection Rate
                    </span>
                    <span style="background:#fdedec; color:#e74c3c; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Liberty Score
                    </span>
                    <span style="background:#fdedec; color:#e74c3c; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        False Positives
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button(
        "🚀 Launch Simulation →",
        key="launch_National_Security",
        use_container_width=True
    ):
        st.switch_page("pages/2_National_Security.py")


with col3:
    st.markdown(f"""
        <div class="module-card" style="--module-color:#2ecc71;">
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <div style="background:linear-gradient(135deg, #2ecc71 0%, #27ae60 100%); 
                            padding:0.8rem; border-radius:12px; margin-right:1rem;">
                    <span style="font-size:1.5rem;">🌱</span>
                </div>
                <div>
                    <h3 style="margin:0; color:#2c3e50;">Sustainable Agrotech</h3>
                    <span class="badge badge-active">Active</span>
                </div>
            </div>
            <p style="color:#7f8c8d; margin-bottom:1.5rem;">
                Explore how bias, climate stress, and data attacks affect 
                AI-driven crop yield predictions and farmer equity.
            </p>
            <div style="margin-bottom:1rem;">
                <span style="font-size:0.8rem; color:#2ecc71;">🌾 Key Metrics:</span>
                <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.5rem;">
                    <span style="background:#e8f8f5; color:#27ae60; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Equity Score
                    </span>
                    <span style="background:#e8f8f5; color:#27ae60; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Sustainability
                    </span>
                    <span style="background:#e8f8f5; color:#27ae60; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Yield Accuracy
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button(
        "🚀 Launch Simulation →",
        key="launch_Sustainable_Agrotech",
        use_container_width=True
    ):
        st.switch_page("pages/3_Sustainable_Agrotech.py")

# Second row of modules
st.markdown("<br>", unsafe_allow_html=True)
col4, col5, col6 = st.columns(3)

with col4:
    st.markdown(f"""
        <div class="module-card" style="--module-color:#3498db;">
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <div style="background:linear-gradient(135deg, #3498db 0%, #2980b9 100%); 
                            padding:0.8rem; border-radius:12px; margin-right:1rem;">
                    <span style="font-size:1.5rem;">🎓</span>
                </div>
                <div>
                    <h3 style="margin:0; color:#2c3e50;">Education Equity</h3>
                    <span class="badge badge-active">Active</span>
                </div>
            </div>
            <p style="color:#7f8c8d; margin-bottom:1.5rem;">
                Explore fairness in admissions, grading, resource allocation, 
                and student support systems across demographic groups.
                Explore Education Equity
            </p>
            <div style="margin-bottom:1rem;">
                <span style="font-size:0.8rem; color:#3498db;">📚 Key Metrics:</span>
                <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.5rem;">
                    <span style="background:#e8f4f8; color:#3498db; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Opportunity Score
                    </span>
                    <span style="background:#e8f4f8; color:#3498db; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Access Equity
                    </span>
                    <span style="background:#e8f4f8; color:#3498db; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Performance Gap
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button(
        "🚀 Launch Simulation →",
        key="launch_Education_Equity",
        use_container_width=True
    ):
        st.switch_page("pages/Education_Equity.py")


with col5:
    st.markdown(f"""
        <div class="module-card" style="--module-color:#9b59b6;">
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <div style="background:linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%); 
                            padding:0.8rem; border-radius:12px; margin-right:1rem;">
                    <span style="font-size:1.5rem;">⚖️</span>
                </div>
                <div>
                    <h3 style="margin:0; color:#2c3e50;">Judicial Systems</h3>
                    <span class="badge badge-active">Active</span>
                </div>
            </div>
            <p style="color:#7f8c8d; margin-bottom:1.5rem;">
                Analyze bias in risk assessment, sentencing algorithms, 
                and their impact on different demographic groups across judicial systems.
            </p>
            <div style="margin-bottom:1rem;">
                <span style="font-size:0.8rem; color:#9b59b6;">⚖️ Key Metrics:</span>
                <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.5rem;">
                    <span style="background:#f4ecf7; color:#9b59b6; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Rule of Law
                    </span>
                    <span style="background:#f4ecf7; color:#9b59b6; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Fairness Index
                    </span>
                    <span style="background:#f4ecf7; color:#9b59b6; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Disparity Impact
                    </span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    if st.button(
        "🚀 Launch Simulation →",
        key="launch_Judicial_System",
        use_container_width=True
    ):
        st.switch_page("pages/Judicial_System.py")

with col6:
    st.markdown(f"""
        <div class="module-card" style="--module-color:#f39c12;">
            <div style="display:flex; align-items:center; margin-bottom:1rem;">
                <div style="background:linear-gradient(135deg, #f39c12 0%, #d35400 100%); 
                            padding:0.8rem; border-radius:12px; margin-right:1rem;">
                    <span style="font-size:1.5rem;">💰</span>
                </div>
                <div>
                    <h3 style="margin:0; color:#2c3e50;">Financial Inclusion</h3>
                    <span class="badge badge-new">New</span>
                </div>
            </div>
            <p style="color:#7f8c8d; margin-bottom:1.5rem;">
                Explore AI fairness in credit scoring, loan approvals, 
                and access to banking services across demographic and income groups.
            </p>
            <div style="margin-bottom:1rem;">
                <span style="font-size:0.8rem; color:#f39c12;">🏦 Key Metrics:</span>
                <div style="display:flex; flex-wrap:wrap; gap:0.5rem; margin-top:0.5rem;">
                    <span style="background:#fef9e7; color:#f39c12; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Inclusion Score
                    </span>
                    <span style="background:#fef9e7; color:#f39c12; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Approval Gap
                    </span>
                    <span style="background:#fef9e7; color:#f39c12; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                        Economic Impact
                    </span>
                </div>
            </div>
        </div>
 
    """, unsafe_allow_html=True)

    if st.button(
        "🚀 Launch Simulation →",
        key="launch_Financial_Inclusion",
        use_container_width=True
    ):
        st.switch_page("pages/Financial_Inclusion.py")

st.divider()

# ───────────────────────────────────────────────
# Quick Start Guide
# ───────────────────────────────────────────────
st.markdown('<a name="quick-start"></a>', unsafe_allow_html=True)
st.markdown("""
<div style="margin-bottom:3rem;">
    <h2 style="color:#2c3e50;">📖 Quick Start Guide</h2>
    <p style="color:#7f8c8d;">
        Get started with GAGS in 3 simple steps
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem;">
        <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    width:60px; height:60px; border-radius:50%; display:flex; 
                    align-items:center; justify-content:center; margin:0 auto 1rem; color:white; font-size:1.5rem;">
            1
        </div>
        <h4 style="color:#2c3e50;">Choose a Module</h4>
        <p style="color:#7f8c8d; font-size:0.9rem;">
            Select from Healthcare, Agrotech, National Security, Education, Judicial, or Financial modules
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem;">
        <div style="background:linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%); 
                    width:60px; height:60px; border-radius:50%; display:flex; 
                    align-items:center; justify-content:center; margin:0 auto 1rem; color:white; font-size:1.5rem;">
            2
        </div>
        <h4 style="color:#2c3e50;">Configure Parameters</h4>
        <p style="color:#7f8c8d; font-size:0.9rem;">
            Adjust bias intensity, attack types, and domain-specific parameters in the sidebar
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="text-align:center; padding:1.5rem;">
        <div style="background:linear-gradient(135deg, #f6d365 0%, #fda085 100%); 
                    width:60px; height:60px; border-radius:50%; display:flex; 
                    align-items:center; justify-content:center; margin:0 auto 1rem; color:white; font-size:1.5rem;">
            3
        </div>
        <h4 style="color:#2c3e50;">Analyze Results</h4>
        <p style="color:#7f8c8d; font-size:0.9rem;">
            Explore visualizations, download results, and implement recommendations
        </p>
    </div>
    """, unsafe_allow_html=True)

# Quick start card
st.markdown("""
<div class="quick-start-card">
    <div style="display:flex; align-items:start; gap:1.5rem;">
        <div style="flex:1;">
            <h3 style="color:#2c3e50; margin-top:0;">🚀 Ready to Start?</h3>
            <p style="color:#2c3e50; opacity:0.9;">
                Begin with our most popular simulation to understand AI governance challenges
            </p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
if st.button(
        "🚀 Start with Healthcare Equity →",
        key="launch_Financial_Inclusion_2",
        use_container_width=True
):
    st.switch_page("pages/02_Healthcare_Equity.py")

st.divider()

# ───────────────────────────────────────────────
# Features & Capabilities
# ───────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:3rem;">
    <h2 style="color:#2c3e50;">✨ Key Features</h2>
    <p style="color:#7f8c8d;">
        What makes GAGS the leading AI governance simulation platform
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="feature-card" style="--card-color:#667eea;">
        <div style="display:flex; align-items:center; margin-bottom:1rem;">
            <div style="background:#667eea; padding:0.5rem; border-radius:10px; margin-right:1rem;">
                <span style="color:white; font-size:1.2rem;">🎯</span>
            </div>
            <h4 style="margin:0; color:#2c3e50;">Realistic Simulations</h4>
        </div>
        <p style="color:#7f8c8d;">
            Domain-specific scenarios with realistic bias and attack patterns. 
            Simulate real-world AI governance challenges across 6 domains.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card" style="--card-color:#2ecc71; margin-top:1.5rem;">
        <div style="display:flex; align-items:center; margin-bottom:1rem;">
            <div style="background:#2ecc71; padding:0.5rem; border-radius:10px; margin-right:1rem;">
                <span style="color:white; font-size:1.2rem;">📊</span>
            </div>
            <h4 style="margin:0; color:#2c3e50;">Advanced Analytics</h4>
        </div>
        <p style="color:#7f8c8d;">
            Comprehensive metrics including fairness scores, accuracy, 
            equity gaps, and impact metrics. Interactive visualizations.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card" style="--card-color:#f39c12;">
        <div style="display:flex; align-items:center; margin-bottom:1rem;">
            <div style="background:#f39c12; padding:0.5rem; border-radius:10px; margin-right:1rem;">
                <span style="color:white; font-size:1.2rem;">🛡️</span>
            </div>
            <h4 style="margin:0; color:#2c3e50;">Attack & Defense</h4>
        </div>
        <p style="color:#7f8c8d;">
            Test against 9+ attack vectors including data poisoning, 
            backdoor attacks, and evasion techniques. Measure resilience.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card" style="--card-color:#9b59b6; margin-top:1.5rem;">
        <div style="display:flex; align-items:center; margin-bottom:1rem;">
            <div style="background:#9b59b6; padding:0.5rem; border-radius:10px; margin-right:1rem;">
                <span style="color:white; font-size:1.2rem;">🤝</span>
            </div>
            <h4 style="margin:0; color:#2c3e50;">Stakeholder Focus</h4>
        </div>
        <p style="color:#7f8c8d;">
            Designed for policymakers, researchers, developers, and 
            educators. Generate actionable recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Interactive Demo
# ───────────────────────────────────────────────
st.divider()

st.markdown("""
<div style="margin-bottom:3rem;">
    <h2 style="color:#2c3e50;">📈 Impact Visualization</h2>
    <p style="color:#7f8c8d;">
        See how different factors affect AI system performance across domains
    </p>
</div>
""", unsafe_allow_html=True)

# Create interactive visualization
col1, col2 = st.columns([2, 1])

with col1:
    # Generate sample data for visualization
    np.random.seed(42)
    domains = ['Healthcare', 'Agrotech', 'National Security', 'Finance', 'Education', 'Judicial']

    # Sample data with realistic patterns
    data = {
        'Domain': domains * 3,
        'Metric': ['Fairness'] * 6 + ['Accuracy'] * 6 + ['Inclusion'] * 6,
        'Score': [
            # Fairness scores
            0.82, 0.78, 0.65, 0.70, 0.75, 0.68,
            # Accuracy scores
            0.88, 0.85, 0.92, 0.80, 0.83, 0.79,
            # Inclusion scores
            0.76, 0.72, 0.60, 0.85, 0.78, 0.65
        ]
    }

    df = pd.DataFrame(data)

    # Create radar-like visualization
    fig = px.line_polar(df, r='Score', theta='Domain', color='Metric',
                        line_close=True, color_discrete_sequence=['#667eea', '#2ecc71', '#f39c12'])

    fig.update_traces(fill='toself')
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        height=400,
        title="AI System Performance Across Domains"
    )

    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    <div style="background:#f8f9fa; padding:1.5rem; border-radius:12px; height:400px; display:flex; flex-direction:column; justify-content:center;">
        <h4 style="color:#2c3e50; text-align:center;">Simulation Impact</h4>
        <div style="text-align:center; margin:1rem 0;">
            <div class="stat-number">78%</div>
            <div style="font-size:0.9rem; color:#7f8c8d;">Average Fairness Score</div>
        </div>
        <div style="text-align:center; margin:1rem 0;">
            <div class="stat-number">85%</div>
            <div style="font-size:0.9rem; color:#7f8c8d;">Average Accuracy</div>
        </div>
        <div style="text-align:center; margin:1rem 0;">
            <div class="stat-number">73%</div>
            <div style="font-size:0.9rem; color:#7f8c8d;">Average Inclusion</div>
        </div>
        <p style="text-align:center; font-size:0.9rem; color:#7f8c8d; margin-top:1rem;">
            Based on 2,489 simulation runs
        </p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ───────────────────────────────────────────────
# News & Updates
# ───────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:3rem;">
    <h2 style="color:#2c3e50;">📰 Latest Updates</h2>
    <p style="color:#7f8c8d;">
        Stay informed about new features and research
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="news-card">
        <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:0.5rem;">
            <h4 style="margin:0; color:#2c3e50;">All Modules Now Live!</h4>
            <span style="background:#2ecc71; color:white; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                RELEASE
            </span>
        </div>
        <p style="color:#7f8c8d; font-size:0.9rem; margin-bottom:0.5rem;">
            All 6 simulation modules are now fully implemented and available. 
            Explore Healthcare, Agrotech, Security, Education, Judicial, and Financial domains.
        </p>
        <span style="font-size:0.8rem; color:#3498db;">February 1, 2024</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="news-card">
        <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:0.5rem;">
            <h4 style="margin:0; color:#2c3e50;">Enhanced Attack Simulations</h4>
            <span style="background:#3498db; color:white; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                UPDATE
            </span>
        </div>
        <p style="color:#7f8c8d; font-size:0.9rem; margin-bottom:0.5rem;">
            Added 3 new attack vectors including backdoor attacks and 
            gradient-based poisoning across all modules.
        </p>
        <span style="font-size:0.8rem; color:#3498db;">January 25, 2024</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="news-card">
        <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:0.5rem;">
            <h4 style="margin:0; color:#2c3e50;">Financial Inclusion Module</h4>
            <span style="background:#f39c12; color:white; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                NEW
            </span>
        </div>
        <p style="color:#7f8c8d; font-size:0.9rem; margin-bottom:0.5rem;">
            New Financial Inclusion module launched. Explore credit scoring fairness, 
            loan approval algorithms, and economic impact analysis.
        </p>
        <span style="font-size:0.8rem; color:#3498db;">January 20, 2024</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="news-card">
        <div style="display:flex; justify-content:space-between; align-items:start; margin-bottom:0.5rem;">
            <h4 style="margin:0; color:#2c3e50;">New Export Features</h4>
            <span style="background:#9b59b6; color:white; padding:0.2rem 0.6rem; border-radius:10px; font-size:0.8rem;">
                FEATURE
            </span>
        </div>
        <p style="color:#7f8c8d; font-size:0.9rem; margin-bottom:0.5rem;">
            Export simulation results in CSV and JSON formats with 
            comprehensive configuration metadata for all modules.
        </p>
        <span style="font-size:0.8rem; color:#3498db;">January 15, 2024</span>
    </div>
    """, unsafe_allow_html=True)

# ───────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────
st.divider()

st.markdown("""
<div style="text-align:center; padding:3rem 0; color:#7f8c8d;">
    <div style="margin-bottom:2rem;">
        <h3 style="color:#2c3e50;">Ready to Transform AI Governance?</h3>
        <p style="max-width:600px; margin:0 auto 2rem;">
            Start simulating, understanding, and improving AI systems today
        </p>
    </div>
    <div style="display:flex; justify-content:center; gap:1rem;">
        <a href="/1_🏥_Healthcare_Equity" target="_self">
            <div class="launch-button" style="--btn-color-start: #667eea; --btn-color-end: #764ba2; width:200px;">
                Start Simulation
            </div>
        </a>
        <a href="/6_💰_Financial_Inclusion" target="_self">
            <div class="launch-button" style="--btn-color-start: #f39c12; --btn-color-end: #d35400; width:200px;">
                Try New Module
            </div>
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# Add floating icons in the background
st.markdown("""
<style>
    .floating-icons {
        position: fixed;
        bottom: 0;
        right: 0;
        width: 300px;
        height: 300px;
        pointer-events: none;
        z-index: -1;
        opacity: 0.1;
    }

    .floating-icons div {
        position: absolute;
        font-size: 2rem;
        animation: float-random 20s infinite linear;
    }

    .floating-icons div:nth-child(1) { top: 10%; left: 20%; animation-delay: 0s; }
    .floating-icons div:nth-child(2) { top: 30%; left: 70%; animation-delay: -5s; }
    .floating-icons div:nth-child(3) { top: 60%; left: 40%; animation-delay: -10s; }
    .floating-icons div:nth-child(4) { top: 80%; left: 80%; animation-delay: -15s; }
    .floating-icons div:nth-child(5) { top: 40%; left: 10%; animation-delay: -20s; }

    @keyframes float-random {
        0% { transform: translate(0, 0) rotate(0deg); }
        25% { transform: translate(20px, 20px) rotate(90deg); }
        50% { transform: translate(0, 40px) rotate(180deg); }
        75% { transform: translate(-20px, 20px) rotate(270deg); }
        100% { transform: translate(0, 0) rotate(360deg); }
    }
</style>

<div class="floating-icons">
    <div>🤖</div>
    <div>⚖️</div>
    <div>🔬</div>
    <div>🌍</div>
    <div>🚀</div>
</div>
""", unsafe_allow_html=True)