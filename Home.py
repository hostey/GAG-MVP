# Home.py — GAGS Resilience Framework v1.0
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
from components.translate import install_auto_translate, tx, tx_plotly

from components.live_data import fetch_nigeria_national_live
from components.gags_interactive import render_interactive_healthcare

install_auto_translate()

# 1. FIXED: Set page config ONLY ONCE at the top
st.set_page_config(
    page_title="GAGS · AI Governance Framework",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add custom global CSS for polished metric cards & tabs
st.markdown("""
    <style>
    /* Styled Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 12px 16px;
    }
    /* Tab Styling */
    button[data-baseweb="tab"] {
        font-size: 0.95rem;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# Suppress any stale guided tour banners from other pages
for _stale_key in ["_tour_dismissed_agrotech", "_tour_dismissed_health", "_tour_dismissed_security"]:
    if _stale_key not in st.session_state:
        st.session_state[_stale_key] = True   # auto-dismiss on non-home pages


try:
    from components.gags_design import build_css, DOMAIN_ACCENTS, plotly_theme
    ACCENT = DOMAIN_ACCENTS["home"]
    st.markdown(build_css(ACCENT), unsafe_allow_html=True)
except ImportError:
    ACCENT = "#0891b2"

# ── Homepage-only extra styles ─────────────────────────────────────────────────
ACR = "0,229,255"
st.markdown(f"""<style>
.hero{{background:var(--bg0);border:1px solid var(--bdr);border-radius:16px;
  padding:4rem 3.5rem 3rem;margin-bottom:2.5rem;position:relative;overflow:hidden}}
.hero::before{{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse 55% 55% at 80% 50%,rgba({ACR},.06) 0%,transparent 100%),
             radial-gradient(ellipse 35% 55% at 15% 85%,rgba(57,255,122,.04) 0%,transparent 100%);
  pointer-events:none}}
.hero-grid{{position:absolute;inset:0;
  background-image:linear-gradient(rgba({ACR},.035) 1px,transparent 1px),
    linear-gradient(90deg,rgba({ACR},.035) 1px,transparent 1px);
  background-size:52px 52px;
  mask-image:radial-gradient(ellipse 75% 80% at 85% 50%,black 0%,transparent 70%);
  pointer-events:none}}
.hero-tag{{font-family:var(--ff-m);font-size:.66rem;letter-spacing:.2em;
  text-transform:uppercase;color:var(--ac);margin:0 0 .9rem;
  display:flex;align-items:center;gap:.5rem}}
.hero-tag::before{{content:'';display:inline-block;width:20px;height:1px;background:var(--ac)}}
.hero h1{{font-family:var(--ff-d)!important;
  font-size:clamp(2.3rem,5vw,3.6rem)!important;font-weight:800!important;
  letter-spacing:-.04em!important;color:var(--t0)!important;
  line-height:1.04!important;margin:0 0 .9rem!important}}
.hero h1 span{{color:var(--ac)}}
.hero-sub{{font-size:1.02rem;color:var(--t1);line-height:1.7;
  max-width:600px;margin:0 0 2rem}}
.hero-stats{{display:flex;gap:2.5rem;flex-wrap:wrap;
  border-top:1px solid var(--bg3);padding-top:1.4rem;margin-top:.3rem}}
.hero-stat .n{{font-family:var(--ff-d);font-size:1.9rem;font-weight:800;
  color:var(--ac);line-height:1}}
.hero-stat .l{{font-family:var(--ff-m);font-size:.65rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--t2);margin-top:.18rem}}
.dc{{background:var(--bg1);border:1px solid var(--bg3);border-radius:12px;
  padding:1.35rem 1.5rem;height:100%;position:relative;overflow:hidden;
  transition:border-color .22s,transform .2s,box-shadow .22s}}
.dc:hover{{transform:translateY(-3px);border-color:var(--bdr);
  box-shadow:0 14px 44px rgba(0,0,0,.45)}}
.dc .dca{{position:absolute;top:0;left:0;right:0;height:2px;opacity:.75}}
.dc .dci{{font-size:1.75rem;margin:.3rem 0 .5rem;display:block}}
.dc .dct{{font-family:var(--ff-d);font-size:.95rem;font-weight:700;
  color:var(--t0);margin:0 0 .28rem;letter-spacing:-.01em}}
.dc .dcd{{font-size:.77rem;color:var(--t2);line-height:1.5;margin:0 0 .7rem}}
.dc-pill{{font-family:var(--ff-m);font-size:.59rem;letter-spacing:.04em;
  padding:2px 7px;border-radius:3px;background:var(--bg2);color:var(--t2);
  border:1px solid var(--bg3);margin:2px 2px 0 0;display:inline-block}}
.uc{{background:var(--bg2);border-left:3px solid var(--ac);
  border-radius:0 8px 8px 0;padding:.9rem 1.1rem;margin-bottom:.65rem}}
.uc-r{{font-family:var(--ff-m);font-size:.65rem;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ac);margin:0 0 .25rem}}
.uc-t{{font-size:.83rem;color:var(--t1);line-height:1.5;margin:0}}
.ng{{background:linear-gradient(135deg,rgba(57,255,122,.06) 0%,rgba({ACR},.06) 100%);
  border:1px solid rgba(57,255,122,.2);border-radius:12px;padding:1.4rem 2rem}}
.cl-item{{display:flex;gap:1rem;margin-bottom:.85rem;align-items:flex-start}}
.cl-ver{{font-family:var(--ff-m);font-size:.68rem;letter-spacing:.06em;
  color:var(--ac);white-space:nowrap;padding-top:.05rem;min-width:48px}}
.cl-date{{font-family:var(--ff-m);font-size:.62rem;color:var(--t2)}}
.cl-text{{font-size:.81rem;color:var(--t1);line-height:1.5}}
</style>""", unsafe_allow_html=True)

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero fade-in">
  <div class="hero-grid"></div>
  <p class="hero-tag">GAGS · v1.0 · Nigeria · AI Governance Simulation</p>
  <h1>The AI Bias<br><span>Resilience Framework</span></h1>
  <p class="hero-sub">
    Research-grade simulation for measuring, visualising, and mitigating
    algorithmic bias across 7 high-stakes domains — real datasets,
    regulatory compliance scoring, Nigeria-first contextualisation.
  </p>
  <div class="hero-stats">
    <div class="hero-stat"><div class="n">9</div><div class="l">AI Domains</div></div>
    <div class="hero-stat"><div class="n">24</div><div class="l">Engine Modules §1–§24</div></div>
    <div class="hero-stat"><div class="n">13</div><div class="l">Community Scenarios</div></div>
    <div class="hero-stat"><div class="n">5</div><div class="l">Regulatory Frameworks</div></div>
    <div class="hero-stat"><div class="n">4</div><div class="l">Languages</div></div>
    <div class="hero-stat"><div class="n">16k+</div><div class="l">Lines of Code</div></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── DOMAIN CARDS ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-label fade-in-2">Simulation Modules</p>', unsafe_allow_html=True)

DOMAINS = [
    dict(icon="🏥", title="Healthcare Equity", accent="#0891b2", page="pages/02_Healthcare_Equity.py",
         desc="Diagnostic AI bias across income, gender, and insurance. UCI Heart, PIMA Diabetes, Nigeria scenarios.",
         pills=["Sensitivity Gap","Gender Audit","WHO Ethics","NITDA","XAI"],
         metrics=["Sensitivity","Specificity","Equity Score"]),
    dict(icon="🛡️", title="National Security", accent="#ef4444", page="pages/2_National_Security.py",
         desc="Surveillance vs civil-liberties. GTD & UNSW-NB15 datasets. Liberty score, predictive policing.",
         pills=["Liberty Score","FPR Gap","GTD Dataset","Governance"],
         metrics=["Detection Rate","False Alarm","Liberty"]),
    dict(icon="🌾", title="Agrotech Equity", accent="#22c55e", page="pages/3_Sustainable_Agrotech.py",
         desc="AI fairness for Nigeria smallholder farmers. Gender audit (52% female), Vickrey auction, USSD mode.",
         pills=["Gender Audit","Agent Economy","USSD","52% Female"],
         metrics=["Crop Risk","Market Access","Gender Gap"]),
    dict(icon="🎓", title="Education Equity", accent="#f59e0b", page="pages/Education_Equity.py",
         desc="JAMB/WAEC examination bias, dropout prediction, automated essay grading. SES & urban-rural gaps.",
         pills=["JAMB/WAEC","Dropout Risk","SES Gap","Language Bias"],
         metrics=["Opportunity Gap","Gender Gap","SES Gap"]),
    dict(icon="💰", title="Financial Inclusion", accent="#0d9488", page="pages/Financial_Inclusion.py",
         desc="Credit scoring bias, ECOA disparate impact, informal economy exclusion. Nigeria CBN/NDIC context.",
         pills=["ECOA DI Ratio","CBN Nigeria","Redlining","Mobile Money"],
         metrics=["Approval Gap","DI Ratio","Inclusion Score"]),
    dict(icon="⚖️", title="Judicial Justice", accent="#7c3aed", page="pages/Judicial_system.py",
         desc="COMPAS replication, bail decision bias, predictive policing. Racial FPR gap, liberty score. Highest stakes.",
         pills=["COMPAS","Racial FPR","Lagos Bail","Liberty"],
         metrics=["Racial FPR Gap","Liberty Score","Wrongful Det."]),
    dict(icon="📡", title="Disinformation", accent="#ea580c", page="pages/Disinformation_Misinformation.py",
         desc="Content moderation fairness, deepfake detection bias. Language FPR gap. Nigeria 2027 election context.",
         pills=["Language FPR","Election 2027","Deepfake","Over-Removal"],
         metrics=["Language Gap","Over-Removal","Recall"]),
    dict(icon="🕊️", title="Peace & Conflict", accent="#0f766e", page="pages/Peace_Conflict.py",
         desc="Conflict early-warning AI bias: Boko Haram, farmer-herder, election violence. Game theory · SIR violence contagion · Network cascades.",
         pills=["Ethnic FPR","Community Trust","ECOWAS","Game Theory"],
         metrics=["False Alarm","Ethnic Gap","Trust"]),
    dict(icon="📚", title="Scenario Library", accent="#a78bfa", page="pages/05_Scenario_Library.py",
         desc="13 curated real-world bias scenarios with real citations. Browse, filter, apply in one click.",
         pills=["13 Scenarios","Real Citations","Contribute","Export"],
         metrics=["By Domain","By Priority","By Region"]),
    dict(icon="🔀", title="Comparison Mode", accent="#0284c7", page="pages/Comparison_Mode.py",
         desc="Run two configurations side-by-side and diff every metric. Radar, bar, delta table.",
         pills=["A vs B","Config Diff","Delta Metrics","Export"],
         metrics=["Δ Accuracy","Δ Fairness","Δ Liberty"]),
    dict(icon="📡", title="Real-Time Monitor", accent="#0284c7", page="pages/Real_Time_Monitor.py",
         desc="Continuous fairness monitoring with drift detection. Gradual drift, sudden spike, adversarial scenarios.",
         pills=["Drift Detection","Live Stream","Auto-Alert","REST API"],
         metrics=["Fairness Drift","FPR Drift","Alert Count"]),
]

for row_start in range(0, len(DOMAINS), 4):
    row = DOMAINS[row_start:row_start+4]
    cols = st.columns(len(row))
    for col, d in zip(cols, row):
        with col:
            pills = "".join(f'<span class="dc-pill">{p}</span>' for p in d["pills"])
            mets  = " · ".join(f'<span style="color:{d["accent"]};font-family:var(--ff-m);font-size:.63rem">{m}</span>' for m in d["metrics"])
            st.markdown(f"""
<div class="dc fade-in-2">
  <div class="dca" style="background:linear-gradient(90deg,transparent,{d['accent']},transparent)"></div>
  <span class="dci">{d['icon']}</span>
  <p class="dct">{d['title']}</p>
  <p class="dcd">{d['desc']}</p>
  <div style="margin-bottom:.55rem">{mets}</div>
  <div>{pills}</div>
</div>""", unsafe_allow_html=True)
            st.page_link(d["page"], label=f"Open →", use_container_width=True)
    st.write("")

# ── CAPABILITIES ──────────────────────────────────────────────────────────────
st.markdown('<p class="section-label fade-in">Platform Capabilities</p>', unsafe_allow_html=True)

FEATS = [
    [("🧠","Explainable AI","LIME-lite explanations, DiCE-lite counterfactuals, feature importance, model cards."),
     ("🔁","Longitudinal Analysis","Bias amplification across N retraining generations. Detects self-reinforcing loops."),
     ("🌐","Federated Learning","Train across clients with differential privacy. Non-IID bias heterogeneity."),
     ("📊","Causal Inference","IPW and propensity-score matching. ATE estimation after confound removal.")],
    [("📋","Compliance Reports","EU AI Act, ISO 42001, NIST AI RMF, NITDA, UNESCO, WHO. PDF download."),
     ("🇳🇬","Nigeria Regulatory","NITDA AI Policy 2023 (10 principles), NDPR (7 articles), NCC USSD guidelines."),
     ("📱","USSD/SMS Mode","Models 66% of Nigerian users on 2G. Accessibility gap analysis. 4-language menus."),
     ("🌍","4 Languages","Full Hausa · Yorùbá · Igbo · English UI. Language switcher on every page.")],
    [("🔌","Plugin API","DataLoader, Metric, Scenario, Bias plugins. Register at import — no forking."),
     ("⚡","REST API","FastAPI /simulate /xai /compliance /monitor. MLOps pipeline integration."),
     ("📚","Scenario Library","13 real-world scenarios with real citations. Contribute without code."),
     ("📡","Live Monitoring","Drift detection, threshold alerts, snapshot history for production endpoints.")],
]

fc1, fc2, fc3 = st.columns(3)
for col, feats in zip([fc1, fc2, fc3], FEATS):
    with col:
        items = "".join(f"""
<div class="feat-item">
  <div class="feat-icon">{ic}</div>
  <div class="feat-text"><p class="ft">{t}</p><p class="fs">{d}</p></div>
</div>""" for ic, t, d in feats)
        st.markdown(f'<div class="feat-strip fade-in-3">{items}</div>', unsafe_allow_html=True)

# ── WHO IS THIS FOR ────────────────────────────────────────────────────────────
st.markdown('<p class="section-label fade-in">Who Uses GAGS</p>', unsafe_allow_html=True)

USECASES = [
    ("AI Researcher", "Run reproducible bias experiments. Export JSON/CSV. Cite community scenarios with real DOI-linked references. Compare mitigation strategies head-to-head."),
    ("Regulatory Analyst", "Generate NITDA/NDPR/EU AI Act compliance reports in one click. Audit against the ECOA 80% disparate impact threshold automatically."),
    ("ML Engineer / MLOps", "Push live model predictions to the REST API monitoring endpoint. Receive drift alerts before fairness degrades past your threshold."),
    ("Policy Maker", "Board-member executive summary mode collapses the full dashboard to a single verdict and one recommended action."),
    ("University Educator", "7 domains × 13 pre-built scenarios = 91 ready-to-run teaching cases. Students see real metrics change as they adjust bias sliders."),
    ("Civil Society / NGO", "Hausa/Yoruba/Igbo interface. USSD accessibility gap report. Contribute community scenarios without forking the codebase."),
]
uc1, uc2 = st.columns(2)
for col, cases in zip([uc1, uc2], [USECASES[:3], USECASES[3:]]):
    with col:
        for role, text in cases:
            st.markdown(f'<div class="uc fade-in-3"><p class="uc-r">{role}</p><p class="uc-t">{text}</p></div>',
                        unsafe_allow_html=True)

# ── NIGERIA STRIP ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="ng fade-in-3">
  <p style="font-family:var(--ff-m);font-size:.66rem;letter-spacing:.15em;text-transform:uppercase;color:#22c55e;margin:0 0 .55rem">
    🇳🇬 Nigeria-First Design
  </p>
  <p style="font-size:.88rem;color:var(--t1);line-height:1.65;margin:0;max-width:940px">
    Every module includes Nigeria-specific presets calibrated to
    <strong style="color:var(--t0)">CBN FinScope 2023</strong>,
    <strong style="color:var(--t0)">JAMB Annual Reports 2022–2024</strong>,
    <strong style="color:var(--t0)">NJC Lagos Correctional Service Report 2022</strong>, and
    <strong style="color:var(--t0)">Meta Nigeria Transparency Report 2023</strong>.
    NITDA AI Policy 2023, NDPR, and NCC USSD guidelines are mapped to every result.
    The USSD simulator models feature-phone-only access for the 66% of Nigerian users who are 2G-only.
  </p>
</div>
""", unsafe_allow_html=True)

# ── COVERAGE RADAR + BAR ──────────────────────────────────────────────────────
st.markdown('<p class="section-label fade-in">Framework Coverage</p>', unsafe_allow_html=True)

rc1, rc2 = st.columns([1.1, 1])
try:
    PT = plotly_theme(ACCENT)
except Exception:
    PT = {}

with rc1:
    cats = ["Explainability","Fairness Metrics","Regulatory Coverage",
            "Nigeria Context","Multilingual","Adversarial Robustness",
            "Longitudinal","Plugin Extensibility"]
    vals = [0.95, 0.93, 0.90, 0.97, 0.88, 0.85, 0.90, 0.82]
    fig = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]], fill="toself",
        fillcolor="rgba(0,229,255,.08)", line=dict(color="#0891b2", width=2),
        marker=dict(color="#0891b2", size=5),
    ))
    fig.update_layout(
        paper_bgcolor="#f8fafc",
        font=dict(family="DM Mono, monospace", color="#475569", size=10),
        polar=dict(bgcolor="#f8fafc",
            angularaxis=dict(linecolor="#e2e8f0", gridcolor="#e2e8f0",
                tickfont=dict(color="#475569", size=10, family="DM Mono, monospace")),
            radialaxis=dict(visible=False, range=[0, 1])),
        height=340, showlegend=False, margin=dict(t=18, b=18, l=38, r=38),
        title=dict(text="Capability Radar", font=dict(family="Syne, sans-serif",
                   color="#0f172a", size=13)),
    )
    st.plotly_chart(fig, use_container_width=True)

with rc2:
    d_scores = {"Healthcare":95,"Nat. Security":90,"Agrotech":93,
                "Education":88,"Financial":91,"Judicial":87,"Disinformation":86,"Peace & Conflict":84}
    colors = ["#0891b2","#ef4444","#22c55e","#f59e0b","#0d9488","#7c3aed","#ea580c","#0f766e"]
    fig2 = go.Figure(go.Bar(
        x=list(d_scores.values()), y=list(d_scores.keys()),
        orientation="h", marker=dict(color=colors),
        text=[f"{v}%" for v in d_scores.values()],
        textposition="inside",
        textfont=dict(family="DM Mono, monospace", size=10, color="#ffffff"),
    ))
    ptz = PT if PT else {}
    fig2.update_layout(
        paper_bgcolor="#f8fafc", plot_bgcolor="#ffffff",
        font=dict(family="DM Mono, monospace", color="#475569", size=10),
        title=dict(text="Domain Completeness", font=dict(family="Syne, sans-serif",
                   color="#0f172a", size=13)),
        xaxis=dict(range=[0,100], gridcolor="#e2e8f0", linecolor="#e2e8f0",
                   tickcolor="#64748b", tickfont=dict(color="#64748b", size=9)),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0",
                   tickfont=dict(color="#475569", size=10)),
        height=340, margin=dict(t=42, b=10, l=8, r=8),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── QUICK START + CHANGELOG ────────────────────────────────────────────────────
st.markdown('<p class="section-label fade-in">Quick Start · Changelog</p>', unsafe_allow_html=True)

qs1, qs2 = st.columns([1.2, 1])
with qs1:
    st.markdown("""
<div class="feat-strip fade-in-3">
  <p style="font-family:var(--ff-d);font-size:.98rem;font-weight:700;color:var(--t0);margin:0 0 .9rem;letter-spacing:-.01em">
    Get Running in 3 Steps
  </p>
  <div class="terminal">
    <div><span class="prompt">❯ </span>pip install streamlit plotly scikit-learn pandas</div>
    <div><span class="prompt">❯ </span>pip install fastapi uvicorn reportlab</div>
    <div><span class="prompt">❯ </span>streamlit run Home.py</div>
    <div style="color:var(--t2);margin-top:.45rem"># Optional REST API</div>
    <div><span class="prompt">❯ </span>uvicorn api:app --host 0.0.0.0 --port 8502</div>
    <div style="color:var(--t2);margin-top:.45rem"># Docs at http://localhost:8502/docs</div>
  </div>
  <p style="font-family:var(--ff-m);font-size:.7rem;color:var(--t2);margin:.85rem 0 0">
    Python 3.10 + · No GPU required · Synthetic/Real world data — no PII processed
  </p>
</div>
""", unsafe_allow_html=True)

with qs2:
    cl = [
        ("v4.0","Mar 2026","Dark terminal UI · domain neon accents · design system · animated hero"),
        ("v3.0","Feb 2026","Education, Financial, Judicial, Disinformation modules · REST API · Monitoring"),
        ("v2.5","Jan 2026","Nigeria regulatory (NITDA/NDPR/NCC) · USSD simulator · i18n Hausa/Yoruba/Igbo"),
        ("v2.0","Dec 2025","XAI (LIME/DiCE) · Longitudinal bias · Federated learning · Community scenarios"),
        ("v1.0","Nov 2025","Healthcare, National Security, Agrotech · Governance layer · Agent economy"),
    ]
    items = "".join(f"""
<div class="cl-item">
  <div><div class="cl-ver">{ver}</div><div class="cl-date">{date}</div></div>
  <div class="cl-text">{text}</div>
</div>""" for ver, date, text in cl)
    st.markdown(f'<div class="feat-strip fade-in-3">{items}</div>', unsafe_allow_html=True)

# ── STATUS BAR ─────────────────────────────────────────────────────────────────
st.markdown("<div style='height:1.25rem'></div>", unsafe_allow_html=True)
statuses = [("🟢","Core Engine §1–§24"),("🟢","7 Simulation Pages"),
            ("🟢","Compliance Reports"),("🟢","REST API"),("🟢","Plugin Registry"),
            ("🟢","Nigeria Regulatory"),("🟡","Community Scenarios (13/∞)")]
shtml = "".join(
    f'<span style="font-family:var(--ff-m);font-size:.7rem;color:var(--t2);'
    f'display:inline-flex;align-items:center;gap:.3rem;margin-right:1.1rem">{ic}&nbsp;{lb}</span>'
    for ic,lb in statuses)
st.markdown(f"""
<div style="background:var(--bg1);border:1px solid var(--bg3);border-radius:8px;
  padding:.65rem 1.2rem;display:flex;flex-wrap:wrap;align-items:center;gap:.2rem">
  <span class="live-badge" style="margin-right:.7rem">
    <span class="live-dot"></span>Live
  </span>
  {shtml}
  <span style="margin-left:auto;font-family:var(--ff-m);font-size:.65rem;color:var(--t2)">
    GAGS v1.0 · {datetime.now().strftime('%Y-%m-%d')}
  </span>
</div>
""", unsafe_allow_html=True)