# pages/5_📱_Disinformation_Misinformation_Advanced.py
"""
GAGS Framework — Disinformation Simulation v5.0
New in this version:
  • Real-time data integration (Twitter/X, Reddit, NewsAPI, RSS, Demo)
  • Live feed dashboard with source health indicators
  • Narrative credibility + virality timeline
  • Auto-seeding from real-world high-virality posts
  • Full XAI suite from v4 retained
  • Audience mode switcher retained
"""

import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import warnings
from datetime import datetime, timedelta, timezone
import json
import time
import copy

warnings.filterwarnings("ignore")

from utils.simulation_disinfo import (
    create_social_graph,
    assign_node_attributes,
    simulate_misinfo_propagation_advanced,
    load_isot_dataset,
    train_text_classifier,
    run_counterfactual_simulations,
)
from utils.realtime_connector import (
    DataFeed, DemoConnector, DEFAULT_RSS_FEEDS, NarrativePost,
)

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Disinformation Simulator • GAGS v5",
    layout="wide",
    page_icon="📡",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Design tokens + CSS
# ─────────────────────────────────────────────────────────────
PALETTE = dict(
    brand="#6C47FF", brand_light="#EDE9FF",
    danger="#E63946", warning="#F4A261",
    success="#2DC653", info="#2196F3",
    surface="#F8F7FF", border="#E2DDFF",
    muted="#6B7280",
)

SOURCE_COLORS = {
    "twitter": "#1DA1F2",
    "reddit":  "#FF4500",
    "newsapi": "#2DC653",
    "rss":     "#F4A261",
    "demo":    "#9B7DFF",
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}

.hero {{
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    border: 1px solid rgba(108,71,255,.4); position: relative; overflow: hidden;
}}
.hero::before {{
    content:''; position:absolute; inset:0;
    background: radial-gradient(ellipse at 70% 50%, rgba(108,71,255,.25) 0%, transparent 60%);
}}
.hero h1 {{ color:#fff; font-size:2rem; font-weight:600; margin:0 0 .4rem; }}
.hero p  {{ color:rgba(255,255,255,.75); font-size:1rem; margin:0; }}
.hero-badge {{
    display:inline-flex; align-items:center; gap:.4rem;
    background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2);
    border-radius:999px; padding:.2rem .75rem; font-size:.75rem;
    color:rgba(255,255,255,.85); margin-top:.8rem; margin-right:.4rem;
}}

.kpi-card {{
    background:{PALETTE['surface']}; border:1px solid {PALETTE['border']};
    border-radius:12px; padding:1.1rem 1.25rem; margin-bottom:.75rem;
}}
.kpi-label {{ font-size:.72rem; font-weight:600; color:{PALETTE['muted']};
               text-transform:uppercase; letter-spacing:.06em; }}
.kpi-value {{ font-size:1.9rem; font-weight:600; line-height:1.1; }}
.kpi-delta {{ font-size:.78rem; color:{PALETTE['muted']}; margin-top:.2rem; }}

.section-head {{
    display:flex; align-items:center; gap:.6rem;
    border-left:3px solid {PALETTE['brand']}; padding-left:.75rem;
    margin:1.5rem 0 .75rem;
}}
.section-head h3 {{ margin:0; font-size:1rem; font-weight:600; }}
.section-head span {{ font-size:.8rem; color:{PALETTE['muted']}; }}

.xai-insight {{
    background:linear-gradient(135deg,{PALETTE['brand_light']} 0%,#fff 100%);
    border:1px solid {PALETTE['border']}; border-left:4px solid {PALETTE['brand']};
    border-radius:0 10px 10px 0; padding:.85rem 1rem;
    margin:.6rem 0; font-size:.88rem; line-height:1.6;
}}
.xai-insight strong {{ color:{PALETTE['brand']}; }}

.ctx-box {{ border-radius:10px; padding:.85rem 1rem;
             font-size:.85rem; margin:.5rem 0 1rem; line-height:1.5; }}
.ctx-research  {{ background:#EDE9FF; border-left:3px solid #6C47FF; }}
.ctx-business  {{ background:#FEF3C7; border-left:3px solid #D97706; }}
.ctx-govt      {{ background:#DCFCE7; border-left:3px solid #16A34A; }}
.ctx-education {{ background:#DBEAFE; border-left:3px solid #2563EB; }}

/* Source health pill */
.src-pill {{
    display:inline-flex; align-items:center; gap:.35rem;
    padding:.25rem .7rem; border-radius:999px; font-size:.75rem;
    font-weight:600; margin:.2rem;
}}
.src-live    {{ background:#DCFCE7; color:#166534; }}
.src-offline {{ background:#FEE2E2; color:#991B1B; }}
.src-demo    {{ background:#EDE9FF; color:#4B2FCC; }}

/* Post card */
.post-card {{
    background:#fff; border:1px solid {PALETTE['border']};
    border-radius:10px; padding:.85rem 1rem; margin:.4rem 0;
    font-size:.85rem; line-height:1.5;
}}
.post-card .meta {{ color:{PALETTE['muted']}; font-size:.75rem; margin-bottom:.3rem; }}
.post-card .cred-bar {{ height:4px; border-radius:2px; margin-top:.5rem; }}

/* Credibility gradient badge */
.cred-high  {{ background:#2DC653; color:#fff; padding:.1rem .5rem;
                border-radius:4px; font-size:.72rem; font-weight:600; }}
.cred-mid   {{ background:#F4A261; color:#fff; padding:.1rem .5rem;
                border-radius:4px; font-size:.72rem; font-weight:600; }}
.cred-low   {{ background:#E63946; color:#fff; padding:.1rem .5rem;
                border-radius:4px; font-size:.72rem; font-weight:600; }}

/* Chain pill */
.chain-pill {{ display:inline-block; padding:.2rem .65rem; border-radius:999px;
                font-size:.75rem; font-weight:500; margin:.15rem;
                font-family:'DM Mono',monospace; }}
.pill-seed {{ background:#6C47FF; color:#fff; }}
.pill-hop1 {{ background:#EF4444; color:#fff; }}
.pill-hop2 {{ background:#F4A261; color:#fff; }}
.pill-hop3 {{ background:#EDE9FF; color:#4B2FCC; }}

div[data-testid="stButton"] > button[kind="primary"] {{
    background:linear-gradient(135deg,#6C47FF,#A855F7) !important;
    border:none !important; color:white !important;
    font-weight:600 !important; border-radius:10px !important;
    padding:.7rem 1.5rem !important;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def kpi(label, value, delta="", color="#1a1a2e"):
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value" style="color:{color}">{value}</div>'
        f'<div class="kpi-delta">{delta}</div></div>',
        unsafe_allow_html=True)


def section(icon, title, sub=""):
    st.markdown(
        f'<div class="section-head"><h3>{icon} {title}</h3>'
        f'<span>{sub}</span></div>', unsafe_allow_html=True)


def audience_ctx(mode, texts):
    css = {"Research":"research","Business":"business",
           "Government":"govt","Education":"education"}
    txt = texts.get(mode, "")
    if txt:
        st.markdown(
            f'<div class="ctx-box ctx-{css.get(mode,"research")}">{txt}</div>',
            unsafe_allow_html=True)


def xai_insight(text):
    st.markdown(f'<div class="xai-insight">{text}</div>',
                unsafe_allow_html=True)


def cred_badge(score: float) -> str:
    if score >= 0.7:
        return f'<span class="cred-high">credible {score:.2f}</span>'
    elif score >= 0.4:
        return f'<span class="cred-mid">uncertain {score:.2f}</span>'
    return f'<span class="cred-low">low cred {score:.2f}</span>'


def source_pill(src: str, live: bool) -> str:
    col = SOURCE_COLORS.get(src, "#888")
    cls = "src-live" if live else "src-demo"
    dot = "●" if live else "○"
    return (f'<span class="src-pill {cls}" '
            f'style="border:1px solid {col}30">{dot} {src}</span>')


# ─────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────
_defaults = {
    "last_graph": None, "last_history": None, "last_metrics": None,
    "last_detection": None, "last_step_states": None, "last_seeds": [],
    "last_graph_pos": None, "isot_dataset": None, "isot_classifier": None,
    "playing": False, "current_step": 0, "counter_results": None,
    "audience_mode": "Research",
    # realtime
    "live_posts": [], "feed_summary": {}, "feed_df": None,
    "data_source_mode": "Demo (no keys needed)",
    "last_fetch_time": None,
    "last_date_from": None,
    "last_date_to": None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

mode = st.session_state.audience_mode

# ─────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <h1>📡 Disinformation Simulation Platform</h1>
  <p>Real-time narrative ingestion · Network SIR modelling ·
     AI amplification · Echo chambers · Explainable AI · Policy counterfactuals</p>
  <span class="hero-badge">v5.0</span>
  <span class="hero-badge">Audience: {mode}</span>
  {"".join(source_pill(s, True) for s in ['twitter','reddit','newsapi','rss','demo'])}
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎯 Audience Mode")
    aud = st.radio("Presentation for:",
                   ["Research","Business","Government","Education"],
                   index=["Research","Business","Government","Education"].index(
                       st.session_state.audience_mode))
    st.session_state.audience_mode = aud
    mode = aud

    st.divider()
    st.markdown("### 📡 Real-Time Data Source")
    data_mode = st.selectbox(
        "Source",
        ["Demo (no keys needed)",
         "RSS feeds (no keys needed)",
         "Twitter/X (Bearer Token)",
         "Reddit (Client ID + Secret)",
         "NewsAPI (API Key)",
         "All sources combined"],
        index=["Demo (no keys needed)",
               "RSS feeds (no keys needed)",
               "Twitter/X (Bearer Token)",
               "Reddit (Client ID + Secret)",
               "NewsAPI (API Key)",
               "All sources combined"].index(
            st.session_state.data_source_mode),
    )
    st.session_state.data_source_mode = data_mode

    # Show credential fields only when relevant
    twitter_bearer = reddit_id = reddit_secret = newsapi_key = ""
    if "Twitter" in data_mode or "All" in data_mode:
        twitter_bearer = st.text_input(
            "Twitter Bearer Token",
            type="password",
            value=st.session_state.get("_tw_bearer",""),
            help="From developer.twitter.com — read-only Bearer Token")
        st.session_state["_tw_bearer"] = twitter_bearer

    if "Reddit" in data_mode or "All" in data_mode:
        reddit_id = st.text_input(
            "Reddit Client ID", type="password",
            value=st.session_state.get("_rd_id",""))
        reddit_secret = st.text_input(
            "Reddit Client Secret", type="password",
            value=st.session_state.get("_rd_sec",""))
        st.session_state["_rd_id"]  = reddit_id
        st.session_state["_rd_sec"] = reddit_secret

    if "NewsAPI" in data_mode or "All" in data_mode:
        newsapi_key = st.text_input(
            "NewsAPI Key", type="password",
            value=st.session_state.get("_na_key",""),
            help="From newsapi.org — free tier allows 100 req/day")
        st.session_state["_na_key"] = newsapi_key

    narrative_query = st.text_input(
        "Search / topic keyword",
        value="misinformation",
        help="Used for Twitter, NewsAPI, and RSS keyword filter")

    if "Reddit" in data_mode or "All" in data_mode:
        subreddit_input = st.text_input(
            "Subreddits (comma-separated)",
            value="worldnews,conspiracy,skeptic,science")
        subreddits = [s.strip() for s in subreddit_input.split(",") if s.strip()]
    else:
        subreddits = ["worldnews","conspiracy","skeptic"]

    max_per_source = st.slider("Max posts per source", 20, 200, 80)

    st.markdown("#### 📅 Date range")
    _today    = datetime.now(timezone.utc).date()
    _week_ago = _today - timedelta(days=7)

    date_col1, date_col2 = st.columns(2)
    with date_col1:
        date_from_input = st.date_input(
            "From", value=_week_ago,
            max_value=_today,
            help="Start date (inclusive)")
    with date_col2:
        date_to_input = st.date_input(
            "To", value=_today,
            max_value=_today,
            help="End date (inclusive)")

    # Validate range
    if date_from_input > date_to_input:
        st.error("⚠️ 'From' must be before 'To'.")
        date_from_input = date_to_input - timedelta(days=1)

    # Convert to UTC datetimes (start of From day, end of To day)
    date_from_dt = datetime.combine(date_from_input,
                                    datetime.min.time()).replace(
                                        tzinfo=timezone.utc)
    date_to_dt   = datetime.combine(date_to_input,
                                    datetime.max.time()).replace(
                                        tzinfo=timezone.utc)
    date_range_days = (date_to_input - date_from_input).days + 1

    # Per-source capability notices
    if "Twitter" in data_mode or "All" in data_mode:
        if date_range_days > 7:
            st.warning("🐦 Twitter free tier: max 7 days back. "
                       "Dates older than 7 days will be ignored.")
    if "NewsAPI" in data_mode or "All" in data_mode:
        if date_range_days > 30:
            st.warning("📰 NewsAPI free tier: max 30 days back. "
                       "Dates older than 30 days will be ignored.")
    if "Reddit" in data_mode or "All" in data_mode:
        if date_range_days > 90:
            st.info("🟠 Reddit: very old posts may not appear in "
                    "hot/new listings. Use 'top + all time' for archives.")

    virality_thresh = st.slider(
        "Seed virality threshold", 0.3, 0.95, 0.65,
        help="Posts above this virality score become initial spreaders")

    fetch_button = st.button("📡 Fetch Live Data", use_container_width=True)

    st.divider()
    st.markdown("### 🌐 Network")
    num_nodes  = st.slider("Users (nodes)", 50, 1000, 300)
    avg_degree = st.slider("Avg connections", 2.0, 15.0, 5.0)

    st.divider()
    st.markdown("### 🦠 Spread Dynamics")
    base_infection = st.slider("Base infection prob.", 0.01, 0.5, 0.15)
    recovery_prob  = st.slider("Recovery prob.", 0.0, 0.2, 0.03)
    initial_amp    = st.slider("Initial AI amplification", 1.0, 4.0, 2.0)
    decay_rate     = st.slider("Amplification decay rate", 0.0, 0.5, 0.1)

    st.divider()
    st.markdown("### 🛡️ Intervention")
    enable_bots = st.toggle("Enable fact-checker bots", value=False)
    if enable_bots:
        bot_coverage      = st.slider("Bot deployment (%)", 1, 20, 5)
        intervention_step = st.slider("Deploy at step", 0, 50, 5)
        bot_strength      = st.slider("Bot strength", 1.0, 5.0, 2.0)
    else:
        bot_coverage, intervention_step, bot_strength = 5, 10, 2.0

    st.divider()
    st.markdown("### 🎯 Simulation")
    num_seeds_manual = st.number_input(
        "Manual seed count (if no live data)", 1, 50, 8)
    max_steps        = st.slider("Max steps", 20, 300, 100)
    collect_xai      = st.checkbox("Train detector + XAI", value=True)
    seed_val         = st.number_input("Random seed", 0, 9999, 42)

    st.divider()
    run_button = st.button("🚀 Run Simulation", type="primary",
                           use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Data feed builder
# ─────────────────────────────────────────────────────────────
def build_feed() -> DataFeed:
    return DataFeed(
        twitter_bearer      = twitter_bearer or None,
        reddit_client_id    = reddit_id or None,
        reddit_client_secret= reddit_secret or None,
        newsapi_key         = newsapi_key or None,
        use_demo_fallback   = True,
    )


# ─────────────────────────────────────────────────────────────
# Fetch live data
# ─────────────────────────────────────────────────────────────
if fetch_button:
    with st.spinner("Fetching data from live sources…"):
        feed  = build_feed()
        posts = feed.fetch(
            query          = narrative_query,
            subreddits     = subreddits,
            rss_feeds      = DEFAULT_RSS_FEEDS if "RSS" in data_mode or "All" in data_mode else None,
            max_per_source = max_per_source,
            date_from      = date_from_dt,
            date_to        = date_to_dt,
            demo_topic     = narrative_query,
            demo_n         = max_per_source * 2,
            demo_seed      = seed_val,
        )
        summary = feed.narrative_summary(posts)
        df_feed = feed.to_dataframe(posts)

        st.session_state.live_posts      = posts
        st.session_state.feed_summary    = summary
        st.session_state.feed_df         = df_feed
        st.session_state.last_fetch_time = datetime.now().strftime("%H:%M:%S")
        st.session_state.last_date_from  = date_from_input
        st.session_state.last_date_to    = date_to_input

    st.success(
        f"✅ Fetched **{len(posts):,} posts** "
        f"from **{len(summary.get('sources',{}))} source(s)** · "
        f"📅 {date_from_input.strftime('%d %b %Y')} → "
        f"{date_to_input.strftime('%d %b %Y')} · "
        f"🕐 {st.session_state.last_fetch_time}"
    )


# ─────────────────────────────────────────────────────────────
# Cached graph
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def get_graph(n, deg, seed):
    np.random.seed(seed)
    G = create_social_graph(n, deg)
    return assign_node_attributes(G)


# ─────────────────────────────────────────────────────────────
# Run simulation
# ─────────────────────────────────────────────────────────────
if run_button:
    with st.spinner("Building network and running simulation…"):
        G    = get_graph(num_nodes, avg_degree, seed_val)
        feed = build_feed()
        posts: list = st.session_state.live_posts

        # If no live data yet, generate demo
        if not posts:
            posts = DemoConnector().fetch(
                topic=narrative_query, n_posts=200, seed=seed_val)
            st.session_state.live_posts   = posts
            st.session_state.feed_summary = feed.narrative_summary(posts)
            st.session_state.feed_df      = feed.to_dataframe(posts)

        # Assign posts to graph nodes and extract seeds
        G_seeded, auto_seeds = feed.seed_graph(
            copy.deepcopy(G), posts, virality_thresh)

        # Fallback to manual random seeds if too few high-virality posts
        if len(auto_seeds) < 2:
            np.random.seed(seed_val)
            auto_seeds = np.random.choice(
                list(G.nodes()),
                size=min(num_seeds_manual, len(G.nodes())),
                replace=False).tolist()

        dataset  = st.session_state.feed_df    # use live data as content
        detector = None                         # NLP detector uses live df

        hist_df, metrics, det_metrics, step_states = \
            simulate_misinfo_propagation_advanced(
                G_seeded, auto_seeds,
                base_infection_prob = base_infection,
                recovery_prob       = recovery_prob,
                max_steps           = max_steps,
                initial_amplification = initial_amp,
                decay_rate          = decay_rate,
                dataset_df          = dataset,
                content_detector    = detector,
                fake_news_boost     = 1.5,
                collect_detection_data = collect_xai,
                enable_bots         = enable_bots,
                bot_coverage        = float(bot_coverage),
                intervention_step   = intervention_step,
                bot_strength        = bot_strength,
            )

        pos = nx.spring_layout(G, seed=seed_val, k=0.5)

        st.session_state.update({
            "last_graph": G, "last_history": hist_df,
            "last_metrics": metrics, "last_detection": det_metrics,
            "last_step_states": step_states, "last_seeds": auto_seeds,
            "last_graph_pos": pos, "current_step": 0, "playing": False,
        })
    st.success(f"✅ Simulation complete — "
               f"{len(auto_seeds)} seed nodes from live data "
               f"(virality ≥ {virality_thresh:.2f})")


# ─────────────────────────────────────────────────────────────
# Main content area — tabs
# ─────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📡 Live Feed",
    "📈 Spread Dynamics",
    "🕸️ Network Playback",
    "🤖 Explainable AI",
    "🔬 Infection Provenance",
    "🌐 Community & Echo Chambers",
    "💰 Cost-Benefit & Export",
    "🧪 Counterfactual Simulator",
])


# ══════════════════════════════════════════════════════════════
# TAB 0 — LIVE FEED
# ══════════════════════════════════════════════════════════════
with tabs[0]:
    section("📡", "Live Narrative Feed",
            "Real-time posts ingested from connected sources")

    audience_ctx(mode, {
        "Research":   "Posts are normalised into NarrativePost objects with "
                      "credibility scores derived from domain reputation and "
                      "virality scores from engagement + reach. "
                      "High-virality posts auto-seed the simulation.",
        "Business":   "💡 These are the real narratives circulating about your "
                      "topic right now. Posts below 0.5 credibility are classified "
                      "as potential misinformation and become the simulation's "
                      "infection seeds.",
        "Government": "💡 The feed shows active narratives across platforms. "
                      "Sort by virality to identify the highest-priority "
                      "counter-messaging targets.",
        "Education":  "🎓 Each post has a credibility score (domain reputation) "
                      "and a virality score (engagement ÷ reach). "
                      "Watch how the most viral posts are often the least credible.",
    })

    posts = st.session_state.live_posts
    summary = st.session_state.feed_summary

    if not posts:
        st.info("Click **📡 Fetch Live Data** in the sidebar to load content, "
                "or click **🚀 Run Simulation** to auto-generate demo data.")
    else:
        # Source health bar + date range badge
        st.markdown("**Active sources**")
        src_html = ""
        for src, count in summary.get("sources", {}).items():
            live = src != "demo"
            src_html += source_pill(src, live) + f" <small>({count} posts)</small> "
        if st.session_state.last_fetch_time:
            src_html += (f'<span style="color:{PALETTE["muted"]};font-size:.75rem">'
                         f'· fetched at {st.session_state.last_fetch_time}</span>')
        st.markdown(src_html, unsafe_allow_html=True)

        # Date range badge
        d_from = st.session_state.get("last_date_from")
        d_to   = st.session_state.get("last_date_to")
        if d_from and d_to:
            days_span = (d_to - d_from).days + 1
            st.markdown(
                f'<div style="display:inline-flex;align-items:center;gap:.5rem;'
                f'background:{PALETTE["brand_light"]};border:1px solid {PALETTE["border"]};'
                f'border-radius:8px;padding:.35rem .85rem;margin:.4rem 0;font-size:.82rem">'
                f'📅 <strong>{d_from.strftime("%d %b %Y")}</strong>'
                f' → '
                f'<strong>{d_to.strftime("%d %b %Y")}</strong>'
                f' &nbsp;·&nbsp; {days_span} day{"s" if days_span != 1 else ""}'
                f'</div>',
                unsafe_allow_html=True
            )

        # KPIs
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1: kpi("Total posts",      f"{summary.get('total_posts',0):,}",
                     "", PALETTE['brand'])
        with k2: kpi("Misinformation",
                     f"{summary.get('fake_count',0):,}",
                     f"{summary.get('fake_count',0)/max(1,summary.get('total_posts',1))*100:.0f}% of feed",
                     PALETTE['danger'])
        with k3: kpi("Credible content",
                     f"{summary.get('real_count',0):,}",
                     "", PALETTE['success'])
        with k4: kpi("Avg virality",
                     f"{summary.get('avg_virality',0):.3f}",
                     "0=dormant · 1=viral", PALETTE['warning'])
        with k5: kpi("Avg credibility",
                     f"{summary.get('avg_credibility',0):.3f}",
                     "0=fake · 1=verified", PALETTE['info'])

        # ── Virality vs credibility scatter ──
        df_feed = st.session_state.feed_df
        if df_feed is not None and len(df_feed) > 0:
            section("📊", "Virality vs Credibility",
                    "Each dot is one post — bottom-right = high-risk")

            fig_scatter = px.scatter(
                df_feed.head(500),
                x="credibility", y="virality_score",
                color="source",
                color_discrete_map=SOURCE_COLORS,
                size_max=10,
                hover_data=["author","text"],
                labels={"credibility": "Credibility score (0=fake, 1=verified)",
                        "virality_score": "Virality score"},
                opacity=0.7,
            )
            # Danger quadrant
            fig_scatter.add_shape(type="rect",
                x0=0, y0=virality_thresh, x1=0.5, y1=1.0,
                fillcolor="rgba(230,57,70,0.06)",
                line=dict(color=PALETTE['danger'], width=1, dash='dot'))
            fig_scatter.add_annotation(
                x=0.25, y=0.97, text="High-risk zone",
                showarrow=False, font=dict(color=PALETTE['danger'], size=11))
            fig_scatter.update_layout(
                plot_bgcolor='white', height=420,
                xaxis=dict(showgrid=True, gridcolor='#f0f0f0', range=[-0.05, 1.05]),
                yaxis=dict(showgrid=True, gridcolor='#f0f0f0', range=[-0.05, 1.05]),
                margin=dict(t=20, b=40),
                legend=dict(orientation='h', y=1.08))
            st.plotly_chart(fig_scatter, use_container_width=True)
            xai_insight(
                "Posts in the <strong>bottom-right</strong> (low credibility + high virality) "
                "are the most dangerous: they spread wide and fast while being "
                "least likely to be true. These become the simulation's seed nodes."
            )

            # ── Credibility distribution ──
            col_dist, col_time = st.columns(2)

            with col_dist:
                section("📉", "Credibility distribution")
                fig_hist = px.histogram(
                    df_feed, x="credibility", nbins=20,
                    color="source", color_discrete_map=SOURCE_COLORS,
                    barmode="overlay", opacity=0.75)
                fig_hist.add_vline(x=0.5, line_dash="dot",
                                   line_color=PALETTE['danger'],
                                   annotation_text="Misinfo threshold")
                fig_hist.update_layout(
                    height=280, plot_bgcolor='white',
                    xaxis_title="Credibility",
                    yaxis_title="Post count",
                    showlegend=False,
                    margin=dict(t=10, b=40))
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_time:
                section("⏱️", "Post volume over time")
                df_time = df_feed.copy()
                df_time['created_at'] = pd.to_datetime(
                    df_time['created_at'], utc=True, errors='coerce')
                df_time = df_time.dropna(subset=['created_at'])
                if len(df_time) > 0:
                    df_time['hour'] = df_time['created_at'].dt.floor('H')
                    vol = df_time.groupby(['hour','source']).size().reset_index(
                        name='count')
                    fig_time = px.bar(
                        vol, x='hour', y='count', color='source',
                        color_discrete_map=SOURCE_COLORS,
                        barmode='stack')
                    fig_time.update_layout(
                        height=280, plot_bgcolor='white',
                        xaxis_title="Hour (UTC)",
                        yaxis_title="Posts",
                        showlegend=False,
                        margin=dict(t=10, b=40))
                    st.plotly_chart(fig_time, use_container_width=True)

            # ── Top authors ──
            section("🏆", "Top 5 most viral authors")
            top_authors = summary.get("top_authors", [])
            if top_authors:
                ta_df = pd.DataFrame(top_authors)
                st.dataframe(ta_df, use_container_width=True, hide_index=True)

            # ── Top tags ──
            top_tags = summary.get("top_tags", [])
            if top_tags:
                section("🏷️", "Top narrative tags")
                tags_html = " ".join(
                    f'<span style="background:{PALETTE["brand_light"]};'
                    f'color:{PALETTE["brand"]};padding:.2rem .6rem;'
                    f'border-radius:999px;font-size:.8rem;margin:.2rem;'
                    f'display:inline-block">{t}</span>'
                    for t in top_tags)
                st.markdown(tags_html, unsafe_allow_html=True)

            # ── Post cards — sample ──
            with st.expander(f"📄 Browse posts (top 20 by virality)"):
                for p in (sorted(posts, key=lambda x: x.virality_score,
                                 reverse=True)[:20]):
                    cred_cls = ("high" if p.credibility >= 0.7
                                else "mid" if p.credibility >= 0.4
                                else "low")
                    bar_w = int(p.credibility * 100)
                    bar_c = (PALETTE['success'] if p.credibility >= 0.7
                             else PALETTE['warning'] if p.credibility >= 0.4
                             else PALETTE['danger'])
                    src_c = SOURCE_COLORS.get(p.source, "#888")
                    st.markdown(
                        f'<div class="post-card">'
                        f'<div class="meta">'
                        f'<span style="color:{src_c};font-weight:600">'
                        f'{p.source.upper()}</span> · '
                        f'@{p.author} · '
                        f'{p.created_at.strftime("%b %d %H:%M UTC")} · '
                        f'engagement {p.engagement:,} · '
                        f'virality {p.virality_score:.2f} · '
                        f'{cred_badge(p.credibility)}'
                        f'</div>'
                        f'{p.text[:280]}{"…" if len(p.text)>280 else ""}'
                        f'<div class="cred-bar" style="background:#e5e7eb">'
                        f'<div style="background:{bar_c};width:{bar_w}%;'
                        f'height:100%;border-radius:2px"></div></div>'
                        f'</div>',
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 1 — SPREAD DYNAMICS
# ══════════════════════════════════════════════════════════════
with tabs[1]:
    section("📈", "SIR Curve", "Susceptible · Infected · Recovered over time")

    audience_ctx(mode, {
        "Research":   "SIR model with heterogeneous PageRank influence and "
                      "susceptibility attributes. Amplification decays exponentially. "
                      "Seeds are derived from high-virality real-world posts.",
        "Business":   "💡 The red peak = maximum simultaneous exposure. "
                      "The seed nodes come from the most viral real-world posts "
                      "about your topic — this is not hypothetical.",
        "Government": "💡 The area under the infected curve represents "
                      "total 'information dose.' The seed nodes are real "
                      "high-engagement posts identified from live data.",
        "Education":  "🎓 These seeds came from real (or realistic) posts. "
                      "The same SIR maths models COVID and disinformation. "
                      "What changes when you increase the amplification factor?",
    })

    if st.session_state.last_history is None:
        st.info("Run the simulation first.")
    else:
        hist_df = st.session_state.last_history
        metrics = st.session_state.last_metrics
        G       = st.session_state.last_graph

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_df['step'], y=hist_df['susceptible'], name='Susceptible',
            mode='lines', line=dict(color='#2196F3', width=2.5),
            fill='tozeroy', fillcolor='rgba(33,150,243,0.06)'))
        fig.add_trace(go.Scatter(
            x=hist_df['step'], y=hist_df['infected'], name='Infected',
            mode='lines', line=dict(color=PALETTE['danger'], width=2.5),
            fill='tozeroy', fillcolor='rgba(230,57,70,0.08)'))
        fig.add_trace(go.Scatter(
            x=hist_df['step'], y=hist_df['recovered'], name='Recovered',
            mode='lines', line=dict(color=PALETTE['success'], width=2.5),
            fill='tozeroy', fillcolor='rgba(45,198,83,0.07)'))

        if enable_bots:
            fig.add_vline(x=intervention_step, line_dash="dash",
                          line_color="#FFD700", line_width=2,
                          annotation_text=f"Bots deployed (step {intervention_step})")

        # Mark seed injection point
        fig.add_vline(x=0, line_dash="dot", line_color=PALETTE['brand'],
                      line_width=1.5,
                      annotation_text="Real-world seeds injected",
                      annotation_position="top right")

        fig.update_layout(
            hovermode='x unified', plot_bgcolor='white',
            xaxis_title="Time step", yaxis_title="Number of users",
            legend=dict(orientation='h', y=1.08),
            margin=dict(t=40, b=40))
        fig.update_xaxes(showgrid=True, gridcolor='#f0f0f0')
        fig.update_yaxes(showgrid=True, gridcolor='#f0f0f0')
        st.plotly_chart(fig, use_container_width=True)

        # R₀
        avg_inf  = np.mean([G.nodes[n]['influence'] for n in G.nodes()])
        avg_susc = np.mean([G.nodes[n]['susceptibility'] for n in G.nodes()])
        r0 = (base_infection * avg_inf * avg_susc
              * initial_amp * avg_degree * (1 - recovery_prob))
        r0_color = PALETTE['danger'] if r0 > 1 else PALETTE['success']

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1: kpi("Total reached",
                      f"{metrics['total_reached']:,}",
                      f"{metrics['total_reached']/metrics['total_nodes']*100:.1f}%",
                      PALETTE['danger'])
        with rc2: kpi("Peak spreaders",
                      f"{metrics['max_infected']:,}", "", PALETTE['warning'])
        with rc3: kpi("Steps to peak",
                      str(metrics['steps_to_peak']), "", PALETTE['brand'])
        with rc4: kpi("Estimated R₀", f"{r0:.2f}",
                      ">1 = epidemic threshold", r0_color)

        # Amplification decay
        with st.expander("📉 AI amplification decay curve"):
            steps_arr = np.arange(max_steps)
            amp_arr   = initial_amp * np.exp(-decay_rate * steps_arr)
            fig_amp = go.Figure(go.Scatter(
                x=steps_arr, y=amp_arr, mode='lines',
                line=dict(color=PALETTE['brand'], width=2.5),
                fill='tozeroy', fillcolor='rgba(108,71,255,0.08)'))
            fig_amp.add_hline(y=1.0, line_dash='dot', line_color='gray',
                              annotation_text='No amplification (1×)')
            fig_amp.update_layout(
                xaxis_title="Step", yaxis_title="Amplification ×",
                plot_bgcolor='white', margin=dict(t=10, b=40))
            st.plotly_chart(fig_amp, use_container_width=True)
            xai_insight(
                f"Amplification starts at <strong>{initial_amp:.1f}×</strong> — "
                "modelling how algorithmic recommendation boosts new viral content — "
                "then decays as organic filters and moderation kick in."
            )


# ══════════════════════════════════════════════════════════════
# TAB 2 — NETWORK PLAYBACK
# ══════════════════════════════════════════════════════════════
with tabs[2]:
    section("🕸️", "Interactive Outbreak Playback")

    if st.session_state.last_step_states is None:
        st.info("Run the simulation first.")
    else:
        G           = st.session_state.last_graph
        step_states = st.session_state.last_step_states
        pos         = st.session_state.last_graph_pos

        if step_states and len(step_states) > 1:
            cp1, cp2 = st.columns([1, 2])

            def toggle_play():
                st.session_state.playing = not st.session_state.playing

            with cp1:
                st.button(
                    "⏸️ Pause" if st.session_state.playing else "▶️ Auto-Play",
                    on_click=toggle_play)
            with cp2:
                speed = st.select_slider(
                    "Speed", ["Slow","Normal","Fast"], "Normal")
                sleep_map = {"Slow":0.8,"Normal":0.4,"Fast":0.1}

            max_step = len(step_states) - 1
            sc = st.slider("Timeline", 0, max_step,
                           st.session_state.current_step, key="step_slider")
            st.session_state.current_step = sc

            if st.session_state.playing and sc < max_step:
                st.session_state.current_step += 1
                time.sleep(sleep_map[speed])
                st.rerun()
            elif sc >= max_step:
                st.session_state.playing = False

            state_now = step_states[sc]
            cnt = {s: sum(1 for v in state_now.values() if v == s)
                   for s in ('S','I','R')}
            sc1, sc2, sc3 = st.columns(3)
            with sc1: kpi("Susceptible", f"{cnt['S']:,}", "", '#2196F3')
            with sc2: kpi("Infected",    f"{cnt['I']:,}", "", PALETTE['danger'])
            with sc3: kpi("Recovered",   f"{cnt['R']:,}", "", PALETTE['success'])

            fig_play = go.Figure()
            ex, ey = [], []
            for e in G.edges():
                x0,y0=pos[e[0]]; x1,y1=pos[e[1]]
                ex.extend([x0,x1,None]); ey.extend([y0,y1,None])
            fig_play.add_trace(go.Scatter(
                x=ex, y=ey, mode='lines',
                line=dict(width=0.4, color='#d1d5db'),
                hoverinfo='none', showlegend=False))

            meta = {'S':('Susceptible','#60A5FA',8),
                    'I':('Infected','#EF4444',14),
                    'R':('Recovered','#34D399',8)}
            for sk,(nm,col,sz) in meta.items():
                ns = [n for n,s in state_now.items() if s==sk]
                if not ns: continue
                src = [G.nodes[n].get('source','—') for n in ns]
                fig_play.add_trace(go.Scatter(
                    x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns],
                    mode='markers', name=nm,
                    marker=dict(color=col, size=sz,
                                line=dict(width=0.8, color='white')),
                    text=[f"Node {n} | {src[i]} | "
                          f"Cred {G.nodes[n].get('credibility',0.5):.2f} | "
                          f"Viral {G.nodes[n].get('virality_score',0):.2f}"
                          for i,n in enumerate(ns)],
                    hoverinfo='text'))

            fig_play.update_layout(
                height=550, plot_bgcolor='white',
                legend=dict(orientation='h', y=1.05),
                xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                margin=dict(t=20,b=10,l=10,r=10))
            st.plotly_chart(fig_play, use_container_width=True)

            audience_ctx(mode, {
                "Education":  "🎓 Hover over nodes to see their credibility and "
                              "virality scores — these came from real post data.",
                "Business":   "💼 Red nodes with high-credibility scores are "
                              "infected despite being credible sources — "
                              "they may be amplifying false content unwittingly.",
                "Government": "🏛️ Watch which communities light up first. "
                              "Those are your highest-priority intervention targets.",
                "Research":   "🔬 Node tooltips show source platform, credibility, "
                              "and virality — all seeded from the live data feed.",
            })


# ══════════════════════════════════════════════════════════════
# TAB 3 — EXPLAINABLE AI
# ══════════════════════════════════════════════════════════════
with tabs[3]:
    section("🤖", "Explainable AI (XAI)",
            "SHAP · Permutation importance · Per-node explanations")

    audience_ctx(mode, {
        "Research":   "Three complementary XAI methods. GBM importance is "
                      "model-native; SHAP is game-theoretic; permutation is "
                      "model-agnostic. Agreement between them validates findings.",
        "Business":   "💡 Every model decision is auditable. This panel shows "
                      "exactly why a node was flagged as a high-risk spreader — "
                      "critical for regulatory compliance and internal governance.",
        "Government": "💡 SHAP values provide the evidential chain for each "
                      "flagging decision — suitable for enforcement documentation.",
        "Education":  "🎓 SHAP = SHapley Additive exPlanations. Borrowed from "
                      "game theory, it fairly distributes 'credit' for each "
                      "prediction across all input features.",
    })

    det_metrics = st.session_state.last_detection
    if not det_metrics:
        st.info("Enable 'Train detector + XAI' and re-run.")
    else:
        nd        = det_metrics.get('node_detection', {})
        shap_data = det_metrics.get('shap', {})
        perm_data = det_metrics.get('permutation_importance', {})
        node_exps = det_metrics.get('node_explanations', [])

        if nd:
            section("📐", "Detector performance")
            m_cols = st.columns(5)
            for col, lbl, key in zip(m_cols,
                    ["Accuracy","Precision","Recall","F1","ROC AUC"],
                    ['accuracy','precision','recall','f1','roc_auc']):
                with col: kpi(lbl, f"{nd.get(key,0):.3f}", "", PALETTE['brand'])

            cl, cr = st.columns([1, 2])
            with cl:
                cm = np.array(nd['confusion_matrix'])
                fig_cm = px.imshow(cm, text_auto=True,
                    labels=dict(x="Predicted",y="Actual"),
                    x=['Not spreading','Spreading'],
                    y=['Not spreading','Spreading'],
                    color_continuous_scale=[[0,'#EDE9FF'],[1,'#6C47FF']])
                fig_cm.update_layout(height=260,
                    margin=dict(t=10,b=10,l=10,r=10),
                    coloraxis_showscale=False)
                st.plotly_chart(fig_cm, use_container_width=True)
            with cr:
                feat_names = nd.get('feature_names',
                    ['Degree','Clustering','Community',
                     'Influence','Susceptibility','Infection step'])
                imps = nd.get('feature_importance', [])
                if imps:
                    imp_df = pd.DataFrame({
                        'Feature': feat_names[:len(imps)],
                        'Importance': imps}).sort_values('Importance')
                    fig_imp = go.Figure(go.Bar(
                        x=imp_df['Importance'], y=imp_df['Feature'],
                        orientation='h',
                        marker_color=[
                            f'rgba(108,71,255,{0.35+0.65*v/max(imps)})'
                            for v in imp_df['Importance']]))
                    fig_imp.update_layout(height=260, plot_bgcolor='white',
                        xaxis_title='Importance',
                        margin=dict(t=10,b=10,l=10,r=10))
                    st.plotly_chart(fig_imp, use_container_width=True)

        if shap_data:
            st.divider()
            section("🔷", "SHAP values",
                    "Game-theoretic feature attribution")
            sn = shap_data.get('feature_names', feat_names)
            ms = shap_data.get('mean_abs_shap', [])
            wv = shap_data.get('waterfall_values', [])
            bv = shap_data.get('base_value', 0.5)

            s1, s2 = st.columns(2)
            with s1:
                st.markdown("**Mean |SHAP| — global**")
                if ms:
                    shap_df = pd.DataFrame({
                        'Feature': sn[:len(ms)],
                        'Mean |SHAP|': ms,
                    }).sort_values('Mean |SHAP|')
                    fig_shap = go.Figure(go.Bar(
                        x=shap_df['Mean |SHAP|'], y=shap_df['Feature'],
                        orientation='h', marker_color='#FF6B6B'))
                    fig_shap.update_layout(height=260, plot_bgcolor='white',
                        xaxis_title='Mean |SHAP value|',
                        margin=dict(t=10,b=10))
                    st.plotly_chart(fig_shap, use_container_width=True)

            with s2:
                st.markdown("**Waterfall — single node**")
                if wv:
                    names = sn[:len(wv)]
                    xv, yv, cols, txts = [], [], [], []
                    for fn, sv in sorted(zip(names, wv),
                                         key=lambda t: abs(t[1]), reverse=True):
                        xv.append(fn); yv.append(sv)
                        cols.append('#EF4444' if sv > 0 else '#2DC653')
                        txts.append(f"+{sv:.3f}" if sv > 0 else f"{sv:.3f}")
                    fig_wf = go.Figure(go.Bar(
                        x=xv, y=yv, marker_color=cols,
                        text=txts, textposition='outside'))
                    fig_wf.add_hline(y=0, line_width=1, line_color='gray')
                    fig_wf.update_layout(height=260, plot_bgcolor='white',
                        yaxis_title='SHAP contribution',
                        xaxis_tickangle=-30,
                        margin=dict(t=10,b=60))
                    st.plotly_chart(fig_wf, use_container_width=True)

            xai_insight(
                f"The waterfall chart shows how each feature pushes the model's "
                f"prediction away from the base rate (<strong>{bv:.3f}</strong>). "
                "Red bars increase the spread probability; green bars decrease it. "
                "This explanation applies to a single node — the one with "
                "the highest influence among infected nodes."
            )

            # Beeswarm
            sv_mat = shap_data.get('shap_matrix', [])
            xs_mat = shap_data.get('X_sample', [])
            if sv_mat and xs_mat:
                st.markdown("**SHAP beeswarm — all sampled nodes**")
                sv_arr = np.array(sv_mat)
                xs_arr = np.array(xs_mat)
                n_feat = min(sv_arr.shape[1], len(sn))
                fig_bee = go.Figure()
                for fi in range(n_feat):
                    fig_bee.add_trace(go.Scatter(
                        x=sv_arr[:, fi],
                        y=[sn[fi]] * len(sv_arr),
                        mode='markers',
                        marker=dict(size=5, color=xs_arr[:, fi],
                                    colorscale='RdBu_r',
                                    showscale=(fi == 0),
                                    colorbar=dict(title='Feature\nvalue',
                                                  thickness=10) if fi == 0 else None),
                        showlegend=False))
                fig_bee.update_layout(height=300, plot_bgcolor='white',
                    xaxis_title='SHAP value',
                    margin=dict(t=10,b=10))
                st.plotly_chart(fig_bee, use_container_width=True)
                xai_insight(
                    "Each dot = one node. <strong>X-position</strong> = SHAP contribution. "
                    "<strong>Colour</strong> = feature value (red = high, blue = low). "
                    "When high-value (red) dots cluster on the right, that feature "
                    "consistently <em>increases</em> spread probability."
                )

        if perm_data:
            st.divider()
            section("🔁", "Permutation importance",
                    "Model-agnostic · shuffles each feature · measures AUC drop")
            pi_df = pd.DataFrame({
                'Feature': perm_data['feature_names'][:len(perm_data['importances_mean'])],
                'AUC drop': perm_data['importances_mean'],
                'Std':      perm_data['importances_std'][:len(perm_data['importances_mean'])],
            }).sort_values('AUC drop')
            fig_pi = go.Figure(go.Bar(
                x=pi_df['AUC drop'], y=pi_df['Feature'],
                orientation='h',
                error_x=dict(type='data', array=pi_df['Std'].tolist()),
                marker_color='#2DC653'))
            fig_pi.update_layout(height=260, plot_bgcolor='white',
                xaxis_title='AUC drop when feature shuffled',
                margin=dict(t=10,b=10))
            st.plotly_chart(fig_pi, use_container_width=True)
            xai_insight(
                "A large AUC drop means the model genuinely relies on that feature. "
                "This cross-validates the SHAP results above — if both methods "
                "agree on the most important feature, the finding is robust."
            )

        if node_exps:
            st.divider()
            section("🧑‍💻", "Top spreader explanations",
                    "Natural-language explanation for top-5 infected nodes")
            for exp in node_exps:
                prob_pct = exp['probability'] * 100
                bar_col  = PALETTE['danger'] if prob_pct > 60 else PALETTE['warning']
                st.markdown(
                    f'<div class="xai-insight">'
                    f'<strong>Node {exp["node_id"]}</strong> '
                    f'— {exp["status"]} '
                    f'— model confidence <strong>{prob_pct:.1f}%</strong><br>'
                    f'<div style="background:#e5e7eb;border-radius:4px;'
                    f'height:5px;margin:.4rem 0 .5rem;overflow:hidden">'
                    f'<div style="background:{bar_col};width:{prob_pct:.0f}%;'
                    f'height:100%"></div></div>'
                    f'{exp["explanation"]}'
                    f'</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# TAB 4 — INFECTION PROVENANCE
# ══════════════════════════════════════════════════════════════
with tabs[4]:
    section("🔬", "Infection Provenance Chain",
            "Trace spread from seed posts outward")

    audience_ctx(mode, {
        "Research":   "BFS-based contact tracing from seed nodes. "
                      "Each node records its parent, hop depth, and infection step.",
        "Business":   "💡 Seed nodes came from real high-virality posts. "
                      "This chain shows exactly how the narrative propagated "
                      "from those origin posts through the network.",
        "Government": "💡 This is the attribution chain for a live narrative. "
                      "Seed nodes are the originating posts; each hop is "
                      "one degree of spread.",
        "Education":  "🎓 Think of this as the family tree of the false narrative. "
                      "Seeds = patient zero; hop 1 = first wave of spread.",
    })

    det_metrics = st.session_state.last_detection
    chain = det_metrics.get('infection_chain', {}) if det_metrics else {}

    if not chain:
        st.info("Enable 'Train detector + XAI' and re-run.")
    else:
        G       = st.session_state.last_graph
        pos     = st.session_state.last_graph_pos
        seeds_s = set(st.session_state.last_seeds)
        posts   = st.session_state.live_posts

        # Build seed → post text lookup
        nodes_list = list(G.nodes())
        seed_post_map = {}
        if posts:
            extended = (posts * (len(nodes_list) // len(posts) + 1))[:len(nodes_list)]
            for node, post in zip(nodes_list, extended):
                if node in seeds_s:
                    seed_post_map[node] = post

        hop_counts = {}
        for nid, info in chain.items():
            h = info['hop']
            hop_counts[h] = hop_counts.get(h, 0) + 1

        p1, p2, p3 = st.columns(3)
        with p1: kpi("Seed nodes",   str(len(seeds_s)),                  "", PALETTE['brand'])
        with p2: kpi("Chain depth",  str(max(hop_counts.keys(), default=0)), "hops", PALETTE['warning'])
        with p3: kpi("Nodes traced", str(len(chain)),                    "", PALETTE['danger'])

        # Seed post previews
        if seed_post_map:
            with st.expander("📄 Origin posts (seeds from live data)"):
                for node, post in list(seed_post_map.items())[:5]:
                    src_c = SOURCE_COLORS.get(post.source, "#888")
                    st.markdown(
                        f'<div class="post-card">'
                        f'<div class="meta">'
                        f'<span style="color:{src_c};font-weight:600">'
                        f'SEED NODE {node}</span> · '
                        f'{post.source.upper()} · @{post.author} · '
                        f'virality {post.virality_score:.2f} · '
                        f'{cred_badge(post.credibility)}'
                        f'</div>'
                        f'{post.text[:240]}{"…" if len(post.text)>240 else ""}'
                        f'</div>', unsafe_allow_html=True)

        # Hop pills
        st.markdown("#### Propagation chain by hop")
        hop_css = {0:'pill-seed',1:'pill-hop1',2:'pill-hop2',3:'pill-hop3'}
        for hop in sorted(hop_counts.keys()):
            nodes_at_hop = [(nid, info) for nid, info in chain.items()
                            if info['hop'] == hop]
            nodes_at_hop.sort(key=lambda x: x[1]['influence'], reverse=True)
            lbl = "Seeds (origin)" if hop == 0 else f"Hop {hop}"
            st.markdown(f"**{lbl}** — {len(nodes_at_hop)} nodes")
            pills = ""
            for nid, info in nodes_at_hop[:25]:
                css_c = hop_css.get(hop, 'pill-hop3')
                step_lbl = f"step {info['step']}" if info['step'] >= 0 else "seed"
                pills += (f'<span class="chain-pill {css_c}">'
                          f'N{nid} · {step_lbl} · inf {info["influence"]:.2f}'
                          f'</span>')
            if len(nodes_at_hop) > 25:
                pills += (f'<span class="chain-pill pill-hop3">'
                          f'+{len(nodes_at_hop)-25} more</span>')
            st.markdown(pills, unsafe_allow_html=True)

        # Network graph coloured by hop
        st.markdown("#### Network coloured by infection hop")
        hop_colors = {0:'#6C47FF',1:'#EF4444',2:'#F4A261',3:'#2DC653',-1:'#D1D5DB'}
        fig_chain = go.Figure()
        ex2, ey2 = [], []
        for e in G.edges():
            x0,y0=pos[e[0]]; x1,y1=pos[e[1]]
            ex2.extend([x0,x1,None]); ey2.extend([y0,y1,None])
        fig_chain.add_trace(go.Scatter(
            x=ex2, y=ey2, mode='lines',
            line=dict(width=0.3, color='#e5e7eb'),
            hoverinfo='none', showlegend=False))
        for hv, col in hop_colors.items():
            if hv == -1:
                ns = [n for n in G.nodes() if n not in chain]
                nm = 'Not reached'
            else:
                ns = [nid for nid,info in chain.items() if info['hop']==hv]
                nm = 'Seed' if hv == 0 else f'Hop {hv}'
            if not ns: continue
            sz = 14 if hv == 0 else max(6, 12-hv*2)
            fig_chain.add_trace(go.Scatter(
                x=[pos[n][0] for n in ns], y=[pos[n][1] for n in ns],
                mode='markers', name=nm,
                marker=dict(color=col, size=sz,
                            line=dict(width=0.8, color='white')),
                text=[f"Node {n} | hop {chain.get(n,{}).get('hop','—')}"
                      for n in ns],
                hoverinfo='text'))
        fig_chain.update_layout(
            height=500, plot_bgcolor='white',
            legend=dict(orientation='h', y=1.05),
            xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
            margin=dict(t=20,b=10))
        st.plotly_chart(fig_chain, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 5 — COMMUNITY & ECHO CHAMBERS
# ══════════════════════════════════════════════════════════════
with tabs[5]:
    section("🌐", "Community Structure & Echo Chamber Analysis")

    if st.session_state.last_graph is None:
        st.info("Run the simulation first.")
    else:
        G           = st.session_state.last_graph
        step_states = st.session_state.last_step_states
        pos         = st.session_state.last_graph_pos

        audience_ctx(mode, {
            "Research":   "Louvain community detection. "
                          "Density × infection rate identifies echo chambers.",
            "Business":   "💡 High-density + high-infection groups are your "
                          "most exposed customer segments.",
            "Government": "💡 Top-right bubble chart quadrant = "
                          "highest-priority intervention communities.",
            "Education":  "🎓 Why do some groups become fully infected while others "
                          "barely notice? Internal density = echo chamber strength.",
        })

        communities = nx.get_node_attributes(G, 'community')
        if communities:
            node_data = [{'Node':n,
                          'Community':f"Group {communities[n]}",
                          'State':step_states[-1][n],
                          'Influence':G.nodes[n].get('influence',0),
                          'Source':G.nodes[n].get('source','—'),
                          'Credibility':round(G.nodes[n].get('credibility',0.5),3)}
                         for n in G.nodes()]
            df_nodes = pd.DataFrame(node_data)
            unique_coms = sorted(df_nodes['Community'].unique())

            eco_rows = []
            for com in unique_coms:
                com_nodes = [n for n in G.nodes()
                             if f"Group {communities[n]}" == com]
                density   = nx.density(G.subgraph(com_nodes))
                impacted  = sum(1 for n in com_nodes
                                if step_states[-1][n] in ('I','R'))
                avg_cred  = np.mean([G.nodes[n].get('credibility',0.5)
                                     for n in com_nodes])
                eco_rows.append({
                    'Community':         com,
                    'Density':           round(density, 4),
                    'Infection Rate (%)': round(impacted/len(com_nodes)*100, 1),
                    'Avg Credibility':   round(avg_cred, 3),
                    'Size':              len(com_nodes),
                })
            df_eco = pd.DataFrame(eco_rows)

            fig_bubble = px.scatter(
                df_eco, x='Density', y='Infection Rate (%)',
                size='Size', color='Avg Credibility',
                hover_name='Community',
                color_continuous_scale='RdYlGn',
                title='Echo chamber risk (size = community size, colour = avg credibility)',
                size_max=40)
            fig_bubble.add_vline(x=df_eco['Density'].median(),
                                 line_dash='dot', line_color='gray')
            fig_bubble.add_hline(y=50, line_dash='dot', line_color='gray',
                                 annotation_text='50% infected')
            fig_bubble.update_layout(plot_bgcolor='white',
                                     margin=dict(t=40,b=40))
            st.plotly_chart(fig_bubble, use_container_width=True)
            xai_insight(
                "Communities in the <strong>top-right</strong> combine high internal "
                "connectivity with high infection. The colour shows average content "
                "credibility — darker red communities consumed lower-credibility "
                "content AND spread more of it."
            )

            cc1, cc2 = st.columns(2)
            with cc1:
                fig_pie = px.pie(df_nodes, names='Community', hole=0.45,
                    color_discrete_sequence=px.colors.qualitative.Pastel)
                fig_pie.update_layout(margin=dict(t=10,b=10))
                st.plotly_chart(fig_pie, use_container_width=True)
            with cc2:
                df_eco_s = df_eco.sort_values('Infection Rate (%)', ascending=True)
                fig_bar = go.Figure(go.Bar(
                    x=df_eco_s['Infection Rate (%)'],
                    y=df_eco_s['Community'], orientation='h',
                    marker=dict(color=df_eco_s['Infection Rate (%)'],
                                colorscale='Reds', showscale=False),
                    text=df_eco_s['Infection Rate (%)'].astype(str)+'%',
                    textposition='outside'))
                fig_bar.update_layout(
                    height=max(200, 40*len(df_eco_s)), plot_bgcolor='white',
                    xaxis=dict(range=[0,105]),
                    margin=dict(t=10,b=10))
                st.plotly_chart(fig_bar, use_container_width=True)

            # Community network
            st.markdown("#### Community network (coloured by group)")
            fig_comm = go.Figure()
            ex3,ey3=[],[]
            for e in G.edges():
                x0,y0=pos[e[0]]; x1,y1=pos[e[1]]
                ex3.extend([x0,x1,None]); ey3.extend([y0,y1,None])
            fig_comm.add_trace(go.Scatter(
                x=ex3, y=ey3, mode='lines',
                line=dict(width=0.3,color='#f0f0f0'),
                hoverinfo='none', showlegend=False))
            pal = px.colors.qualitative.Pastel
            for i, cid in enumerate(unique_coms):
                cn = df_nodes[df_nodes['Community']==cid]['Node'].tolist()
                fig_comm.add_trace(go.Scatter(
                    x=[pos[n][0] for n in cn],
                    y=[pos[n][1] for n in cn],
                    mode='markers', name=str(cid),
                    marker=dict(color=pal[i%len(pal)], size=9,
                                line=dict(width=0.5,color='white')),
                    text=[f"Node {n} | Inf {G.nodes[n].get('influence',0):.3f} | "
                          f"Source {G.nodes[n].get('source','—')}" for n in cn]))
            fig_comm.update_layout(
                height=500, plot_bgcolor='white',
                xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
                legend=dict(orientation='h',y=1.04),
                margin=dict(t=10,b=10))
            st.plotly_chart(fig_comm, use_container_width=True)

            # Top influencers
            section("🏆", "Top 10 influencers")
            centrality = nx.pagerank(G)
            top10 = sorted(centrality.items(), key=lambda x:x[1], reverse=True)[:10]
            top_df = pd.DataFrame([{
                "Node": n, "Influence": f"{s:.4f}",
                "Community": G.nodes[n]['community'],
                "Source": G.nodes[n].get('source','—'),
                "Credibility": f"{G.nodes[n].get('credibility',0.5):.3f}",
                "Final status": step_states[-1][n],
            } for n,s in top10])
            st.dataframe(top_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# TAB 6 — COST-BENEFIT & EXPORT
# ══════════════════════════════════════════════════════════════
with tabs[6]:
    section("💰", "Cost-Benefit Analysis & Export")

    if st.session_state.last_metrics is None:
        st.info("Run the simulation first.")
    else:
        metrics  = st.session_state.last_metrics
        hist_df  = st.session_state.last_history
        det_metrics = st.session_state.last_detection

        audience_ctx(mode, {
            "Business":   "Enter your organisation's cost assumptions. "
                          "The model calculates intervention ROI.",
            "Government": "💡 These figures justify counter-disinformation "
                          "programme budgets against harm prevented.",
            "Research":   "Adjust harm valuation to match your study's "
                          "economic model.",
            "Education":  "🎓 Early intervention often shows surprising ROI — "
                          "small early cost vs large late damage.",
        })

        cb1, cb2, cb3 = st.columns(3)
        with cb1: cost_per_bot    = st.number_input("Cost per bot ($)", 1, 500, 50)
        with cb2: harm_per_inf    = st.number_input("Harm per infection ($)", 1, 1000, 150)
        with cb3: platform_budget = st.number_input("Budget ($)", 1000, 50000, 10000)

        total_bots  = metrics.get('bots_deployed', 0)
        total_cost  = total_bots * cost_per_bot
        actual_harm = metrics['total_reached'] * harm_per_inf
        harm_saved  = (metrics['total_nodes'] - metrics['total_reached']) * harm_per_inf
        roi         = ((harm_saved - total_cost) / total_cost * 100) if total_cost > 0 else 0

        r1,r2,r3,r4 = st.columns(4)
        with r1: kpi("Bot cost",        f"${total_cost:,}",
                     f"Budget: ${platform_budget:,}", PALETTE['danger'])
        with r2: kpi("Actual harm",     f"${actual_harm:,}",
                     f"{metrics['total_reached']:,} people", PALETTE['warning'])
        with r3: kpi("Harm prevented",  f"${harm_saved:,}",
                     f"{metrics['total_nodes']-metrics['total_reached']:,} people",
                     PALETTE['success'])
        with r4: kpi("ROI",             f"{roi:.1f}%",
                     "positive = cost-effective", PALETTE['brand'])

        fig_roi = go.Figure(go.Bar(
            x=['Bot cost','Actual harm','Harm prevented'],
            y=[total_cost, actual_harm, harm_saved],
            marker_color=[PALETTE['brand'], PALETTE['danger'], PALETTE['success']],
            text=[f"${v:,.0f}" for v in [total_cost, actual_harm, harm_saved]],
            textposition='outside'))
        fig_roi.update_layout(plot_bgcolor='white', yaxis_title='USD ($)',
                              margin=dict(t=20,b=40))
        st.plotly_chart(fig_roi, use_container_width=True)

        if total_cost > platform_budget:
            st.error(f"🚨 Budget overrun: ${total_cost-platform_budget:,} over limit")
        elif roi > 50:
            st.balloons()
            st.success("✨ Highly efficient — harm prevented far exceeds bot costs.")

        st.divider()
        section("📥", "Export")
        e1, e2, e3, e4 = st.columns(4)
        with e1:
            st.download_button("📊 Spread history (CSV)",
                hist_df.to_csv(index=False).encode(),
                f"spread_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "text/csv", use_container_width=True)
        config = {
            "num_nodes":num_nodes,"avg_degree":avg_degree,
            "base_infection":base_infection,"recovery_prob":recovery_prob,
            "initial_amp":initial_amp,"decay_rate":decay_rate,
            "num_seeds":len(st.session_state.last_seeds),
            "max_steps":max_steps,"seed":seed_val,"metrics":metrics,
        }
        with e2:
            st.download_button("⚙️ Config (JSON)",
                json.dumps(config, indent=2, default=str).encode(),
                f"config_{datetime.now():%Y%m%d_%H%M%S}.json",
                "application/json", use_container_width=True)
        if det_metrics:
            det_export = {k:v for k,v in det_metrics.items()
                          if k not in ('shap','infection_chain')}
            with e3:
                st.download_button("🤖 XAI metrics (JSON)",
                    json.dumps(det_export, indent=2, default=str).encode(),
                    f"xai_{datetime.now():%Y%m%d_%H%M%S}.json",
                    "application/json", use_container_width=True)
        feed_df = st.session_state.feed_df
        if feed_df is not None and len(feed_df) > 0:
            with e4:
                st.download_button("📡 Live feed (CSV)",
                    feed_df.to_csv(index=False).encode(),
                    f"feed_{datetime.now():%Y%m%d_%H%M%S}.csv",
                    "text/csv", use_container_width=True)


# ══════════════════════════════════════════════════════════════
# TAB 7 — COUNTERFACTUAL SIMULATOR
# ══════════════════════════════════════════════════════════════
with tabs[7]:
    section("🧪", "AI Governance Counterfactual Simulator",
            "5 platform policy scenarios — seeded from live data")

    audience_ctx(mode, {
        "Research":   "Each scenario uses a fresh graph copy to avoid "
                      "cross-contamination from echo chamber rewiring.",
        "Business":   "💡 Build the business case for content moderation investment. "
                      "'Aggressive moderation' typically shows the largest "
                      "reduction in total reach.",
        "Government": "💡 Map each scenario to a real regulatory option: "
                      "amplification caps, early moderation, algorithmic dampening.",
        "Education":  "🎓 Which policy is most effective? Which has the best "
                      "tradeoff between reach reduction and intervention cost?",
    })

    if not st.session_state.last_seeds:
        st.info("Run the main simulation first to enable counterfactual analysis.")
    else:
        if st.button("▶️ Run all 5 policy scenarios", type="primary",
                     use_container_width=True):
            with st.spinner("Simulating policy scenarios…"):
                G_cf = get_graph(num_nodes, avg_degree, seed_val)
                feed_df = st.session_state.feed_df
                cdf = run_counterfactual_simulations(
                    G_cf, st.session_state.last_seeds,
                    base_infection, recovery_prob, max_steps,
                    initial_amp, decay_rate,
                    feed_df, None, 1.5)
                st.session_state.counter_results = cdf

        if st.session_state.counter_results is not None:
            cdf = st.session_state.counter_results
            baseline = cdf.iloc[0]
            cdf['vs Baseline'] = (
                (cdf['Total Reached'] - baseline['Total Reached'])
                / baseline['Total Reached'] * 100
            ).round(1).astype(str) + '%'
            st.dataframe(cdf, use_container_width=True, hide_index=True)

            cf1, cf2 = st.columns(2)
            with cf1:
                fig_bar = px.bar(cdf, x='Scenario', y='Total Reached',
                    color='Total Reached', color_continuous_scale='Reds',
                    title='Total users reached by scenario')
                fig_bar.update_layout(showlegend=False, plot_bgcolor='white',
                    xaxis_tickangle=-25, margin=dict(t=40,b=80))
                st.plotly_chart(fig_bar, use_container_width=True)
            with cf2:
                fig_line = go.Figure(go.Scatter(
                    x=cdf['Scenario'], y=cdf['Peak Spread'],
                    mode='lines+markers',
                    line=dict(color=PALETTE['brand'],width=2.5),
                    marker=dict(size=8)))
                fig_line.update_layout(title='Peak spreaders by scenario',
                    plot_bgcolor='white', xaxis_tickangle=-25,
                    margin=dict(t=40,b=80))
                st.plotly_chart(fig_line, use_container_width=True)

            best     = cdf.sort_values('Total Reached').iloc[0]
            baseline = cdf.iloc[0]
            reduction = ((baseline['Total Reached'] - best['Total Reached'])
                         / baseline['Total Reached'] * 100)
            st.success(
                f"**Best policy:** {best['Scenario']} — "
                f"**{reduction:.1f}% reduction** in total reach "
                f"({best['Total Reached']:,} vs {baseline['Total Reached']:,})")
            xai_insight(
                f"Baseline reaches <strong>{baseline['Total Reached']:,}</strong> users. "
                f"Best policy — <strong>{best['Scenario']}</strong> — "
                f"cuts this to <strong>{best['Total Reached']:,}</strong>, "
                "quantifying the exact value of each platform policy lever "
                "when seeded from <strong>real-world post data</strong>."
            )
            st.download_button("📥 Download counterfactual results (CSV)",
                cdf.to_csv(index=False).encode(),
                f"counterfactual_{datetime.now():%Y%m%d_%H%M%S}.csv",
                "text/csv", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Welcome screen
# ─────────────────────────────────────────────────────────────
if (st.session_state.last_history is None
        and not st.session_state.live_posts):
    st.markdown("### 👋 Getting started")
    audience_ctx(mode, {
        "Research":   "1. Click <strong>📡 Fetch Live Data</strong> to ingest posts "
                      "(demo mode works without any API keys). "
                      "2. Click <strong>🚀 Run Simulation</strong>. "
                      "3. Enable XAI for full SHAP analysis.",
        "Business":   "1. Enter your topic (e.g. your brand name) in the keyword field. "
                      "2. Fetch live data. 3. Run the simulation. "
                      "4. Use the Cost-Benefit tab to quantify exposure risk.",
        "Government": "1. Enter a policy-relevant keyword. "
                      "2. Fetch data. 3. Run simulation. "
                      "4. Use the Counterfactual tab to test five interventions.",
        "Education":  "1. Use Demo mode (no keys needed). "
                      "2. Try topic = 'vaccines'. 3. Run the simulation. "
                      "4. Change the amplification slider and re-run. "
                      "What changes?",
    })
    cols = st.columns(4)
    features = [
        ("📡","Real-time ingestion",    "Twitter · Reddit · NewsAPI · RSS · Demo"),
        ("📈","SIR + amplification",    "Heterogeneous nodes, decaying AI boost"),
        ("🤖","Explainable AI",         "SHAP · Permutation · Per-node reasoning"),
        ("🧪","Policy simulator",       "5 counterfactual scenarios from live seeds"),
    ]
    for col,(ic,title,sub) in zip(cols, features):
        with col:
            st.markdown(
                f'<div class="kpi-card" style="text-align:center">'
                f'<div style="font-size:1.8rem">{ic}</div>'
                f'<div style="font-weight:600;margin:.3rem 0 .2rem">{title}</div>'
                f'<div style="font-size:.8rem;color:{PALETTE["muted"]}">{sub}</div>'
                f'</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f'<div style="text-align:center;color:{PALETTE["muted"]};'
    f'padding:1rem 0;font-size:.78rem">'
    f'<strong>GAGS Framework · Disinformation Simulation v5.0</strong> · '
    f'Real-time ingestion · SIR · XAI · Echo chambers · '
    f'Policy counterfactuals · For research and educational purposes only'
    f'</div>', unsafe_allow_html=True)