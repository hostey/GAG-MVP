# pages/06_📡_Real_Time_Monitor.py
"""
GAGS Real-Time Monitoring Mode — v1.0
Continuous fairness monitoring with drift detection, alert thresholds,
and a simulated live data stream for demonstration.
"""
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from components.translate import install_auto_translate, tx, tx_plotly
install_auto_translate()

from components.governance_logic import (
    generate_synthetic_data,
    apply_bias,
    simulate_data_poisoning,
    calculate_fairness_metrics,
    AttackSeverity,
)
from utils.config import simulation_config, settings

try:
    from components.i18n import t, get_lang
except ImportError:
    def t(key, lang=None): return key
    def get_lang(): return "en"

# ── Page setup ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Real-Time Monitor • GAGS",
    page_icon="📡",
    layout="wide",
)

# ── Auto-dismiss stale guided tour banners from other pages ───────────────────
for _tk in ["_tour_dismissed_agrotech","_tour_dismissed_health","_tour_dismissed_security"]:
    if _tk not in st.session_state:
        st.session_state[_tk] = True


# ── Design system ──────────────────────────────────────────────────────────────
try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, page_header, plotly_theme as _ptheme
    inject_css("monitor")
    ACCENT = DOMAIN_ACCENTS["monitor"]
except ImportError:
    ACCENT = "#0284c7"


st.markdown(
    f"""<div class="page-header" style="--ac:#00ffd0;"><p style="font-family:'DM Mono',monospace;font-size:.69rem;letter-spacing:.16em;text-transform:uppercase;opacity:.5;margin:0 0 .55rem;display:flex;align-items:center;gap:.45rem;"><span style="width:16px;height:1px;background:#00ffd0;opacity:.55;display:inline-block;"></span>REAL-TIME MONITOR · GAGS v3.0</p><h1 style="font-family:'Syne',sans-serif!important;font-size:2.5rem!important;font-weight:800!important;line-height:1.08!important;letter-spacing:-.03em!important;margin:0 0 .6rem!important;">Real-Time Fairness Monitor</h1><p style="margin:0;opacity:.72;font-size:.96rem;max-width:660px;line-height:1.65;">Continuous fairness monitoring with drift detection — 4 drift scenarios, configurable alert thresholds, REST API push.</p><div style="margin-top:.9rem;"><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#00ffd0;">Live Stream</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#00ffd0;">Drift Detection</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#00ffd0;">Auto-Alert</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#00ffd0;">4 Scenarios</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#00ffd0;">REST API Integration</span><span style="display:inline-flex;align-items:center;padding:.2rem .68rem;border-radius:99px;font-family:'DM Mono',monospace;font-size:.67rem;letter-spacing:.05em;font-weight:500;border:1px solid;text-transform:uppercase;margin:.18rem .12rem 0 0;background:rgba(var(--acr,255,255,255),.11);border-color:rgba(var(--acr,255,255,255),.32);color:#00ffd0;">Configurable Thresholds</span></div></div>""",
    unsafe_allow_html=True
)

# ── Session state ──────────────────────────────────────────────────────────────
_MON_DEFAULTS = {
    "mon_snapshots":    [],
    "mon_alerts":       [],
    "mon_running":      False,
    "mon_tick":         0,
    "mon_model_id":     "model-gags-v1",
}
for k, v in _MON_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Thresholds ─────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "fairness_score":                ("above", 0.70, "Fairness Score"),
    "demographic_parity_difference": ("below", 0.10, "Demographic Parity Gap"),
    "equalized_odds_difference":     ("below", 0.10, "Equalized Odds Gap"),
    "accuracy":                      ("above", 0.55, "Accuracy"),
    "fpr":                           ("below", 0.20, "False Positive Rate"),
}

def _is_breached(metric: str, value: float) -> bool:
    if metric not in THRESHOLDS: return False
    direction, threshold, _ = THRESHOLDS[metric]
    return (direction == "above" and value < threshold) or \
           (direction == "below" and value > threshold)

def _simulate_snapshot(tick: int, drift_scenario: str, domain: str) -> Dict:
    """Generate one simulated monitoring snapshot."""
    n   = 400
    rng = np.random.RandomState(tick)

    X, y, demo = generate_synthetic_data(n_samples=n, n_features=10,
        decision_boundary=simulation_config.DECISION_BOUNDARY)
    X = X.astype(np.float64)

    # Inject drift based on scenario
    drift_intensity = 0.0
    if drift_scenario == "Gradual bias drift":
        drift_intensity = min(0.05 + tick * 0.04, 0.7)
    elif drift_scenario == "Sudden bias spike":
        drift_intensity = 0.6 if tick >= 5 else 0.1
    elif drift_scenario == "Adversarial attack":
        drift_intensity = 0.15
        if tick >= 4:
            X, y, demo = simulate_data_poisoning(X, y, 0.15, attack_type="label_flipping",
                demographic_info=demo, targeted=False)
    elif drift_scenario == "Stable (no drift)":
        drift_intensity = 0.08

    if drift_intensity > 0:
        X, y, demo = apply_bias(X, y, "demographic", drift_intensity,
            demographic_info=demo, severity=AttackSeverity.MEDIUM)

    # Quick model evaluation
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X_sc, y, demo, test_size=0.3, random_state=tick,
        stratify=y if (len(np.unique(y)) > 1 and np.bincount(y.astype(int)).min() >= 2) else None)
    clf = LogisticRegression(max_iter=300, class_weight="balanced", random_state=tick)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    fair = calculate_fairness_metrics(y_te, y_pred, d_te)
    acc  = float(accuracy_score(y_te, y_pred))
    fpr  = float(np.mean(y_pred[y_te==0]==1)) if (y_te==0).any() else 0.0

    metrics = {
        "fairness_score":                round(fair.get("fairness_score",0.5), 4),
        "demographic_parity_difference": round(fair.get("demographic_parity_difference",0), 4),
        "equalized_odds_difference":     round(fair.get("equalized_odds_difference",0), 4),
        "accuracy":                      round(acc, 4),
        "fpr":                           round(fpr, 4),
    }

    alerts = [f"{THRESHOLDS[m][2]} breached: {v:.3f} (threshold: {THRESHOLDS[m][1]})"
              for m, v in metrics.items() if _is_breached(m, v)]

    return {
        "snapshot_id":  f"snap-{tick:04d}",
        "tick":         tick,
        "timestamp":    (datetime.now() - timedelta(minutes=(10-tick)*2)).isoformat(),
        "n_samples":    len(y_te),
        "metrics":      metrics,
        "alerts":       alerts,
        "drift_intensity": round(drift_intensity, 3),
        "domain":       domain,
    }

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Monitor Configuration")

    mon_domain = st.selectbox("Domain", ["healthcare","national_security","agrotech"],
        format_func=lambda x: {"healthcare":"🏥 Healthcare","national_security":"🛡️ Security",
                               "agrotech":"🌾 Agrotech"}[x])
    drift_scenario = st.selectbox("Drift Scenario", [
        "Stable (no drift)", "Gradual bias drift",
        "Sudden bias spike", "Adversarial attack"])
    n_ticks = st.slider("Snapshots to simulate", 3, 15, 8)
    tick_delay = st.slider("Delay between snapshots (sec)", 0.0, 2.0, 0.3, 0.1)

    st.divider()
    st.markdown("### Alert Thresholds")
    for metric, (direction, default, label) in THRESHOLDS.items():
        new_val = st.slider(
            f"{label} ({'min' if direction=='above' else 'max'})",
            0.0, 1.0, float(default), 0.05,
            key=f"_thresh_{metric}",
        )
        THRESHOLDS[metric] = (direction, new_val, label)

    st.divider()
    c1, c2 = st.columns(2)
    run_btn = c1.button("▶ Start", type="primary", use_container_width=True)
    clr_btn = c2.button("🗑 Clear",               use_container_width=True)

if clr_btn:
    st.session_state.mon_snapshots = []
    st.session_state.mon_alerts    = []
    st.session_state.mon_tick      = 0
    st.rerun()

# ── Run simulation stream ──────────────────────────────────────────────────────
if run_btn:
    st.session_state.mon_snapshots = []
    st.session_state.mon_alerts    = []
    st.session_state.mon_tick      = 0

    prog = st.progress(0, text="Initialising monitoring stream…")
    status_ph = st.empty()

    for tick in range(n_ticks):
        prog.progress((tick+1)/n_ticks, text=f"Snapshot {tick+1}/{n_ticks}…")
        snap = _simulate_snapshot(tick, drift_scenario, mon_domain)
        st.session_state.mon_snapshots.append(snap)
        for alert in snap["alerts"]:
            st.session_state.mon_alerts.append({
                "tick": tick, "message": alert, "timestamp": snap["timestamp"]})
        st.session_state.mon_tick = tick + 1
        if tick_delay > 0:
            time.sleep(tick_delay)

    prog.empty()
    st.session_state.mon_running = False

# ── Display results ────────────────────────────────────────────────────────────
snaps  = st.session_state.mon_snapshots
alerts = st.session_state.mon_alerts

if snaps:
    df: pd.DataFrame = pd.DataFrame()  # safe default; overwritten below
    df = pd.DataFrame([{"tick": s["tick"], "timestamp": s["timestamp"][:16],
                        **s["metrics"], "n_alerts": len(s["alerts"])}
                       for s in snaps])

    # ── Live status bar ─────────────────────────────────────────────────────
    latest  = snaps[-1]["metrics"]
    n_alert = len(snaps[-1]["alerts"])
    is_ok   = n_alert == 0

    dot_cls = "live-green" if is_ok else "live-red"
    status_label = "All systems nominal" if is_ok else f"{n_alert} threshold(s) breached"
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:.75rem;"
        f"background:var(--color-background-secondary);"
        f"border:0.5px solid var(--color-border-tertiary);"
        f"border-radius:var(--border-radius-md);padding:.5rem 1rem;"
        f"margin-bottom:1rem;'>"
        f"<span class='live-dot {dot_cls}'></span>"
        f"<strong style='font-size:.9rem;'>Live Monitor: {st.session_state.mon_model_id}</strong>"
        f"<span style='font-size:.85rem;color:var(--color-text-secondary);'>"
        f"  {status_label} · {len(snaps)} snapshots · "
        f"  Scenario: {drift_scenario}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── KPIs ────────────────────────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    for col, (mkey, mlabel) in zip([k1,k2,k3,k4,k5], [
        ("fairness_score","Fairness Score"),
        ("demographic_parity_difference","Parity Gap"),
        ("equalized_odds_difference","Eq. Odds Gap"),
        ("accuracy","Accuracy"),
        ("fpr","False Positive Rate"),
    ]):
        val   = latest.get(mkey, 0)
        delta = val - snaps[0]["metrics"].get(mkey, val) if len(snaps) > 1 else None
        breached = _is_breached(mkey, val)
        col.metric(
            mlabel, f"{val:.3f}",
            f"{delta:+.3f}" if delta is not None else None,
            delta_color="inverse" if mkey in ("demographic_parity_difference",
                                               "equalized_odds_difference","fpr") else "normal",
        )
        if breached:
            col.markdown(
                f"<div class='alert-item alert-critical'>⚠️ Threshold breached</div>",
                unsafe_allow_html=True,
            )

    st.divider()

    # ── Time-series charts ──────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📈 Metric Trends", "⚠️ Alert Timeline", "📋 Raw Snapshots"])

    with tab1:
        fig = make_subplots(rows=2, cols=2,
            subplot_titles=("Fairness Score", "Demographic Parity Gap",
                            "Accuracy", "False Positive Rate"))
        colour_map = {"fairness_score":"#16a34a","demographic_parity_difference":"#ef4444",
                      "accuracy":"#1d4ed8","fpr":"#e67e22"}

        for (row, col, mkey, threshold_dir, threshold_val) in [
            (1,1,"fairness_score","above",THRESHOLDS["fairness_score"][1]),
            (1,2,"demographic_parity_difference","below",THRESHOLDS["demographic_parity_difference"][1]),
            (2,1,"accuracy","above",THRESHOLDS["accuracy"][1]),
            (2,2,"fpr","below",THRESHOLDS["fpr"][1]),
        ]:
            colour = colour_map[mkey]
            fig.add_trace(go.Scatter(
                x=df["tick"], y=df[mkey], mode="lines+markers",
                name=mkey.replace("_"," ").title(),
                line=dict(color=colour, width=2),
                marker=dict(size=7, color=colour),
            ), row=row, col=col)
            # Threshold line
            fig.add_hline(y=threshold_val, line_dash="dot", line_color="#888",
                          line_width=1, row=row, col=col)

        fig.update_layout(height=420, showlegend=False, margin=dict(t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)

        # Drift intensity
        drift_vals = [s.get("drift_intensity", 0) for s in snaps]
        fig_drift = px.area(x=list(range(len(drift_vals))), y=drift_vals,
            title="Injected Drift Intensity Over Time",
            labels={"x":"Snapshot","y":"Drift Intensity"},
            color_discrete_sequence=["#ef4444"])
        fig_drift.update_layout(height=200, margin=dict(t=40,b=20))
        st.plotly_chart(fig_drift, use_container_width=True)

    with tab2:
        if not alerts:
            st.markdown('<div class="alert-item alert-ok">✅ No alerts triggered across all snapshots.</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f"**{len(alerts)} alert(s) triggered:**")
            for a in reversed(alerts):
                st.markdown(
                    f"<div class='alert-item alert-critical'>"
                    f"⚠️ Tick {a['tick']} — {a['message']}</div>",
                    unsafe_allow_html=True,
                )

        # Alert rate chart
        alert_counts = [len(s["alerts"]) for s in snaps]
        if any(c > 0 for c in alert_counts):
            fig_a = px.bar(x=list(range(len(alert_counts))), y=alert_counts,
                title="Alerts per Snapshot",
                labels={"x":"Snapshot","y":"Alert Count"},
                color=alert_counts,
                color_continuous_scale="Reds")
            fig_a.update_layout(height=220, margin=dict(t=40,b=20))
            st.plotly_chart(fig_a, use_container_width=True)

        # Auto-response recommendation
        total_alerts = len(alerts)
        if total_alerts == 0:
            st.success("✅ Model operating within all thresholds. Continue monitoring.")
        elif total_alerts <= 3:
            st.warning("⚠️ Minor drift detected. Investigate bias source and schedule retraining.")
        else:
            st.error("❌ Significant drift. Consider pausing model and triggering bias mitigation.")

    with tab3:
        display_df = df.copy()
        display_df.columns = [c.replace("_"," ").title() for c in display_df.columns]
        st.dataframe(
            display_df.style
                .background_gradient(subset=["Fairness Score"], cmap="RdYlGn")
                .background_gradient(subset=["Accuracy"], cmap="Greens")
                .background_gradient(subset=["N Alerts"], cmap="Reds"),
            use_container_width=True,
        )
        st.download_button("📥 Download snapshots CSV",
            df.to_csv(index=False).encode(),
            "gags_monitor_snapshots.csv", "text/csv",
            use_container_width=True)

    # ── API integration note ──────────────────────────────────────────────
    st.divider()
    st.markdown("### 🔌 Connect to a Live Model via REST API")
    with st.expander("How to push real model predictions to this monitor"):
        st.markdown("""
        Start the GAGS REST API, then push predictions from your model-serving layer:

        ```python
        import requests

        # After each batch of predictions from your production model:
        requests.post("http://localhost:8502/monitor/snapshot", json={
            "model_id":         "my-production-model-v2",
            "y_true":           y_true.tolist(),
            "y_pred":           y_pred.tolist(),
            "demographic_info": demographic_groups.tolist(),
            "metadata": {
                "batch_id":  "batch-2024-001",
                "n_samples": len(y_true),
            }
        })
        ```

        Start the API:
        ```bash
        pip install fastapi uvicorn
        uvicorn api:app --host 0.0.0.0 --port 8502
        ```

        Get monitoring status:
        ```bash
        curl http://localhost:8502/monitor/status
        ```
        """)

else:
    # Welcome state
    st.markdown("## 📡 Welcome to Real-Time Monitoring")
    st.markdown(
        "Configure a drift scenario in the sidebar and click **▶ Start** "
        "to simulate continuous monitoring. Connect your production model "
        "via the REST API to monitor live predictions."
    )

    c1, c2, c3 = st.columns(3)
    for col, (title, color, desc) in zip([c1,c2,c3],[
        ("Gradual Drift",    "#f39c12", "Bias slowly compounds over retraining cycles. Common in automated ML pipelines."),
        ("Sudden Spike",     "#ef4444", "Abrupt bias injection simulates dataset poisoning or sudden population shift."),
        ("Adversarial Attack","#6d28d9","Targeted label flipping at snapshot 4 — tests attack detection latency."),
    ]):
        col.markdown(
            f"<div style='border-left:4px solid {color};padding:.75rem 1rem;"
            f"background:var(--color-background-secondary);border-radius:0 8px 8px 0;'>"
            f"<strong>{title}</strong>"
            f"<p style='font-size:.83rem;color:var(--color-text-secondary);margin:.3rem 0 0;'>{desc}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.divider()
st.markdown("<div style='text-align:center;color:#7f8c8d;font-size:.82rem;padding:.75rem 0;'>"
            "📡 Real-Time Monitor · GAGS v3.0 · Continuous Fairness Monitoring</div>",
            unsafe_allow_html=True)
