"""
components/gags_lifecycle_ui.py — GAGS Lifecycle & Sustainability UI v1.0
==========================================================================
Renders the 🔄 Lifecycle & 🌱 Eco tabs in all 8 simulation modules.
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

from components.gags_lifecycle import (
    ModelRegistry, ContinuousMonitor, ComplianceAutomation,
    EnergyConsumptionModeller, ALGO_ENV_PROFILES, ECO_TIER_COLOURS,
    NIGERIA_GRID_CARBON_INTENSITY,
)

C_NAVY  = "#1e3a5f"
C_GREEN = "#16a34a"
C_AMBER = "#d97706"
C_RED   = "#dc2626"
C_INFO  = "#0891b2"
C_LIGHT = "#f8fafc"


def _score_c(v: float) -> str:
    return C_GREEN if v >= 0.75 else C_AMBER if v >= 0.55 else C_RED


def _sev_c(s: str) -> str:
    return {"warning": C_AMBER, "critical": C_RED}.get(s, C_INFO)


def _card(title: str, value: str, sub: str, colour: str) -> None:
    st.markdown(
        f"<div style='background:{colour}12;border:1px solid {colour}40;"
        f"border-left:4px solid {colour};border-radius:8px;padding:10px 14px;'>"
        f"<p style='font-size:.68rem;font-weight:700;color:{colour};text-transform:uppercase;"
        f"letter-spacing:.07em;margin:0 0 4px;'>{title}</p>"
        f"<p style='font-size:1.5rem;font-weight:700;color:#0f172a;margin:0 0 2px;'>{value}</p>"
        f"<p style='font-size:.72rem;color:#6b7280;margin:0;'>{sub}</p>"
        f"</div>", unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE TAB
# ══════════════════════════════════════════════════════════════════════════════

# Inside components/gags_lifecycle_ui.py

def render_lifecycle_tab(domain_key: str = "health", lifecycle_report: dict | str | None = None) -> None:
    """
    Renders the Model Lifecycle Management diagnostic component:
    - Model Registry
    - Continuous Monitoring & Drift Detection
    - Compliance Audit Certificate
    """
    # ── 1. State Retrieval & Input Normalisation ──────────────────────────────
    # Auto-resolve from st.session_state if report is not passed explicitly
    if lifecycle_report is None:
        lifecycle_report = st.session_state.get(
            f"{domain_key}_lifecycle_report",
            st.session_state.get("health_lifecycle_report", {})
        )

    if isinstance(lifecycle_report, str):
        try:
            import json
            lifecycle_report = json.loads(lifecycle_report)
        except Exception:
            lifecycle_report = {}

    if not isinstance(lifecycle_report, dict):
        lifecycle_report = {}

    pillars = lifecycle_report.get("pillars", {})

    # ── 2. Empty State Fallback ────────────────────────────────────────────────
    if not pillars:
        st.info(
            "⚠️ **No Lifecycle Report Available**\n\n"
            "Enable **Lifecycle Management** under *Analytical Modules* in the sidebar and run the simulation."
        )
        return

    # ── 3. Component Header ───────────────────────────────────────────────────
    # Uses Streamlit subheader instead of raw H2 for clean container/expander nesting
    st.subheader("🔄 Model Lifecycle Management")
    st.caption(
        f"Domain: **{domain_key.replace('_', ' ').title()}** | "
        "Registry · Continuous Monitoring · Regulatory Audit"
    )

    # ── 4. Sub-Tab Diagnostics ────────────────────────────────────────────────
    tab_reg, tab_mon, tab_audit = st.tabs([
        "📦 Model Registry",
        "📡 Continuous Monitoring",
        "📜 Compliance Audit"
    ])

    with tab_reg:
        _render_registry(
            reg=pillars.get("registry", {}),
            domain=domain_key
        )

    with tab_mon:
        _render_monitoring(
            mon=pillars.get("monitoring", {}),
            domain=domain_key
        )

    with tab_audit:
        _render_audit(
            audit=pillars.get("audit", {})
        )
# ── Model Registry ────────────────────────────────────────────────────────────

def _render_registry(reg: dict, domain: str) -> None:
    st.markdown("### 📦 Model Version Registry")
    if not reg:
        st.info("No versions registered yet. Run a simulation first.")
        return

    current = reg.get("current_version", {})
    all_v   = reg.get("all_versions", [])
    drift   = reg.get("fairness_drift")

    # Current version card
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        _card("Active Version", current.get("version_id", "—"),
              current.get("algo_label", ""), C_GREEN)
    with c2:
        _card("Status", current.get("status", "—").upper(),
              current.get("timestamp", "")[:10], C_INFO)
    with c3:
        fs = current.get("fairness_metrics", {}).get("fairness_score", 0)
        _card("Fairness Score", f"{fs:.3f}",
              "✅ meets NITDA ≥ 0.75" if fs >= 0.75 else "❌ below threshold",
              C_GREEN if fs >= 0.75 else C_RED)
    with c4:
        _card("Eco Score", f"{current.get('eco_score', 0):.3f}",
              "Environmental efficiency", C_GREEN if current.get("eco_score", 0) >= 0.70 else C_AMBER)

    st.divider()

    # Fairness drift
    if drift:
        st.markdown("#### 📉 Fairness Drift vs Baseline")
        direction = drift.get("overall_direction", "stable")
        col = C_GREEN if direction == "improving" else C_RED
        st.markdown(f"""<div style='background:{col}12;border-left:4px solid {col};
        border-radius:6px;padding:8px 14px;margin-bottom:12px;font-size:.85rem;'>
        <b>Baseline:</b> {drift['baseline_version']} →
        <b>Active:</b> {drift['active_version']} |
        Overall direction: <b style='color:{col}'>{direction.upper()}</b>
        </div>""", unsafe_allow_html=True)

        metric_drift = drift.get("metric_drift", {})
        if metric_drift:
            drift_df = pd.DataFrame([
                {"Metric": k.replace("_"," ").title(),
                 "Δ Change": round(v, 4),
                 "Direction": "▲ Improving" if v > 0 else "▼ Degrading" if v < 0 else "— Stable"}
                for k, v in metric_drift.items()
            ])
            st.dataframe(drift_df.style.background_gradient(subset=["Δ Change"], cmap="RdYlGn"),
                        use_container_width=True)

    st.divider()

    # Version history table
    st.markdown(f"#### 🗂️ All Versions for {domain.title()} ({len(all_v)} total)")
    if all_v:
        df_v = pd.DataFrame([{
            "Version":       v["version_id"],
            "Algorithm":     v["algo_label"],
            "Status":        v["status"].upper(),
            "Fairness":      v.get("fairness_metrics", {}).get("fairness_score", 0),
            "Safety":        v.get("safety_score", 0),
            "Eco Score":     v.get("eco_score", 0),
            "Timestamp":     v.get("timestamp", "")[:19].replace("T", " "),
        } for v in all_v])

        st.dataframe(
            df_v.style
            .format({"Fairness": "{:.3f}", "Safety": "{:.3f}", "Eco Score": "{:.3f}"})
            .background_gradient(subset=["Fairness", "Safety"], cmap="RdYlGn"),
            use_container_width=True,
        )

        # Rollback selector
        with st.expander("⏪ Rollback to Previous Version", expanded=False):
            archived = [v for v in all_v if v["status"] == "archived"]
            if not archived:
                st.info("No archived versions available for rollback.")
            else:
                sel = st.selectbox(
                    "Select version to restore as active:",
                    [v["version_id"] for v in archived],
                    key=f"_rollback_{domain}",
                )
                if st.button("⏪ Rollback Now", key=f"_rollback_btn_{domain}"):
                    if ModelRegistry.rollback(sel):
                        st.success(f"✅ Rolled back to {sel}. Previous version archived.")
                        st.rerun()
                    else:
                        st.error("Rollback failed — version not found.")


# ── Continuous Monitoring ─────────────────────────────────────────────────────

def _render_monitoring(mon: dict, domain: str) -> None:
    st.markdown("### 📡 Continuous Monitoring & Drift Detection")
    if not mon:
        st.info("No monitoring data yet.")
        return

    all_alerts   = mon.get("all_alerts", [])
    new_alerts   = mon.get("new_alerts", [])
    n_snaps      = mon.get("domain_snapshots", 0)
    trend_data   = mon.get("trend_analysis", {})

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    crit = len([a for a in all_alerts if a.get("severity") == "critical"])
    warn = len([a for a in all_alerts if a.get("severity") == "warning"])
    with c1:
        _card("Total Snapshots",  str(n_snaps),      "This session",       C_INFO)
    with c2:
        _card("Critical Alerts",  str(crit),          "Require action",     C_RED if crit else C_GREEN)
    with c3:
        _card("Warnings",         str(warn),          "Monitor closely",    C_AMBER if warn else C_GREEN)
    with c4:
        drifting = sum(1 for t in trend_data.values() if t.get("drift_detected"))
        _card("Drifting Metrics", str(drifting),      "Monotonic degradation", C_RED if drifting else C_GREEN)

    st.divider()

    # Trend charts
    if n_snaps >= 2:
        st.markdown("#### 📈 Metric Trends Across Runs")
        snaps = [s for s in ContinuousMonitor._snapshots() if s["domain"] == domain]
        if snaps:
            metrics_to_plot = ["fairness_score", "accuracy", "demographic_parity"]
            fig = go.Figure()
            for m in metrics_to_plot:
                vals = [s["metrics"].get(m) for s in snaps if s["metrics"].get(m) is not None]
                if vals:
                    colour = {"fairness_score": C_GREEN, "accuracy": C_INFO,
                              "demographic_parity": C_RED}.get(m, "#94a3b8")
                    fig.add_scatter(
                        x=list(range(1, len(vals)+1)),
                        y=vals, mode="lines+markers",
                        name=m.replace("_"," ").title(),
                        line=dict(color=colour, width=2),
                        marker=dict(size=6),
                    )
            fig.update_layout(
                title="Fairness & Performance Trend",
                xaxis_title="Run #", yaxis_title="Value",
                height=280, margin=dict(t=40, b=30, l=30, r=10),
                legend=dict(orientation="h", y=1.15),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, use_container_width=True, key=f"_mon_trend_{domain}")

    # Trend analysis
    if trend_data:
        st.markdown("#### 🔍 Drift Detection")
        for metric, result in trend_data.items():
            if result.get("drift_detected"):
                st.markdown(
                    f"<div style='background:{C_RED}12;border-left:4px solid {C_RED};"
                    f"border-radius:6px;padding:8px 14px;margin:4px 0;font-size:.83rem;'>"
                    f"⚠️ <b>{metric.replace('_',' ').title()}</b> is monotonically degrading "
                    f"(slope: {result.get('trend_slope', 0):+.6f})</div>",
                    unsafe_allow_html=True
                )

    st.divider()

    # Alert timeline
    if all_alerts:
        st.markdown("#### ⚠️ Alert Timeline")
        for alert in reversed(all_alerts[-10:]):
            sev   = alert.get("severity", "warning")
            col   = _sev_c(sev)
            icon  = "⛔" if sev == "critical" else "⚠️"
            st.markdown(
                f"<div style='background:{col}10;border-left:4px solid {col};"
                f"border-radius:6px;padding:8px 14px;margin:4px 0;'>"
                f"<div style='display:flex;justify-content:space-between;'>"
                f"<span style='font-weight:700;font-size:.83rem;'>{icon} {alert.get('message','')}</span>"
                f"<span style='font-size:.72rem;color:#6b7280;'>{alert.get('timestamp','')[:19]}</span>"
                f"</div>"
                f"<p style='font-size:.75rem;color:#374151;margin:4px 0 0;'>"
                f"🎯 {alert.get('action','')}</p></div>",
                unsafe_allow_html=True
            )

        if st.button("🗑️ Clear Alerts", key=f"_clear_alerts_{domain}"):
            n = ContinuousMonitor.clear_alerts(domain)
            st.success(f"Cleared {n} alerts.")
            st.rerun()
    else:
        st.success("✅ No active alerts — all metrics within thresholds.")


# ── Audit Certificate ─────────────────────────────────────────────────────────

def _render_audit(audit: dict) -> None:
    st.markdown("### 📜 Compliance Audit Certificate")
    if not audit:
        st.info("No audit report generated yet.")
        return

    cert  = audit.get("certification", "PENDING")
    col   = {
        "CERTIFIED":   C_GREEN,
        "CONDITIONAL": C_AMBER,
        "REJECTED":    C_RED,
        "PENDING":     C_INFO,
    }.get(cert, C_INFO)
    icon  = {"CERTIFIED": "✅", "CONDITIONAL": "⚠️", "REJECTED": "❌", "PENDING": "🔄"}.get(cert, "🔄")

    # Certificate banner
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{C_NAVY},{C_INFO});
    border-radius:12px;padding:20px 24px;margin-bottom:16px;'>
    <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <p style='color:#7dd3fc;font-size:.7rem;font-weight:700;letter-spacing:.1em;
      text-transform:uppercase;margin:0 0 4px;'>GAGS AI BIAS RESILIENCE FRAMEWORK</p>
      <p style='color:#f1f5f9;font-size:1.1rem;font-weight:700;margin:0 0 4px;'>
      Regulatory Audit Certificate</p>
      <p style='color:#94a3b8;font-size:.78rem;margin:0;'>
      Report ID: {audit.get('report_id','—')}</p>
    </div>
    <div style='text-align:center;background:rgba(255,255,255,.15);
    border-radius:10px;padding:12px 20px;'>
      <p style='color:{col};font-size:2rem;margin:0;'>{icon}</p>
      <p style='color:#f1f5f9;font-weight:700;font-size:1rem;margin:0;'>{cert}</p>
      <p style='color:#7dd3fc;font-size:.65rem;margin:0;'>
      Valid until: {audit.get('valid_until','N/A')}</p>
    </div>
    </div></div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Regulatory Compliance Scores**")
        reg = audit.get("regulatory_scores", {})
        for framework, score in reg.items():
            bar = int(score * 100)
            c   = C_GREEN if score >= 0.75 else C_AMBER if score >= 0.60 else C_RED
            st.markdown(
                f"<div style='margin:5px 0;'>"
                f"<div style='display:flex;justify-content:space-between;"
                f"font-size:.78rem;margin-bottom:2px;'>"
                f"<span>{framework}</span><b style='color:{c};'>{score:.1%}</b></div>"
                f"<div style='height:8px;background:#e2e8f0;border-radius:4px;'>"
                f"<div style='height:8px;width:{bar}%;background:{c};border-radius:4px;'>"
                f"</div></div></div>", unsafe_allow_html=True
            )

    with c2:
        st.markdown("**Safety Pillars**")
        safety = audit.get("safety_pillars", {})
        for pillar, score in safety.items():
            c = C_GREEN if score >= 0.75 else C_AMBER if score >= 0.55 else C_RED
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:4px 8px;border-bottom:1px solid #f1f5f9;font-size:.8rem;'>"
                f"<span>{pillar}</span>"
                f"<b style='color:{c};'>{score:.3f}</b></div>",
                unsafe_allow_html=True
            )

    # Conditions
    conditions = audit.get("conditions", [])
    if conditions:
        st.markdown("#### ⚠️ Outstanding Conditions")
        for c in conditions:
            st.markdown(
                f"<div style='background:{C_AMBER}12;border-left:4px solid {C_AMBER};"
                f"border-radius:4px;padding:6px 12px;margin:3px 0;font-size:.82rem;'>"
                f"• {c}</div>", unsafe_allow_html=True
            )

    # Certifying body
    st.caption(
        f"🏛️ Certifying Body: {audit.get('certifying_body','NITDA')} · "
        f"Algo: {audit.get('algo_label','—')} · "
        f"Digital Signature: `{audit.get('signature','—')}`"
    )

    # Download
    narrative = audit.get("full_narrative", "")
    if narrative:
        st.download_button(
            label="📄 Download Full Audit Report (.txt)",
            data=narrative.encode("utf-8"),
            file_name=f"gags_audit_{audit.get('report_id','report')}.txt",
            mime="text/plain",
            use_container_width=True,
            key=f"_audit_dl_{audit.get('report_id','x')}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# ECO TAB
# ══════════════════════════════════════════════════════════════════════════════

def render_eco_tab(lifecycle_report: dict, algo_key: str,
                   n_samples: int, n_runs: int, domain: str = "x") -> None:
    if not lifecycle_report or not lifecycle_report.get("pillars", {}).get("eco"):
        st.markdown("""
        <div style='background:#f8fafc;border:2px dashed #e2e8f0;border-radius:12px;
        padding:32px;text-align:center;'>
        <p style='font-size:2rem;margin:0 0 8px;'>🌱</p>
        <p style='font-weight:700;color:#374151;'>Environmental Analysis Not Run</p>
        <p style='color:#6b7280;font-size:.85rem;'>Enable <strong>🌱 Eco Analysis</strong>
        in the sidebar and run the simulation.</p></div>""", unsafe_allow_html=True)
        return

    eco = lifecycle_report["pillars"]["eco"]
    _render_eco_report(eco, algo_key, n_samples, n_runs, domain=domain)


def _render_eco_report(eco: dict, algo_key: str, n_samples: int, n_runs: int, domain: str = "x") -> None:
    st.markdown("### 🌱 Environmental Sustainability Analysis")

    # Eco score banner
    tier  = eco.get("eco_tier", "B")
    score = eco.get("eco_score", 0.5)
    colour = ECO_TIER_COLOURS.get(tier, C_AMBER)

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#052e16,#14532d);
    border-radius:12px;padding:16px 20px;margin-bottom:16px;'>
    <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <p style='color:#86efac;font-size:.7rem;font-weight:700;letter-spacing:.1em;
      text-transform:uppercase;margin:0 0 4px;'>🌍 Green AI Eco-Score</p>
      <p style='color:#f0fdf4;font-size:1.1rem;font-weight:700;margin:0 0 4px;'>
      {eco.get('algo_label','Algorithm')}</p>
      <p style='color:#6ee7b7;font-size:.8rem;margin:0;'>{eco.get('recommendation','')}</p>
    </div>
    <div style='text-align:center;background:rgba(255,255,255,.1);
    border-radius:10px;padding:10px 18px;'>
      <p style='color:{colour};font-weight:700;font-size:2.2rem;margin:0;'>
      {tier}</p>
      <p style='color:#f0fdf4;font-size:.78rem;margin:0;'>Tier</p>
      <p style='color:{colour};font-weight:700;font-size:1.1rem;margin:4px 0 0;'>
      {score:.0%}</p>
    </div>
    </div></div>""", unsafe_allow_html=True)

    # KPI metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Energy",    f"{eco.get('total_kwh', 0)*1000:.4f} Wh",
              help="Wh consumed for training + inference")
    c2.metric("CO₂ Emitted",    f"{eco.get('total_co2_g', 0):.3f} g",
              help=f"Based on Nigeria grid: {NIGERIA_GRID_CARBON_INTENSITY} gCO₂/kWh")
    c3.metric("Inference Speed", f"{eco.get('infer_time_ms', 0):.1f} ms",
              help="Time to score all test samples")
    c4.metric("Model Size",      f"{eco.get('model_size_kb', 0):,.0f} KB",
              help="Serialised model size in memory")

    st.info(f"🌍 **Equivalent:** {eco.get('equivalent', '')}")

    col_bar, col_detail = st.columns([3, 2])

    with col_bar:
        # CO₂ breakdown chart
        fig = go.Figure(go.Bar(
            x=["Training CO₂", "Inference CO₂"],
            y=[eco.get("train_co2_g", 0), eco.get("infer_co2_g", 0)],
            marker_color=[C_AMBER, C_INFO],
            text=[f"{eco.get('train_co2_g',0):.4f}g", f"{eco.get('infer_co2_g',0):.4f}g"],
            textposition="outside",
        ))
        fig.update_layout(
            title="CO₂ Breakdown (g)", height=250,
            margin=dict(t=40, b=20, l=20, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"_eco_co2_{domain}")

    with col_detail:
        st.markdown("**Efficiency Metrics**")
        rows = [
            ("Training Time",    f"{eco.get('train_time_s', 0):.2f} s"),
            ("Inference Time",   f"{eco.get('infer_time_ms', 0):.1f} ms"),
            ("Model Parameters", f"{eco.get('params_k', 0):.1f}k"),
            ("FLOPs/prediction", f"{eco.get('flops_k', 0):.1f}k"),
            ("Train Energy",     f"{eco.get('train_kwh',0)*1e6:.3f} µWh"),
            ("Infer Energy",     f"{eco.get('infer_kwh',0)*1e9:.3f} nWh"),
        ]
        for label, val in rows:
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:3px 8px;border-bottom:1px solid #f1f5f9;font-size:.8rem;'>"
                f"<span style='color:#6b7280;'>{label}</span><b>{val}</b></div>",
                unsafe_allow_html=True
            )

    st.divider()

    # Algorithm eco-ranking
    st.markdown("#### 🏆 Algorithm Eco-Score Rankings")
    st.caption(f"Comparison for {n_samples:,} samples × {n_runs} run(s) on Nigeria CPU grid")

    all_reports = EnergyConsumptionModeller.compare_all_algorithms(n_samples, n_runs)

    df_eco = pd.DataFrame([{
        "Algorithm":    r.algo_label,
        "Tier":         r.eco_tier,
        "Eco Score":    r.eco_score,
        "CO₂ (g)":      r.total_co2_g,
        "Energy (µWh)": round(r.total_kwh * 1e6, 4),
        "Speed (ms)":   r.infer_time_ms,
        "Size (KB)":    r.model_size_kb,
        "Current":      "⭐" if r.algo_key == algo_key else "",
    } for r in all_reports])

    # Highlight current algorithm
    def highlight_current(row):
        bg = "#dcfce7" if row["Current"] == "⭐" else ""
        return [f"background:{bg}" for _ in row]

    st.dataframe(
        df_eco.style
        .apply(highlight_current, axis=1)
        .format({"Eco Score": "{:.0%}", "CO₂ (g)": "{:.4f}",
                 "Energy (µWh)": "{:.4f}", "Speed (ms)": "{:.1f}"}),
        use_container_width=True, height=380,
    )

    # Radar chart
    top5 = all_reports[:5]
    categories = ["Eco Score", "Speed", "Size Efficiency", "CO₂ Efficiency", "FLOPs Efficiency"]

    def normalise_inv(vals):
        mn, mx = min(vals), max(vals)
        return [1 - (v - mn) / (mx - mn + 1e-9) for v in vals]

    speeds    = normalise_inv([r.infer_time_ms    for r in top5])
    sizes     = normalise_inv([r.model_size_kb    for r in top5])
    co2s      = normalise_inv([r.total_co2_g      for r in top5])
    flops     = normalise_inv([r.flops_k          for r in top5])

    fig_radar = go.Figure()
    for i, r in enumerate(top5):
        vals = [r.eco_score, speeds[i], sizes[i], co2s[i], flops[i]]
        vals += [vals[0]]
        cats  = categories + [categories[0]]
        mark  = "⭐ " if r.algo_key == algo_key else ""
        fig_radar.add_scatterpolar(
            r=vals, theta=cats,
            fill="toself", name=mark + r.algo_label,
            opacity=0.8 if r.algo_key == algo_key else 0.4,
        )
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        title="Top-5 Eco Algorithms (normalised)",
        showlegend=True, height=380,
        margin=dict(t=50, b=20, l=20, r=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig_radar, use_container_width=True, key=f"_eco_radar_{domain}")

    # Nigeria context
    st.markdown("""
    <div style='background:#fff7ed;border-left:4px solid #d97706;border-radius:6px;
    padding:10px 14px;margin-top:8px;font-size:.82rem;color:#374151;'>
    <strong>🇳🇬 Nigeria Grid Context:</strong>
    Nigeria's electricity grid emits <strong>420 gCO₂/kWh</strong> (NERC 2023) — higher than
    the EU average (326 g) but lower than coal-heavy grids. Deploying Naïve Bayes instead of
    Stacking for a 100k-sample production system saves approximately <strong>270 gCO₂</strong>
    per day — equivalent to charging 22 smartphones.
    </div>""", unsafe_allow_html=True)
