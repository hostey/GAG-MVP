"""
components/gags_safety_ui.py — GAGS AI Safety & Robustness UI Renderer
=======================================================================
Renders the AI Safety tab in all 8 simulation modules.
Call: render_safety_tab(safety_report, domain)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


# ── Colour palette ──────────────────────────────────────────────────────────
SAFE_C   = "#16a34a"
WARN_C   = "#d97706"
CRIT_C   = "#dc2626"
INFO_C   = "#0891b2"
NAVY_C   = "#1e3a5f"
LIGHT_BG = "#f8fafc"


def _severity_colour(sev: str) -> str:
    return {"low": SAFE_C, "medium": WARN_C, "high": "#ea580c",
            "critical": CRIT_C}.get(str(sev).lower(), INFO_C)


def _score_colour(score: float) -> str:
    return SAFE_C if score >= 0.75 else WARN_C if score >= 0.55 else CRIT_C


def _gauge(value: float, title: str, colour: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(value * 100, 1),
        title={"text": title, "font": {"size": 13, "color": "#374151"}},
        number={"suffix": "%", "font": {"size": 18, "color": colour}},
        gauge={
            "axis":  {"range": [0, 100], "tickfont": {"size": 10}},
            "bar":   {"color": colour, "thickness": 0.22},
            "bgcolor": "#f1f5f9",
            "steps": [
                {"range": [0, 55],  "color": "#fee2e2"},
                {"range": [55, 75], "color": "#fef9c3"},
                {"range": [75, 100],"color": "#dcfce7"},
            ],
            "threshold": {"line": {"color": colour, "width": 3},
                          "thickness": 0.8, "value": value * 100},
        },
    ))
    fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=160,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SAFETY SUMMARY CARD
# ══════════════════════════════════════════════════════════════════════════════

def render_safety_summary(report: dict, domain: str = "x") -> None:
    summary = report.get("safety_summary", {})
    if not summary:
        st.info("Run the simulation with AI Safety enabled to see results.")
        return

    verdict = summary.get("verdict", "")
    overall = summary.get("overall_score", 0)
    colour  = _score_colour(overall)

    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{NAVY_C},{INFO_C});
    border-radius:12px;padding:16px 20px;margin-bottom:16px;'>
    <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <p style='color:#7dd3fc;font-size:.7rem;font-weight:700;
      letter-spacing:.1em;text-transform:uppercase;margin:0 0 4px;'>
      🛡️ AI Safety & Robustness Verdict</p>
      <p style='color:#f1f5f9;font-size:1.3rem;font-weight:700;margin:0;'>{verdict}</p>
    </div>
    <div style='background:rgba(255,255,255,.15);border-radius:10px;
    padding:8px 16px;text-align:center;'>
      <p style='color:#f1f5f9;font-size:1.8rem;font-weight:700;margin:0;'>
      {overall:.0%}</p>
      <p style='color:#7dd3fc;font-size:.65rem;margin:0;'>Overall Safety Score</p>
    </div>
    </div></div>""", unsafe_allow_html=True)

    # Pillar score gauges
    pillar_scores = summary.get("pillar_scores", {})
    if pillar_scores:
        cols = st.columns(len(pillar_scores))
        for col, (name, score) in zip(cols, pillar_scores.items()):
            with col:
                st.plotly_chart(_gauge(score, name, _score_colour(score)),
                                use_container_width=True, key=f"_sg_{domain}_{name}")

    # Priority actions
    actions = report.get("priority_actions", [])
    if actions:
        st.markdown("#### 🎯 Priority Actions")
        for a in actions:
            colour_map = {"🔴": CRIT_C, "🟠": "#ea580c", "🟡": WARN_C, "✅": SAFE_C}
            bg = next((v for k, v in colour_map.items() if a.startswith(k)), "#e2e8f0")
            st.markdown(f"""<div style='background:{bg}18;border-left:4px solid {bg};
            border-radius:4px;padding:7px 12px;margin:4px 0;font-size:.82rem;'>
            {a}</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 — ADVERSARIAL ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════

def render_adversarial_robustness(rob: dict, domain: str = "x") -> None:
    st.markdown("### 🎯 Adversarial Robustness")

    if not rob:
        st.warning("Adversarial robustness analysis not available.")
        return

    # Narrative
    narrative = rob.get("domain_narrative", "")
    if narrative:
        st.markdown(f"""<div style='background:#fff7ed;border-left:4px solid {WARN_C};
        border-radius:6px;padding:10px 14px;margin-bottom:12px;font-size:.83rem;
        color:#374151;line-height:1.5;'>
        <strong>⚠️ Attack Scenario:</strong> {narrative}</div>""",
        unsafe_allow_html=True)

    # Key metrics row
    rs     = rob.get("robustness_score", 0)
    asr    = rob.get("overall_asr", 0)
    worst  = rob.get("worst_severity", "unknown")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Robustness Score",    f"{rs:.1%}", delta=f"{rs-0.75:.1%} vs threshold",
              delta_color="normal")
    c2.metric("Attack Success Rate", f"{asr:.1%}", delta=f"{-asr:.1%}",
              delta_color="inverse")
    c3.metric("Worst Severity",      worst.capitalize())

    cert = rob.get("certified", {})
    c4.metric("Certified Accuracy",  f"{cert.get('certified_accuracy',0):.1%}",
              help="Fraction of samples with certified robustness (Cohen 2019)")

    st.markdown("---")

    # Attack comparison chart
    fgsm_l = rob.get("fgsm_low", {})
    fgsm_h = rob.get("fgsm_high", {})
    pgd    = rob.get("pgd", {})

    if fgsm_l and fgsm_h and pgd:
        attacks = ["FGSM ε=0.05", "FGSM ε=0.15", "PGD ε=0.05"]
        clean   = [fgsm_l.get("clean_accuracy",0),
                   fgsm_h.get("clean_accuracy",0),
                   pgd.get("clean_accuracy",0)]
        robust  = [fgsm_l.get("robust_accuracy",0),
                   fgsm_h.get("robust_accuracy",0),
                   pgd.get("robust_accuracy",0)]
        asr_v   = [fgsm_l.get("attack_success_rate",0),
                   fgsm_h.get("attack_success_rate",0),
                   pgd.get("attack_success_rate",0)]

        col_chart, col_table = st.columns([3, 2])
        with col_chart:
            fig = go.Figure()
            fig.add_bar(name="Clean Accuracy",  x=attacks, y=[v*100 for v in clean],
                        marker_color=SAFE_C)
            fig.add_bar(name="Robust Accuracy", x=attacks, y=[v*100 for v in robust],
                        marker_color=CRIT_C)
            fig.update_layout(barmode="group", height=280,
                               title="Clean vs Robust Accuracy by Attack",
                               yaxis_title="Accuracy (%)",
                               legend=dict(orientation="h", y=1.12),
                               margin=dict(t=50, b=20, l=20, r=10),
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key=f"_rob_{domain}")

        with col_table:
            st.markdown("**Attack Success Rate**")
            for atk, rate in zip(attacks, asr_v):
                bar_width = int(rate * 100)
                colour    = CRIT_C if rate > 0.3 else WARN_C if rate > 0.15 else SAFE_C
                st.markdown(f"""
                <div style='margin:6px 0;'>
                <div style='font-size:.75rem;color:#374151;'>{atk}</div>
                <div style='display:flex;align-items:center;gap:8px;'>
                  <div style='height:14px;width:{bar_width}%;background:{colour};
                  border-radius:3px;min-width:4px;'></div>
                  <span style='font-size:.78rem;font-weight:700;color:{colour};'>
                  {rate:.1%}</span>
                </div></div>""", unsafe_allow_html=True)

    # Certified robustness details
    with st.expander("🔐 Certified Robustness Details (Randomised Smoothing)", expanded=False):
        if cert:
            ce1, ce2, ce3 = st.columns(3)
            ce1.metric("Certified Accuracy",      f"{cert.get('certified_accuracy',0):.1%}")
            ce2.metric("Avg Certified Radius",    f"{cert.get('avg_certified_radius',0):.3f}")
            ce3.metric("Fraction Certifiable",    f"{cert.get('fraction_certifiable',0):.1%}")
            st.caption(f"Smoothing σ = {cert.get('sigma', 0.25)}. Based on Cohen et al. (2019) randomised smoothing. "
                      f"A certified radius r means the prediction is guaranteed unchanged for all perturbations ‖δ‖₂ < r.")

    # Fairness impact of attacks
    fi = fgsm_h.get("fairness_impact", {}) if fgsm_h else {}
    if fi:
        st.markdown("**🌍 Fairness Impact of Adversarial Attack**")
        st.caption("How much does FGSM ε=0.15 increase the error rate for each group?")
        for group, delta in fi.items():
            colour = CRIT_C if delta > 0.05 else WARN_C if delta > 0.02 else SAFE_C
            st.markdown(f"""<div style='display:flex;justify-content:space-between;
            padding:4px 8px;background:{LIGHT_BG};border-radius:4px;margin:3px 0;'>
            <span style='font-size:.8rem;'>{group.title()}</span>
            <span style='font-weight:700;color:{colour};font-size:.8rem;'>
            {'+' if delta > 0 else ''}{delta:.1%} error increase</span>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 2 — OOD DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def render_ood_detection(ood: dict, domain: str = "x") -> None:
    st.markdown("### 🔍 Out-of-Distribution (OOD) Detection")

    if not ood:
        st.warning("OOD detection analysis not available.")
        return

    drift_narrative = ood.get("drift_narrative", "")
    if drift_narrative:
        st.markdown(f"""<div style='background:#eff6ff;border-left:4px solid {INFO_C};
        border-radius:6px;padding:10px 14px;margin-bottom:12px;font-size:.83rem;
        color:#1e3a5f;line-height:1.5;'>
        <strong>🌍 Deployment Shift Scenario:</strong> {drift_narrative}
        </div>""", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    ood_rate  = ood.get("composite_ood_rate", 0)
    drop      = ood.get("avg_accuracy_drop", 0)
    dep_risk  = ood.get("deployment_risk", "unknown")
    ks        = ood.get("ks_drift", {})
    drift_frac= ks.get("drift_fraction", 0)

    c1.metric("OOD Detection Rate", f"{ood_rate:.1%}",
              delta=f"{ood_rate - 0.80:.1%} vs 80% target",
              delta_color="normal")
    c2.metric("Accuracy Drop (OOD)", f"{drop:.1%}",
              delta=f"{-drop:.1%}", delta_color="inverse")
    c3.metric("Deployment Risk",    dep_risk.capitalize())
    c4.metric("Features Drifted",  f"{drift_frac:.0%}",
              help="KS test: fraction of features with significant distribution shift")

    st.markdown("---")

    col_mah, col_eng = st.columns(2)
    mah = ood.get("mahalanobis", {})
    eng = ood.get("energy", {})

    with col_mah:
        st.markdown("**📐 Mahalanobis Distance Detector**")
        if mah:
            rows = [
                ("OOD Detection Rate", f"{mah.get('ood_detection_rate',0):.1%}"),
                ("In-dist Accuracy",   f"{mah.get('in_dist_accuracy',0):.1%}"),
                ("OOD Accuracy",       f"{mah.get('ood_accuracy',0):.1%}"),
                ("Accuracy Drop",      f"{mah.get('accuracy_drop',0):.1%}"),
                ("Threshold (95p)",    f"{mah.get('threshold',0):.2f}"),
                ("Drift Severity",     mah.get('drift_severity','').capitalize()),
            ]
            for label, val in rows:
                st.markdown(f"""<div style='display:flex;justify-content:space-between;
                padding:3px 8px;border-bottom:1px solid #f1f5f9;font-size:.8rem;'>
                <span style='color:#6b7280;'>{label}</span>
                <b>{val}</b></div>""", unsafe_allow_html=True)

    with col_eng:
        st.markdown("**⚡ Energy-Based Detector**")
        if eng:
            rows = [
                ("OOD Detection Rate", f"{eng.get('ood_detection_rate',0):.1%}"),
                ("In-dist Accuracy",   f"{eng.get('in_dist_accuracy',0):.1%}"),
                ("OOD Accuracy",       f"{eng.get('ood_accuracy',0):.1%}"),
                ("Accuracy Drop",      f"{eng.get('accuracy_drop',0):.1%}"),
                ("Threshold (5p)",     f"{eng.get('threshold',0):.2f}"),
                ("Drift Severity",     eng.get('drift_severity','').capitalize()),
            ]
            for label, val in rows:
                st.markdown(f"""<div style='display:flex;justify-content:space-between;
                padding:3px 8px;border-bottom:1px solid #f1f5f9;font-size:.8rem;'>
                <span style='color:#6b7280;'>{label}</span>
                <b>{val}</b></div>""", unsafe_allow_html=True)

    # KS test results
    if ks:
        with st.expander("📊 Feature-Level Drift (Kolmogorov-Smirnov)", expanded=False):
            dl = ks.get("drift_level", "unknown")
            st.markdown(f"**Drift Level:** `{dl.upper()}` — "
                        f"{ks.get('features_drifted',0)} / {ks.get('features_tested',0)} "
                        f"features show significant shift (p < 0.05)")
            top = ks.get("top_drifted", [])
            if top:
                df_data = {
                    "Feature":  [f"Feature {r['feature_idx']}" for r in top],
                    "KS Stat":  [r["ks_stat"] for r in top],
                    "p-value":  [r["p_value"] for r in top],
                    "Drifted":  ["Yes" for r in top],
                }
                import pandas as pd
                st.dataframe(pd.DataFrame(df_data), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 3 — UNCERTAINTY QUANTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def render_uncertainty(uq: dict, domain: str = "x") -> None:
    st.markdown("### 📊 Uncertainty Quantification")

    if not uq:
        st.warning("Uncertainty quantification not available.")
        return

    mc = uq.get("mc_dropout", {})
    if not mc:
        st.warning("MC Dropout results not available.")
        return

    # Verdict banner
    verdict = uq.get("calibration_verdict", "")
    col  = SAFE_C if "Well" in verdict else WARN_C if "Moderate" in verdict else CRIT_C
    st.markdown(f"""<div style='background:{col}18;border-left:4px solid {col};
    border-radius:6px;padding:8px 14px;margin-bottom:12px;font-size:.85rem;font-weight:600;'>
    {verdict}</div>""", unsafe_allow_html=True)

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    ece = uq.get("ece", 0)
    c1.metric("ECE",                f"{ece:.4f}",
              delta=f"{ece-0.10:.4f} vs 0.10 threshold",
              delta_color="inverse",
              help="Expected Calibration Error. 0 = perfect. < 0.10 = well calibrated.")
    c2.metric("Brier Score",        f"{uq.get('brier_score',0):.4f}",
              help="Mean squared error of probability predictions. Lower is better.")
    c3.metric("Avg Interval Width", f"{uq.get('avg_interval_width',0):.3f}",
              help="Average width of 90% prediction interval. Narrower = more confident.")
    c4.metric("High Uncertainty",   f"{mc.get('high_uncertainty_frac',0):.1%}",
              help="Fraction of samples with prediction entropy > 0.5 (the model is unsure).")

    st.markdown("---")

    # Calibration plot (reliability diagram)
    bins = mc.get("calibration_bins", [])
    if bins:
        conf  = [b["confidence"] for b in bins]
        acc   = [b["accuracy"]   for b in bins]

        fig = go.Figure()
        fig.add_scatter(x=[0,1], y=[0,1], mode="lines",
                        line=dict(dash="dash", color="#94a3b8", width=1),
                        name="Perfect calibration")
        fig.add_scatter(x=conf, y=acc, mode="lines+markers",
                        line=dict(color=INFO_C, width=2),
                        marker=dict(size=8, color=INFO_C),
                        name="Model calibration")
        fig.add_bar(x=conf, y=[b["n_samples"]/sum(b["n_samples"] for b in bins) for b in bins],
                    yaxis="y2", marker_color="#e2e8f0", name="Sample fraction",
                    opacity=0.5)
        fig.update_layout(
            title="Reliability Diagram (Calibration Plot)",
            xaxis_title="Mean Predicted Confidence",
            yaxis=dict(title="Fraction Positive", range=[0,1]),
            yaxis2=dict(title="Sample Fraction", overlaying="y", side="right",
                       range=[0, 0.5], showgrid=False),
            height=320, legend=dict(orientation="h", y=1.15),
            margin=dict(t=50, b=40, l=40, r=40),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"_uq_{domain}")

    # Per-bin calibration error
    if bins:
        with st.expander("📋 Per-Bin Calibration Details", expanded=False):
            import pandas as pd
            df = pd.DataFrame([{
                "Conf. Range": f"{b['bin_lower']:.1f}–{b['bin_upper']:.1f}",
                "Confidence":  f"{b['confidence']:.3f}",
                "Accuracy":    f"{b['accuracy']:.3f}",
                "Cal. Error":  f"{b['cal_error']:.4f}",
                "N Samples":   b["n_samples"],
            } for b in bins])
            st.dataframe(df.style.background_gradient(subset=["Cal. Error"], cmap="Reds"),
                        use_container_width=True)

    # Uncertainty narrative
    narrative = uq.get("uncertainty_narrative", "")
    if narrative:
        st.info(narrative)


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 4 — SAFETY CHECKLISTS
# ══════════════════════════════════════════════════════════════════════════════

def render_safety_checklists(cl: dict, domain: str = "x") -> None:
    st.markdown("### ✅ AI Safety Checklists")

    if not cl:
        st.warning("Safety checklist results not available.")
        return

    # Combined verdict
    comb_score   = cl.get("combined_score", 0)
    comb_verdict = cl.get("combined_verdict", "")
    col = _score_colour(comb_score)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div style='background:{col}18;border:1px solid {col};
        border-radius:10px;padding:14px;text-align:center;'>
        <p style='font-size:.68rem;font-weight:700;color:{col};text-transform:uppercase;
        letter-spacing:.07em;margin:0 0 6px;'>Combined Safety Score</p>
        <p style='font-size:2rem;font-weight:700;color:{col};margin:0;'>
        {comb_score:.0%}</p>
        <p style='font-size:.78rem;color:#374151;margin:4px 0 0;'>{comb_verdict}</p>
        </div>""", unsafe_allow_html=True)

    with c2:
        gaps = cl.get("critical_gaps", [])
        if gaps:
            st.markdown(f"""<div style='background:{CRIT_C}10;border:1px solid {CRIT_C};
            border-radius:10px;padding:14px;'>
            <p style='font-weight:700;color:{CRIT_C};margin:0 0 6px;font-size:.85rem;'>
            ⚠️ Critical Gaps ({len(gaps)})</p>""" +
            "".join(f"<span style='background:{CRIT_C}20;color:{CRIT_C};padding:2px 8px;"
                    f"border-radius:10px;font-size:.72rem;margin:2px;display:inline-block;'>"
                    f"{g}</span>" for g in gaps[:8]) +
            "</div>", unsafe_allow_html=True)
        else:
            st.success("No critical gaps identified across both frameworks.")

    st.markdown("---")

    tab_nist, tab_iso = st.tabs(["🇺🇸 NIST AI RMF 1.0", "🌍 ISO 42001:2023"])

    # NIST AI RMF
    with tab_nist:
        nist = cl.get("nist_ai_rmf", {})
        if nist:
            st.markdown(f"**Overall: {nist.get('overall_verdict','')}** "
                        f"({nist.get('overall_score',0):.1%})")
            functions = nist.get("functions", {})
            for fn_name, fn_data in functions.items():
                sc  = fn_data.get("score", 0)
                col = _score_colour(sc)
                with st.expander(
                    f"{fn_data.get('verdict','')}  {fn_name}  ({sc:.1%})",
                    expanded=sc < 0.60
                ):
                    for item in fn_data.get("items", []):
                        st.markdown(
                            f"<div style='padding:4px 8px;border-bottom:1px solid #f1f5f9;"
                            f"display:flex;justify-content:space-between;font-size:.79rem;'>"
                            f"<span><code style='font-size:.72rem;color:{INFO_C};'>{item['id']}</code>"
                            f"  {item['item']}</span>"
                            f"<span style='white-space:nowrap;margin-left:12px;'>{item['status']}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

    # ISO 42001
    with tab_iso:
        iso = cl.get("iso_42001", {})
        if iso:
            st.markdown(f"**Overall: {iso.get('overall_verdict','')}** "
                        f"({iso.get('overall_score',0):.1%})")
            clauses = iso.get("clauses", {})
            for clause_name, cl_data in clauses.items():
                sc  = cl_data.get("score", 0)
                with st.expander(
                    f"{cl_data.get('verdict','')}  Clause: {clause_name}  ({sc:.1%})",
                    expanded=sc < 0.60
                ):
                    for item in cl_data.get("items", []):
                        st.markdown(
                            f"<div style='padding:4px 8px;border-bottom:1px solid #f1f5f9;"
                            f"display:flex;justify-content:space-between;font-size:.79rem;'>"
                            f"<span><code style='font-size:.72rem;color:{INFO_C};'>{item['id']}</code>"
                            f"  {item['item']}</span>"
                            f"<span style='white-space:nowrap;margin-left:12px;'>{item['status']}</span>"
                            f"</div>",
                            unsafe_allow_html=True
                        )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_safety_tab(safety_report: dict, domain: str = "health") -> None:
    """
    Main renderer — call this inside the AI Safety tab:

        with tab_safety:
            render_safety_tab(st.session_state.get("safety_report", {}), "health")
    """
    if not safety_report or not safety_report.get("pillars"):
        st.markdown("""
        <div style='background:#f8fafc;border:2px dashed #e2e8f0;border-radius:12px;
        padding:32px;text-align:center;'>
        <p style='font-size:2rem;margin:0 0 8px;'>🛡️</p>
        <p style='font-weight:700;color:#374151;font-size:1rem;margin:0 0 6px;'>
        AI Safety Analysis Not Run</p>
        <p style='color:#6b7280;font-size:.85rem;margin:0;'>
        Enable <strong>AI Safety & Robustness</strong> in the sidebar and run the simulation.
        </p></div>""", unsafe_allow_html=True)
        return

    # Summary card first
    render_safety_summary(safety_report, domain)

    st.divider()

    pillars = safety_report.get("pillars", {})

    # Sub-tabs for four pillars
    sub_tabs = st.tabs([
        "🎯 Adversarial Robustness",
        "🔍 OOD Detection",
        "📊 Uncertainty",
        "✅ Safety Checklists",
    ])

    with sub_tabs[0]:
        render_adversarial_robustness(pillars.get("adversarial_robustness", {}), domain)

    with sub_tabs[1]:
        render_ood_detection(pillars.get("ood_detection", {}), domain)

    with sub_tabs[2]:
        render_uncertainty(pillars.get("uncertainty", {}), domain)

    with sub_tabs[3]:
        render_safety_checklists(pillars.get("safety_checklists", {}), domain)
