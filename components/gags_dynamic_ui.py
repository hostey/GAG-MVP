"""
components/gags_dynamic_ui.py — Dynamic Systems Theory UI Renderer v1.0
=========================================================================
Renders the 🔮 Dynamic Systems tab across all 8 GAGS modules.
Six theory pillars, each with full interactive Plotly visualisation.
"""
from __future__ import annotations
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any
try:
    from components.gags_dynamic_systems import run_dynamic_systems_suite as _run_ds
except ImportError:
    _run_ds = None

C_NAVY   = "#1e3a5f"
C_GREEN  = "#16a34a"
C_AMBER  = "#d97706"
C_RED    = "#dc2626"
C_PURPLE = "#7c3aed"
C_CYAN   = "#0891b2"
C_ORANGE = "#ea580c"
C_LIGHT  = "#f8fafc"

LOOP_COLOURS = {"reinforcing": C_RED, "balancing": C_GREEN}


def _score_c(v: float) -> str:
    return C_GREEN if v >= 0.75 else C_AMBER if v >= 0.55 else C_RED


def _theory_header(icon: str, title: str, citation: str, colour: str) -> None:
    st.markdown(
        f"<div style='border-left:4px solid {colour};padding:6px 14px;"
        f"margin-bottom:12px;background:{colour}08;border-radius:0 8px 8px 0;'>"
        f"<span style='font-size:1.1rem;'>{icon}</span> "
        f"<strong style='color:{colour};font-size:.9rem;'>{title}</strong>"
        f"<span style='color:#6b7280;font-size:.72rem;margin-left:10px;'>{citation}</span>"
        f"</div>", unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 — SYSTEM DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

def render_system_dynamics(sd: dict, domain: str) -> None:
    _theory_header("♾️", "System Dynamics — Stock & Flow Model",
                   "Forrester (1961) · Sterman (2000) · Meadows (2008)", C_CYAN)

    if not sd:
        st.info("System dynamics not computed.")
        return

    cld = sd.get("causal_loops", {})
    stocks_def = cld.get("stocks", {})
    flows_def  = cld.get("flows", {})

    # Stability card
    stab   = sd.get("system_stability", "stable")
    tip_m  = sd.get("tipping_message", "")
    s_col  = C_RED if stab == "unstable" else C_GREEN
    st.markdown(
        f"<div style='background:{s_col}12;border:1px solid {s_col};border-radius:8px;"
        f"padding:10px 16px;margin-bottom:12px;'>"
        f"<b style='color:{s_col};'>System Status: {stab.upper()}</b><br>"
        f"<span style='font-size:.82rem;color:#374151;'>{tip_m}</span>"
        f"</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Bias Stock",    f"{sd.get('final_bias', 0):.3f}",
              delta=f"{sd.get('final_bias',0)-sd.get('initial_bias',0):+.3f}",
              delta_color="inverse")
    c2.metric("Tipping Point",       f"{sd.get('tipping_point', 0.55):.2f}")
    c3.metric("Reinforcing Loops",   str(sd.get("n_reinforcing", 0)),
              help="R-loops amplify bias over time")
    c4.metric("Balancing Loops",     str(sd.get("n_balancing", 0)),
              help="B-loops dampen bias — governance levers")

    # Stock time series chart
    history = sd.get("history", [])
    if history:
        df_h = pd.DataFrame(history)
        stock_cols = [c for c in df_h.columns if c != "step"]
        fig = go.Figure()
        colours_s = [C_RED, C_AMBER, C_PURPLE, C_CYAN]
        for i, col in enumerate(stock_cols):
            fig.add_scatter(x=df_h["step"], y=df_h[col],
                            name=stocks_def.get(col, {}).get("label", col),
                            mode="lines", line=dict(width=2.5, color=colours_s[i % len(colours_s)]))

        # Mark tipping point
        tp   = sd.get("tipping_point", 0.55)
        tc   = sd.get("tipping_crossed")
        fig.add_hline(y=tp, line_dash="dot", line_color=C_RED,
                      annotation_text=f"Tipping point ({tp:.2f})")
        if tc is not None:
            fig.add_vline(x=tc, line_dash="dash", line_color=C_RED,
                          annotation_text=f"System tips at step {tc}")

        fig.update_layout(
            title=cld.get("title", "Causal Loop Dynamics"),
            xaxis_title="Time Steps (Quarters)", yaxis_title="Stock Level [0–1]",
            height=320, margin=dict(t=40, b=30),
            legend=dict(orientation="h", y=1.15),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(fig, use_container_width=True,
                        key=f"_sd_stocks_{domain}")

    # Causal loops detail
    st.markdown("#### 🔄 Causal Loop Detail")
    col_r, col_b = st.columns(2)
    r_loops = {k: v for k, v in flows_def.items() if v.get("type") == "reinforcing"}
    b_loops = {k: v for k, v in flows_def.items() if v.get("type") == "balancing"}

    with col_r:
        st.markdown(f"**🔴 Reinforcing Loops (amplify bias)**")
        for k, v in r_loops.items():
            with st.expander(v.get("label", k), expanded=False):
                st.markdown(f"*Strength:* `{v.get('strength', 0):.2f}`")
                st.markdown(v.get("description", ""))

    with col_b:
        st.markdown(f"**🟢 Balancing Loops (governance levers)**")
        for k, v in b_loops.items():
            with st.expander(v.get("label", k), expanded=False):
                st.markdown(f"*Strength:* `{v.get('strength', 0):.2f}`")
                st.markdown(v.get("description", ""))

    leverage = sd.get("policy_leverage", "")
    if leverage:
        st.info(f"🎯 **Highest Leverage Policy Point:** {leverage}")


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 2 — MDP
# ══════════════════════════════════════════════════════════════════════════════

def render_mdp(mdp: dict, domain: str) -> None:
    _theory_header("🎯", "Markov Decision Process — Optimal Deployment Policy",
                   "Bellman (1957) · Sutton & Barto (2018)", C_PURPLE)

    if not mdp:
        st.info("MDP not computed.")
        return

    st.markdown(f"*{mdp.get('interpretation', '')}*")
    st.markdown(f"**Converged after {mdp.get('converged_iter', 0)} value iterations**")

    # Policy grid heatmap
    p_grid = mdp.get("policy_grid", [])
    v_grid = mdp.get("value_grid", [])
    labels = mdp.get("action_labels", ["Deploy","Audit","Retrain","Suspend"])

    if p_grid:
        action_to_num = {a: i for i, a in enumerate(labels)}
        num_grid = [[action_to_num.get(cell, 0) for cell in row] for row in p_grid]
        fig = go.Figure(go.Heatmap(
            z=num_grid,
            colorscale=[[0,"#16a34a"],[0.33,"#0891b2"],[0.66,"#d97706"],[1,"#dc2626"]],
            text=p_grid, texttemplate="%{text}",
            xaxis="x", yaxis="y",
            colorbar=dict(tickvals=[0,1,2,3], ticktext=labels, title="Action"),
            showscale=True,
        ))
        fig.update_layout(
            title="Optimal Policy: State → Action",
            xaxis=dict(title="Bias Level →", ticktext=["Low","","Med","","High"],
                       tickvals=list(range(5))),
            yaxis=dict(title="↑ Fairness Level", ticktext=["Low","","Med","","High"],
                       tickvals=list(range(5)), autorange="reversed"),
            height=340, margin=dict(t=40, b=40, l=60, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"_mdp_policy_{domain}")
        st.caption("Each cell = optimal action for that (fairness, bias) state combination. "
                   "Green = Deploy · Blue = Audit · Amber = Retrain · Red = Suspend")

    # Convergence
    conv = mdp.get("convergence", [])
    if conv:
        with st.expander("📉 Value Iteration Convergence", expanded=False):
            fig_c = go.Figure(go.Scatter(y=conv, mode="lines",
                line=dict(color=C_PURPLE, width=2)))
            fig_c.update_layout(xaxis_title="Iteration", yaxis_title="Max ΔV",
                height=220, margin=dict(t=10, b=30),
                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_c, use_container_width=True, key=f"_mdp_conv_{domain}")


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 3 — INFORMATION THEORY
# ══════════════════════════════════════════════════════════════════════════════

def render_information_theory(info: dict, domain: str) -> None:
    """
    Renders Information-Theoretic Fairness metrics.
    Core quantities: Mutual Information I(Ŷ;A), Normalised MI, KL-divergence,
    and a composite Info Fairness Score.
    """
    _theory_header(
        "📡",
        "Information-Theoretic Fairness",
        "Shannon (1948) · Dwork et al. (2012) · Ghassami et al. (2018)",
        C_ORANGE,
    )

    if not info or not isinstance(info, dict):
        st.info("Information-theoretic metrics not computed.")
        return

    # ── Safe extraction ──────────────────────────────────────────────────────
    mi   = float(info.get("mutual_information", 0) or 0)
    nmi  = float(info.get("normalised_mi", 0) or 0)
    kl   = float(info.get("kl_divergence", 0) or 0)
    ifs  = float(info.get("info_fairness_score", 0) or 0)
    h_y  = float(info.get("prediction_entropy", 0) or 0)
    h_ya = float(info.get("conditional_entropy", 0) or 0)

    interpretation = info.get("interpretation", "")
    if interpretation:
        st.markdown(f"*{interpretation}*")

    # ── Headline metrics ─────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Mutual Information I(Ŷ;A)",
        f"{mi:.4f} bits",
        help="Bits of information about the sensitive attribute contained in the prediction. 0 = perfect statistical independence."
    )
    c2.metric(
        "Normalised MI",
        f"{nmi:.4f}",
        help="Mutual information scaled to [0, 1]. Easier to compare across tasks."
    )
    c3.metric(
        "KL-Divergence",
        f"{kl:.4f}",
        help="Statistical distance between outcome distributions of different groups."
    )
    c4.metric(
        "Info Fairness Score",
        f"{ifs:.3f}",
        help="Composite score (higher is fairer). Typically 1 − normalised MI."
    )

    # ── Plain-language guidance ──────────────────────────────────────────────
    if mi < 0.01:
        st.success(
            "✅ Very low leakage — predictions carry almost no information about the sensitive attribute."
        )
    elif mi < 0.05:
        st.info(
            "ℹ️ Low-to-moderate leakage. Generally acceptable for screening tasks; "
            "review carefully for high-stakes clinical decisions."
        )
    else:
        st.warning(
            "⚠️ Notable information leakage detected. "
            "Consider adversarial debiasing, reweighting, or fairness constraints."
        )

    # ── Visualisations ───────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        # Entropy decomposition
        fig = go.Figure(
            go.Bar(
                x=["H(Ŷ) — Total Entropy", "H(Ŷ|A) — Cond. Entropy", "I(Ŷ;A) — Mutual Info"],
                y=[h_y, h_ya, mi],
                marker_color=[C_ORANGE, C_CYAN, C_RED],
                text=[f"{v:.4f}" for v in [h_y, h_ya, mi]],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Entropy Decomposition (bits)",
            height=280,
            margin=dict(t=40, b=30, l=20, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Bits",
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True, key=f"_info_entropy_{domain}")

    with col_right:
        # Data Shapley / group contribution
        shapley = info.get("data_shapley", {})
        if shapley and isinstance(shapley, dict):
            st.markdown("**Data Shapley — Group Value Attribution**")
            st.caption(
                "Estimated contribution of each group's data to overall model performance."
            )

            # Sort by absolute contribution for clearer display
            sorted_items = sorted(
                shapley.items(),
                key=lambda x: abs(float(x[1])),
                reverse=True,
            )

            for g, val in sorted_items:
                val = float(val)
                colour = C_GREEN if val >= 0 else C_RED

                # Flexible group labelling
                try:
                    g_int = int(g)
                    label = f"Group {g} ({'Advantaged' if g_int == 1 else 'Disadvantaged'})"
                except (ValueError, TypeError):
                    label = f"Group {g}"

                st.markdown(
                    f"<div style='display:flex;justify-content:space-between;"
                    f"padding:6px 10px;border-bottom:1px solid #f1f5f9;font-size:.83rem;'>"
                    f"<span>{label}</span>"
                    f"<b style='color:{colour};'>{val:+.4f}</b></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("Data Shapley values not available for this run.")

    # ── Optional expander for non-technical readers ──────────────────────────
    with st.expander("What does this mean for patients?", expanded=False):
        st.markdown(
            """
            - **Mutual Information ≈ 0** → The model’s prediction does not reveal 
              whether a patient belongs to a sensitive group (e.g., rural, low-income, female).
            - **Higher Mutual Information** → The model is (intentionally or unintentionally) 
              using group membership as a signal. This can produce unequal under-diagnosis rates.
            - In clinical settings we usually want **very low** mutual information 
              between the prediction and protected attributes, especially for high-stakes decisions.
            """
        )


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 4 — CAUSAL FAIRNESS
# ══════════════════════════════════════════════════════════════════════════════

def render_causal_fairness(causal: dict, domain: str) -> None:
    _theory_header("🔗", "Causal Fairness — Pearl's do-Calculus & Counterfactual Fairness",
                   "Pearl (2009) · Kusner et al. (2017) · Kilbertus et al. (2017)", C_NAVY)

    if not causal:
        st.info("Causal analysis not computed.")
        return

    st.markdown(f"*{causal.get('interpretation', '')}*")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Causal Effect", f"{causal.get('total_causal_effect', 0):+.4f}")
    c2.metric("Direct Effect",       f"{causal.get('direct_effect', 0):+.4f}",
              help="Bias not mediated through proxy variables")
    c3.metric("Indirect Effect",     f"{causal.get('indirect_effect', 0):+.4f}",
              help="Bias via proxy variables (geography, insurance etc.)")
    c4.metric("Counterfactual Fairness", f"{causal.get('counterfactual_fairness', 0):.3f}",
              help="1.0 = decision invariant to sensitive attribute in counterfactual world")

    # Effect decomposition
    d = causal.get("direct_effect", 0)
    i = causal.get("indirect_effect", 0)
    s = causal.get("spurious_association", 0)

    fig = go.Figure(go.Bar(
        x=["Direct\n(unexplained)", "Indirect\n(via proxies)", "Spurious\n(confounding)"],
        y=[abs(d), abs(i), abs(s)],
        marker_color=[C_RED, C_AMBER, "#94a3b8"],
        text=[f"{v:+.4f}" for v in [d, i, s]],
        textposition="outside"))
    fig.update_layout(title="Bias Effect Decomposition (Pearl's Mediation Analysis)",
                      height=280, yaxis_title="|Effect Size|",
                      margin=dict(t=40, b=30),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=f"_causal_decomp_{domain}")

    # DAG display
    dag  = causal.get("causal_dag", {})
    path = causal.get("policy", "")
    cfq  = causal.get("counterfactual_q", "")
    if path:
        st.markdown(f"🔴 **Primary bias path:** `{path}`")
    if cfq:
        st.markdown(f"""
        <div style='background:#eff6ff;border-left:4px solid {C_NAVY};
        border-radius:6px;padding:10px 14px;font-size:.83rem;'>
        <strong>Counterfactual Question:</strong> {cfq}
        </div>""", unsafe_allow_html=True)

    if dag.get("edges"):
        with st.expander("🔗 Full Causal DAG", expanded=False):
            edges_df = pd.DataFrame(
                [{"From": e[0], "To": e[1], "Mechanism": e[2]}
                 for e in dag["edges"]])
            st.dataframe(edges_df, use_container_width=True, hide_index=True)
            conf = dag.get("confounders", [])
            med  = dag.get("mediators", [])
            if conf:
                st.markdown(f"**Confounders (backdoor paths):** {', '.join(conf)}")
            if med:
                st.markdown(f"**Mediators (indirect paths):** {', '.join(med)}")


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 5 — EVOLUTIONARY GAME THEORY
# ══════════════════════════════════════════════════════════════════════════════

def render_evolutionary(evo: dict, domain: str) -> None:
    _theory_header("🧬", "Evolutionary Game Theory — Replicator Dynamics",
                   "Maynard Smith (1982) · Taylor & Jonker (1978) · Axelrod (1984)", C_GREEN)

    if not evo:
        st.info("Evolutionary analysis not computed.")
        return

    strategies  = evo.get("strategies", [])
    final_share = evo.get("final_shares", {})
    ess         = evo.get("ess", "")
    ess_msg     = evo.get("ess_message", "")
    history     = evo.get("history", [])

    ess_col = C_GREEN if evo.get("ess_stable") else C_AMBER
    st.markdown(
        f"<div style='background:{ess_col}12;border-left:4px solid {ess_col};"
        f"border-radius:6px;padding:8px 14px;margin-bottom:12px;font-size:.83rem;'>"
        f"<strong>ESS:</strong> {ess_msg}</div>",
        unsafe_allow_html=True)

    col_pie, col_evo = st.columns([1, 2])

    with col_pie:
        if final_share:
            fig = go.Figure(go.Pie(
                labels=list(final_share.keys()),
                values=list(final_share.values()),
                hole=0.45,
                marker_colors=[C_GREEN, C_PURPLE, C_AMBER, C_CYAN, C_ORANGE],
            ))
            fig.update_layout(title="Final Strategy Shares",
                              height=260, margin=dict(t=40, b=10, l=10, r=10),
                              paper_bgcolor="rgba(0,0,0,0)",
                              showlegend=True,
                              legend=dict(font=dict(size=10)))
            st.plotly_chart(fig, use_container_width=True, key=f"_evo_pie_{domain}")

    with col_evo:
        if history and strategies:
            fig = go.Figure()
            colours_e = [C_GREEN, C_PURPLE, C_AMBER, C_CYAN, C_ORANGE]
            for i, s in enumerate(strategies):
                vals = [h[i] for h in history]
                fig.add_scatter(x=list(range(len(vals))), y=vals,
                                name=s, mode="lines",
                                line=dict(color=colours_e[i % len(colours_e)], width=2),
                                fill="tozeroy" if i == 0 else "tonexty",
                                stackgroup="one", fillcolor=f"rgba({int(colours_e[i % len(colours_e)][1:3],16)},{int(colours_e[i % len(colours_e)][3:5],16)},{int(colours_e[i % len(colours_e)][5:7],16)},0.18)")
            fig.update_layout(
                title="Replicator Dynamics — Strategy Population Shares",
                xaxis_title="Evolutionary Steps",
                yaxis_title="Population Share",
                yaxis=dict(range=[0, 1]),
                height=260, margin=dict(t=40, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(size=9), orientation="h", y=1.2))
            st.plotly_chart(fig, use_container_width=True, key=f"_evo_dyn_{domain}")

    st.markdown(
        f"**Regulatory pressure:** {evo.get('regulatory_pressure', 0.5):.1f} · "
        f"**Market pressure:** {evo.get('market_pressure', 0.5):.1f}")


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 6 — COMPLEX ADAPTIVE SYSTEMS
# ══════════════════════════════════════════════════════════════════════════════

def render_cas(cas: dict, domain: str) -> None:
    _theory_header("🌐", "Complex Adaptive Systems — Emergent Fairness Dynamics",
                   "Holland (1992) · Axelrod (2006) · Mitchell (2009)", C_AMBER)

    if not cas:
        st.info("CAS analysis not computed.")
        return

    amp     = cas.get("amplification", 1.0)
    history = cas.get("history", [])
    pts     = cas.get("phase_transitions", [])

    amp_col = C_RED if amp > 1.2 else C_AMBER if amp > 1.0 else C_GREEN
    st.markdown(
        f"<div style='background:{amp_col}12;border-left:4px solid {amp_col};"
        f"border-radius:6px;padding:8px 14px;margin-bottom:12px;font-size:.83rem;'>"
        f"{cas.get('emergence_message', '')}</div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("N Agents",          f"{cas.get('n_agents', 0):,}")
    c2.metric("Initial Gap",       f"{cas.get('initial_gap', 0):.3f}")
    c3.metric("Final Gap",         f"{cas.get('final_gap', 0):.3f}",
              delta=f"{cas.get('final_gap',0)-cas.get('initial_gap',0):+.3f}",
              delta_color="inverse")
    c4.metric("Amplification",     f"{amp:.2f}×",
              delta_color="inverse" if amp > 1.0 else "normal")

    if history:
        df_h = pd.DataFrame(history)
        fig  = go.Figure()
        fig.add_scatter(x=df_h["step"], y=df_h["advantaged_rate"],
                        name=f"Advantaged {cas.get('agent_label','')}",
                        mode="lines", line=dict(color=C_CYAN, width=2.5))
        fig.add_scatter(x=df_h["step"], y=df_h["disadvantaged_rate"],
                        name=f"Disadvantaged {cas.get('agent_label','')}",
                        mode="lines", line=dict(color=C_RED, width=2.5))
        fig.add_scatter(x=df_h["step"], y=df_h["access_gap"],
                        name="Access Gap",
                        mode="lines", line=dict(color=C_AMBER, width=2, dash="dash"))

        for pt in pts:
            fig.add_vline(x=pt["step"], line_dash="dot", line_color=C_PURPLE,
                          annotation_text=f"Phase transition (Δ={pt['gini_jump']:+.3f})")

        fig.update_layout(
            title=f"Agent Adaptation Dynamics — {cas.get('decision_label','').title()} Access Over Time",
            xaxis_title="Adaptation Steps", yaxis_title="Approval Rate",
            yaxis=dict(range=[0, 1]),
            height=320, margin=dict(t=40, b=30),
            legend=dict(orientation="h", y=1.15),
            paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True, key=f"_cas_adapt_{domain}")

        # Gini over time
        with st.expander("📊 Gini Coefficient of Access Over Time", expanded=False):
            fig_g = go.Figure(go.Scatter(
                y=cas.get("gini_history", []), mode="lines",
                line=dict(color=C_ORANGE, width=2),
                fill="tozeroy", fillcolor="rgba(234,88,12,0.08)"))
            fig_g.add_hline(y=0.4, line_dash="dot", line_color=C_RED,
                            annotation_text="High inequality threshold (0.40)")
            fig_g.update_layout(xaxis_title="Step", yaxis_title="Gini Coefficient",
                                height=220, margin=dict(t=10, b=30),
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g, use_container_width=True, key=f"_cas_gini_{domain}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY CARD
# ══════════════════════════════════════════════════════════════════════════════

def render_dynamic_summary(report: dict, domain: str) -> None:
    comp   = report.get("composite_score", 0)
    scores = report.get("pillar_scores", {})
    n_t    = report.get("theory_count", 0)

    comp_col = _score_c(comp)
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{C_NAVY},{C_PURPLE});
    border-radius:12px;padding:16px 20px;margin-bottom:16px;'>
    <div style='display:flex;justify-content:space-between;align-items:center;'>
    <div>
      <p style='color:#a78bfa;font-size:.68rem;font-weight:700;
      letter-spacing:.1em;text-transform:uppercase;margin:0 0 4px;'>
      🔮 Dynamic Systems Analysis — {domain.title()}</p>
      <p style='color:#f1f5f9;font-size:.9rem;margin:0;'>
      {n_t} theoretical frameworks · System Dynamics · MDP · Information Theory ·
      Causal Fairness · Evolutionary Game Theory · Complex Adaptive Systems</p>
    </div>
    <div style='text-align:center;background:rgba(255,255,255,.12);
    border-radius:10px;padding:8px 16px;'>
      <p style='color:{comp_col};font-size:1.8rem;font-weight:700;margin:0;'>
      {comp:.0%}</p>
      <p style='color:#c4b5fd;font-size:.65rem;margin:0;'>Composite Score</p>
    </div>
    </div></div>""", unsafe_allow_html=True)

    if scores:
        cols = st.columns(len(scores))
        for col, (name, val) in zip(cols, scores.items()):
            c = _score_c(val)
            col.markdown(
                f"<div style='background:{c}12;border:1px solid {c}40;border-radius:8px;"
                f"padding:8px;text-align:center;'>"
                f"<p style='font-size:.7rem;color:{c};font-weight:700;margin:0 0 4px;'>{name}</p>"
                f"<p style='font-size:1.2rem;font-weight:700;color:#0f172a;margin:0;'>{val:.0%}</p>"
                f"</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def render_dynamic_systems_tab(
    report: dict,
    domain: str,
    bias_intensity: float = 0.3,
    governance_strength: float = 0.5,
    regulatory_pressure: float = 0.6,
    market_pressure: float = 0.4,
    n_agents: int = 150,
    ds_key: str = "",
) -> None:
    """
    Main renderer for 🔮 Dynamic Systems tab.
    Includes an inline Run button so it runs independently of the main simulation.
    """
    # ── Show error if previous run failed ─────────────────────────────────────
    if report and report.get("error") and not report.get("pillars"):
        st.error(f"⚠️ Dynamic Systems error: {report['error']}")
        st.caption("Check that gags_dynamic_systems.py is in components/")

    # ── If no results yet (or error), show run panel ───────────────────────────
    if not report or not report.get("pillars"):
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#1e1b4b,#312e81);"
            f"border-radius:12px;padding:24px 28px;margin-bottom:16px;text-align:center;'>"
            f"<p style='font-size:2rem;margin:0 0 8px;'>🔮</p>"
            f"<p style='color:#e0e7ff;font-weight:700;font-size:1.05rem;margin:0 0 6px;'>"
            f"Dynamic Systems Analysis</p>"
            f"<p style='color:#a5b4fc;font-size:.83rem;margin:0 0 14px;'>"
            f"6 theoretical frameworks · System Dynamics · MDP · Information Theory · "
            f"Causal Fairness · Evolutionary Game Theory · Complex Adaptive Systems</p>"
            f"</div>", unsafe_allow_html=True
        )

        # Inline configuration sliders
        with st.expander("⚙️ Analysis Configuration", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                bias_intensity     = st.slider("Bias Intensity",      0.1, 1.0, bias_intensity, 0.05, key=f"_ds_bias_{domain}")
                governance_strength= st.slider("Governance Strength", 0.1, 1.0, governance_strength, 0.05, key=f"_ds_gov_{domain}")
            with col2:
                regulatory_pressure= st.slider("Regulatory Pressure", 0.1, 1.0, regulatory_pressure, 0.05, key=f"_ds_reg_{domain}")
                market_pressure    = st.slider("Market Pressure",     0.1, 1.0, market_pressure, 0.05, key=f"_ds_mkt_{domain}")
            n_agents = st.slider("N Agents (CAS)", 50, 300, n_agents, 50, key=f"_ds_agents_{domain}")

        if st.button("▶️ Run Dynamic Systems Analysis", type="primary",
                     use_container_width=True, key=f"_ds_run_{domain}"):
            with st.spinner("Running 6 theoretical frameworks…"):
                try:
                    rng  = np.random.default_rng(42)
                    X_ds = rng.standard_normal((300, 10))
                    y_ds = (X_ds[:, 0] > 0).astype(int)
                    s_ds = (X_ds[:, 1] > 0).astype(int)
                    if _run_ds is None:
                        st.error("Dynamic systems engine not available. Check components/gags_dynamic_systems.py")
                        return
                    _rep = _run_ds(
                        domain=domain,
                        y_true=y_ds, y_pred=y_ds, sensitive=s_ds,
                        bias_intensity=float(bias_intensity),
                        governance_strength=float(governance_strength),
                        regulatory_pressure=float(regulatory_pressure),
                        market_pressure=float(market_pressure),
                        n_agents=int(n_agents),
                    )
                    # Store in the domain-specific session state key
                    _store_key = ds_key or f"{domain}_ds_report"
                    st.session_state[_store_key] = _rep
                    st.rerun()
                except Exception as _e:
                    st.error(f"Analysis failed: {_e}")
                    import traceback
                    st.code(traceback.format_exc())
        return

    render_dynamic_summary(report, domain)
    st.divider()

    pillars = report.get("pillars", {})

    tabs = st.tabs([
        "♾️ System Dynamics",
        "🎯 MDP Policy",
        "📡 Information Theory",
        "🔗 Causal Fairness",
        "🧬 Evolutionary GT",
        "🌐 Complex Adaptive",
    ])

    with tabs[0]:
        render_system_dynamics(pillars.get("system_dynamics", {}), domain)
    with tabs[1]:
        render_mdp(pillars.get("mdp", {}), domain)
    with tabs[2]:
        render_information_theory(pillars.get("information", {}), domain)
    with tabs[3]:
        render_causal_fairness(pillars.get("causal", {}), domain)
    with tabs[4]:
        render_evolutionary(pillars.get("evolutionary", {}), domain)
    with tabs[5]:
        render_cas(pillars.get("cas", {}), domain)
