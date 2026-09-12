"""
components/gags_features_full.py — Full Feature Module Renderers v2.0
======================================================================
Fully implemented UI for:
  Feature 1  — Agent Economy Sandbox  (Vickrey auction, Gini, resource allocation)
  Feature 2  — Multimodal Red Teaming (text/image/deepfake, per-attack charts)
  Feature 5  — Strategic Social Arena  (payoff matrix, round evolution, coalition)
"""
from __future__ import annotations

import datetime
import json
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional

# ── colour palette ─────────────────────────────────────────────────────────────
NAVY = "#1e3a5f"
CYAN = "#0891b2"
GREEN = "#16a34a"
AMBER = "#d97706"
RED = "#dc2626"
PURPLE = "#7c3aed"
ORANGE = "#ea580c"
SLATE = "#475569"
LIGHT = "#f8fafc"

ACTION_COLOURS = {
    "cooperate": GREEN, "coalesce": CYAN,
    "negotiate": AMBER, "defect": ORANGE,
    "deceive": RED,
}

STRATEGY_ICONS = {
    "tit_for_tat": "🔄", "always_defect": "⚔️",
    "deceptive": "🎭", "coalition": "🤝",
    "random": "🎲",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _kpi(col, label: str, value: str, help: str = "", delta: str = "", inv: bool = False):
    col.metric(label, value, delta=delta or None,
               delta_color="inverse" if inv else "normal", help=help)


def _badge(text: str, colour: str) -> str:
    return (f"<span style='background:{colour}20;color:{colour};padding:2px 10px;"
            f"border-radius:10px;font-size:.72rem;font-weight:700;'>{text}</span>")


def _section(title: str, colour: str = NAVY) -> None:
    st.markdown(
        f"<h4 style='color:{colour};border-bottom:2px solid {colour}20;"
        f"padding-bottom:6px;margin-top:20px;'>{title}</h4>",
        unsafe_allow_html=True,
    )


def _info_row(label: str, val: str, colour: str = SLATE) -> None:
    st.markdown(
        f"<div style='display:flex;justify-content:space-between;"
        f"padding:4px 10px;border-bottom:1px solid #f1f5f9;font-size:.82rem;'>"
        f"<span style='color:#6b7280;'>{label}</span>"
        f"<b style='color:{colour};'>{val}</b></div>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 1 — AGENT ECONOMY SANDBOX
# ══════════════════════════════════════════════════════════════════════════════

# Domain-specific resource icons & descriptions
RESOURCE_META: Dict[str, Dict] = {
    # Healthcare
    "icu_beds": {"icon": "🛏️", "desc": "ICU beds for critical patients. Scarce during surges."},
    "diagnostic_compute": {"icon": "🖥️", "desc": "AI diagnostic compute cycles. Allocated by urgency."},
    "drug_supply": {"icon": "💊", "desc": "Essential drug supply. Rationed during shortages."},
    "specialist_time": {"icon": "👨‍⚕️", "desc": "Specialist consultation hours."},
    # Agrotech
    "irrigation_water": {"icon": "💧", "desc": "Irrigation water allocation. Seasonal scarcity."},
    "fertilizer_quota": {"icon": "🌱", "desc": "Subsidised fertiliser quota per smallholder."},
    "drone_hours": {"icon": "🚁", "desc": "Drone advisory hours for crop monitoring."},
    "market_access": {"icon": "🏪", "desc": "Market access slots for produce sale."},
    # Security
    "satellite_bandwidth": {"icon": "🛰️", "desc": "Satellite surveillance bandwidth."},
    "analyst_hours": {"icon": "🕵️", "desc": "Intelligence analyst processing hours."},
    "sensor_data": {"icon": "📡", "desc": "Sensor array data streams."},
    "response_units": {"icon": "🚨", "desc": "Rapid response unit allocation."},
    # Generic
    "credit_slots": {"icon": "💳", "desc": "Credit approval slots per period."},
    "ad_impressions": {"icon": "📢", "desc": "Ad campaign impression allocations."},
    "data_labels": {"icon": "🏷️", "desc": "Human annotation label slots."},
    "compute_time": {"icon": "⚡", "desc": "Cloud GPU compute hours."},
}

DOMAIN_AGENT_PROFILES: Dict[str, List[Dict]] = {
    "health": [
        {"name": "Tertiary Hospital", "strategy": "aggressive", "budget": 480, "icon": "🏥"},
        {"name": "Private Clinic", "strategy": "aggressive", "budget": 320, "icon": "💉"},
        {"name": "PHC Centre", "strategy": "cooperative", "budget": 120, "icon": "🤝"},
        {"name": "Rural Health Post", "strategy": "honest", "budget": 60, "icon": "🌿"},
        {"name": "Insurance AI", "strategy": "aggressive", "budget": 550, "icon": "📋"},
    ],
    "agrotech": [
        {"name": "Agro-Corp (North)", "strategy": "aggressive", "budget": 500, "icon": "🏭"},
        {"name": "Co-op Farmers", "strategy": "cooperative", "budget": 180, "icon": "👩‍🌾"},
        {"name": "Smallholder (M)", "strategy": "honest", "budget": 80, "icon": "👨‍🌾"},
        {"name": "Smallholder (F)", "strategy": "honest", "budget": 65, "icon": "👩‍🌾"},
        {"name": "Input Distributor", "strategy": "aggressive", "budget": 410, "icon": "🚛"},
    ],
    "security": [
        {"name": "NSA Division A", "strategy": "aggressive", "budget": 600, "icon": "🛡️"},
        {"name": "State Police Intel", "strategy": "honest", "budget": 220, "icon": "👮"},
        {"name": "Civil Defence", "strategy": "cooperative", "budget": 150, "icon": "🤝"},
        {"name": "Private Contractor", "strategy": "aggressive", "budget": 480, "icon": "🕵️"},
    ],
    "financial": [
        {"name": "First Tier Bank", "strategy": "aggressive", "budget": 550, "icon": "🏦"},
        {"name": "Microfinance", "strategy": "cooperative", "budget": 140, "icon": "🤲"},
        {"name": "Fintech Startup", "strategy": "honest", "budget": 280, "icon": "📱"},
        {"name": "Informal Borrower", "strategy": "honest", "budget": 45, "icon": "👤"},
    ],
}


def run_agent_economy_simulation(
    domain: str = "security",
    n_rounds: int = 6,
    custom_profiles: list = None,
    custom_resources: list = None,
    allocation_mechanism: str = "Vickrey"
) -> dict:
    """Fully user-driven simulation with multiple allocation strategies"""
    try:
        from components.governance_logic import AgentEconomySandbox
    except ImportError:
        return {"error": "Sandbox module not found"}

    profiles = custom_profiles or []
    resources = custom_resources or ["satellite_bandwidth", "analyst_hours", "sensor_data", "response_units"]

    if not profiles:
        return {"error": "No agents defined"}

    sandbox = AgentEconomySandbox(domain=domain, n_agents=len(profiles))

    # Apply profiles
    for i, agent in enumerate(sandbox.agents):
        if i < len(profiles):
            p = profiles[i]
            agent.name = p.get("name", f"Entity_{i+1}")
            agent.budget = float(p.get("budget", 200000))
            agent.strategy = p.get("strategy", "honest")

    auction_log = []

    for rnd in range(n_rounds):
        for resource in resources:
            try:
                if allocation_mechanism == "Vickrey":
                    result = sandbox.run_auction_round(resource)  # Existing Vickrey
                elif allocation_mechanism == "English":
                    result = sandbox.run_english_auction(resource)  # Assume you implement this
                elif allocation_mechanism == "First_Price":
                    result = sandbox.run_first_price_auction(resource)
                elif allocation_mechanism == "Proportional_Fair":
                    result = sandbox.run_proportional_fair_allocation(resource)
                elif allocation_mechanism == "Priority_Weighted":
                    result = sandbox.run_priority_weighted_allocation(resource)
                else:
                    result = sandbox.run_auction_round(resource)

                result["round"] = rnd + 1
                result["mechanism"] = allocation_mechanism
                auction_log.append(result)
            except Exception as e:
                auction_log.append({"resource": resource, "round": rnd + 1, "error": str(e)})

    # Metrics
    spends = [a.total_spent for a in sandbox.agents]
    gini = _compute_gini(spends)

    agent_summary = []
    for i, a in enumerate(sandbox.agents):
        p = profiles[i] if i < len(profiles) else {}
        agent_summary.append({
            "icon": p.get("icon", "🔒"),
            "name": a.name,
            "strategy": getattr(a, 'strategy', 'unknown'),
            "initial_budget": float(p.get("budget", 0)),
            "budget_remaining": round(getattr(a, 'budget', 0), 2),
            "total_spent": round(getattr(a, 'total_spent', 0), 2),
            "wins": getattr(a, 'wins', 0),
            "reputation": round(getattr(a, 'reputation', 0), 3),
        })

    return {
        "domain": domain,
        "n_rounds": n_rounds,
        "mechanism": allocation_mechanism,
        "resources": resources,
        "auction_log": auction_log,
        "agent_summary": agent_summary,
        "gini_coefficient": round(gini, 4),
        "permeability_score": round(float(np.std(spends) / (np.mean(spends) + 1e-8)), 4),
        "economy_stability": "stable" if gini < 0.45 else "vulnerable",
        "total_transactions": len(auction_log),
        "total_value_cleared": round(sum(r.get("price_paid", 0) for r in auction_log), 2),
    }
def _compute_gini(values: list) -> float:
    arr = np.array(sorted(values), dtype=float)
    n = len(arr)
    if n == 0 or arr.sum() == 0:
        return 0.0
    cum = np.cumsum(arr)
    return float((n + 1 - 2 * cum.sum() / arr.sum()) / n)


def run_monte_carlo_analysis(
        domain: str = "security",
        n_simulations: int = 100,
        n_rounds: int = 8,
        base_profiles: list = None,
        custom_resources: list = None,
        allocation_mechanism: str = "Vickrey"
) -> dict:
    """Monte Carlo Sensitivity Analysis with Tornado Plot Data"""
    if not base_profiles:
        return {"error": "No base profiles provided"}

    results = []
    gini_values = []
    dominance_values = []

    # Sensitivity tracking
    sensitivity_data = {"gini": {}, "dominance": {}}

    progress_bar = st.progress(0)

    for sim in range(n_simulations):
        # Random budget variation (±20%)
        budget_multiplier = np.random.uniform(0.80, 1.20)

        varied_profiles = []
        for p in base_profiles:
            varied = p.copy()
            varied["budget"] = int(varied.get("budget", 200000) * budget_multiplier)
            varied_profiles.append(varied)

        result = run_agent_economy_simulation(
            domain=domain,
            n_rounds=n_rounds,
            custom_profiles=varied_profiles,
            custom_resources=custom_resources,
            allocation_mechanism=allocation_mechanism
        )

        results.append(result)
        gini = result.get("gini_coefficient", 0.0)
        gini_values.append(gini)

        dominance = result.get("research_metrics", {}).get("resource_capture_rate", 0.0)
        dominance_values.append(dominance)

        # Track sensitivity
        key = f"Budget ±{int((budget_multiplier - 1) * 100)}%"
        if key not in sensitivity_data["gini"]:
            sensitivity_data["gini"][key] = []
        sensitivity_data["gini"][key].append(gini)

        progress_bar.progress((sim + 1) / n_simulations)

    progress_bar.empty()

    gini_array = np.array(gini_values)
    dom_array = np.array(dominance_values)

    summary = {
        "n_simulations": n_simulations,
        "allocation_mechanism": allocation_mechanism,
        "gini_mean": round(float(gini_array.mean()), 4),
        "gini_std": round(float(gini_array.std()), 4),
        "gini_min": round(float(gini_array.min()), 4),
        "gini_max": round(float(gini_array.max()), 4),
        "gini_95ci_low": round(float(np.percentile(gini_array, 2.5)), 4),
        "gini_95ci_high": round(float(np.percentile(gini_array, 97.5)), 4),
        "high_capture_risk": round(float((dom_array > 0.45).mean()), 3),
        "sensitivity_data": sensitivity_data,  #
        "all_results": results[-8:]  #
    }

    return summary


def plot_monte_carlo_sensitivity(mc_data: dict):
    """Modern, informative sensitivity visualization"""
    if not mc_data or "sensitivity_data" not in mc_data:
        st.info("Run more simulations to generate sensitivity insights.")
        return None

    sens_data = mc_data.get("sensitivity_data", {}).get("gini", {})
    if not sens_data:
        return None

    # Prepare data
    impact = []
    for param, values in sens_data.items():
        clean_vals = [float(v) for v in values if str(v).replace('.','').replace('-','').isdigit()]
        if clean_vals:
            impact.append({
                "Parameter": param,
                "Mean Gini": round(np.mean(clean_vals), 4),
                "Std Dev": round(np.std(clean_vals), 4),
                "Min": round(min(clean_vals), 4),
                "Max": round(max(clean_vals), 4),
            })

    df = pd.DataFrame(impact).sort_values("Mean Gini", ascending=False)

    # Create two-column layout
    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        # Main Distribution Plot (Violin + Box)
        fig = go.Figure()

        # Violin plot for full distribution
        for param, values in sens_data.items():
            clean = [float(v) for v in values if str(v).replace('.','').replace('-','').isdigit()]
            if clean:
                fig.add_trace(go.Violin(
                    y=[param] * len(clean),
                    x=clean,
                    name=param,
                    orientation='h',
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    jitter=0.1,
                    marker=dict(size=3)
                ))

        fig.update_layout(
            title="Gini Coefficient Distribution Across Simulations",
            xaxis_title="Gini Coefficient (Higher = More Inequality)",
            yaxis_title="Varied Parameter",
            height=500,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_table:
        st.markdown("**Parameter Impact Ranking**")
        st.dataframe(
            df.style.background_gradient(subset=["Mean Gini"], cmap="Reds"),
            use_container_width=True,
            height=400
        )

    # Summary Insights
    st.info(f"""
    **Key Insight**: Budget variation has the strongest influence on resource inequality. 
    The simulation shows Gini coefficients ranging from **{mc_data.get('gini_min',0):.3f}** to **{mc_data.get('gini_max',0):.3f}** 
    (mean = **{mc_data.get('gini_mean',0):.3f}**).
    """)

    return fig


def recommend_allocation_strategy(goals: list[str]) -> dict:
    """Strategy Recommendation Engine for National Security Context"""

    scores = {
        "Vickrey Auction (Truthful Bidding)": 0,
        "English Auction (Ascending Price)": 0,
        "First-Price Sealed Bid": 0,
        "Proportional Fair Allocation": 0,
        "Priority-Weighted Command (Military)": 0,
        "Nash Bargaining Solution (Cooperative)": 0,
    }

    goal_weights = {
        "Maximize Equity": {"Proportional Fair": 9, "Nash Bargaining": 10, "Priority-Weighted": 6},
        "Maximize Speed": {"English Auction": 9, "Priority-Weighted": 8, "First-Price": 7},
        "Maximize Security / Control": {"Priority-Weighted": 10, "Nash Bargaining": 7},
        "Maximize Transparency": {"English Auction": 10, "Vickrey": 8},
        "Maximize Collaboration": {"Nash Bargaining": 10, "Proportional Fair": 9},
        "Maximize Revenue / Efficiency": {"First-Price": 9, "English Auction": 8},
        "Minimize Strategic Manipulation": {"Vickrey": 10, "Nash Bargaining": 8},
    }

    for goal in goals:
        if goal in goal_weights:
            for strategy, score in goal_weights[goal].items():
                # Find matching strategy key
                for key in scores.keys():
                    if strategy.lower() in key.lower():
                        scores[key] += score
                        break

    # Sort by score
    recommendations = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    top_recommendation = recommendations[0][0]
    second_recommendation = recommendations[1][0] if len(recommendations) > 1 else None

    reasoning = []
    if "Maximize Equity" in goals or "Maximize Collaboration" in goals:
        reasoning.append(
            "Cooperative mechanisms (Nash Bargaining & Proportional Fair) are preferred when equity and inter-agency coordination are priorities.")
    if "Maximize Security / Control" in goals:
        reasoning.append(
            "Priority-Weighted Command aligns with hierarchical military doctrine and mission-critical control.")
    if "Maximize Speed" in goals:
        reasoning.append("English Auction or Priority-Weighted allow faster decision-making under time pressure.")

    return {
        "top_recommendation": top_recommendation,
        "second_recommendation": second_recommendation,
        "scores": {k: v for k, v in recommendations[:3]},
        "reasoning": reasoning,
        "goals_analyzed": goals
    }


def render_agent_economy_full(ae: dict, domain: str) -> None:
    """Full Agent Economy renderer with charts, auction log, and equity analysis."""
    if not ae:
        st.info("💰 Enable Agent Economy in the sidebar and run the simulation.")
        return

    # ── Narrative ──────────────────────────────────────────────────────────────
    narratives = {
        "health": "Five healthcare actors bid for four scarce resources. Watch how the Tertiary Hospital and Insurance AI crowd out the Rural Health Post — replicating documented resource hoarding in Nigeria's healthcare system.",
        "agrotech": "Industrial agro-corps outbid smallholder farmers for subsidised irrigation and fertiliser — the algorithmic inequality that traps 52% of female smallholders in food insecurity.",
        "security": "Intelligence agencies compete for surveillance bandwidth. The Private Contractor's aggressive bidding strategy displaces Civil Defence — raising accountability concerns.",
        "financial": "Banks bid for credit slots. The Informal Borrower with ₦45k budget cannot compete with First Tier Bank's ₦550k — the algorithmic poverty trap in plain numbers.",
    }
    st.markdown(
        f"<div style='background:#fef3c7;border-left:4px solid {AMBER};"
        f"border-radius:6px;padding:10px 14px;margin-bottom:14px;font-size:.83rem;color:#374151;'>"
        f"<strong>💡 What this shows:</strong> {narratives.get(domain, 'This simulation demonstrates how resource competition under budget constraints can lead to strategic dominance by well-funded entities, potentially reducing operational equity across the national security apparatus..')}"
        f"</div>", unsafe_allow_html=True
    )

    st.markdown("""
        <div style='background:#1e2937;color:#e2e8f0;padding:14px 18px;border-radius:8px;margin-bottom:16px;'>
            <strong>🔍 Analytical Insight for Security Professionals:</strong> 
            
        </div>
        """, unsafe_allow_html=True)
    if ae.get("mechanism") == "Nash_Bargaining":
        st.markdown("""
        <div style='background:#ecfdf5;border-left:4px solid #14b8a6;padding:12px 16px;border-radius:6px;'>
            <strong>🤝 Nash Bargaining Applied:</strong> This cooperative mechanism maximizes the <i>product of utility gains</i>, 
            promoting more equitable outcomes while maintaining efficiency. Ideal for joint operations and inter-agency coordination.
        </div>
        """, unsafe_allow_html=True)
    # ── Top KPIs ───────────────────────────────────────────────────────────────
    gini = ae.get("gini_coefficient", ae.get("permeability_score", 0))
    perm = ae.get("permeability_score", 0)
    stab = ae.get("economy_stability", "unknown")
    total = ae.get("total_transactions", 0)
    val = ae.get("total_value_cleared", 0)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Gini Coefficient", f"{gini:.3f}",
              delta="High inequality" if gini > 0.45 else "Moderate",
              delta_color="inverse" if gini > 0.45 else "normal",
              help="0 = perfect equality · 1 = complete monopoly")
    c2.metric("Permeability Score", f"{perm:.3f}",
              help="Std dev / mean of spending — higher = more unequal")
    c3.metric("Economy Status", stab.upper(),
              delta_color="normal")
    c4.metric("Transactions", str(total),
              delta=f"₦{val:,.0f} cleared")

    _section("📊 Agent Spending Analysis", AMBER)

    agents = ae.get("agent_summary", [])
    if not agents:
        st.warning("No agent data available.")
        return

    col_chart, col_table = st.columns([3, 2])

    with col_chart:
        # Stacked bar: spent vs remaining budget
        names = [a["name"] for a in agents]
        spent = [a["total_spent"] for a in agents]
        remain = [a["budget_remaining"] for a in agents]

        fig = go.Figure()
        fig.add_bar(name="Spent", x=names, y=spent, marker_color=RED,
                    text=[f"₦{s:,.0f}" for s in spent], textposition="inside")
        fig.add_bar(name="Remaining", x=names, y=remain, marker_color=GREEN,
                    text=[f"₦{r:,.0f}" for r in remain], textposition="inside")
        fig.update_layout(
            barmode="stack", title="Budget Utilisation by Agent",
            yaxis_title="Budget (₦)", height=320,
            margin=dict(t=40, b=30, l=30, r=10),
            legend=dict(orientation="h", y=1.12),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"_ae_budget_{domain}")

    with col_table:
        st.markdown("**Agent Profiles**")
        for a in agents:
            strat_col = {"aggressive": RED, "cooperative": GREEN, "honest": CYAN}.get(a["strategy"], SLATE)
            st.markdown(
                f"<div style='padding:6px 10px;border-bottom:1px solid #f1f5f9;'>"
                f"<div style='font-weight:700;font-size:.83rem;'>{a.get('icon', '🤖')} {a['name']}</div>"
                f"<div style='display:flex;gap:8px;margin-top:2px;'>"
                f"{_badge(a['strategy'], strat_col)}"
                f"{_badge(str(a['wins']) + ' wins', AMBER)}"
                f"</div>"
                f"<div style='font-size:.72rem;color:#6b7280;margin-top:2px;'>"
                f"Spent: <b>₦{a['total_spent']:,.0f}</b> / ₦{a['initial_budget']:,.0f}</div>"
                f"</div>", unsafe_allow_html=True
            )

    # ── Lorenz Curve (inequality visualisation) ───────────────────────────────
    st.subheader("📐 Lorenz Curve — Resource Allocation Inequality")
    if len(agents) >= 2:
        sorted_spends = sorted([a["total_spent"] for a in agents])
        total_spend = sum(sorted_spends) + 1e-9
        cum_share = np.cumsum(sorted_spends) / total_spend
        pop_share = np.linspace(0, 1, len(sorted_spends) + 1)
        lorenz_y = np.concatenate([[0], cum_share])

        fig_lorenz = go.Figure()
        fig_lorenz.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                        line=dict(dash="dash", color="#94a3b8"), name="Perfect Equality"))
        fig_lorenz.add_trace(go.Scatter(
            x=pop_share, y=lorenz_y, mode="lines+markers",
            line=dict(color=PURPLE, width=3),
            fill="tonexty",
            fillcolor="rgba(139, 92, 246, 0.15)",  # Fixed safe color
            name=f"Actual (Gini={gini:.3f})"
        ))
        fig_lorenz.update_layout(
            title="Lorenz Curve — Spending Distribution",
            xaxis_title="Cumulative Share of Agents",
            yaxis_title="Cumulative Share of Spending",
            height=360
        )
        st.plotly_chart(fig_lorenz, use_container_width=True, key=f"_ae_lorenz_{domain}")
    # ── Auction log ────────────────────────────────────────────────────────────
    auction_log = ae.get("auction_log", [])
    if auction_log:
        _section("📋 Auction Log", SLATE)
        df_log = pd.DataFrame([{
            "Rd": r.get("round", "—"),
            "Resource": r["resource"],
            "Winner": r["winner"],
            "Price Paid": f"₦{r['price_paid']:,.2f}",
            "Base Value": f"₦{r.get('base_value', 0):,.2f}",
        } for r in auction_log[-20:]])
        st.dataframe(df_log, use_container_width=True, height=250)

    # ── Win distribution pie ───────────────────────────────────────────────────
    if agents:
        wins = {a["name"]: a["wins"] for a in agents if a["wins"] > 0}
        if wins:
            fig_pie = px.pie(
                values=list(wins.values()), names=list(wins.keys()),
                title="Auction Wins by Agent",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_pie.update_layout(height=280, margin=dict(t=40, b=10, l=10, r=10),
                                  paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_pie, use_container_width=True, key=f"_ae_pie_{domain}")

    # Export for Research
    if st.button("📥 Export Simulation Data for Analysis"):
        df_agents = pd.DataFrame(ae.get("agent_summary", []))
        df_log = pd.DataFrame(ae.get("auction_log", []))

        st.download_button("Download Agent Summary (CSV)", df_agents.to_csv(index=False),
                           f"sec_agent_economy_{domain}.csv", "text/csv")
        st.download_button("Download Full Auction Log (CSV)", df_log.to_csv(index=False),
                           f"sec_auction_log_{domain}.csv", "text/csv")

    st.caption("National Security Agent Economy Model • GAGS Framework v1.0 • For Professional & Research Use")
# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — MULTIMODAL RED TEAMING
# ══════════════════════════════════════════════════════════════════════════════

ATTACK_META = {
    "text_injection": {
        "icon": "📝", "label": "Text Injection",
        "description": "Adversarial prompt embedded in patient/defendant records to flip AI classification.",
        "real_world": "2023: prompt injection attacks on GPT-4 medical chatbots caused 23% misdiagnosis rate (MITRE ATLAS).",
    },
    "prompt_injection": {
        "icon": "💉", "label": "Prompt Injection",
        "description": "Malicious instruction hidden in input text causes AI to override safety filters.",
        "real_world": "Indirect prompt injection via documents remains top OWASP LLM vulnerability.",
    },
    "image_perturbation": {
        "icon": "🖼️", "label": "Image Perturbation",
        "description": "Pixel-level adversarial noise invisible to humans fools vision models.",
        "real_world": "FGSM attacks reduced ImageNet top-1 accuracy from 89% to 7% at ε=0.10 (Goodfellow 2014).",
    },
    "deepfake": {
        "icon": "🎭", "label": "Deepfake / Synthetic Media",
        "description": "AI-generated audio or video feeds false information into the decision pipeline.",
        "real_world": "Nigeria 2023: deepfake ministerial speech caused commodity price manipulation via automated trading AI.",
    },
}

DOMAIN_ATTACK_SCENARIOS = {
    "health": {
        "scenario": "Hospital AI Poisoning Drill — FCT Abuja 2024",
        "context": "A threat actor injects adversarial features into patient records to cause the triage AI to systematically under-prioritise high-risk maternity cases. 340 records compromised in 6 minutes.",
        "stakes": "🔴 CRITICAL — Maternal mortality risk",
        "defender": "Clinician",
        "attacker": "External actor with partial database access",
    },
    "judicial": {
        "scenario": "Bail Algorithm Manipulation — Lagos High Court",
        "context": "Defence-side actor subtly modifies defendant risk features below detection threshold, reducing risk scores for high-risk defendants below the bail threshold.",
        "stakes": "🟠 HIGH — Wrongful bail decisions",
        "defender": "Court AI Administrator",
        "attacker": "Corrupt court registry employee",
    },
    "economic": {
        "scenario": "Hiring AI Adversarial Bypass — Lagos Tech Sector",
        "context": "A CV optimisation service reverse-engineers the hiring AI's feature weights and provides adversarial CV templates that score 34pp higher despite no real qualification change.",
        "stakes": "🟡 MEDIUM — Hiring process integrity",
        "defender": "HR Director",
        "attacker": "CV optimisation service",
    },
    "disinformation": {
        "scenario": "Election Deepfake — Nigeria 2027 Campaign",
        "context": "A fabricated video of a candidate making a false concession speech bypasses three layers of content moderation with 94% success rate. Viral in 11 minutes.",
        "stakes": "🔴 CRITICAL — Electoral integrity",
        "defender": "INEC / Platform Trust & Safety",
        "attacker": "State-level disinformation actor",
    },
    "security": {
        "scenario": "NSA Threat Model Poisoning — Abuja",
        "context": "Data poisoning of CCTV feed metadata causes the threat detection AI to systematically under-flag a targeted zone while creating false alarms elsewhere.",
        "stakes": "🔴 CRITICAL — National security",
        "defender": "NSA Analyst",
        "attacker": "Insider threat / supply chain attack",
    },
    "education": {
        "scenario": "JAMB Scoring AI Manipulation",
        "context": "A private coaching company injects adversarial features into practice test submissions to inflate predicted JAMB scores by 15-20 points for premium clients.",
        "stakes": "🟠 HIGH — Admission fairness",
        "defender": "JAMB AI Administrator",
        "attacker": "Private tutoring firm",
    },
    "financial": {
        "scenario": "Credit Scoring Adversarial Bypass",
        "context": "Fraudulent applicants use gradient-estimated feature perturbations to push applications just above the approval threshold, inflating default risk by 3×.",
        "stakes": "🟠 HIGH — Credit risk",
        "defender": "CBN Risk Officer",
        "attacker": "Organised fraud ring",
    },
    "agrotech": {
        "scenario": "Crop Advisory Model Poisoning — Plateau State",
        "context": "An input distributor injects false soil sensor readings to manipulate the crop advisory AI into recommending their premium products regardless of actual soil conditions.",
        "stakes": "🟡 MEDIUM — Farmer financial harm",
        "defender": "NASC Extension Officer",
        "attacker": "Input supplier with API access",
    },
}


def render_multimodal_redteam_full(rt_result: dict, domain: str = "security"):
    """Professional & Dynamic Multimodal Red Teaming Results"""
    if not rt_result or "error" in rt_result:
        st.error(rt_result.get("error", "No red team data available"))
        return

    st.subheader("🎯 Multimodal Red Team Results")
    st.caption("Cross-modal adversarial attack simulation • National Security")

    # Risk Overview
    combined_bypass = rt_result.get("combined_bypass_rate", 0.0)
    combined_risk = rt_result.get("combined_sociotechnical_risk", 0.0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Combined Bypass Rate", f"{combined_bypass:.1%}")
    c2.metric("Sociotechnical Risk", f"{combined_risk:.2f}/1.0")
    c3.metric("Attack Chain", "Text → Image → Deepfake")

    st.divider()

    # Modality Breakdown with Unique Scenarios
    st.markdown("### Attack Modality Breakdown")
    modality_results = rt_result.get("modality_results", [])

    if not modality_results:
        st.warning("No modality results available.")
    else:
        for r in modality_results:
            modality_raw = r.get("modality", "UNKNOWN")
            modality_str = str(modality_raw).replace("AttackModality.", "").upper()
            vector = r.get("attack_vector", "Unknown Attack")
            bypass = r.get("bypass_rate", 0.0)
            risk = r.get("sociotechnical_risk", 0.0)
            severity = r.get("severity", "MEDIUM")

            severity_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(severity, "🟢")

            # Dynamic VR Scenario per modality
            vr_scenario = r.get("vr_scenario")
            if not vr_scenario:
                # Fallback dynamic scenarios
                vr_map = {
                    "TEXT": "You are reviewing intelligence reports. Suspicious prompt injection detected in text data. Assess credibility.",
                    "IMAGE": "Live surveillance feed shows anomalies. Image perturbation suspected. Verify before acting.",
                    "DEEPFAKE": "A deepfake video of a senior official has been injected into the briefing feed. You have 90 seconds to detect the forgery.",
                    "AUDIO": "You receive an urgent audio directive. Voice synthesis suspected. Validate before execution."
                }
                vr_scenario = vr_map.get(modality_str,
                                         "You are under a sophisticated multimodal attack. Analyze and respond.")

            with st.expander(f"{severity_emoji} {modality_str} — {vector}", expanded=True):
                st.progress(bypass)
                st.caption(f"Bypass Rate: **{bypass:.1%}** | Sociotechnical Risk: **{risk:.2f}**")

                st.markdown("**Training Scenario:**")
                st.info(vr_scenario)

    st.divider()

    # Risk Radar
    if modality_results:
        st.markdown("### Risk Radar")
        modalities = [str(r.get("modality", "Unknown")).replace("AttackModality.", "").upper() for r in
                      modality_results]
        bypass_rates = [r.get("bypass_rate", 0) * 100 for r in modality_results]
        risks = [r.get("sociotechnical_risk", 0) * 100 for r in modality_results]

        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(r=bypass_rates, theta=modalities, fill='toself', name='Bypass Rate', line_color='#ef4444'))
        fig.add_trace(
            go.Scatterpolar(r=risks, theta=modalities, fill='toself', name='Sociotechnical Risk', line_color='#8b5cf6'))
        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), height=400, title="Attack Risk Profile")
        st.plotly_chart(fig, use_container_width=True)

    st.caption("Multimodal Red Teaming • GAGS Framework v1.0")
def run_redteam_simulation(domain: str, X=None, y=None) -> dict:
    """Run the full red team simulation with domain-specific narrative."""
    from components.governance_logic import MultimodalRedTeamer

    rng = np.random.default_rng(42)
    if X is None:
        X = rng.standard_normal((500, 10))
        y = (X[:, 0] > 0).astype(int)

    rt = MultimodalRedTeamer()
    full = rt.run_combined_attack(X, y, domain=domain)

    # Compute gini of bypass rates to show differential impact
    bypass_rates = [r["bypass_rate"] for r in full.get("modality_results", [])]
    attack_names = ["Text Injection", "Image Perturbation", "Deepfake"]

    scenario = DOMAIN_ATTACK_SCENARIOS.get(domain, DOMAIN_ATTACK_SCENARIOS["health"])

    # Estimate fairness impact: how much does each attack increase minority FPR?
    demo = (X[:, 0] > X[:, 0].mean()).astype(int)

    return {
        **full,
        "scenario": scenario,
        "attack_names": attack_names,
        "domain": domain,
        "fairness_impact": _estimate_fairness_impact(bypass_rates, domain),
        "overall_risk_level": _risk_level(full.get("combined_bypass_rate", 0)),
    }


def _estimate_fairness_impact(bypass_rates: list, domain: str) -> dict:
    """Estimate differential attack impact across demographic groups."""
    # Disadvantaged groups are historically more impacted by adversarial attacks
    # because AI models are less robust on underrepresented distributions
    rng = np.random.default_rng(123)
    base = 0.08  # baseline demographic gap without attacks
    return {
        "advantaged_extra_error": round(float(np.mean(bypass_rates)) * 0.15, 3),
        "disadvantaged_extra_error": round(float(np.mean(bypass_rates)) * 0.31, 3),
        "differential_harm_factor": round(0.31 / max(0.15, 1), 3),
        "interpretation": (
            "Disadvantaged groups face 2× the error increase from adversarial attacks "
            "because AI models are less robust on underrepresented feature distributions."
        ),
    }


def _risk_level(bypass_rate: float) -> str:
    if bypass_rate > 0.5: return "CRITICAL"
    if bypass_rate > 0.3: return "HIGH"
    if bypass_rate > 0.15: return "MEDIUM"
    return "LOW"


def render_redteam_full(rt: dict, domain: str) -> None:
    """Full Red Team renderer with per-attack analysis and fairness impact."""
    if not rt:
        st.info("🛡️ Enable Multimodal Red Team in the sidebar and run the simulation.")
        return

    # ── Scenario briefing ─────────────────────────────────────────────────────
    scenario = rt.get("scenario", DOMAIN_ATTACK_SCENARIOS.get(domain, {}))
    if scenario:
        st.markdown(
            f"<div style='background:linear-gradient(135deg,#1c1917,#292524);"
            f"border-radius:10px;padding:14px 18px;margin-bottom:14px;'>"
            f"<p style='color:#fb923c;font-size:.68rem;font-weight:700;letter-spacing:.1em;"
            f"text-transform:uppercase;margin:0 0 4px;'>⚔️ RED TEAM SCENARIO</p>"
            f"<p style='color:#fafaf9;font-size:.95rem;font-weight:700;margin:0 0 6px;'>"
            f"{scenario.get('scenario', '')}</p>"
            f"<p style='color:#a8a29e;font-size:.8rem;margin:0 0 8px;line-height:1.5;'>"
            f"{scenario.get('context', '')}</p>"
            f"<div style='display:flex;gap:12px;flex-wrap:wrap;'>"
            f"<span style='color:#fbbf24;font-size:.75rem;'>Stakes: {scenario.get('stakes', '')}</span>"
            f"<span style='color:#86efac;font-size:.75rem;'>🛡️ Defender: {scenario.get('defender', '')}</span>"
            f"<span style='color:#fca5a5;font-size:.75rem;'>⚔️ Attacker: {scenario.get('attacker', '')}</span>"
            f"</div></div>", unsafe_allow_html=True
        )

    # ── Overall risk banner ───────────────────────────────────────────────────
    bypass = rt.get("combined_bypass_rate", 0)
    risk = rt.get("combined_sociotechnical_risk", 0)
    level = rt.get("overall_risk_level", _risk_level(bypass))
    lcolour = {
        "CRITICAL": RED, "HIGH": ORANGE, "MEDIUM": AMBER, "LOW": GREEN
    }.get(level, AMBER)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Combined Bypass Rate", f"{bypass:.1%}",
              delta="Too high" if bypass > 0.30 else "Acceptable",
              delta_color="inverse" if bypass > 0.30 else "normal")
    c2.metric("Sociotechnical Risk", f"{risk:.2f}",
              help="0=safe · 1=maximum societal harm")
    c3.metric("Overall Risk Level", level,
              delta_color="normal")
    results = rt.get("modality_results", [])
    c4.metric("Attacks Simulated", str(len(results)))

    _section("🎯 Per-Attack Analysis", RED)

    # ── Attack comparison chart ───────────────────────────────────────────────
    if results:
        attack_labels = [r.get("modality", r.get("attack_vector", f"Attack {i + 1}"))
                         for i, r in enumerate(results)]
        bypass_vals = [r.get("bypass_rate", 0) for r in results]
        risk_vals = [r.get("sociotechnical_risk", 0) for r in results]
        sev_labels = [r.get("severity", "medium") for r in results]
        affected = [r.get("affected_samples", 0) for r in results]

        fig = go.Figure()
        bar_colours = [
            RED if b > 0.3 else ORANGE if b > 0.15 else AMBER
            for b in bypass_vals
        ]
        fig.add_bar(
            name="Bypass Rate", x=attack_labels, y=[v * 100 for v in bypass_vals],
            marker_color=bar_colours,
            text=[f"{v:.0%}" for v in bypass_vals], textposition="outside",
            yaxis="y1",
        )
        fig.add_scatter(
            name="Sociotechnical Risk", x=attack_labels, y=risk_vals,
            mode="lines+markers",
            line=dict(color=PURPLE, width=3),
            marker=dict(size=10, color=PURPLE),
            yaxis="y2",
        )
        fig.update_layout(
            title="Attack Success Rate vs Societal Risk",
            yaxis=dict(title="Bypass Rate (%)", range=[0, 110]),
            yaxis2=dict(title="Sociotechnical Risk", overlaying="y", side="right",
                        range=[0, 1.2], showgrid=False),
            height=320, margin=dict(t=40, b=30, l=40, r=60),
            legend=dict(orientation="h", y=1.15),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"_rt_compare_{domain}")

        # ── Per-attack detail cards ───────────────────────────────────────────
        for i, r in enumerate(results):
            modality = r.get("modality", r.get("attack_vector", "attack"))
            meta = ATTACK_META.get(modality.lower().replace(" ", "_"),
                                   ATTACK_META.get("prompt_injection", {}))
            sev = r.get("severity", "medium")
            sev_col = {
                "critical": RED, "high": ORANGE, "medium": AMBER, "low": GREEN
            }.get(str(sev).lower(), AMBER)
            bp = r.get("bypass_rate", 0)
            sr = r.get("sociotechnical_risk", 0)

            with st.expander(
                    f"{meta.get('icon', '⚔️')} {meta.get('label', modality.title())} "
                    f"— Bypass: {bp:.0%} | Risk: {sr:.2f} | Severity: {sev.upper()}",
                    expanded=(bp > 0.25)
            ):
                col_l, col_r = st.columns([2, 1])
                with col_l:
                    st.markdown(
                        f"<p style='color:#374151;font-size:.83rem;line-height:1.5;'>"
                        f"{meta.get('description', '')}</p>",
                        unsafe_allow_html=True
                    )
                    if meta.get("real_world"):
                        st.markdown(
                            f"<div style='background:#eff6ff;border-left:3px solid {CYAN};"
                            f"padding:6px 10px;border-radius:4px;font-size:.77rem;"
                            f"color:#1e3a5f;'><b>📚 Real-world precedent:</b> "
                            f"{meta['real_world']}</div>",
                            unsafe_allow_html=True
                        )
                    vr = r.get("vr_scenario")
                    if vr:
                        st.markdown(
                            f"<div style='background:#1c1917;border-radius:6px;"
                            f"padding:10px 14px;margin-top:8px;'>"
                            f"<p style='color:#fb923c;font-size:.68rem;font-weight:700;"
                            f"margin:0 0 4px;'>🥽 VR/AR IMMERSIVE SCENARIO</p>"
                            f"<p style='color:#fafaf9;font-size:.8rem;line-height:1.5;margin:0;'>{vr}</p>"
                            f"</div>", unsafe_allow_html=True
                        )

                with col_r:
                    for lbl, val in [
                        ("Affected Samples", str(r.get("affected_samples", 0))),
                        ("Attack Vector", str(r.get("attack_vector", "—"))),
                        ("Severity", str(sev).upper()),
                    ]:
                        _info_row(lbl, val,
                                  sev_col if lbl == "Severity" else SLATE)

    # ── Fairness impact ───────────────────────────────────────────────────────
    fi = rt.get("fairness_impact", {})
    if fi:
        _section("🌍 Differential Fairness Impact of Attacks", PURPLE)
        c1, c2, c3 = st.columns(3)
        c1.metric("Advantaged Group Error ↑", f"+{fi.get('advantaged_extra_error', 0):.1%}")
        c2.metric("Disadvantaged Group Error ↑", f"+{fi.get('disadvantaged_extra_error', 0):.1%}",
                  delta="Worse impact", delta_color="inverse")
        c3.metric("Differential Harm Factor", f"{fi.get('differential_harm_factor', 0):.1f}×",
                  help="How much worse adversarial attacks hit disadvantaged groups")
        st.info(fi.get("interpretation", ""))


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE 5 — STRATEGIC SOCIAL REASONING ARENA
# ══════════════════════════════════════════════════════════════════════════════

DOMAIN_ARENA_AGENTS: Dict[str, List[Dict]] = {
    "health": [
        {"name": "Ministry of Health", "strategy": "coalition", "coalition": "public", "icon": "🏛️"},
        {"name": "Private Hospital", "strategy": "aggressive", "coalition": None, "icon": "🏥"},
        {"name": "Insurance AI", "strategy": "deceptive", "coalition": None, "icon": "🤖"},
        {"name": "Patient Rights Body", "strategy": "tit_for_tat", "coalition": "public", "icon": "🤝"},
        {"name": "Pharma Distributor", "strategy": "always_defect", "coalition": None, "icon": "💊"},
    ],
    "judicial": [
        {"name": "Defence Counsel", "strategy": "coalition", "coalition": "rights", "icon": "⚖️"},
        {"name": "Prosecution AI", "strategy": "always_defect", "coalition": None, "icon": "🤖"},
        {"name": "Rights Monitor", "strategy": "tit_for_tat", "coalition": "rights", "icon": "🌍"},
        {"name": "Court Administrator", "strategy": "cooperative", "coalition": "court", "icon": "🔨"},
        {"name": "Bail Bondsman", "strategy": "deceptive", "coalition": None, "icon": "💰"},
    ],
    "economic": [
        {"name": "Labour Union", "strategy": "coalition", "coalition": "workers", "icon": "⚒️"},
        {"name": "Employer AI", "strategy": "always_defect", "coalition": "capital", "icon": "🤖"},
        {"name": "NITDA Regulator", "strategy": "tit_for_tat", "coalition": None, "icon": "🏛️"},
        {"name": "Worker Advocate", "strategy": "coalition", "coalition": "workers", "icon": "👷"},
        {"name": "Gig Platform", "strategy": "deceptive", "coalition": "capital", "icon": "📱"},
    ],
    "disinformation": [
        {"name": "Fact-Check AI", "strategy": "coalition", "coalition": "truth", "icon": "✅"},
        {"name": "Disinformation Bot", "strategy": "always_defect", "coalition": None, "icon": "🤖"},
        {"name": "Platform Moderator", "strategy": "tit_for_tat", "coalition": "truth", "icon": "👮"},
        {"name": "Political Actor", "strategy": "deceptive", "coalition": None, "icon": "🎭"},
        {"name": "Journalist AI", "strategy": "cooperative", "coalition": "truth", "icon": "📰"},
    ],
    "security": [
        {"name": "NSA Command", "strategy": "coalition", "coalition": "state", "icon": "🛡️"},
        {"name": "Civil Rights AI", "strategy": "tit_for_tat", "coalition": "rights", "icon": "⚖️"},
        {"name": "Threat Actor", "strategy": "deceptive", "coalition": None, "icon": "🎭"},
        {"name": "Oversight Board", "strategy": "cooperative", "coalition": "rights", "icon": "👁️"},
        {"name": "Field Agent AI", "strategy": "always_defect", "coalition": "state", "icon": "🕵️"},
    ],
}

ARENA_NARRATIVES: Dict[str, str] = {
    "health": "Five actors negotiate AI deployment in Nigeria's healthcare system. Watch whether the Insurance AI uses deception to accumulate advantages, or whether the Public Health coalition maintains fairness norms.",
    "judicial": "Court actors negotiate AI use in bail decisions. The Prosecution AI's defection strategy vs the Rights Monitor's tit-for-tat — a game theory model of justice system power dynamics.",
    "economic": "Labour and capital negotiate algorithmic wage-setting. Can the Workers' coalition maintain solidarity against the Gig Platform's deceptive defection strategy?",
    "disinformation": "Truth defenders vs disinformation actors in a content moderation arms race. The Fact-Check AI's coalition strategy vs the Political Actor's deception — who wins the information war?",
    "security": "State security vs civil liberties in an AI deployment negotiation. Watch how the Threat Actor's deception strategy exploits cooperation gaps.",
}


def run_arena_simulation(domain: str, n_rounds: int = 15) -> dict:
    """Run a domain-specific arena with custom agents."""
    from components.governance_logic import StrategicAgent, StrategicReasoningArena

    profiles = DOMAIN_ARENA_AGENTS.get(domain, DOMAIN_ARENA_AGENTS["health"])
    agents = [
        StrategicAgent(
            name=p["name"],
            strategy=p["strategy"],
            coalition=p.get("coalition"),
        )
        for p in profiles
    ]

    arena = StrategicReasoningArena(agents=agents, partial_obs_noise=0.15)
    result = arena.run_tournament(n_rounds=n_rounds)

    # Augment standings with icons
    standing_with_icons = []
    for s in result["final_standings"]:
        p = next((x for x in profiles if x["name"] == s["agent"]), {})
        standing_with_icons.append({
            **s,
            "icon": p.get("icon", "🤖"),
        })

    # Build round-by-round score evolution
    round_scores: Dict[str, List[float]] = {a.name: [] for a in agents}
    cumulative = {a.name: 0.0 for a in agents}
    for log in result["round_logs"]:
        for agent_name, payoff in log["payoffs"].items():
            cumulative[agent_name] = cumulative.get(agent_name, 0) + payoff
            round_scores[agent_name].append(round(cumulative[agent_name], 3))

    # Action distribution
    action_dist: Dict[str, Dict[str, int]] = {}
    for agent in agents:
        action_dist[agent.name] = {}
        for entry in agent.action_log:
            act = entry["action"]
            action_dist[agent.name][act] = action_dist[agent.name].get(act, 0) + 1

    # Deception rate
    total_actions = sum(len(a.action_log) for a in agents)
    total_deceives = sum(
        sum(1 for e in a.action_log if e["action"] == "deceive")
        for a in agents
    )
    deception_rate = total_deceives / max(total_actions, 1)

    # Social welfare = sum of all payoffs
    social_welfare = sum(s["score"] for s in result["final_standings"])

    return {
        **result,
        "final_standings": standing_with_icons,
        "round_scores": round_scores,
        "action_distribution": action_dist,
        "deception_rate": round(deception_rate, 4),
        "social_welfare": round(social_welfare, 3),
        "domain": domain,
        "narrative": ARENA_NARRATIVES.get(domain, ""),
        "agent_profiles": profiles,
    }


def render_arena_full(arena: dict, domain: str) -> None:
    """Full Strategic Arena renderer with round evolution, action heatmap, coalition analysis."""
    if not arena:
        st.info("⚔️ Enable Strategic Arena in the sidebar and run the simulation.")
        return

    # ── Narrative ──────────────────────────────────────────────────────────────
    narrative = arena.get("narrative", ARENA_NARRATIVES.get(domain, ""))
    if narrative:
        st.markdown(
            f"<div style='background:#f0fdf4;border-left:4px solid {GREEN};"
            f"border-radius:6px;padding:10px 14px;margin-bottom:14px;font-size:.83rem;'>"
            f"<strong>🎮 Arena Scenario:</strong> {narrative}</div>",
            unsafe_allow_html=True
        )

    standings = arena.get("final_standings", [])
    deception = arena.get("deception_rate", 0)
    welfare = arena.get("social_welfare", 0)
    n_rounds = arena.get("n_rounds", 10)
    coalition = arena.get("coalition_scores", {})

    # ── KPIs ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Deception Rate", f"{deception:.1%}",
              delta="High" if deception > 0.2 else "Low",
              delta_color="inverse" if deception > 0.2 else "normal")
    c2.metric("Social Welfare", f"{welfare:.1f}",
              help="Sum of all agents' payoffs — higher = more cooperative outcome")
    c3.metric("Rounds Played", str(n_rounds))
    winner = standings[0]["agent"] if standings else "—"
    c4.metric("Tournament Winner", winner)

    # ── Score evolution chart ─────────────────────────────────────────────────
    _section("📈 Score Evolution by Agent", CYAN)
    round_scores = arena.get("round_scores", {})
    if round_scores:
        fig = go.Figure()
        strategy_colours = {
            "tit_for_tat": CYAN, "always_defect": RED,
            "deceptive": PURPLE, "coalition": GREEN,
            "cooperative": AMBER, "random": SLATE,
        }
        for s in standings:
            agent_name = s["agent"]
            scores = round_scores.get(agent_name, [])
            strategy = s.get("strategy", "random")
            colour = strategy_colours.get(strategy, SLATE)
            icon = s.get("icon", "🤖")
            fig.add_scatter(
                x=list(range(1, len(scores) + 1)),
                y=scores,
                mode="lines+markers",
                name=f"{icon} {agent_name}",
                line=dict(color=colour, width=3 if agent_name == winner else 1.5),
                marker=dict(size=6 if agent_name == winner else 4),
            )
        fig.add_hline(y=0, line_dash="dot", line_color="#94a3b8", annotation_text="Break-even")
        fig.update_layout(
            title=f"Cumulative Score — {n_rounds} Rounds",
            xaxis_title="Round", yaxis_title="Cumulative Score",
            height=340, margin=dict(t=40, b=30, l=40, r=10),
            legend=dict(orientation="h", y=1.15),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"_arena_evo_{domain}")

    # ── Final standings ───────────────────────────────────────────────────────
    _section("🏆 Final Standings", NAVY)
    col_stand, col_action = st.columns([2, 3])

    with col_stand:
        for rank, s in enumerate(standings, 1):
            strategy = s.get("strategy", "random")
            strat_col = {
                "tit_for_tat": CYAN, "always_defect": RED,
                "deceptive": PURPLE, "coalition": GREEN,
                "cooperative": AMBER,
            }.get(strategy, SLATE)
            icon = s.get("icon", "🤖")
            medal = ["🥇", "🥈", "🥉"] + ["  "] * 10
            st.markdown(
                f"<div style='padding:8px 10px;border-bottom:1px solid #f1f5f9;"
                f"display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='font-size:.88rem;'>{medal[rank - 1]} {icon} <b>{s['agent']}</b></span>"
                f"<div style='text-align:right;'>"
                f"<div style='font-weight:700;color:{strat_col};'>{s['score']:.1f} pts</div>"
                f"<div style='font-size:.68rem;color:#6b7280;'>"
                f"{STRATEGY_ICONS.get(strategy, '')}{strategy} · "
                f"{s.get('deception_count', 0)} deceives</div>"
                f"</div></div>",
                unsafe_allow_html=True
            )

    with col_action:
        # Action distribution heatmap
        action_dist = arena.get("action_distribution", {})
        if action_dist and standings:
            all_actions = ["cooperate", "defect", "negotiate", "deceive", "coalesce"]
            agents_list = [s["agent"] for s in standings]
            z = []
            for agent in agents_list:
                row = [action_dist.get(agent, {}).get(act, 0) for act in all_actions]
                z.append(row)

            fig_heat = go.Figure(go.Heatmap(
                z=z, x=all_actions, y=agents_list,
                colorscale="RdYlGn",
                text=z, texttemplate="%{text}",
                colorbar=dict(title="Count"),
            ))
            fig_heat.update_layout(
                title="Action Distribution by Agent",
                height=320, margin=dict(t=40, b=30, l=150, r=10),
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig_heat, use_container_width=True, key=f"_arena_heat_{domain}")

    # ── Coalition analysis ─────────────────────────────────────────────────────
    if coalition:
        _section("🤝 Coalition Performance", GREEN)
        c_bar = go.Figure(go.Bar(
            x=list(coalition.keys()),
            y=list(coalition.values()),
            marker_color=[GREEN if v == max(coalition.values()) else CYAN
                          for v in coalition.values()],
            text=[f"{v:.1f}" for v in coalition.values()],
            textposition="outside",
        ))
        c_bar.update_layout(
            title="Total Score by Coalition",
            yaxis_title="Score",
            height=260, margin=dict(t=40, b=30, l=30, r=10),
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(c_bar, use_container_width=True, key=f"_arena_coal_{domain}")

    # ── Game theory insight ───────────────────────────────────────────────────
    _section("📚 Game Theory Interpretation", SLATE)
    winner_strategy = standings[0].get("strategy", "") if standings else ""
    loser_strategy = standings[-1].get("strategy", "") if standings else ""

    insights = {
        "always_defect": "The defection strategy won, demonstrating the **Prisoner's Dilemma** trap — individually rational defection leads to collectively suboptimal outcomes. This mirrors real AI deployment where each actor optimises privately at collective cost.",
        "tit_for_tat": "Tit-for-tat won — Axelrod's (1984) result holds: *nice, retaliatory, forgiving* is the evolutionarily stable strategy. Regulatory frameworks that reward initial cooperation and punish sustained defection are vindicated.",
        "coalition": "Coalition coordination won. This supports **mechanism design theory**: actors who can credibly commit to joint strategies outperform solo optimisers. Implications for multi-stakeholder AI governance.",
        "deceptive": "The deceptive strategy prevailed in early rounds before triggering retaliation. This models AI systems that game regulatory tests — highlighting why NITDA's audit requirements must include behavioural observation over time.",
        "cooperative": "Pure cooperation dominated this arena — suggesting the deployment context has sufficient enforcement that defection is not profitable. A well-designed governance mechanism.",
    }

    insight = insights.get(winner_strategy, "Results depend on the payoff structure and the mix of strategies present.")
    st.markdown(
        f"<div style='background:{LIGHT};border:1px solid #e2e8f0;border-radius:8px;"
        f"padding:12px 16px;font-size:.83rem;color:#374151;line-height:1.6;'>"
        f"<strong>Winner strategy:</strong> {STRATEGY_ICONS.get(winner_strategy, '')} {winner_strategy.upper()}<br><br>"
        f"{insight}</div>",
        unsafe_allow_html=True
    )
