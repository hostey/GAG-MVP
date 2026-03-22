"""
GAGS Charts v1.0
=================
Publication-quality, research + industry grade chart library.
Every chart function returns a Plotly figure ready for st.plotly_chart().

Design principles:
  • Dark terminal theme (matches gags_design.py)
  • Annotation-rich — every chart explains itself
  • Research-grade: confidence intervals, statistical markers, benchmark overlays
  • Industry-grade: KPI glanceable, colour-blind accessible, print-ready
  • No chart is ever "just a bar chart" — every one tells a specific story
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Shared theme ───────────────────────────────────────────────────────────────
BG0 = "#ffffff"
BG1 = "#f8fafc"
BG2 = "#f1f5f9"
BG3 = "#e2e8f0"
T0  = "#0f172a"
T1  = "#334155"
T2  = "#64748b"

PALETTE = ["#2563eb","#ff3b5c","#39ff7a","#f5a623","#c084fc",
           "#00d4aa","#fb923c","#38bdf8","#a78bfa","#fbbf24"]

ALERT_RED   = "#ef4444"
ALERT_AMBER = "#f59e0b"
ALERT_GREEN = "#22c55e"
MONO_FONT   = "DM Mono, monospace"
DISP_FONT   = "Syne, sans-serif"


def _base_layout(accent: str = "#00e5ff", height: int = 400,
                 title: str = "", showlegend: bool = True) -> dict:
    return dict(
        paper_bgcolor=BG1, plot_bgcolor=BG0,
        height=height,
        title=dict(text=title, font=dict(family=DISP_FONT, color=T0, size=13),
                   x=0.01, xanchor="left"),
        font=dict(family=MONO_FONT, color=T1, size=11),
        showlegend=showlegend,
        legend=dict(bgcolor="rgba(15,21,32,.88)", bordercolor=BG3,
                    borderwidth=1, font=dict(size=10)),
        xaxis=dict(gridcolor=BG3, linecolor=BG3,
                   tickcolor=T2, tickfont=dict(color=T2, size=10),
                   title_font=dict(color=T1, size=11)),
        yaxis=dict(gridcolor=BG3, linecolor=BG3,
                   tickcolor=T2, tickfont=dict(color=T2, size=10),
                   title_font=dict(color=T1, size=11)),
        margin=dict(t=48, b=36, l=36, r=16),
        hoverlabel=dict(bgcolor=BG2, bordercolor=BG3,
                        font=dict(family=MONO_FONT, size=11)),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WATERFALL CHART — Feature contribution / SHAP-style
# ═══════════════════════════════════════════════════════════════════════════════

def waterfall_feature_contributions(
    feature_names: List[str],
    contributions: List[float],
    base_value:    float = 0.5,
    instance_label: str = "This individual",
    title: str = "Why did the AI make this decision?",
    accent: str = "#00e5ff",
    height: int = 420,
) -> go.Figure:
    """
    SHAP-style waterfall chart showing how each feature pushed the decision
    up (positive) or down (negative) from the base rate.

    Better than a plain bar chart because it shows cumulative flow.
    """
    # Sort by absolute magnitude
    sorted_idx = np.argsort(np.abs(contributions))[::-1][:10]
    names  = [feature_names[i] for i in sorted_idx]
    contribs = [contributions[i] for i in sorted_idx]

    # Build waterfall: running total
    running = [base_value]
    for c in contribs:
        running.append(running[-1] + c)
    final = running[-1]

    measures = ["absolute"] + ["relative"] * len(contribs) + ["total"]
    x_labels = ["Base rate"] + names + [instance_label]
    y_values = [base_value] + contribs + [final]

    colors = []
    for i, v in enumerate(y_values):
        if i == 0 or i == len(y_values)-1:
            colors.append(accent)
        else:
            colors.append(ALERT_GREEN if v > 0 else ALERT_RED)

    fig = go.Figure(go.Waterfall(
        orientation="h",
        measure=measures,
        x=y_values,
        y=x_labels,
        textposition="outside",
        text=[f"{v:+.3f}" if i not in (0, len(y_values)-1)
              else f"{v:.3f}" for i, v in enumerate(y_values)],
        textfont=dict(family=MONO_FONT, size=10, color=T1),
        connector=dict(line=dict(color=BG3, width=1, dash="dot")),
        increasing=dict(marker_color=ALERT_GREEN),
        decreasing=dict(marker_color=ALERT_RED),
        totals=dict(marker_color=accent),
    ))

    # Threshold line at 0.5
    fig.add_vline(x=0.5, line_dash="dash", line_color=T2,
                  annotation_text="Decision boundary",
                  annotation_font=dict(color=T2, size=9))

    fig.update_layout(**_base_layout(accent, height, title, showlegend=False))
    fig.update_xaxes(range=[max(0, min(y_values)-0.15),
                             min(1, max(y_values)+0.15)],
                     title="Predicted probability")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BENCHMARK COMPARISON CHART
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_comparison_bar(
    simulation_metrics: Dict[str, float],
    benchmark_metrics:  Dict[str, float],
    benchmark_name:     str,
    metric_labels:      Optional[Dict[str, str]] = None,
    accent: str = "#00e5ff",
    height: int = 380,
    lower_is_better: Optional[List[str]] = None,
) -> go.Figure:
    """
    Grouped horizontal bar comparing simulation vs published real-world benchmark.
    Green = simulation better. Red = simulation worse.
    """
    lower_better = set(lower_is_better or ["fpr","gap","exclusion","penalty","error","wrongful"])
    shared = sorted(set(simulation_metrics) & set(benchmark_metrics))
    if not shared:
        fig = go.Figure()
        fig.add_annotation(text="No comparable metrics", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(color=T2))
        fig.update_layout(**_base_layout(accent, height, "Benchmark Comparison"))
        return fig

    labels = [metric_labels.get(m, m.replace("_"," ").title()) if metric_labels else
              m.replace("_"," ").title() for m in shared]
    sim_vals = [simulation_metrics[m] for m in shared]
    bm_vals  = [benchmark_metrics[m]  for m in shared]

    # Colour each simulation bar: green=better, red=worse, grey=comparable
    def _sim_color(m, sv, bv):
        delta = sv - bv
        lib = any(k in m for k in lower_better)
        if abs(delta) < 0.03: return T2
        return ALERT_GREEN if (delta < 0 if lib else delta > 0) else ALERT_RED

    bar_colors = [_sim_color(m, sv, bv) for m, sv, bv in zip(shared, sim_vals, bm_vals)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=f"📚 {benchmark_name[:35]}",
        y=labels, x=bm_vals, orientation="h",
        marker=dict(color=accent, opacity=0.35,
                    line=dict(color=accent, width=1)),
        text=[f"{v:.3f}" for v in bm_vals],
        textposition="outside",
        textfont=dict(color=T2, size=9, family=MONO_FONT),
    ))
    fig.add_trace(go.Bar(
        name="🔬 Your Simulation",
        y=labels, x=sim_vals, orientation="h",
        marker=dict(color=bar_colors, opacity=0.85),
        text=[f"{v:.3f}" for v in sim_vals],
        textposition="outside",
        textfont=dict(color=T0, size=9, family=MONO_FONT),
    ))

    fig.update_layout(**_base_layout(accent, height, "Simulation vs Real-World Benchmark"),
                      barmode="group")
    fig.update_xaxes(range=[0, min(1.05, max(sim_vals + bm_vals) * 1.25)])
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ANIMATED BIAS DRIFT TIMELINE
# ═══════════════════════════════════════════════════════════════════════════════

def animated_bias_drift(
    generation_metrics: List[Dict],
    metric_keys: List[str] = None,
    accent: str = "#00e5ff",
    height: int = 400,
    title: str = "Bias Amplification Across Retraining Cycles",
) -> go.Figure:
    """
    Animated line chart showing how bias evolves across generations.
    Plays automatically — compelling for presentations and research talks.
    """
    if not generation_metrics:
        fig = go.Figure()
        fig.update_layout(**_base_layout(accent, height, title))
        return fig

    keys = metric_keys or [k for k in generation_metrics[0].keys()
                            if k not in ("generation","random_state")]
    gens = [m.get("generation", i) for i, m in enumerate(generation_metrics)]
    colors = [ALERT_RED, ALERT_GREEN, accent, ALERT_AMBER, "#c084fc"]

    # Build frames for animation
    frames = []
    for frame_end in range(1, len(gens)+1):
        frame_data = []
        for ki, key in enumerate(keys[:5]):
            vals = [m.get(key, 0) for m in generation_metrics[:frame_end]]
            frame_data.append(go.Scatter(
                x=gens[:frame_end], y=vals,
                mode="lines+markers",
                name=key.replace("_"," ").title(),
                line=dict(color=colors[ki % len(colors)], width=2),
                marker=dict(size=6, color=colors[ki % len(colors)]),
            ))
        frames.append(go.Frame(data=frame_data, name=str(frame_end)))

    # Initial data (first frame)
    fig = go.Figure(data=frames[0].data if frames else [], frames=frames)

    # Threshold lines
    fig.add_hline(y=0.10, line_dash="dot", line_color=ALERT_AMBER,
                  annotation_text="Moderate bias threshold",
                  annotation_font=dict(color=ALERT_AMBER, size=9))
    fig.add_hline(y=0.20, line_dash="dot", line_color=ALERT_RED,
                  annotation_text="Critical threshold",
                  annotation_font=dict(color=ALERT_RED, size=9))

    fig.update_layout(
        **_base_layout(accent, height, title),
        updatemenus=[dict(
            type="buttons", showactive=False,
            buttons=[
                dict(label="▶ Play", method="animate",
                     args=[None, dict(frame=dict(duration=400, redraw=True),
                                      fromcurrent=True)]),
                dict(label="⏸ Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
            x=0.01, y=1.18, bgcolor=BG2, bordercolor=BG3,
            font=dict(family=MONO_FONT, size=10, color=T1),
        )],
        sliders=[dict(
            steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                               mode="immediate")],
                         method="animate", label=str(f.name))
                   for f in frames],
            currentvalue=dict(prefix="Cycle: ", font=dict(color=T1, family=MONO_FONT)),
            pad=dict(t=10), bgcolor=BG2, bordercolor=BG3,
            tickcolor=T2, font=dict(color=T2, size=9),
        )],
    )
    fig.update_xaxes(title="Retraining Cycle")
    fig.update_yaxes(range=[0, 0.55], title="Metric Value")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SANKEY — Economic / Decision Flow
# ═══════════════════════════════════════════════════════════════════════════════

def sankey_decision_flow(
    flow_data: Dict,
    title: str = "Who Gets What? — AI Decision Flow",
    accent: str = "#00e5ff",
    height: int = 420,
) -> go.Figure:
    """
    Sankey diagram showing how population flows through the AI decision system
    to outcomes, broken down by demographic group.

    flow_data keys:
        nodes: list of node labels
        source: list of source indices
        target: list of target indices
        value:  list of flow sizes
        colors: list of node colors (optional)
    """
    nodes  = flow_data.get("nodes", [])
    source = flow_data.get("source", [])
    target = flow_data.get("target", [])
    value  = flow_data.get("value",  [])
    node_colors = flow_data.get("colors",
        [accent if i < 2 else ALERT_GREEN if i % 3 == 0
         else ALERT_RED if i % 3 == 1 else ALERT_AMBER
         for i in range(len(nodes))])

    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            pad=18, thickness=18,
            line=dict(color=BG3, width=0.5),
            label=nodes,
            color=[
                "rgba({},{},{},0.80)".format(
                    int(c.lstrip("#")[0:2], 16),
                    int(c.lstrip("#")[2:4], 16),
                    int(c.lstrip("#")[4:6], 16),
                )
                for c in node_colors
            ],
            hovertemplate="%{label}: %{value:,.0f}<extra></extra>",
        ),
        link=dict(
            source=source, target=target, value=value,
            color=[
            "rgba({},{},{},0.25)".format(
                int(node_colors[s].lstrip("#")[0:2], 16),
                int(node_colors[s].lstrip("#")[2:4], 16),
                int(node_colors[s].lstrip("#")[4:6], 16),
            )
            for s in source
        ],
            hovertemplate="From %{source.label} → %{target.label}: %{value:,.0f}<extra></extra>",
        ),
    ))
    fig.update_layout(**_base_layout(accent, height, title, showlegend=False))
    return fig


def make_economic_sankey(
    n_total:      int,
    advantaged_pct:  float,
    outcome_rate_adv: float,
    outcome_rate_dis: float,
    advantaged_label: str = "Formal / Majority",
    disadvantaged_label: str = "Informal / Minority",
    positive_label: str = "Positive Outcome",
    negative_label: str = "Negative Outcome",
    accent: str = "#00e5ff",
    title: str = "AI Decision Flow by Group",
) -> go.Figure:
    """Pre-built Sankey for economic AI outcomes."""
    n_adv = int(n_total * advantaged_pct)
    n_dis = n_total - n_adv
    adv_pos = int(n_adv * outcome_rate_adv)
    adv_neg = n_adv - adv_pos
    dis_pos = int(n_dis * outcome_rate_dis)
    dis_neg = n_dis - dis_pos

    return sankey_decision_flow({
        "nodes":  ["Total Population", advantaged_label, disadvantaged_label,
                   positive_label + " (Adv)", negative_label + " (Adv)",
                   positive_label + " (Dis)", negative_label + " (Dis)"],
        "source": [0, 0, 1, 1, 2, 2],
        "target": [1, 2, 3, 4, 5, 6],
        "value":  [n_adv, n_dis, adv_pos, adv_neg, dis_pos, dis_neg],
        "colors": [accent, accent, ALERT_AMBER,
                   ALERT_GREEN, ALERT_RED, ALERT_GREEN, ALERT_RED],
    }, title=title, accent=accent, height=380)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MULTI-METRIC RADAR WITH CONFIDENCE BANDS
# ═══════════════════════════════════════════════════════════════════════════════

def radar_with_benchmark(
    metrics: Dict[str, float],
    benchmark: Optional[Dict[str, float]] = None,
    benchmark_label: str = "Real-World Benchmark",
    simulation_label: str = "Your Simulation",
    accent: str = "#00e5ff",
    height: int = 380,
    title: str = "Fairness Radar",
) -> go.Figure:
    """Radar chart with optional benchmark overlay and threshold ring."""
    keys   = list(metrics.keys())
    labels = [k.replace("_"," ").title() for k in keys]
    vals   = [metrics[k] for k in keys]

    fig = go.Figure()

    # Threshold ring at 0.70
    theta_full = labels + [labels[0]]
    fig.add_trace(go.Scatterpolar(
        r=[0.70]*len(theta_full), theta=theta_full,
        fill="toself",
        fillcolor="rgba(57,255,122,0.04)",
        line=dict(color=ALERT_GREEN, width=1, dash="dot"),
        name="Minimum threshold (0.70)",
        hoverinfo="skip",
    ))

    # Benchmark
    if benchmark:
        bm_vals = [benchmark.get(k, 0) for k in keys]
        h = accent.lstrip("#")
        fc = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.06)"
        fig.add_trace(go.Scatterpolar(
            r=bm_vals + [bm_vals[0]], theta=theta_full,
            fill="toself", fillcolor=fc,
            line=dict(color=accent, width=1, dash="dash"),
            name=benchmark_label,
        ))

    # Simulation
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=theta_full,
        fill="toself",
        fillcolor="rgba(57,255,122,0.10)",
        line=dict(color=ALERT_GREEN, width=2.5),
        marker=dict(size=7, color=ALERT_GREEN),
        name=simulation_label,
    ))

    fig.update_layout(
        paper_bgcolor=BG1, font=dict(family=MONO_FONT, color=T1, size=10),
        height=height,
        title=dict(text=title, font=dict(family=DISP_FONT, color=T0, size=13),
                   x=0.01, xanchor="left"),
        polar=dict(
            bgcolor=BG0,
            angularaxis=dict(linecolor=BG3, gridcolor=BG3,
                             tickfont=dict(color=T1, size=10, family=MONO_FONT)),
            radialaxis=dict(visible=True, range=[0,1], gridcolor=BG3,
                            linecolor=BG3, tickfont=dict(color=T2, size=8)),
        ),
        legend=dict(bgcolor="rgba(15,21,32,.9)", bordercolor=BG3,
                    borderwidth=1, font=dict(size=10)),
        margin=dict(t=48, b=20, l=40, r=40),
        hoverlabel=dict(bgcolor=BG2, bordercolor=BG3,
                        font=dict(family=MONO_FONT, size=11)),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HEATMAP — Cross-domain or cross-group fairness
# ═══════════════════════════════════════════════════════════════════════════════

def fairness_heatmap(
    data: pd.DataFrame,
    title: str = "Fairness Metric Heatmap",
    accent: str = "#00e5ff",
    height: int = 380,
    colorscale: str = "RdYlGn",
    zmin: float = 0.0,
    zmax: float = 1.0,
    annotate: bool = True,
) -> go.Figure:
    """
    Heatmap of fairness metrics × groups/runs.
    Green = good. Red = bad. Tells the whole story at a glance.
    """
    fig = go.Figure(go.Heatmap(
        z=data.values,
        x=list(data.columns),
        y=list(data.index),
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        text=data.applymap(lambda v: f"{v:.3f}").values if annotate else None,
        texttemplate="%{text}" if annotate else "",
        textfont=dict(size=10, family=MONO_FONT, color=BG0),
        hovertemplate="Group: %{y}<br>Metric: %{x}<br>Value: %{z:.3f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Score", font=dict(color=T1, size=10)),
            tickfont=dict(color=T2, size=9, family=MONO_FONT),
            bgcolor=BG1, bordercolor=BG3,
        ),
    ))
    fig.update_layout(**_base_layout(accent, height, title, showlegend=False))
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=9))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CONFIDENCE INTERVAL / MULTI-RUN DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

def multi_run_distribution(
    run_results:  List[Dict],
    metrics:      List[str],
    metric_labels: Optional[Dict[str,str]] = None,
    accent: str = "#00e5ff",
    height: int = 400,
    title: str = "Statistical Distribution Across Simulation Runs",
) -> go.Figure:
    """
    Box + violin plot per metric across multiple runs.
    Shows median, IQR, outliers — research-grade statistical display.
    """
    if not run_results:
        fig = go.Figure()
        fig.update_layout(**_base_layout(accent, height, title))
        return fig

    df = pd.DataFrame(run_results)
    avail = [m for m in metrics if m in df.columns]
    colors = [accent, ALERT_RED, ALERT_GREEN, ALERT_AMBER, "#c084fc",
              "#00d4aa", "#fb923c"]

    fig = go.Figure()
    for ki, metric in enumerate(avail):
        vals = df[metric].dropna().tolist()
        label = (metric_labels or {}).get(metric, metric.replace("_"," ").title())
        color = colors[ki % len(colors)]
        h = color.lstrip("#")
        fc = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.25)"

        fig.add_trace(go.Violin(
            y=vals, x=[label]*len(vals),
            name=label,
            box_visible=True,
            meanline_visible=True,
            fillcolor=fc,
            line_color=color,
            marker=dict(color=color, size=5, opacity=0.6),
        ))

    fig.update_layout(**_base_layout(accent, height, title))
    fig.update_yaxes(range=[0, 1.05], title="Value")
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 8. LOLLIPOP CHART — Gap analysis (cleaner than bar charts for gaps)
# ═══════════════════════════════════════════════════════════════════════════════

def lollipop_gap_chart(
    groups:      List[str],
    values_a:    List[float],
    values_b:    List[float],
    label_a:     str = "Advantaged",
    label_b:     str = "Disadvantaged",
    metric_name: str = "Outcome Rate",
    accent:      str = "#00e5ff",
    height:      int = 380,
    threshold:   Optional[float] = None,
) -> go.Figure:
    """
    Lollipop chart showing two groups' performance side by side per category.
    The connecting line visually encodes the gap magnitude.
    Far more readable than grouped bars for gap analysis.
    """
    fig = go.Figure()

    for i, (grp, va, vb) in enumerate(zip(groups, values_a, values_b)):
        # Connecting line
        fig.add_trace(go.Scatter(
            x=[va, vb], y=[grp, grp],
            mode="lines",
            line=dict(color=BG3, width=3),
            showlegend=False, hoverinfo="skip",
        ))
        # Gap fill colour
        gap_color = ALERT_RED if abs(va-vb) > 0.15 else ALERT_AMBER if abs(va-vb) > 0.08 else ALERT_GREEN

        # Dots
        for val, label, color, sym in [(va, label_a, accent, "circle"),
                                        (vb, label_b, gap_color, "diamond")]:
            fig.add_trace(go.Scatter(
                x=[val], y=[grp],
                mode="markers+text",
                name=label if i == 0 else None,
                showlegend=(i == 0),
                marker=dict(color=color, size=14, symbol=sym,
                            line=dict(color=BG0, width=1.5)),
                text=[f"{val:.1%}"],
                textposition="middle right" if val == max(va, vb) else "middle left",
                textfont=dict(color=T1, size=9, family=MONO_FONT),
                hovertemplate=f"{label}: {val:.3f}<extra></extra>",
            ))

    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dot", line_color=ALERT_GREEN,
                      annotation_text=f"Threshold {threshold:.0%}",
                      annotation_font=dict(color=ALERT_GREEN, size=9))

    layout = _base_layout(accent, height,
                          f"{metric_name} — {label_a} vs {label_b}")
    layout["xaxis"].update(range=[0, 1.05], tickformat=".0%",
                            title=metric_name)
    layout["yaxis"].update(title="")
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 9. REAL-WORLD TIMELINE — Historical AI bias incidents
# ═══════════════════════════════════════════════════════════════════════════════

def ai_bias_incident_timeline(
    accent: str = "#00e5ff",
    height: int = 400,
    domain_filter: Optional[str] = None,
) -> go.Figure:
    """
    Interactive timeline of landmark AI bias incidents.
    Click any point to get full context. Ideal for research and teaching.
    """
    INCIDENTS = [
        (2014, "hiring",   "Amazon CV AI", "Amazon's CV screening AI penalises women's colleges.",             ALERT_RED),
        (2016, "judicial", "COMPAS",        "ProPublica: Black defendants 2× mislabelled high-risk.",          ALERT_RED),
        (2018, "hiring",   "HireVue Gender","Video interview AI penalises female speech patterns.",             ALERT_AMBER),
        (2019, "healthcare","Optum/Epic",   "Healthcare AI underestimates illness in Black patients.",          ALERT_RED),
        (2019, "finance",  "Apple Card",    "Women receive lower credit limits than men with same income.",     ALERT_AMBER),
        (2019, "policy",   "Robodebt AUS",  "Australia: 470,000 unlawful welfare debt notices generated.",     ALERT_RED),
        (2020, "policy",   "SyRI NL",       "Dutch court bans SyRI welfare AI — violates human rights.",       ALERT_RED),
        (2020, "education","UK A-Levels",   "Ofqual algorithm deflates 39% of state school grades.",           ALERT_RED),
        (2021, "gig",      "Uber Race Gap", "NBER: Black Uber drivers earn 10% less due to rating bias.",      ALERT_AMBER),
        (2022, "education","JAMB Nigeria",  "22pp gap between private/state school UTME scores in Nigeria.",   ALERT_AMBER),
        (2022, "hiring",   "LinkedIn Bias", "LinkedIn job ads shown to different genders algorithmically.",    ALERT_AMBER),
        (2023, "finance",  "FinScope NGA",  "61% of Nigerian informal workers excluded from credit AI.",       ALERT_RED),
        (2023, "policy",   "Robodebt Royal","Royal Commission confirms Robodebt unlawful — AUD 1.8B remediation.",ALERT_RED),
        (2023, "gig",      "Bolt Africa",   "Fairwork: 58% of Bolt Africa drivers earn below minimum wage.",   ALERT_RED),
        (2024, "judicial", "EU AI Act",     "EU classifies recidivism AI as Prohibited — Article 5.",          ALERT_GREEN),
    ]

    if domain_filter:
        INCIDENTS = [i for i in INCIDENTS if i[1] == domain_filter]

    domain_colors = {
        "hiring": "#f5a623", "judicial": "#c084fc", "healthcare": "#00e5ff",
        "finance": "#00d4aa", "policy": "#ff3b5c", "education": "#fb923c",
        "gig": "#38bdf8",
    }

    fig = go.Figure()
    domains_done = set()
    for year, domain, name, text, _ in sorted(INCIDENTS):
        color = domain_colors.get(domain, accent)
        show_leg = domain not in domains_done
        domains_done.add(domain)
        fig.add_trace(go.Scatter(
            x=[year], y=[domain.title()],
            mode="markers+text",
            name=domain.title() if show_leg else None,
            showlegend=show_leg,
            marker=dict(size=16, color=color, symbol="circle",
                        line=dict(color=BG0, width=2)),
            text=[name],
            textposition="top center",
            textfont=dict(size=8, color=T1, family=MONO_FONT),
            customdata=[[text, year]],
            hovertemplate="<b>%{text}</b> (%{x})<br>%{customdata[0]}<extra></extra>",
        ))

    layout = _base_layout(accent, height, "Landmark AI Bias Incidents — 2014–2024")
    layout["xaxis"].update(title="Year", dtick=1, range=[2013.5, 2024.5],
                            gridcolor=BG3, tickfont=dict(size=10))
    layout["yaxis"].update(title="", tickfont=dict(size=10))
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 10. STACKED AREA — Cumulative impact / longitudinal
# ═══════════════════════════════════════════════════════════════════════════════

def stacked_area_groups(
    generations:   List[int],
    group_data:    Dict[str, List[float]],
    title:         str = "Outcome Distribution Across Retraining Cycles",
    accent:        str = "#00e5ff",
    height:        int = 380,
    yaxis_title:   str = "Outcome Rate",
) -> go.Figure:
    """
    Stacked area chart showing how group outcomes evolve over time.
    Reveals divergence and convergence patterns in longitudinal bias analysis.
    """
    colors = [accent, ALERT_RED, ALERT_GREEN, ALERT_AMBER, "#c084fc"]
    fig = go.Figure()
    for ki, (group, vals) in enumerate(group_data.items()):
        color = colors[ki % len(colors)]
        h = color.lstrip("#")
        fc = f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},0.25)"
        fig.add_trace(go.Scatter(
            x=generations, y=vals,
            name=group,
            fill="tonexty" if ki > 0 else "tozeroy",
            mode="lines",
            line=dict(color=color, width=2),
            fillcolor=fc,
            hovertemplate=f"{group}: %{{y:.3f}}<extra></extra>",
        ))
    layout = _base_layout(accent, height, title)
    layout["yaxis"].update(title=yaxis_title, range=[0, 1.05])
    layout["xaxis"].update(title="Retraining Generation")
    fig.update_layout(**layout)
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 11. GAUGE CLUSTER — Executive KPI dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def gauge_cluster(
    metrics: List[Tuple[str, float, float, str]],
    # Each tuple: (label, value, threshold, bar_color)
    height: int = 280,
    cols:   int = 4,
) -> go.Figure:
    """
    Production-grade gauge cluster. Better than Streamlit st.metric for dashboards.
    threshold = red line position.
    """
    n = len(metrics)
    fig = make_subplots(
        rows=1, cols=n,
        specs=[[{"type":"indicator"}]*n],
        subplot_titles=[m[0] for m in metrics],
    )
    for ci, (label, value, threshold, color) in enumerate(metrics, 1):
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=value * 100,
            number=dict(suffix="%", font=dict(family=DISP_FONT, size=22, color=color)),
            delta=dict(reference=threshold*100, increasing=dict(color=ALERT_GREEN),
                       decreasing=dict(color=ALERT_RED),
                       font=dict(family=MONO_FONT, size=11)),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(color=T2, size=8)),
                bar=dict(color=color, thickness=0.28),
                bgcolor=BG2,
                bordercolor=BG3,
                steps=[
                    dict(range=[0, threshold*100], color=BG2),
                    dict(range=[threshold*100, 100], color=BG3),
                ],
                threshold=dict(
                    line=dict(color=ALERT_RED, width=3),
                    thickness=0.85,
                    value=threshold * 100,
                ),
            ),
        ), row=1, col=ci)

    fig.update_layout(
        paper_bgcolor=BG1,
        font=dict(family=MONO_FONT, color=T1, size=10),
        height=height,
        showlegend=False,
        margin=dict(t=48, b=8, l=8, r=8),
        hoverlabel=dict(bgcolor=BG2, bordercolor=BG3,
                        font=dict(family=MONO_FONT, size=11)),
    )
    return fig