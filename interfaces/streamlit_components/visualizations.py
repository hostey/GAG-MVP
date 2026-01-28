# interfaces/streamlit_components/visualizations.py
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, List, Any, Optional
import streamlit as st

from config.constants import COLOR_SCHEMES, THRESHOLDS


def create_metric_cards(result, metric_key: str, title: str, icon: str) -> None:
    """Create professional metric cards."""
    value = getattr(result, metric_key, 0)

    # Determine status based on thresholds
    if 'accuracy' in title.lower() or 'safety' in title.lower():
        thresholds = THRESHOLDS['accuracy']
    elif 'fairness' in title.lower():
        thresholds = THRESHOLDS['fairness']
    else:
        thresholds = {'excellent': 90, 'good': 80, 'fair': 70, 'poor': 60}

    # Determine color and status
    if value >= thresholds['excellent']:
        color = "success"
        status = "Excellent"
    elif value >= thresholds['good']:
        color = "success"
        status = "Good"
    elif value >= thresholds['fair']:
        color = "warning"
        status = "Fair"
    else:
        color = "danger"
        status = "Needs Improvement"

    st.markdown(f"""
        <div class="metric-card {color}">
            <div style="font-size: 1.8rem; margin-bottom: 12px;">{icon}</div>
            <div style="font-size: 0.85rem; color: #666; text-transform: uppercase; 
                       letter-spacing: 0.5px; margin-bottom: 8px;">
                {title}
            </div>
            <div style="font-size: 2.4rem; font-weight: 800; color: #333; margin: 12px 0;">
                {value:.1f}{'%' if '%' not in title else ''}
            </div>
            <div style="font-size: 0.8rem; color: #777; font-weight: 600;">
                {status}
            </div>
        </div>
    """, unsafe_allow_html=True)


def create_performance_radar(result) -> go.Figure:
    """Create radar chart for performance metrics."""
    categories = ['Safety', 'Liberty', 'Resilience', 'Fairness', 'Accuracy', 'Stability']

    # Get values from result
    values = [
        result.safety_score,
        result.liberty_score,
        result.resilience_score,
        result.fairness_score,
        result.metrics.get('accuracy', 0) * 100,
        85.0  # Placeholder for stability
    ]

    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(30, 136, 229, 0.2)',
        line=dict(color='#1E88E5', width=2),
        name='Current'
    ))

    # Add optimal region (target)
    target_values = [90] * len(categories)
    fig.add_trace(go.Scatterpolar(
        r=target_values + [target_values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor='rgba(76, 175, 80, 0.1)',
        line=dict(color='#4CAF50', width=1, dash='dash'),
        name='Target'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                rotation=90
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=450,
        margin=dict(t=50, b=50, l=50, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig


def create_trend_chart(data: Dict[str, List[float]], title: str = "Training History") -> go.Figure:
    """Create trend chart for training history."""
    fig = go.Figure()

    colors = ['#1E88E5', '#FF9800', '#4CAF50', '#F44336']

    for i, (key, values) in enumerate(data.items()):
        if values:
            fig.add_trace(go.Scatter(
                y=values,
                mode='lines',
                name=key.replace('_', ' ').title(),
                line=dict(color=colors[i % len(colors)], width=2.5),
                hovertemplate='%{y:.4f}<extra></extra>'
            ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        xaxis_title="Epoch",
        yaxis_title="Value",
        height=400,
        hovermode='x unified',
        margin=dict(t=50, b=50, l=50, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            zerolinecolor='rgba(0,0,0,0.1)'
        ),
        yaxis=dict(
            gridcolor='rgba(0,0,0,0.1)',
            zerolinecolor='rgba(0,0,0,0.1)'
        )
    )

    return fig


def create_confusion_matrix(cm: np.ndarray, title: str = "Confusion Matrix") -> go.Figure:
    """Create confusion matrix heatmap."""
    labels = ['Negative', 'Positive']

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=labels,
        y=labels,
        aspect='auto'
    )

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        height=400,
        margin=dict(t=50, b=50, l=50, r=50),
        coloraxis_showscale=True
    )

    fig.update_traces(
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
    )

    return fig


def create_feature_importance(feature_importance: Dict[str, float],
                              title: str = "Feature Importance") -> go.Figure:
    """Create horizontal bar chart for feature importance."""
    if not feature_importance:
        return go.Figure()

    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    # Sort by importance
    sorted_idx = np.argsort(importance)
    features = [features[i] for i in sorted_idx]
    importance = [importance[i] for i in sorted_idx]

    fig = go.Figure(data=go.Bar(
        x=importance,
        y=features,
        orientation='h',
        marker=dict(
            color=importance,
            colorscale='Blues',
            showscale=True,
            colorbar=dict(title="Importance")
        ),
        hovertemplate='%{y}: %{x:.4f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=16)
        ),
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=max(300, len(features) * 25),
        margin=dict(t=50, b=50, l=50, r=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            automargin=True,
            tickfont=dict(size=10)
        )
    )

    return fig


def create_comparison_chart(results: List[Dict[str, Any]],
                            metric: str = 'safety_score') -> go.Figure:
    """Create comparison chart for multiple results."""
    if len(results) < 2:
        return go.Figure()

    labels = [f"Run {i + 1}" for i in range(len(results))]
    values = [getattr(r, metric, 0) for r in results]

    fig = go.Figure(data=go.Bar(
        x=labels,
        y=values,
        marker=dict(
            color=values,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Score")
        ),
        text=[f"{v:.1f}" for v in values],
        textposition='auto',
        hovertemplate='%{x}: %{y:.1f}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text=f"{metric.replace('_', ' ').title()} Comparison",
            font=dict(size=16)
        ),
        xaxis_title="Simulation Run",
        yaxis_title="Score",
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig


def create_gauge_chart(value: float, title: str, min_val: float = 0, max_val: float = 100) -> go.Figure:
    """Create a gauge/dial chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        gauge={
            'axis': {'range': [min_val, max_val]},
            'bar': {'color': "#1E88E5"},
            'steps': [
                {'range': [min_val, max_val * 0.6], 'color': "lightgray"},
                {'range': [max_val * 0.6, max_val * 0.8], 'color': "gray"},
                {'range': [max_val * 0.8, max_val], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_val * 0.9
            }
        }
    ))

    fig.update_layout(
        height=300,
        margin=dict(t=50, b=50, l=50, r=50),
        paper_bgcolor='rgba(0,0,0,0)'
    )

    return fig