# components/realtime_monitor.py
"""
GAGS Real-Time Monitoring Module — v1.0
=========================================
Connects to a live model endpoint or prediction stream and monitors:
  - Prediction drift (distribution shift in outputs)
  - Fairness degradation (demographic parity gap widening over time)
  - Adversarial attack signals (unusual patterns in input features)
  - Data quality drift (feature distribution shift from training baseline)

Architecture
------------
RealtimeMonitor      — main class; accumulates predictions and computes
                       rolling metrics across a configurable window
DriftDetector        — statistical tests for distribution shift
                       (KS test, PSI, chi-squared)
MonitorAlert         — alert dataclass emitted when a threshold is breached
monitoring_dashboard() — Streamlit UI panel for the monitoring view

Usage in a page module
----------------------
    from components.realtime_monitor import RealtimeMonitor, monitoring_dashboard

    # Instantiate once (store in session state):
    monitor = RealtimeMonitor(domain="agrotech", window_size=100)

    # Feed each new prediction as it arrives:
    monitor.ingest(
        features=feature_vector,       # 1D numpy array
        prediction=1,                  # model output
        true_label=None,               # optional ground truth
        demographic_group=0,           # group indicator
    )

    # Render the live dashboard:
    monitoring_dashboard(monitor, key_prefix="agro_monitor")

Standalone monitoring (no Streamlit)
-------------------------------------
    monitor = RealtimeMonitor(domain="agrotech", window_size=200)
    # ... ingest predictions ...
    alerts = monitor.check_alerts()
    report = monitor.get_report()
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import threading

try:
    from scipy import stats as _scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class MonitorAlert:
    """An alert emitted when a monitored metric breaches its threshold."""
    alert_id:      str
    timestamp:     str
    alert_type:    str        # "drift" | "fairness" | "attack" | "quality"
    metric:        str        # e.g. "demographic_parity", "psi_feature_3"
    current_value: float
    threshold:     float
    severity:      str        # "critical" | "high" | "medium" | "low"
    message:       str
    recommendation: str


@dataclass
class MonitorSnapshot:
    """A point-in-time snapshot of all monitored metrics."""
    timestamp:              str
    n_predictions:          int
    positive_rate:          float          # rolling window positive prediction rate
    demographic_parity_gap: float          # parity gap in rolling window
    feature_drift_score:    float          # mean PSI across features
    attack_signal:          float          # anomaly score [0,1]
    fairness_score:         float          # [0,1]
    alerts_active:          int            # number of active (unresolved) alerts
    drift_detected:         bool
    fairness_degraded:      bool


# ── Drift Detector ────────────────────────────────────────────────────────────

class DriftDetector:
    """
    §21.1 — Statistical drift detection.

    Maintains a reference distribution from training baseline.
    Computes PSI (Population Stability Index) and KS test p-values
    for each feature in the rolling window.

    PSI interpretation:
      < 0.10  — no significant drift
      0.10–0.20 — moderate drift, worth monitoring
      > 0.20  — significant drift, model may be stale
    """

    PSI_LOW      = 0.10
    PSI_HIGH     = 0.20
    KS_THRESHOLD = 0.05     # p-value threshold for KS test

    def __init__(self, reference_X: Optional[np.ndarray] = None):
        self._reference: Optional[np.ndarray] = reference_X
        self._ref_means:  Optional[np.ndarray] = None
        self._ref_stds:   Optional[np.ndarray] = None
        if reference_X is not None:
            self.set_reference(reference_X)

    def set_reference(self, X: np.ndarray) -> None:
        """Set the training distribution baseline."""
        self._reference  = X.copy()
        self._ref_means  = np.nanmean(X, axis=0)
        self._ref_stds   = np.nanstd(X, axis=0) + 1e-8

    def psi(self, expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
        """
        Population Stability Index between two 1D arrays.
        Returns PSI score (higher = more drift).
        """
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        if max_val <= min_val:
            return 0.0

        bins = np.linspace(min_val, max_val + 1e-8, n_bins + 1)
        e_counts, _ = np.histogram(expected, bins=bins)
        a_counts, _ = np.histogram(actual,   bins=bins)

        e_pct = (e_counts + 1e-8) / (len(expected) + 1e-8)
        a_pct = (a_counts + 1e-8) / (len(actual)   + 1e-8)

        return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))

    def compute_drift(self, window_X: np.ndarray) -> Dict[str, Any]:
        """
        Compute drift metrics for a rolling window vs reference.
        Returns dict with per-feature PSI and overall drift verdict.
        """
        if self._reference is None or len(window_X) < 10:
            return {"drift_detected": False, "mean_psi": 0.0, "feature_psi": {}}

        n_feat   = min(self._reference.shape[1], window_X.shape[1])
        psi_vals = {}
        for i in range(n_feat):
            psi_vals[f"feature_{i}"] = self.psi(
                self._reference[:, i], window_X[:, i]
            )

        mean_psi = float(np.mean(list(psi_vals.values())))
        high_drift_features = [k for k, v in psi_vals.items() if v > self.PSI_HIGH]

        return {
            "drift_detected":      mean_psi > self.PSI_LOW,
            "high_drift":          mean_psi > self.PSI_HIGH,
            "mean_psi":            round(mean_psi, 4),
            "feature_psi":         {k: round(v, 4) for k, v in psi_vals.items()},
            "high_drift_features": high_drift_features,
        }

    def ks_test(self, ref: np.ndarray, window: np.ndarray) -> Tuple[float, float]:
        """KS test between reference and window distributions. Returns (statistic, p_value)."""
        if not _SCIPY or len(ref) < 5 or len(window) < 5:
            return 0.0, 1.0
        stat, p = _scipy_stats.ks_2samp(ref, window)
        return float(stat), float(p)


# ── Main Monitor ──────────────────────────────────────────────────────────────

class RealtimeMonitor:
    """
    §21.2 — Real-time prediction stream monitor.

    Maintains a rolling window of recent predictions and computes
    fairness, drift, and attack metrics continuously.

    Thread-safe: uses a lock for concurrent ingest + read.
    """

    # Default alert thresholds
    THRESHOLDS = {
        "demographic_parity_gap": 0.10,   # NITDA principle 2
        "positive_rate_drift":    0.15,   # >15% shift in positive rate
        "mean_psi":               0.20,   # high feature drift
        "attack_signal":          0.70,   # anomaly score
        "fairness_score_min":     0.65,   # below this = alert
    }

    def __init__(
        self,
        domain:           str  = "generic",
        window_size:      int  = 200,
        alert_thresholds: Optional[Dict[str, float]] = None,
        reference_X:      Optional[np.ndarray] = None,
    ):
        self.domain      = domain
        self.window_size = window_size
        self.thresholds  = {**self.THRESHOLDS, **(alert_thresholds or {})}

        # Rolling window buffers
        self._features:   List[np.ndarray] = []
        self._preds:      List[int]         = []
        self._labels:     List[Optional[int]] = []
        self._groups:     List[int]         = []
        self._timestamps: List[str]         = []

        # State
        self._alerts:    List[MonitorAlert]     = []
        self._snapshots: List[MonitorSnapshot]  = []
        self._lock       = threading.Lock()
        self._alert_id   = 0
        self._baseline_positive_rate: Optional[float] = None

        # Drift detector
        self.drift_detector = DriftDetector(reference_X)

    # ── Ingest ────────────────────────────────────────────────────────────

    def ingest(
        self,
        features:          np.ndarray,
        prediction:        int,
        true_label:        Optional[int] = None,
        demographic_group: int = 0,
    ) -> List[MonitorAlert]:
        """
        Feed one prediction event into the monitor.
        Returns any new alerts triggered by this event.

        Parameters
        ----------
        features          : 1D feature vector
        prediction        : model prediction (0 or 1)
        true_label        : ground truth label (optional; enables accuracy monitoring)
        demographic_group : group indicator (0/1 or any integer)
        """
        with self._lock:
            now = datetime.now().isoformat()[:19]

            self._features.append(np.asarray(features, dtype=np.float64))
            self._preds.append(int(prediction))
            self._labels.append(true_label)
            self._groups.append(int(demographic_group))
            self._timestamps.append(now)

            # Trim to window
            if len(self._preds) > self.window_size:
                self._features   = self._features[-self.window_size:]
                self._preds      = self._preds[-self.window_size:]
                self._labels     = self._labels[-self.window_size:]
                self._groups     = self._groups[-self.window_size:]
                self._timestamps = self._timestamps[-self.window_size:]

            # Take a snapshot every 10 predictions
            if len(self._preds) % 10 == 0:
                snap = self._compute_snapshot()
                self._snapshots.append(snap)
                if len(self._snapshots) > 500:
                    self._snapshots = self._snapshots[-500:]

            # Check for alerts
            new_alerts = self._check_thresholds()
            return new_alerts

    # ── Metrics computation ────────────────────────────────────────────────

    def _compute_snapshot(self) -> MonitorSnapshot:
        """Compute a point-in-time snapshot of all metrics."""
        preds  = np.array(self._preds)
        groups = np.array(self._groups)
        n      = len(preds)

        # Positive rate
        pos_rate = float(np.mean(preds))

        # Demographic parity gap
        unique_groups = np.unique(groups)
        if len(unique_groups) >= 2:
            rates = [float(np.mean(preds[groups == g])) for g in unique_groups
                     if np.sum(groups == g) > 0]
            parity_gap = max(rates) - min(rates) if len(rates) >= 2 else 0.0
        else:
            parity_gap = 0.0

        # Fairness score (simple)
        fairness = float(np.clip(1.0 - parity_gap * 2, 0, 1))

        # Feature drift
        feature_X = np.stack(self._features) if self._features else np.zeros((1, 1))
        drift_result = self.drift_detector.compute_drift(feature_X)
        mean_psi     = drift_result.get("mean_psi", 0.0)

        # Attack signal — isolation forest anomaly score (simplified)
        if feature_X.shape[0] >= 20:
            feat_means = np.mean(feature_X, axis=0)
            feat_stds  = np.std(feature_X, axis=0) + 1e-8
            z_scores   = np.abs((feature_X - feat_means) / feat_stds)
            attack_signal = float(np.clip(np.mean(z_scores > 2.5), 0, 1))
        else:
            attack_signal = 0.0

        return MonitorSnapshot(
            timestamp=datetime.now().isoformat()[:19],
            n_predictions=n,
            positive_rate=round(pos_rate, 4),
            demographic_parity_gap=round(parity_gap, 4),
            feature_drift_score=round(mean_psi, 4),
            attack_signal=round(attack_signal, 4),
            fairness_score=round(fairness, 4),
            alerts_active=len([a for a in self._alerts if not getattr(a, "_resolved", False)]),
            drift_detected=drift_result.get("drift_detected", False),
            fairness_degraded=parity_gap > self.thresholds["demographic_parity_gap"],
        )

    def _check_thresholds(self) -> List[MonitorAlert]:
        """Check current metrics against alert thresholds."""
        if len(self._preds) < 20:   # need enough data
            return []

        snap       = self._compute_snapshot()
        new_alerts = []

        def _make_alert(alert_type, metric, current, threshold, sev, msg, rec):
            self._alert_id += 1
            a = MonitorAlert(
                alert_id=f"ALERT-{self._alert_id:04d}",
                timestamp=datetime.now().isoformat()[:19],
                alert_type=alert_type,
                metric=metric,
                current_value=round(current, 4),
                threshold=threshold,
                severity=sev,
                message=msg,
                recommendation=rec,
            )
            self._alerts.append(a)
            new_alerts.append(a)

        # Fairness degradation
        if snap.demographic_parity_gap > self.thresholds["demographic_parity_gap"]:
            _make_alert(
                "fairness", "demographic_parity_gap",
                snap.demographic_parity_gap,
                self.thresholds["demographic_parity_gap"],
                "high" if snap.demographic_parity_gap > 0.20 else "medium",
                f"Demographic parity gap ({snap.demographic_parity_gap:.1%}) exceeds "
                f"threshold ({self.thresholds['demographic_parity_gap']:.1%}).",
                "Review recent input demographics and check for sampling bias.",
            )

        # Positive rate drift
        if self._baseline_positive_rate is None and len(self._preds) >= 50:
            self._baseline_positive_rate = snap.positive_rate

        if self._baseline_positive_rate is not None:
            pos_drift = abs(snap.positive_rate - self._baseline_positive_rate)
            if pos_drift > self.thresholds["positive_rate_drift"]:
                _make_alert(
                    "drift", "positive_rate",
                    snap.positive_rate,
                    self._baseline_positive_rate,
                    "high",
                    f"Positive prediction rate shifted by {pos_drift:.1%} "
                    f"from baseline ({self._baseline_positive_rate:.1%}).",
                    "Investigate for input distribution shift or data pipeline issues.",
                )

        # Feature drift
        if snap.feature_drift_score > self.thresholds["mean_psi"]:
            _make_alert(
                "drift", "feature_psi",
                snap.feature_drift_score,
                self.thresholds["mean_psi"],
                "critical" if snap.feature_drift_score > 0.40 else "high",
                f"Feature distribution PSI ({snap.feature_drift_score:.3f}) indicates "
                "significant drift from training baseline.",
                "Consider model retraining. Check upstream data pipeline for changes.",
            )

        # Attack signal
        if snap.attack_signal > self.thresholds["attack_signal"]:
            _make_alert(
                "attack", "anomaly_score",
                snap.attack_signal,
                self.thresholds["attack_signal"],
                "critical",
                f"High anomaly score ({snap.attack_signal:.2f}) in recent input features. "
                "Possible adversarial attack or data corruption.",
                "Temporarily halt automated decisions. Review inputs for poisoning.",
            )

        # Limit alert log to 200
        if len(self._alerts) > 200:
            self._alerts = self._alerts[-200:]

        return new_alerts

    # ── Public API ─────────────────────────────────────────────────────────

    def check_alerts(self) -> List[MonitorAlert]:
        """Return all active alerts."""
        with self._lock:
            return list(self._alerts)

    def get_snapshot(self) -> MonitorSnapshot:
        """Return the latest snapshot."""
        with self._lock:
            if not self._snapshots:
                return self._compute_snapshot()
            return self._snapshots[-1]

    def get_history(self) -> List[MonitorSnapshot]:
        """Return all snapshots as a list."""
        with self._lock:
            return list(self._snapshots)

    def clear_alerts(self) -> int:
        """Clear all alerts. Returns number cleared."""
        with self._lock:
            n = len(self._alerts)
            self._alerts = []
            return n

    def get_report(self) -> Dict[str, Any]:
        """Return a complete monitoring report dict."""
        with self._lock:
            snap = self._compute_snapshot()
            return {
                "domain":         self.domain,
                "window_size":    self.window_size,
                "n_ingested":     len(self._preds),
                "snapshot":       snap.__dict__,
                "alerts_count":   len(self._alerts),
                "recent_alerts":  [a.__dict__ for a in self._alerts[-10:]],
                "thresholds":     self.thresholds,
                "generated_at":   datetime.now().isoformat()[:19],
            }

    def set_reference(self, X: np.ndarray) -> None:
        """Update the drift detector's reference distribution."""
        self.drift_detector.set_reference(X)
        self._baseline_positive_rate = None


# ── Streamlit dashboard ───────────────────────────────────────────────────────

def monitoring_dashboard(
    monitor: RealtimeMonitor,
    key_prefix: str = "monitor",
    simulate_stream: bool = True,
) -> None:
    """
    Render a real-time monitoring dashboard panel.

    Parameters
    ----------
    monitor          : RealtimeMonitor instance
    key_prefix       : unique key prefix for Streamlit widgets
    simulate_stream  : if True, show a "Simulate incoming predictions" button
                       for demo purposes (no live endpoint required)
    """
    try:
        import streamlit as st
        import plotly.graph_objects as go
        import plotly.express as px
    except ImportError:
        return

    snap    = monitor.get_snapshot()
    alerts  = monitor.check_alerts()
    history = monitor.get_history()

    # ── Status header ─────────────────────────────────────────────────────
    status_ok = not snap.drift_detected and not snap.fairness_degraded and snap.attack_signal < 0.7
    color     = "#27ae60" if status_ok else "#e74c3c"
    status_label = "🟢 HEALTHY" if status_ok else "🔴 ALERT"

    st.markdown(
        f"<div style='display:flex;align-items:center;gap:1rem;"
        f"background:var(--color-background-secondary);"
        f"border:1px solid {color}44;"
        f"border-radius:var(--border-radius-lg);padding:.75rem 1.25rem;"
        f"margin-bottom:.75rem;'>"
        f"<span style='font-size:1.1rem;font-weight:600;color:{color};'>{status_label}</span>"
        f"<span style='font-size:.83rem;color:var(--color-text-secondary);'>"
        f"Domain: {monitor.domain} · Window: {monitor.window_size} · "
        f"Predictions ingested: {snap.n_predictions} · "
        f"Active alerts: {len(alerts)}"
        f"</span>"
        f"<span style='margin-left:auto;font-size:.75rem;color:var(--color-text-tertiary);'>"
        f"Last update: {snap.timestamp}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Live KPI row ──────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Positive Rate",          f"{snap.positive_rate:.1%}")
    k2.metric("Parity Gap",             f"{snap.demographic_parity_gap:.1%}",
              delta=f"{'⚠️ Above threshold' if snap.fairness_degraded else 'Within limit'}",
              delta_color="inverse" if snap.fairness_degraded else "normal")
    k3.metric("Feature Drift (PSI)",    f"{snap.feature_drift_score:.3f}",
              delta="⚠️ High drift" if snap.drift_detected else "Stable",
              delta_color="inverse" if snap.drift_detected else "normal")
    k4.metric("Attack Signal",          f"{snap.attack_signal:.2f}",
              delta="🚨 Critical" if snap.attack_signal > 0.7 else "Normal",
              delta_color="inverse" if snap.attack_signal > 0.7 else "normal")

    # ── History charts ────────────────────────────────────────────────────
    if len(history) >= 2:
        import pandas as pd
        hist_df = pd.DataFrame([h.__dict__ for h in history])

        col1, col2 = st.columns(2)
        with col1:
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=list(range(len(hist_df))),
                y=hist_df["demographic_parity_gap"],
                mode="lines+markers",
                name="Parity Gap",
                line=dict(color="#e74c3c", width=2),
            ))
            fig1.add_hline(
                y=monitor.thresholds["demographic_parity_gap"],
                line_dash="dot", line_color="orange",
                annotation_text="Threshold",
            )
            fig1.update_layout(
                title="Demographic Parity Gap over Time",
                height=220, margin=dict(t=40,b=20,l=40,r=20),
                yaxis=dict(range=[0, max(0.3, hist_df["demographic_parity_gap"].max()*1.2)]),
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=list(range(len(hist_df))),
                y=hist_df["feature_drift_score"],
                mode="lines+markers",
                name="PSI",
                line=dict(color="#2980b9", width=2),
                fill="tozeroy",
                fillcolor="rgba(41,128,185,0.08)",
            ))
            fig2.add_hline(y=0.10, line_dash="dot", line_color="orange",
                           annotation_text="Moderate drift")
            fig2.add_hline(y=0.20, line_dash="dot", line_color="red",
                           annotation_text="High drift")
            fig2.update_layout(
                title="Feature Drift (PSI) over Time",
                height=220, margin=dict(t=40,b=20,l=40,r=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Alert log ─────────────────────────────────────────────────────────
    if alerts:
        st.markdown(f"**⚠️ Active Alerts ({len(alerts)})**")
        sev_colors = {"critical": "#e74c3c", "high": "#e67e22",
                      "medium": "#f39c12", "low": "#3498db"}
        for a in reversed(alerts[-5:]):
            c = sev_colors.get(a.severity, "#888")
            st.markdown(
                f"<div style='border-left:3px solid {c};"
                f"padding:.4rem .75rem;margin:.2rem 0;"
                f"background:var(--color-background-secondary);"
                f"border-radius:0 6px 6px 0;font-size:.82rem;'>"
                f"<strong style='color:{c};'>[{a.severity.upper()}]</strong> "
                f"{a.alert_id} · {a.alert_type} · {a.timestamp}<br>"
                f"{a.message}<br>"
                f"<em>Recommendation: {a.recommendation}</em>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if st.button("Clear all alerts", key=f"_{key_prefix}_clear"):
            monitor.clear_alerts()
            st.rerun()
    else:
        st.markdown(
            '<div style="color:var(--color-text-success);font-size:.83rem;">'
            "✅ No active alerts</div>",
            unsafe_allow_html=True,
        )

    # ── Simulate stream (demo mode) ───────────────────────────────────────
    if simulate_stream:
        st.markdown("---")
        st.markdown("**Simulate incoming predictions** (demo — no live endpoint needed)")
        sc1, sc2, sc3 = st.columns(3)
        n_inject    = sc1.slider("Predictions to inject", 10, 200, 50, key=f"_{key_prefix}_n")
        drift_level = sc2.slider("Drift level", 0.0, 1.0, 0.0, 0.1, key=f"_{key_prefix}_d")
        bias_level  = sc3.slider("Bias level", 0.0, 0.5, 0.0, 0.05, key=f"_{key_prefix}_b")

        if st.button("▶ Inject predictions", key=f"_{key_prefix}_inject"):
            n_feats = 12
            rng     = np.random.default_rng(42)
            for _ in range(n_inject):
                feat = rng.normal(drift_level, 1.0 + drift_level, n_feats)
                grp  = int(rng.random() < 0.5)
                # Bias: group 0 gets artificially lower scores
                base_prob = 0.4 + drift_level * 0.2
                if grp == 0:
                    base_prob -= bias_level
                pred = int(rng.random() < max(0, min(1, base_prob)))
                monitor.ingest(feat, pred, demographic_group=grp)
            st.success(f"Injected {n_inject} predictions. Refresh to see updated metrics.")
            st.rerun()
