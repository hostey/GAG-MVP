import numpy as np
import pandas as pd
import streamlit as st

def compute_clinical_impact_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    demographic_mask: np.ndarray
) -> dict:
    """
    Calculates clinical impact metrics comparing privileged vs underserved cohorts,
    evaluating under-diagnosis risk against NDPR & FMOH AI Health Guidelines.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    demographic_mask = np.asarray(demographic_mask, dtype=bool)

    def get_metrics(yt, yp):
        tp = np.sum((yt == 1) & (yp == 1))
        fn = np.sum((yt == 1) & (yp == 0))
        fp = np.sum((yt == 0) & (yp == 1))
        tn = np.sum((yt == 0) & (yp == 0))

        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Under-diagnosis rate
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Over-diagnosis rate
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # Sensitivity
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Precision

        return {
            "FNR": fnr,
            "FPR": fpr,
            "TPR": tpr,
            "PPV": ppv,
            "Count": len(yt),
        }

    underserved_metrics = get_metrics(y_true[demographic_mask], y_pred[demographic_mask])
    privileged_metrics = get_metrics(y_true[~demographic_mask], y_pred[~demographic_mask])

    fnr_disparity = underserved_metrics["FNR"] - privileged_metrics["FNR"]
    equal_opportunity_ratio = (
        underserved_metrics["TPR"] / privileged_metrics["TPR"]
        if privileged_metrics["TPR"] > 0
        else 0.0
    )

    # Compliance according to NDPR (Data Equity) and FMOH National AI Guidelines
    is_ndpr_compliant = abs(fnr_disparity) < 0.05
    is_fmoh_compliant = 0.80 <= equal_opportunity_ratio <= 1.25

    return {
        "underserved": underserved_metrics,
        "privileged": privileged_metrics,
        "fnr_disparity": fnr_disparity,
        "equal_opportunity_ratio": equal_opportunity_ratio,
        "is_ndpr_compliant": is_ndpr_compliant,
        "is_fmoh_compliant": is_fmoh_compliant,
        "is_national_ai_compliant": is_ndpr_compliant and is_fmoh_compliant
    }

def apply_sample_reweighing(y: np.ndarray, is_underserved: np.ndarray) -> np.ndarray:
    """Computes sample weights using Kamiran & Calders reweighing strategy."""
    y = np.asarray(y)
    is_underserved = np.asarray(is_underserved, dtype=bool)

    n_total = len(y)
    n_underserved = np.sum(is_underserved)
    n_privileged = n_total - n_underserved

    n_pos = np.sum(y == 1)
    n_neg = n_total - n_pos

    weights = np.ones(n_total, dtype=float)

    for group_mask, group_count in [(is_underserved, n_underserved), (~is_underserved, n_privileged)]:
        for label, label_count in [(1, n_pos), (0, n_neg)]:
            cell_mask = group_mask & (y == label)
            n_cell = np.sum(cell_mask)

            if n_cell > 0 and group_count > 0:
                expected_prob = (group_count / n_total) * (label_count / n_total)
                actual_prob = n_cell / n_total
                weights[cell_mask] = expected_prob / actual_prob

    return weights

def render_national_compliance_panel(impact_matrix: dict):
    """Renders the NDPR and FMOH Health AI Governance audit scorecard."""
    st.subheader("📋 NDPR & National Health AI Governance Scorecard")
    st.caption("Real-time fairness audit mapped against Nigerian Federal Ministry of Health (FMOH) and NDPR guidance.")

    underserved = impact_matrix["underserved"]
    priv = impact_matrix["privileged"]
    fnr_gap = impact_matrix["fnr_disparity"]
    eq_opp = impact_matrix["equal_opportunity_ratio"]
    is_compliant = impact_matrix["is_national_ai_compliant"]

    if is_compliant:
        st.success("✅ **Regulatory Status: COMPLIANT (NDPR / FMOH)** — Under-diagnosis disparity is within national risk thresholds (< 5%).")
    else:
        st.error("⚠️ **Regulatory Status: AUDIT WARNING** — Disparate under-diagnosis risk detected in underserved demographic groups.")

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="Under-Diagnosis Gap (FNR)",
            value=f"{fnr_gap * 100:+.2f}%",
            delta="Acceptable" if abs(fnr_gap) < 0.05 else "High Disparity",
            delta_color="normal" if abs(fnr_gap) < 0.05 else "inverse"
        )
    with col2:
        st.metric(
            label="Equal Opportunity Ratio",
            value=f"{eq_opp:.2f}",
            delta="Fair" if 0.8 <= eq_opp <= 1.25 else "Unfair",
            delta_color="normal" if 0.8 <= eq_opp <= 1.25 else "inverse"
        )
    with col3:
        st.metric(
            label="Underserved Cohort FNR",
            value=f"{underserved['FNR'] * 100:.1f}%"
        )
    with col4:
        st.metric(
            label="Privileged Cohort FNR",
            value=f"{priv['FNR'] * 100:.1f}%"
        )

    st.markdown("---")
    st.markdown("### 📊 Subgroup Performance Breakdown")
    comparison_df = pd.DataFrame({
        "Metric": ["False Negative Rate (Under-diagnosis)", "False Positive Rate (Over-diagnosis)", "True Positive Rate (Sensitivity)", "Positive Predictive Value (Precision)", "Sample Size"],
        "Underserved Cohort": [f"{underserved['FNR']*100:.1f}%", f"{underserved['FPR']*100:.1f}%", f"{underserved['TPR']*100:.1f}%", f"{underserved['PPV']*100:.1f}%", underserved['Count']],
        "Privileged Cohort": [f"{priv['FNR']*100:.1f}%", f"{priv['FPR']*100:.1f}%", f"{priv['TPR']*100:.1f}%", f"{priv['PPV']*100:.1f}%", priv['Count']]
    })
    st.dataframe(comparison_df, use_container_width=True)