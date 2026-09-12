import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from components.governance_logic import (
    generate_synthetic_data,
    generate_africa_centric_data,
    apply_bias,
    simulate_data_poisoning,
    calculate_fairness_metrics,
    run_gender_equity_audit,
)
from components.health.equity_audit_engine import build_intersectional_features


def compute_live_metrics(df: pd.DataFrame, target_col: str, proba_col: str, group_col: str, threshold: float = 0.5):
    """Calculates performance and group fairness metrics dynamically."""
    df_clean = df.dropna(subset=[target_col, proba_col, group_col]).copy()
    df_clean['dynamic_pred'] = (df_clean[proba_col] >= threshold).astype(int)

    y_true = df_clean[target_col]
    y_pred = df_clean['dynamic_pred']

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)

    groups = df_clean[group_col].unique()
    group_metrics = {}

    for g in groups:
        g_sub = df_clean[df_clean[group_col] == g]
        group_metrics[str(g)] = {
            "selection_rate": g_sub['dynamic_pred'].mean(),
            "recall_tpr": recall_score(g_sub[target_col], g_sub['dynamic_pred'], zero_division=0),
            "count": len(g_sub)
        }

    sel_rates = [m["selection_rate"] for m in group_metrics.values()]
    max_rate = max(sel_rates) if sel_rates else 1.0
    min_rate = min(sel_rates) if sel_rates else 0.0
    disparate_impact = (min_rate / max_rate) if max_rate > 0 else 1.0

    return {
        "global": {"accuracy": acc, "precision": prec, "recall": rec, "threshold": threshold},
        "disparate_impact": disparate_impact,
        "group_metrics": group_metrics,
        "df_evaluated": df_clean
    }


def _train_and_score(X, y, random_state=42):
    if len(np.unique(y)) < 2:
        return None, None, None, None
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(
        Xs, y, test_size=0.3, random_state=random_state,
        stratify=y if (len(np.unique(y)) > 1 and np.bincount(y.astype(int)).min() >= 2) else None
    )
    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=random_state)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    y_proba = clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba") else yp

    metrics = {
        "accuracy": float(accuracy_score(yte, yp)),
        "recall": float(recall_score(yte, yp, zero_division=0)),
        "precision": float(precision_score(yte, yp, zero_division=0)),
        "f1": float(f1_score(yte, yp, zero_division=0)),
        "fpr": float(np.mean(yp[yte == 0] == 1)) if (yte == 0).any() else 0.0
    }
    return metrics, clf, scaler, (Xtr, Xte, ytr, yte, y_proba)


def run_simulation_step(
        data_source, n_samples, selected_biases, bias_intensity,
        poison_rate, access_inequality, run_idx,
        enable_gender_audit=True, session_state=None
):
    """Executes a single simulation run iteration."""
    try:
        X, y, demo = generate_synthetic_data(n_samples=n_samples, n_features=10)
        if data_source.startswith("abuja") or data_source == "africa_centric":
            X, y, demo, _ = generate_africa_centric_data(scenario="healthcare", n_samples=n_samples)
    except Exception:
        X, y, demo = generate_synthetic_data(n_samples=n_samples, n_features=10)

    # Process features through intersectional engine
    raw_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    raw_df["income_level"] = X[:, 0] if X.shape[1] > 0 else 0.5
    raw_df["is_rural"] = (demo == 0).astype(int)
    featured_df = build_intersectional_features(raw_df)
    demo = featured_df["is_underserved_cohort"].values

    X = X.astype(np.float64)

    if access_inequality > 0:
        mask = demo == 1
        if mask.any():
            X[mask] += np.random.normal(0, access_inequality * 0.3, (mask.sum(), X.shape[1]))

    for bt in selected_biases:
        try:
            X, y, demo = apply_bias(X, y, bt, bias_intensity, demographic_info=demo)
        except Exception:
            pass

    try:
        X, y, demo = simulate_data_poisoning(X, y, poison_rate, attack_type="label_flipping", demographic_info=demo)
    except Exception:
        pass

    metrics, clf, scaler, splits = _train_and_score(X, y, random_state=42 + max(run_idx, 0))
    if clf is None:
        return {"accuracy": 0, "recall": 0, "equity_score": 0, "warnings": ["Model convergence failed."]}

    X_tr, X_te, y_tr, y_te, y_proba = splits
    y_pred = clf.predict(X_te)
    demo_te = demo[:len(y_te)]

    fair = calculate_fairness_metrics(y_te, y_pred, demo_te)
    gender_audit = None
    if enable_gender_audit:
        try:
            gender_audit = run_gender_equity_audit(y_te, y_pred, demo_te, 0.34)
        except Exception:
            pass

    adv_mask = demo_te == 0
    dis_mask = demo_te == 1

    def _grp_acc(m):
        return float(accuracy_score(y_te[m], y_pred[m])) if m.any() else metrics["accuracy"]

    return {
        "run_id": run_idx + 1,
        "data_source": data_source,
        "accuracy": metrics["accuracy"],
        "recall": metrics["recall"],
        "sensitivity": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "acc_hi": _grp_acc(adv_mask),
        "acc_lo": _grp_acc(dis_mask),
        "adv_hi": _grp_acc(adv_mask),
        "adv_lo": _grp_acc(dis_mask),
        "equity_score": fair.get("fairness_score", 0.5),
        "demographic_parity": fair.get("demographic_parity_difference", 0),
        "y_true": y_te,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "demographic_mask": dis_mask,
        "gender_gap": gender_audit.overall_gender_gap if gender_audit else 0.0,
        "warnings": []
    }