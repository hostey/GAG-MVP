"""
components/lazy_imports.py
===========================
Provides lazy-loaded versions of heavy ML libraries.
Import this instead of importing sklearn/scipy directly at module level.

Instead of (slow — runs on every page load):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

Use (fast — imports only when first called):
    from components.lazy_imports import get_sklearn_ensemble, get_sklearn_metrics

These are cached per-process so the cost is paid only once.
"""
import streamlit as st
import functools


@st.cache_resource(show_spinner=False)
def get_sklearn_ensemble():
    from sklearn.ensemble import (
        GradientBoostingClassifier, RandomForestClassifier,
        ExtraTreesClassifier, AdaBoostClassifier,
        HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
    )
    from sklearn.calibration import CalibratedClassifierCV
    return {
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "ExtraTreesClassifier": ExtraTreesClassifier,
        "AdaBoostClassifier": AdaBoostClassifier,
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier,
        "StackingClassifier": StackingClassifier,
        "VotingClassifier": VotingClassifier,
        "CalibratedClassifierCV": CalibratedClassifierCV,
    }


@st.cache_resource(show_spinner=False)
def get_sklearn_metrics():
    from sklearn.metrics import (
        accuracy_score, recall_score, precision_score,
        f1_score, roc_auc_score
    )
    return {
        "accuracy_score": accuracy_score,
        "recall_score": recall_score,
        "precision_score": precision_score,
        "f1_score": f1_score,
        "roc_auc_score": roc_auc_score,
    }


@st.cache_resource(show_spinner=False)
def get_sklearn_preprocessing():
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    return StandardScaler, train_test_split
