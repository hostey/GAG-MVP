# plugins/example_plugin.py
"""
GAGS Plugin Example — v1.0
===========================
Demonstrates how hospitals, NGOs, and government departments can add
custom data sources, metrics, and scenarios without forking GAGS.

Copy this file into your project and register it at app startup:

    # In your app entrypoint or Home.py:
    import plugins.example_plugin   # registration happens at import time
"""

import numpy as np
from components.governance_logic import (
    plugin_registry,
    DataLoaderPlugin,
    MetricPlugin,
    ScenarioPlugin,
    BiasPlugin,
    CommunityScenario,
    GagsPlugin,
)
from dataclasses import field as dc_field
from typing import List


# ── Example 1: Custom Data Loader ────────────────────────────────────────────
# Replace load() with your real EHR / database query

@plugin_registry.register
class LagosTHTDataPlugin(DataLoaderPlugin):
    """
    Simulates a Lagos University Teaching Hospital synthetic EHR dataset.
    Replace with your actual data loading logic.
    """
    name        = "Lagos UTH Synthetic EHR"
    domain      = "healthcare"
    version     = "1.0"
    contributor = "Lagos University Teaching Hospital Research Unit"
    description = "Synthetic patient records calibrated to LUTH demographic distributions"
    tags        = ["Nigeria", "Lagos", "EHR", "healthcare"]

    def load(self, n_samples: int = 5000, **kwargs):
        """
        Return (X, y, demographic_info) arrays.
        X shape: (n_samples, n_features) — all numeric
        y shape: (n_samples,)            — binary 0/1
        demographic_info: (n_samples,)   — group labels (0 = disadvantaged)
        """
        rng = np.random.RandomState(kwargs.get("random_state", 42))
        n   = min(n_samples, 10_000)

        # Simulate: age, income_proxy, insurance, facility_access, prior_visits,
        #           bmi, systolic_bp, hba1c_proxy, malaria_prev, typhoid_prev
        X = np.column_stack([
            rng.normal(42, 15, n).clip(15, 95),   # age
            rng.uniform(0, 1, n),                   # income_proxy
            rng.binomial(1, 0.55, n),               # has_insurance (55% in Lagos)
            rng.uniform(0.2, 1, n),                 # facility_access_score
            rng.poisson(1.2, n),                    # prior_visits_year
            rng.normal(26, 6, n).clip(14, 55),      # bmi
            rng.normal(132, 22, n).clip(70, 220),   # systolic_bp_mmHg
            rng.uniform(4.5, 12, n),                # hba1c_proxy
            rng.binomial(1, 0.18, n),               # malaria_last_12m
            rng.binomial(1, 0.09, n),               # typhoid_last_12m
        ])

        # Outcome: readmission risk (correlated with income, insurance, prior visits)
        risk_score = (
            (1 - X[:, 1]) * 0.3      # low income increases risk
            + (1 - X[:, 2]) * 0.2    # uninsured increases risk
            + X[:, 4] / 5 * 0.2      # more prior visits = higher chronic burden
            + (X[:, 7] > 7) * 0.2    # high HbA1c proxy
            + rng.normal(0, 0.1, n)
        )
        y = (risk_score > np.percentile(risk_score, 60)).astype(int)

        # Demographic: 0 = low-income/uninsured, 1 = insured/higher-income
        demo = ((X[:, 1] > 0.4) & (X[:, 2] == 1)).astype(int)

        return X, y, demo

    def describe(self):
        return {
            "name":          self.name,
            "n_features":    10,
            "feature_names": ["age","income_proxy","insurance","facility_access",
                               "prior_visits","bmi","systolic_bp","hba1c_proxy",
                               "malaria_12m","typhoid_12m"],
            "positive_label": "30-day readmission",
            "source":        "Synthetic — calibrated to LUTH 2022 cohort statistics",
            "contributor":   self.contributor,
        }


# ── Example 2: Custom Metric ─────────────────────────────────────────────────

@plugin_registry.register
class HealthcareAccessGapMetric(MetricPlugin):
    """
    Measures the access gap: difference in positive prediction rates
    between insured (group=1) and uninsured (group=0) patients.
    Useful for Nigeria where insurance status strongly correlates with outcomes.
    """
    name        = "Healthcare Access Gap"
    domain      = "healthcare"
    version     = "1.0"
    contributor = "GAGS Research Team"
    description = "Positive prediction rate gap between insured and uninsured patients"
    tags        = ["Nigeria","insurance","access","equity"]

    def compute(self, y_true, y_pred, demographic_info, **kwargs) -> float:
        insured   = demographic_info == 1
        uninsured = demographic_info == 0
        rate_ins   = float(np.mean(y_pred[insured]))   if insured.any()   else 0.5
        rate_unins = float(np.mean(y_pred[uninsured])) if uninsured.any() else 0.5
        return round(abs(rate_ins - rate_unins), 4)

    def describe(self):
        return {
            "name":    self.name,
            "definition": "Absolute difference in positive prediction rates between insured and uninsured patients",
            "formula":    "gap = |positive_rate_insured − positive_rate_uninsured|",
            "target":     "Gap ≤ 0.05 (5%) for equitable deployment",
            "risk":       "High gap indicates the model treats patients differently based on insurance status",
        }


# ── Example 3: Scenario Plugin ────────────────────────────────────────────────

@plugin_registry.register
class RiversStateAgrotechPlugin(ScenarioPlugin):
    """
    Contributes two agrotech scenarios specific to Rivers State, Nigeria.
    Scenarios are automatically merged into COMMUNITY_SCENARIOS at registration.
    """
    name        = "Rivers State Agrotech Scenarios"
    domain      = "agrotech"
    version     = "1.0"
    contributor = "Rivers State Ministry of Agriculture"
    description = "Bias scenarios for crop advisory AI in Rivers State Niger Delta context"
    tags        = ["Nigeria","Rivers State","Niger Delta","agrotech"]

    def get_scenarios(self) -> List[CommunityScenario]:
        return [
            CommunityScenario(
                id="AG-RS-001",
                name="Niger Delta Fishing Community Market Access",
                domain="agrotech",
                region="Nigeria (Rivers State — Niger Delta)",
                description=(
                    "AI market access model trained on inland farming data "
                    "systematically under-serves coastal fishing-farming communities "
                    "whose livelihood patterns differ from the training distribution."
                ),
                bias_types=["geographic", "demographic", "measurement"],
                bias_intensity=0.38,
                poison_rate=0.04,
                expected_fairness_range=(0.42, 0.62),
                real_world_analogue=(
                    "Digital market platforms targeting Nigerian farmers have been "
                    "documented to have lower engagement and worse price predictions "
                    "for coastal communities vs inland areas."
                ),
                citation="AfDB (2022) Digital Agriculture Inclusion Report; FAO Nigeria Country Note 2023",
                mitigation_priority="high",
                mitigation_strategies=[
                    "Collect representative data from Niger Delta coastal communities",
                    "Add water-body proximity and tidal calendar as features",
                    "Validate with Rivers State ADP extension officers",
                ],
                contributor="Rivers State Ministry of Agriculture",
                version="1.0",
                tags=["Niger Delta","coastal","fishing","market access","Rivers State"],
            ),
            CommunityScenario(
                id="AG-RS-002",
                name="Palm Oil Small-Scale Processor Gender Gap",
                domain="agrotech",
                region="Nigeria (Rivers State)",
                description=(
                    "Palm oil value chain AI trained predominantly on large-scale "
                    "commercial processor data. Small-scale processors — disproportionately "
                    "female-led — receive lower quality yield forecasts and market linkages."
                ),
                bias_types=["gender", "socioeconomic", "historical"],
                bias_intensity=0.32,
                poison_rate=0.03,
                expected_fairness_range=(0.45, 0.65),
                real_world_analogue=(
                    "Women constitute ~70% of palm oil small-scale processors in Rivers State "
                    "but are under-represented in commercial datasets used to train AgriTech AI."
                ),
                citation="IFAD (2021) Women in Nigerian Palm Oil Value Chains; CGIAR Gender Program (2023)",
                mitigation_priority="high",
                mitigation_strategies=[
                    "Stratify training data by processing scale and gender",
                    "Partner with women's cooperatives for ground-truth validation",
                    "Design features that capture small-scale processing indicators",
                ],
                contributor="Rivers State Ministry of Agriculture",
                version="1.0",
                tags=["palm oil","gender","women","small-scale","Rivers State"],
            ),
        ]


# ── Example 4: Custom Bias Plugin ─────────────────────────────────────────────

@plugin_registry.register
class InfrastructureAccessBias(BiasPlugin):
    """
    Custom bias type: infrastructure_access.
    Simulates systematic degradation for users in areas with poor
    infrastructure (electricity, roads, connectivity) — relevant for
    Nigerian deployment contexts where infrastructure access is highly
    unequal and correlates with socioeconomic outcomes.
    """
    name        = "Infrastructure Access Bias"
    domain      = "generic"
    version     = "1.0"
    contributor = "GAGS Research Team"
    description = "Degrades model inputs for low-infrastructure users"
    bias_type   = "infrastructure_access"
    tags        = ["Nigeria","infrastructure","access","digital divide"]

    def inject(self, X, y, demographic_info, intensity, **kwargs):
        """
        Simulate data quality degradation for low-infrastructure users.
        Assumes demographic_info=0 marks low-infrastructure group.
        """
        rng = np.random.RandomState(kwargs.get("random_state", 42))
        X_out    = X.copy().astype(np.float64)
        low_mask = demographic_info == 0

        if not low_mask.any():
            return X_out, y, demographic_info

        n_low = int(low_mask.sum())
        # Feature dropout: randomly zero out features (simulates missing data
        # from poor connectivity / offline form submission)
        dropout_prob = intensity * 0.4
        dropout_mask = rng.binomial(1, dropout_prob, (n_low, X_out.shape[1])).astype(bool)
        X_out[low_mask][dropout_mask] = 0.0

        # Add noise (simulates manual data entry errors)
        noise_std = intensity * 0.3
        X_out[low_mask] += rng.normal(0, noise_std, (n_low, X_out.shape[1]))

        # Label noise for low-infrastructure group (human supervisor overrides)
        if intensity > 0.2:
            flip_n = int(n_low * intensity * 0.15)
            flip_idx = rng.choice(np.where(low_mask)[0], size=min(flip_n, n_low), replace=False)
            y_out = y.copy()
            y_out[flip_idx] = 1 - y_out[flip_idx]
            return X_out, y_out, demographic_info

        return X_out, y, demographic_info