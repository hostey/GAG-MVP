"""
Core simulation and governance logic for GAGS Resilience Framework.
Handles synthetic data generation, bias injection, attacks, and scoring.

Enhanced with:
- Comprehensive type hints
- Better error handling
- More realistic bias implementations
- Multiple fairness metrics
- Attack severity levels
- Detailed logging capabilities

NEW FEATURE MODULES (v3.0):
- AI Agent Economy Sandbox         (Feature 1)
- Multimodal Red Teaming           (Feature 2)
- Africa-Centric / Gender Equity   (Feature 3)
- Hybrid Human-AI Governance       (Feature 4)
- Strategic Social Reasoning       (Feature 5)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import json
import uuid
from datetime import datetime

# ── Sklearn imports moved to top level (was inside function body) ──────────────
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_val_predict

from utils.config import simulation_config, settings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(simulation_config.DATA_SEED)


# ───────────────────────────────────────────────
# Enums and Data Classes
# ───────────────────────────────────────────────

class BiasType(str, Enum):
    DEMOGRAPHIC     = "demographic"
    HISTORICAL      = "historical"
    SELECTION       = "selection"
    REPRESENTATION  = "representation"
    MEASUREMENT     = "measurement"
    TEMPORAL        = "temporal"
    GEOGRAPHIC      = "geographic"
    SOCIOECONOMIC   = "socioeconomic"
    ALGORITHMIC     = "algorithmic"
    GENDER          = "gender"           # Feature 3: explicit gender-equity bias type
    LINGUISTIC      = "linguistic"       # Feature 3: multilingual fairness bias


class AttackSeverity(str, Enum):
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class AttackModality(str, Enum):
    """Feature 2: multimodal attack surface types."""
    TEXT      = "text"
    IMAGE     = "image"
    AUDIO     = "audio"
    DEEPFAKE  = "deepfake"
    COMBINED  = "combined"


class GovernanceVoteOutcome(str, Enum):
    """Feature 4: outcomes from citizen-assembly votes."""
    APPROVED  = "approved"
    REJECTED  = "rejected"
    DEFERRED  = "deferred"
    REVERSED  = "reversed"


@dataclass
class SimulationResult:
    """Container for comprehensive simulation results."""
    accuracy: float
    fairness_score: float
    fairness_metrics: Dict[str, Any]
    poisoned_samples: int
    applied_biases: List[str]
    sample_size_after_bias: int
    data_distribution: Dict[str, Any]
    performance_history: List[float]
    # Fixed: was datetime (not JSON-serialisable) — now ISO string
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 1 — AI Agent Economy Sandbox
# "Sandbox-within-a-sandbox" where autonomous agents negotiate resource
# allocation, with controlled budgets and ZK-proof-inspired identity hashing.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentIdentity:
    """
    Zero-knowledge-proof-inspired agent identity.
    The real identity is never exposed; only the commitment hash is shared.
    """
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    commitment_hash: str = field(init=False)
    _secret: str = field(default_factory=lambda: str(uuid.uuid4()), repr=False)

    def __post_init__(self):
        # Commitment = H(secret || agent_id) — mimics ZKP commitment scheme
        raw = f"{self._secret}:{self.agent_id}"
        self.commitment_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]

    def verify(self, claimed_secret: str) -> bool:
        """Verify agent identity without revealing the secret."""
        raw = f"{claimed_secret}:{self.agent_id}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16] == self.commitment_hash


@dataclass
class EconomyAgent:
    """Autonomous agent participating in the resource-allocation sandbox."""
    name: str
    budget: float
    reputation: float = 1.0           # [0, 2] — affects bid weight
    strategy: str = "honest"          # honest | aggressive | cooperative
    identity: AgentIdentity = field(default_factory=AgentIdentity)
    bid_history: List[Dict] = field(default_factory=list)
    wins: int = 0
    total_spent: float = 0.0

    def place_bid(self, resource: str, base_value: float) -> float:
        """
        Generate a bid for a resource based on strategy and remaining budget.
        Returns bid amount (capped at current budget).
        """
        if self.budget <= 0:
            return 0.0

        noise = np.random.normal(0, 0.05)

        if self.strategy == "aggressive":
            bid = min(self.budget, base_value * (1.3 + noise) * self.reputation)
        elif self.strategy == "cooperative":
            bid = min(self.budget * 0.4, base_value * (0.8 + noise) * self.reputation)
        else:  # honest
            bid = min(self.budget * 0.6, base_value * (1.0 + noise) * self.reputation)

        bid = max(0.0, bid)
        self.bid_history.append({"resource": resource, "bid": round(bid, 4), "budget_remaining": round(self.budget, 4)})
        return bid

    def update_reputation(self, outcome: str) -> None:
        """Adjust reputation after an auction round."""
        delta = {"win": 0.05, "loss": -0.01, "cheat_detected": -0.5}.get(outcome, 0.0)
        self.reputation = float(np.clip(self.reputation + delta, 0.1, 2.0))


class AgentEconomySandbox:
    """
    Feature 1 — AI Agent Economy Sandbox.

    Runs a sealed-bid second-price (Vickrey) auction for shared resources
    (e.g., irrigation water, diagnostic compute, satellite bandwidth).
    Tracks permeability: how agent decisions affect the broader simulation.
    """

    RESOURCE_CATALOG = {
        "agrotech":          ["irrigation_water", "fertilizer_quota", "drone_hours", "market_access"],
        "healthcare":        ["icu_beds", "diagnostic_compute", "drug_supply", "specialist_time"],
        "national_security": ["satellite_bandwidth", "analyst_hours", "sensor_data", "response_units"],
    }

    def __init__(self, domain: str = "agrotech", n_agents: int = 4, budget_range: Tuple[float, float] = (100.0, 500.0)):
        self.domain = domain
        self.resources = self.RESOURCE_CATALOG.get(domain, self.RESOURCE_CATALOG["agrotech"])
        self.agents: List[EconomyAgent] = self._create_agents(n_agents, budget_range)
        self.auction_log: List[Dict] = []
        self.permeability_score: float = 0.0

    def _create_agents(self, n: int, budget_range: Tuple[float, float]) -> List[EconomyAgent]:
        strategies = ["honest", "aggressive", "cooperative"]
        agents = []
        for i in range(n):
            agents.append(EconomyAgent(
                name=f"Agent_{i+1}",
                budget=float(np.random.uniform(*budget_range)),
                strategy=strategies[i % len(strategies)],
            ))
        return agents

    def run_auction_round(self, resource: str, reserve_price: float = 10.0) -> Dict[str, Any]:
        """
        Run one Vickrey auction for a single resource.
        Winner pays second-highest bid (strategy-proof mechanism).
        """
        base_value = reserve_price * np.random.uniform(1.2, 3.0)
        bids = {agent.name: agent.place_bid(resource, base_value) for agent in self.agents}

        sorted_bids = sorted(bids.items(), key=lambda x: x[1], reverse=True)
        winner_name, winner_bid = sorted_bids[0]
        price_paid = sorted_bids[1][1] if len(sorted_bids) > 1 else reserve_price

        # Update agent states
        for agent in self.agents:
            if agent.name == winner_name:
                agent.budget -= price_paid
                agent.total_spent += price_paid
                agent.wins += 1
                agent.update_reputation("win")
            else:
                agent.update_reputation("loss")

        result = {
            "resource": resource,
            "winner": winner_name,
            "price_paid": round(price_paid, 4),
            "all_bids": {k: round(v, 4) for k, v in bids.items()},
            "base_value": round(base_value, 4),
        }
        self.auction_log.append(result)
        logger.info(f"[AgentEconomy] Auction — {resource}: winner={winner_name}, paid={price_paid:.2f}")
        return result

    def run_full_simulation(self, rounds: int = 3) -> Dict[str, Any]:
        """Run multiple auction rounds across all resources."""
        results = []
        for _ in range(rounds):
            for resource in self.resources:
                results.append(self.run_auction_round(resource))

        # Permeability: std of total_spent across agents (higher = more unequal impact)
        spends = [a.total_spent for a in self.agents]
        self.permeability_score = float(np.std(spends) / (np.mean(spends) + 1e-8))

        agent_summary = [
            {
                "name": a.name,
                "strategy": a.strategy,
                "wins": a.wins,
                "total_spent": round(a.total_spent, 2),
                "budget_remaining": round(a.budget, 2),
                "reputation": round(a.reputation, 3),
                "identity_hash": a.identity.commitment_hash,
            }
            for a in self.agents
        ]

        return {
            "domain": self.domain,
            "rounds": rounds,
            "auction_results": results,
            "agent_summary": agent_summary,
            "permeability_score": round(self.permeability_score, 4),
            "economy_stability": "stable" if self.permeability_score < 0.5 else "unstable",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 2 — Multimodal Red Teaming
# Extends attacks to cover text, image, audio, and deepfake modalities.
# Includes a VR/AR stress-scenario descriptor for immersive role-play metadata.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MultimodalAttackResult:
    modality: AttackModality
    attack_vector: str
    severity: AttackSeverity
    affected_samples: int
    bypass_rate: float          # Estimated % of defences bypassed
    sociotechnical_risk: float  # [0,1] human-in-the-loop vulnerability
    vr_scenario: Optional[str]  # Optional immersive scenario description
    metadata: Dict[str, Any] = field(default_factory=dict)


class MultimodalRedTeamer:
    """
    Feature 2 — Multimodal Red Teaming.

    Simulates cross-modal adversarial attacks and produces a
    risk report enriched with VR/AR scenario metadata for
    immersive human-AI role-play sessions.
    """

    # Sociotechnical risk weights per modality (how much human bias is amplified)
    SOCIOTECHNICAL_WEIGHTS = {
        AttackModality.TEXT:     0.4,
        AttackModality.IMAGE:    0.6,
        AttackModality.AUDIO:    0.5,
        AttackModality.DEEPFAKE: 0.9,
        AttackModality.COMBINED: 1.0,
    }

    VR_SCENARIO_TEMPLATES = {
        "healthcare": (
            "You are a clinician in a virtual A&E department. "
            "Diagnostic AI flags a patient as low-risk. An image prompt-injection "
            "has subtly altered the X-ray metadata. Decide whether to override."
        ),
        "national_security": (
            "You are an analyst in a virtual operations centre. "
            "A deepfake video of an official has been fed into the briefing pipeline. "
            "Identify the forgery under time pressure before an incorrect order is issued."
        ),
        "agrotech": (
            "You are a regional agricultural officer. "
            "An audio deepfake of a market price broadcast has mis-priced staple crops. "
            "Smallholder agents are already responding. Can you intervene in time?"
        ),
    }

    def simulate_text_injection(
        self, X: np.ndarray, y: np.ndarray, injection_rate: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, MultimodalAttackResult]:
        """Simulate adversarial prompt injection via text features."""
        n_inject = int(len(X) * injection_rate)
        inject_idx = np.random.choice(len(X), n_inject, replace=False)

        X_attacked = X.copy()
        y_attacked = y.copy()

        # Inject adversarial perturbation into text-proxy features (cols 4–6)
        text_cols = [4, 5, 6] if X.shape[1] > 6 else list(range(min(3, X.shape[1])))
        X_attacked[inject_idx[:, None], text_cols] += np.random.normal(2.5, 0.8, (n_inject, len(text_cols)))
        y_attacked[inject_idx] = 1 - y_attacked[inject_idx]

        bypass_rate = float(np.clip(injection_rate * 1.8, 0, 1))
        result = MultimodalAttackResult(
            modality=AttackModality.TEXT,
            attack_vector="prompt_injection",
            severity=AttackSeverity.HIGH if injection_rate > 0.15 else AttackSeverity.MEDIUM,
            affected_samples=n_inject,
            bypass_rate=round(bypass_rate, 3),
            sociotechnical_risk=self.SOCIOTECHNICAL_WEIGHTS[AttackModality.TEXT],
            vr_scenario=None,
            metadata={"injection_rate": injection_rate, "text_cols": text_cols},
        )
        logger.info(f"[Multimodal] Text injection: {n_inject} samples affected, bypass={bypass_rate:.1%}")
        return X_attacked, y_attacked, result

    def simulate_image_attack(
        self, X: np.ndarray, y: np.ndarray, noise_scale: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray, MultimodalAttackResult]:
        """Simulate adversarial image perturbation (pixel-space proxy)."""
        n_samples = len(X)
        X_attacked = X.copy()

        # Image features proxied by high-index columns
        img_cols = list(range(min(7, X.shape[1]), X.shape[1]))
        if img_cols:
            adversarial_noise = np.sign(np.random.randn(n_samples, len(img_cols))) * noise_scale
            X_attacked[:, img_cols] += adversarial_noise

        severity = AttackSeverity.HIGH if noise_scale > 2.0 else AttackSeverity.MEDIUM
        result = MultimodalAttackResult(
            modality=AttackModality.IMAGE,
            attack_vector="adversarial_pixel_perturbation",
            severity=severity,
            affected_samples=n_samples,
            bypass_rate=round(float(np.clip(noise_scale / 5.0, 0, 1)), 3),
            sociotechnical_risk=self.SOCIOTECHNICAL_WEIGHTS[AttackModality.IMAGE],
            vr_scenario=None,
            metadata={"noise_scale": noise_scale},
        )
        logger.info(f"[Multimodal] Image attack: noise_scale={noise_scale}")
        return X_attacked, y, result

    def simulate_deepfake(
        self, X: np.ndarray, y: np.ndarray, domain: str = "healthcare"
    ) -> Tuple[np.ndarray, np.ndarray, MultimodalAttackResult]:
        """
        Simulate deepfake injection.
        High sociotechnical risk — humans are most likely to be deceived.
        """
        n_fake = int(len(X) * 0.2)
        fake_idx = np.random.choice(len(X), n_fake, replace=False)

        X_attacked = X.copy()
        y_attacked = y.copy()

        # Deepfake: plausible but wrong — features shifted toward opposite class
        X_attacked[fake_idx] = X_attacked[fake_idx] * 0.3 + np.random.normal(0, 2.0, (n_fake, X.shape[1]))
        y_attacked[fake_idx] = 1 - y_attacked[fake_idx]

        vr_scenario = self.VR_SCENARIO_TEMPLATES.get(domain, self.VR_SCENARIO_TEMPLATES["healthcare"])

        result = MultimodalAttackResult(
            modality=AttackModality.DEEPFAKE,
            attack_vector="synthetic_identity_substitution",
            severity=AttackSeverity.CRITICAL,
            affected_samples=n_fake,
            bypass_rate=0.78,
            sociotechnical_risk=self.SOCIOTECHNICAL_WEIGHTS[AttackModality.DEEPFAKE],
            vr_scenario=vr_scenario,
            metadata={"domain": domain, "fake_rate": 0.2},
        )
        logger.info(f"[Multimodal] Deepfake attack: {n_fake} samples, domain={domain}")
        return X_attacked, y_attacked, result

    def run_combined_attack(
        self, X: np.ndarray, y: np.ndarray, domain: str = "healthcare"
    ) -> Dict[str, Any]:
        """Run all modalities sequentially and aggregate risk."""
        X1, y1, r1 = self.simulate_text_injection(X, y)
        X2, y2, r2 = self.simulate_image_attack(X1, y1)
        X3, y3, r3 = self.simulate_deepfake(X2, y2, domain)

        combined_risk = float(np.mean([r.sociotechnical_risk for r in [r1, r2, r3]]))
        combined_bypass = float(np.mean([r.bypass_rate for r in [r1, r2, r3]]))

        return {
            "modality_results": [
                {
                    "modality": r.modality.value,
                    "attack_vector": r.attack_vector,
                    "severity": r.severity.value,
                    "affected_samples": r.affected_samples,
                    "bypass_rate": r.bypass_rate,
                    "sociotechnical_risk": r.sociotechnical_risk,
                    "vr_scenario": r.vr_scenario,
                }
                for r in [r1, r2, r3]
            ],
            "combined_sociotechnical_risk": round(combined_risk, 3),
            "combined_bypass_rate": round(combined_bypass, 3),
            "final_X": X3,
            "final_y": y3,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 3 — Africa-Centric Customisation & Gender Equity Lens
# Abuja-grounded scenario parameters, UNESCO Women4EthicalAI audit, and
# digital-inclusion metrics for underrepresented groups.
# ═══════════════════════════════════════════════════════════════════════════════

# Abuja-specific scenario presets
AFRICA_SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "smallholder_agrotech": {
        "description": "Climate-resilient agriculture for smallholders in Plateau State, Nigeria",
        "languages": ["Hausa", "Yoruba", "Igbo", "Berom", "English"],
        "gender_distribution": {"female": 0.52, "male": 0.48},
        "digital_inclusion_baseline": 0.34,   # 34% smartphone access
        "climate_stress_factor": 0.65,
        "feature_overrides": {
            "income_mean": 1.8,     # log-normal mean (lower than global default)
            "income_sigma": 0.7,
            "age_mean": 38.0,
            "connectivity_score": 0.3,
        },
    },
    "multilingual_healthcare": {
        "description": "Equitable healthcare delivery in multilingual Abuja FCT",
        "languages": ["Hausa", "Yoruba", "Igbo", "English"],
        "gender_distribution": {"female": 0.51, "male": 0.49},
        "digital_inclusion_baseline": 0.45,
        "climate_stress_factor": 0.2,
        "feature_overrides": {
            "income_mean": 2.2,
            "income_sigma": 0.6,
            "age_mean": 35.0,
            "connectivity_score": 0.55,
        },
    },
}


@dataclass
class GenderEquityAuditResult:
    """Result of a UNESCO Women4EthicalAI-aligned bias audit."""
    overall_gender_gap: float           # Accuracy gap between gender groups
    representation_score: float        # How well gender groups are represented [0,1]
    digital_inclusion_score: float     # Access parity [0,1]
    intersectional_disparities: Dict[str, float]
    incentive_recommendations: List[str]
    audit_passed: bool                  # True if all gaps below threshold
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def generate_africa_centric_data(
    scenario: str = "smallholder_agrotech",
    n_samples: int = 1000,
    n_features: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Feature 3 — Generate data using Africa-centric scenario presets.

    Returns:
        (X, y, demographic_info, scenario_metadata)
    """
    if scenario not in AFRICA_SCENARIO_PRESETS:
        logger.warning(f"Unknown scenario '{scenario}'. Using 'smallholder_agrotech'.")
        scenario = "smallholder_agrotech"

    preset = AFRICA_SCENARIO_PRESETS[scenario]
    overrides = preset["feature_overrides"]

    X = np.zeros((n_samples, n_features))

    # Feature 0: Age — local distribution
    X[:, 0] = np.random.normal(overrides["age_mean"], 12, n_samples)
    X[:, 0] = np.clip(X[:, 0], 18, 80)

    # Feature 1: Income — local log-normal
    X[:, 1] = np.random.lognormal(overrides["income_mean"], overrides["income_sigma"], n_samples)
    X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min() + 1e-8)

    # Feature 2: Connectivity / digital-inclusion score
    X[:, 2] = np.random.beta(2, 5, n_samples) * overrides["connectivity_score"]

    # Remaining features: generic
    for i in range(3, n_features):
        X[:, i] = np.random.uniform(0, 1, n_samples)

    # Labels: risk driven by age + low-income + low-connectivity
    risk = 0.4 * (1 - X[:, 1]) + 0.3 * (X[:, 0] / 80) + 0.3 * (1 - X[:, 2])
    y = (risk + np.random.normal(0, 0.1, n_samples) > 0.5).astype(int)

    # Demographic: encode gender using preset distribution
    f_ratio = preset["gender_distribution"]["female"]
    demographic_info = np.random.choice([0, 1], size=n_samples, p=[f_ratio, 1 - f_ratio])

    logger.info(f"[Africa-Centric] Generated {n_samples} samples for scenario: {scenario}")
    return X, y, demographic_info, preset


def run_gender_equity_audit(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    demographic_info: np.ndarray,
    digital_inclusion_baseline: float = 0.34,
    gap_threshold: float = 0.05,
) -> GenderEquityAuditResult:
    """
    Feature 3 — UNESCO Women4EthicalAI-aligned gender equity audit.

    Measures accuracy gaps, representation, digital-inclusion parity,
    and intersectional disparities. Returns incentive recommendations.
    """
    groups = np.unique(demographic_info)
    group_accuracy: Dict[int, float] = {}
    group_sizes: Dict[int, int] = {}

    for g in groups:
        mask = demographic_info == g
        group_accuracy[int(g)] = float(np.mean(y_pred[mask] == y_true[mask]))
        group_sizes[int(g)] = int(np.sum(mask))

    accuracies = list(group_accuracy.values())
    gender_gap = float(max(accuracies) - min(accuracies)) if len(accuracies) >= 2 else 0.0

    # Representation: Gini-based measure of how balanced group sizes are
    sizes = np.array(list(group_sizes.values()), dtype=float)
    sizes /= sizes.sum()
    n = len(sizes)
    representation_score = float(1.0 - np.sum(np.abs(sizes[:, None] - sizes[None, :])) / (2 * n))

    # Digital inclusion: penalise if female group (group 0) has lower predicted-positive rate
    female_pos_rate = float(np.mean(y_pred[demographic_info == 0])) if 0 in groups else 0.5
    male_pos_rate   = float(np.mean(y_pred[demographic_info == 1])) if 1 in groups else 0.5
    inclusion_gap   = abs(female_pos_rate - male_pos_rate)
    digital_inclusion_score = float(np.clip(digital_inclusion_baseline + (1 - inclusion_gap), 0, 1))

    # Intersectional disparities
    intersectional_disparities = {f"group_{k}_accuracy": round(v, 4) for k, v in group_accuracy.items()}
    intersectional_disparities["gender_gap"] = round(gender_gap, 4)
    intersectional_disparities["inclusion_gap"] = round(inclusion_gap, 4)

    # Incentive recommendations
    recommendations: List[str] = []
    if gender_gap > gap_threshold:
        recommendations.append("Apply gender-stratified resampling or reweighting before training.")
    if representation_score < 0.8:
        recommendations.append("Recruit additional data from underrepresented gender groups.")
    if digital_inclusion_score < 0.5:
        recommendations.append("Deploy USSD/SMS fallback interfaces to increase digital access parity.")
    if not recommendations:
        recommendations.append("Equity metrics within acceptable bounds. Continue monitoring quarterly.")

    audit_passed = gender_gap <= gap_threshold and representation_score >= 0.8

    logger.info(f"[GenderAudit] gap={gender_gap:.3f}, representation={representation_score:.3f}, passed={audit_passed}")
    return GenderEquityAuditResult(
        overall_gender_gap=round(gender_gap, 4),
        representation_score=round(representation_score, 4),
        digital_inclusion_score=round(digital_inclusion_score, 4),
        intersectional_disparities=intersectional_disparities,
        incentive_recommendations=recommendations,
        audit_passed=audit_passed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 4 — Hybrid Human-AI Governance Layer
# Citizen-assembly voting with AI drift detection and blockchain-style audit log.
# Supports reversible decisions via chained ledger entries.
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GovernanceLedgerEntry:
    """
    Immutable ledger entry — previous_hash chains entries into a
    tamper-evident log (lightweight blockchain analogue).
    """
    entry_id: str
    policy: str
    vote_outcome: GovernanceVoteOutcome
    vote_tally: Dict[str, int]
    ai_flags: List[str]
    previous_hash: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    hash: str = field(init=False)

    def __post_init__(self):
        payload = json.dumps({
            "entry_id": self.entry_id,
            "policy": self.policy,
            "vote_outcome": self.vote_outcome.value,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
        }, sort_keys=True)
        self.hash = hashlib.sha256(payload.encode()).hexdigest()[:24]


class HybridGovernanceLayer:
    """
    Feature 4 — Hybrid Human-AI Governance.

    Simulates a citizen-assembly vote on a policy proposal.
    AI monitors for bias drift and flags unintended consequences.
    All decisions are recorded in a chained ledger enabling reversals.
    """

    DRIFT_THRESHOLDS = {
        "accuracy_drop": 0.05,
        "fairness_drop": 0.08,
        "demographic_shift": 0.10,
    }

    def __init__(self):
        self.ledger: List[GovernanceLedgerEntry] = []
        self._genesis_hash = "0" * 24

    @property
    def _last_hash(self) -> str:
        return self.ledger[-1].hash if self.ledger else self._genesis_hash

    def simulate_citizen_vote(
        self,
        policy: str,
        n_voters: int = 100,
        ai_influence: float = 0.2,
    ) -> Dict[str, int]:
        """
        Simulate a citizen-assembly vote.
        `ai_influence` shifts some undecided votes toward the AI-recommended option.
        """
        base_for = int(n_voters * np.random.uniform(0.35, 0.65))
        base_against = int((n_voters - base_for) * 0.7)
        abstain = n_voters - base_for - base_against

        # AI nudges undecided votes
        ai_nudge = int(abstain * ai_influence)
        base_for += ai_nudge
        abstain -= ai_nudge

        return {"for": base_for, "against": base_against, "abstain": max(0, abstain)}

    def detect_bias_drift(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
    ) -> List[str]:
        """AI drift-detection: compare current metrics against baseline."""
        flags: List[str] = []

        acc_drop = baseline_metrics.get("accuracy", 1.0) - current_metrics.get("accuracy", 1.0)
        if acc_drop > self.DRIFT_THRESHOLDS["accuracy_drop"]:
            flags.append(f"Accuracy drift detected: -{acc_drop:.3f} (threshold {self.DRIFT_THRESHOLDS['accuracy_drop']})")

        fair_drop = baseline_metrics.get("fairness_score", 1.0) - current_metrics.get("fairness_score", 1.0)
        if fair_drop > self.DRIFT_THRESHOLDS["fairness_drop"]:
            flags.append(f"Fairness drift detected: -{fair_drop:.3f} (threshold {self.DRIFT_THRESHOLDS['fairness_drop']})")

        demo_shift = abs(
            baseline_metrics.get("demographic_parity", 0.0) - current_metrics.get("demographic_parity", 0.0)
        )
        if demo_shift > self.DRIFT_THRESHOLDS["demographic_shift"]:
            flags.append(f"Demographic parity shift: {demo_shift:.3f} (threshold {self.DRIFT_THRESHOLDS['demographic_shift']})")

        return flags

    def propose_and_vote(
        self,
        policy: str,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float],
        n_voters: int = 100,
        ai_influence: float = 0.2,
    ) -> GovernanceLedgerEntry:
        """
        Full governance cycle:
        1. Citizens vote on policy
        2. AI scans for drift/unintended outcomes
        3. Result recorded on tamper-evident ledger
        4. Entry auto-flagged for reversal if AI detects critical drift
        """
        tally = self.simulate_citizen_vote(policy, n_voters, ai_influence)
        ai_flags = self.detect_bias_drift(baseline_metrics, current_metrics)

        # Determine outcome
        if tally["for"] > tally["against"]:
            outcome = GovernanceVoteOutcome.APPROVED
        elif tally["against"] > tally["for"]:
            outcome = GovernanceVoteOutcome.REJECTED
        else:
            outcome = GovernanceVoteOutcome.DEFERRED

        # Override to REVERSED if AI detects critical drift and policy was approved
        if ai_flags and outcome == GovernanceVoteOutcome.APPROVED:
            outcome = GovernanceVoteOutcome.REVERSED
            ai_flags.append("Policy auto-reversed: AI detected critical bias drift post-vote.")

        entry = GovernanceLedgerEntry(
            entry_id=str(uuid.uuid4())[:8],
            policy=policy,
            vote_outcome=outcome,
            vote_tally=tally,
            ai_flags=ai_flags,
            previous_hash=self._last_hash,
        )
        self.ledger.append(entry)
        logger.info(f"[Governance] Policy='{policy}' → {outcome.value} | flags={len(ai_flags)}")
        return entry

    def get_ledger_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "entry_id": e.entry_id,
                "policy": e.policy,
                "outcome": e.vote_outcome.value,
                "tally": e.vote_tally,
                "ai_flags": e.ai_flags,
                "hash": e.hash,
                "previous_hash": e.previous_hash,
                "timestamp": e.timestamp,
            }
            for e in self.ledger
        ]


# ═══════════════════════════════════════════════════════════════════════════════
# FEATURE 5 — Strategic Social Reasoning Arena
# Game-like arena where AI agents practice deception, negotiation,
# and coalition-building under partial observability.
# ═══════════════════════════════════════════════════════════════════════════════

class AgentAction(str, Enum):
    COOPERATE  = "cooperate"
    DEFECT     = "defect"
    NEGOTIATE  = "negotiate"
    DECEIVE    = "deceive"
    COALESCE   = "coalesce"


@dataclass
class StrategicAgent:
    """Agent in the social-reasoning arena."""
    name: str
    strategy: str = "tit_for_tat"  # tit_for_tat | always_defect | random | coalition
    score: float = 0.0
    belief_state: Dict[str, float] = field(default_factory=dict)  # partial observability
    action_log: List[Dict] = field(default_factory=list)
    coalition: Optional[str] = None

    def choose_action(self, opponent_name: str, round_num: int) -> AgentAction:
        """Select action based on strategy and partial belief state."""
        last_opp_action = self.belief_state.get(f"{opponent_name}_last_action", "cooperate")

        if self.strategy == "tit_for_tat":
            if round_num == 0:
                return AgentAction.COOPERATE
            return AgentAction.COOPERATE if last_opp_action == "cooperate" else AgentAction.DEFECT

        elif self.strategy == "always_defect":
            return AgentAction.DEFECT

        elif self.strategy == "coalition":
            # Cooperate with coalition members, defect/negotiate with others
            if self.coalition and opponent_name.startswith(self.coalition):
                return AgentAction.COALESCE
            return AgentAction.NEGOTIATE

        elif self.strategy == "deceptive":
            # Cooperate early, then defect after building trust
            if round_num < 3:
                return AgentAction.COOPERATE
            return AgentAction.DECEIVE if np.random.rand() > 0.4 else AgentAction.DEFECT

        else:  # random
            return np.random.choice(list(AgentAction))

    def update_belief(self, opponent_name: str, observed_action: str, partial_obs_noise: float = 0.2) -> None:
        """
        Update belief about opponent's action under partial observability.
        With probability `partial_obs_noise`, the observation is corrupted.
        """
        if np.random.rand() < partial_obs_noise:
            # Corrupted observation — agent sees wrong action
            actions = [a.value for a in AgentAction]
            observed_action = np.random.choice([a for a in actions if a != observed_action])
        self.belief_state[f"{opponent_name}_last_action"] = observed_action


# Payoff matrix (row player, col player)
PAYOFF_MATRIX: Dict[Tuple[str, str], Tuple[float, float]] = {
    ("cooperate",  "cooperate"):  (3.0,  3.0),
    ("cooperate",  "defect"):     (-1.0, 5.0),
    ("defect",     "cooperate"):  (5.0, -1.0),
    ("defect",     "defect"):     (0.0,  0.0),
    ("negotiate",  "negotiate"):  (2.0,  2.0),
    ("negotiate",  "cooperate"):  (2.5,  2.5),
    ("negotiate",  "defect"):     (0.5,  1.5),
    ("deceive",    "cooperate"):  (4.5, -0.5),
    ("deceive",    "defect"):     (-0.5, 0.5),
    ("deceive",    "deceive"):    (-1.0,-1.0),
    ("coalesce",   "coalesce"):   (4.0,  4.0),
    ("coalesce",   "defect"):     (1.0,  2.0),
}

def _get_payoff(a1: str, a2: str) -> Tuple[float, float]:
    """Look up payoff, falling back to (0, 0) for unlisted combinations."""
    return PAYOFF_MATRIX.get((a1, a2), PAYOFF_MATRIX.get((a2, a1), (0.0, 0.0)))


class StrategicReasoningArena:
    """
    Feature 5 — Strategic Social Reasoning Arena.

    Runs a multi-agent iterated game with partial observability.
    Logs every action and belief update for post-hoc analysis.
    """

    def __init__(
        self,
        agents: Optional[List[StrategicAgent]] = None,
        partial_obs_noise: float = 0.2,
    ):
        self.agents = agents or self._default_agents()
        self.partial_obs_noise = partial_obs_noise
        self.round_logs: List[Dict] = []

    def _default_agents(self) -> List[StrategicAgent]:
        return [
            StrategicAgent("Alpha", strategy="tit_for_tat",   coalition="team_A"),
            StrategicAgent("Beta",  strategy="deceptive",      coalition="team_B"),
            StrategicAgent("Gamma", strategy="coalition",      coalition="team_A"),
            StrategicAgent("Delta", strategy="always_defect",  coalition=None),
        ]

    def run_round(self, round_num: int) -> Dict[str, Any]:
        """Run one round: all agents play pairwise, update beliefs and scores."""
        round_actions: Dict[str, str] = {}
        round_payoffs: Dict[str, float] = {a.name: 0.0 for a in self.agents}

        # Each agent chooses action vs every other agent
        for i, agent_a in enumerate(self.agents):
            for agent_b in self.agents[i+1:]:
                action_a = agent_a.choose_action(agent_b.name, round_num)
                action_b = agent_b.choose_action(agent_a.name, round_num)

                payoff_a, payoff_b = _get_payoff(action_a.value, action_b.value)
                agent_a.score += payoff_a
                agent_b.score += payoff_b
                round_payoffs[agent_a.name] += payoff_a
                round_payoffs[agent_b.name] += payoff_b

                # Update beliefs (with partial observability noise)
                agent_a.update_belief(agent_b.name, action_b.value, self.partial_obs_noise)
                agent_b.update_belief(agent_a.name, action_a.value, self.partial_obs_noise)

                pair_key = f"{agent_a.name}_vs_{agent_b.name}"
                round_actions[pair_key] = {"action_a": action_a.value, "action_b": action_b.value}

                agent_a.action_log.append({"round": round_num, "opponent": agent_b.name, "action": action_a.value, "payoff": payoff_a})
                agent_b.action_log.append({"round": round_num, "opponent": agent_a.name, "action": action_b.value, "payoff": payoff_b})

        round_log = {
            "round": round_num,
            "actions": round_actions,
            "payoffs": {k: round(v, 3) for k, v in round_payoffs.items()},
        }
        self.round_logs.append(round_log)
        return round_log

    def run_tournament(self, n_rounds: int = 10) -> Dict[str, Any]:
        """Run a full tournament and return comprehensive analysis."""
        for r in range(n_rounds):
            self.run_round(r)

        # Deception detection: flag agents whose stated action differed from belief
        deception_counts = {a.name: 0 for a in self.agents}
        for a in self.agents:
            deception_counts[a.name] = sum(1 for log in a.action_log if log["action"] == "deceive")

        # Coalition effectiveness
        coalition_scores: Dict[str, float] = {}
        for a in self.agents:
            if a.coalition:
                coalition_scores[a.coalition] = coalition_scores.get(a.coalition, 0.0) + a.score

        final_standings = sorted(
            [{"agent": a.name, "strategy": a.strategy, "score": round(a.score, 3),
              "deception_count": deception_counts[a.name], "coalition": a.coalition}
             for a in self.agents],
            key=lambda x: x["score"], reverse=True
        )

        logger.info(f"[Arena] Tournament complete ({n_rounds} rounds). Winner: {final_standings[0]['agent']}")
        return {
            "n_rounds": n_rounds,
            "final_standings": final_standings,
            "coalition_scores": {k: round(v, 3) for k, v in coalition_scores.items()},
            "deception_counts": deception_counts,
            "round_logs": self.round_logs,
            "partial_obs_noise": self.partial_obs_noise,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Original Core — preserved and improved
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic_data(
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 10,
        feature_range: Tuple[float, float] = simulation_config.FEATURE_RANGE,
        decision_boundary: float = simulation_config.DECISION_BOUNDARY,
        noise_level: float = 0.1,
        demographic_groups: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate realistic synthetic data for healthcare fairness analysis."""
    try:
        X = np.zeros((n_samples, n_features))

        # Correlated features
        for i in range(0, n_features, 3):
            if i + 1 < n_features:
                correlation = 0.6
                X[:, i + 1] = correlation * X[:, i] + (1 - correlation) * np.random.normal(0, 1, n_samples)

        # Fill any empty columns (fixed: was re-filling already-correlated cols)
        for i in range(n_features):
            if np.all(X[:, i] == 0):
                X[:, i] = np.random.uniform(feature_range[0], feature_range[1], n_samples)

        # Healthcare-realistic features
        X[:, 0] = np.clip(np.random.normal(50, 15, n_samples), feature_range[0], feature_range[1])
        X[:, 1] = np.random.lognormal(mean=3.0, sigma=0.5, size=n_samples)
        X[:, 1] = (X[:, 1] - X[:, 1].min()) / (X[:, 1].max() - X[:, 1].min())
        X[:, 1] = X[:, 1] * (feature_range[1] - feature_range[0]) + feature_range[0]

        age_effect   = 1 / (1 + np.exp(-(X[:, 0] - 50) / 10))
        income_effect = np.log(X[:, 1] + 1) / 5
        other_effect  = np.sum(X[:, 2:5], axis=1) / 3

        risk_score = (age_effect * 0.4 + income_effect * 0.3 + other_effect * 0.3
                      + noise_level * np.random.normal(0, 1, n_samples))
        y = (risk_score > decision_boundary).astype(int)
        demographic_info = np.random.randint(0, demographic_groups, n_samples)

        logger.info(f"Generated {n_samples} samples, {n_features} features. "
                    f"Positive class: {np.mean(y):.1%}")
        return X, y, demographic_info

    except Exception as e:
        logger.error(f"Error generating synthetic data: {e}")
        raise


class BiasInjector:
    """Centralised bias injection with severity levels."""

    @staticmethod
    def demographic_bias(X, y, bias_factor, demographic_info):
        minority_mask = (demographic_info == 0)
        if np.any(minority_mask):
            X[minority_mask, :3] *= (1 - bias_factor * 2.0)
            minority_pos = minority_mask & (y == 1)
            if np.any(minority_pos):
                flip_mask = np.random.rand(np.sum(minority_pos)) < bias_factor * 0.8
                y[minority_pos] = np.where(flip_mask, 0, 1)
        return X, y

    @staticmethod
    def historical_bias(X, y, bias_factor, demographic_info):
        for group in np.unique(demographic_info):
            group_mask = (demographic_info == group)
            if np.any(group_mask):
                noise_proba = min(0.5, bias_factor * (1 + group * 0.5))
                noise_mask  = np.random.rand(np.sum(group_mask)) < noise_proba
                noisy_indices = np.where(group_mask)[0][noise_mask]
                y[noisy_indices] = 1 - y[noisy_indices]
        return X, y

    @staticmethod
    def selection_bias(X, y, bias_factor, demographic_info):
        n_samples = len(X)
        keep_probabilities = np.ones(n_samples)
        for group in np.unique(demographic_info):
            for outcome in [0, 1]:
                mask = (demographic_info == group) & (y == outcome)
                if np.any(mask):
                    if group == 0 and outcome == 1:
                        keep_probabilities[mask] = 1 - bias_factor * 0.9
                    elif group == 1 and outcome == 0:
                        keep_probabilities[mask] = 1 - bias_factor * 0.3
        keep_mask = np.random.rand(n_samples) < keep_probabilities
        logger.info(f"Selection bias: kept {np.sum(keep_mask)}/{n_samples} samples")
        return X[keep_mask], y[keep_mask], demographic_info[keep_mask]

    @staticmethod
    def measurement_bias(X, y, bias_factor, demographic_info):
        for group in np.unique(demographic_info):
            group_mask = (demographic_info == group)
            if np.any(group_mask):
                noise_scale = simulation_config.NOISE_STD * (1 + group * bias_factor)
                X[group_mask] += np.random.normal(0, noise_scale, (np.sum(group_mask), X.shape[1]))
        return X, y

    # Feature 3: gender bias
    @staticmethod
    def gender_bias(X, y, bias_factor, demographic_info):
        """Reduce model accuracy for female group (group 0) — for audit testing."""
        female_mask = (demographic_info == 0)
        if np.any(female_mask):
            X[female_mask, :2] *= (1 - bias_factor * 1.5)
            flip = np.random.rand(np.sum(female_mask)) < bias_factor * 0.6
            y[female_mask] = np.where(flip, 1 - y[female_mask], y[female_mask])
        return X, y

    # Feature 3: linguistic bias
    @staticmethod
    def linguistic_bias(X, y, bias_factor, demographic_info):
        """Simulate measurement noise from multilingual input inconsistencies."""
        for group in np.unique(demographic_info):
            mask = (demographic_info == group)
            if np.any(mask):
                lang_noise = bias_factor * (1 + 0.3 * group)
                X[mask, 3:7] += np.random.normal(0, lang_noise, (np.sum(mask), min(4, X.shape[1] - 3)))
        return X, y

    # Feature 1–5: temporal / geographic / socioeconomic — now proper methods
    @staticmethod
    def temporal_bias(X, y, bias_factor, demographic_info):
        drift = np.linspace(0, bias_factor * 2, len(X))[:, np.newaxis]
        X += drift * np.random.normal(0, 0.5, X.shape)
        return X, y

    @staticmethod
    def geographic_bias(X, y, bias_factor, demographic_info):
        geo_factor = bias_factor * (X[:, 1] - X[:, 1].mean()) / (X[:, 1].std() + 1e-8)
        X[:, 2:] += geo_factor[:, np.newaxis]
        return X, y

    @staticmethod
    def socioeconomic_bias(X, y, bias_factor, demographic_info):
        X[:, 3:6] *= (1 + bias_factor * np.random.uniform(-0.5, 0.5, (len(X), 3)))
        return X, y


def apply_bias(
        X: np.ndarray,
        y: np.ndarray,
        bias_type: str,
        bias_factor: float = simulation_config.MAX_BIAS_FACTOR,
        demographic_info: Optional[np.ndarray] = None,
        severity: AttackSeverity = AttackSeverity.MEDIUM
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply specified bias to dataset with configurable severity."""
    if bias_factor <= 0 or bias_type not in simulation_config.BIAS_TYPES:
        return X, y, demographic_info

    if demographic_info is None:
        demographic_info = np.zeros(len(X))

    severity_multiplier = {
        AttackSeverity.LOW: 0.5, AttackSeverity.MEDIUM: 1.0,
        AttackSeverity.HIGH: 1.5, AttackSeverity.CRITICAL: 2.0,
    }
    adjusted_bias = bias_factor * severity_multiplier.get(severity, 1.0)
    inj = BiasInjector()

    try:
        dispatch = {
            BiasType.DEMOGRAPHIC:    lambda: inj.demographic_bias(X, y, adjusted_bias, demographic_info),
            BiasType.HISTORICAL:     lambda: inj.historical_bias(X, y, adjusted_bias, demographic_info),
            BiasType.MEASUREMENT:    lambda: inj.measurement_bias(X, y, adjusted_bias, demographic_info),
            BiasType.TEMPORAL:       lambda: inj.temporal_bias(X, y, adjusted_bias, demographic_info),
            BiasType.GEOGRAPHIC:     lambda: inj.geographic_bias(X, y, adjusted_bias, demographic_info),
            BiasType.SOCIOECONOMIC:  lambda: inj.socioeconomic_bias(X, y, adjusted_bias, demographic_info),
            BiasType.GENDER:         lambda: inj.gender_bias(X, y, adjusted_bias, demographic_info),
            BiasType.LINGUISTIC:     lambda: inj.linguistic_bias(X, y, adjusted_bias, demographic_info),
        }

        if bias_type in [BiasType.SELECTION, BiasType.REPRESENTATION]:
            return inj.selection_bias(X, y, adjusted_bias, demographic_info)

        fn = dispatch.get(bias_type)
        if fn:
            result = fn()
            # Some methods return 2 values, others 3
            if len(result) == 3:
                return result
            X, y = result
        logger.info(f"Applied {bias_type} bias with factor {adjusted_bias:.3f}")

    except Exception as e:
        logger.error(f"Error applying {bias_type} bias: {e}")

    return X, y, demographic_info


def simulate_data_poisoning(
        X: np.ndarray,
        y: np.ndarray,
        poison_rate: float = simulation_config.ATTACK_TYPES["data_poisoning"]["default"],
        attack_type: str = "label_flipping",
        targeted: bool = False,
        demographic_info: Optional[np.ndarray] = None,
        target_group: Optional[int] = None,
        attack_sophistication: str = "medium",
        feature_columns: Optional[List[int]] = None,
        **kwargs
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Simulate various data poisoning attacks. (Unchanged logic — fixed return type only.)"""
    if poison_rate <= 0:
        return X, y, demographic_info

    n_samples = len(X)
    n_features = X.shape[1]
    n_poison = int(n_samples * poison_rate)
    if n_poison == 0:
        return X, y, demographic_info

    sophistication_params = {
        "low":      {"noise_scale": 0.5,  "pattern_strength": 0.3,  "strategic": False},
        "medium":   {"noise_scale": 1.0,  "pattern_strength": 0.6,  "strategic": True},
        "high":     {"noise_scale": 1.5,  "pattern_strength": 0.9,  "strategic": True},
        "advanced": {"noise_scale": 2.0,  "pattern_strength": 1.2,  "strategic": True},
    }
    attack_params = sophistication_params.get(attack_sophistication.lower(), sophistication_params["medium"])

    if targeted and demographic_info is not None:
        if target_group is None:
            unique_groups, counts = np.unique(demographic_info, return_counts=True)
            target_group = int(unique_groups[np.argmin(counts)])
        target_indices = np.where(demographic_info == target_group)[0]
        if len(target_indices) == 0:
            logger.warning(f"Target group {target_group} not found. Switching to random poisoning.")
            poison_idx = np.random.choice(n_samples, n_poison, replace=False)
        else:
            n_poison = min(n_poison, len(target_indices))
            poison_idx = np.random.choice(target_indices, n_poison, replace=False)
    else:
        poison_idx = np.random.choice(n_samples, n_poison, replace=False)

    X_poisoned = X.copy()
    y_poisoned = y.copy()
    feature_columns = feature_columns or list(range(n_features))

    if attack_type == "label_flipping":
        y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]

    elif attack_type == "feature_noise":
        noise = np.random.normal(0, attack_params["noise_scale"], (n_poison, len(feature_columns)))
        if attack_params["strategic"]:
            for i, col in enumerate(feature_columns):
                fm = np.mean(X[:, col])
                noise[:, i] *= (X_poisoned[poison_idx, col] - fm) / (np.std(X[:, col]) + 1e-8)
        X_poisoned[np.ix_(poison_idx, feature_columns)] += noise

    elif attack_type == "backdoor":
        pattern_strength = attack_params["pattern_strength"]
        n_bd = max(1, int(n_features * 0.3))
        bd_feats = np.random.choice(feature_columns, n_bd, replace=False)
        bd_pattern = np.zeros(n_features)
        for i, f in enumerate(bd_feats):
            bd_pattern[f] = pattern_strength * (1 if i % 2 == 0 else -1)
        X_poisoned[poison_idx] += bd_pattern
        y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]

    elif attack_type == "outlier_injection":
        outlier_strength = attack_params["noise_scale"] * 3
        for col in feature_columns:
            fm, fs = np.mean(X[:, col]), np.std(X[:, col])
            outliers = np.random.choice([-1, 1], n_poison) * outlier_strength * fs
            X_poisoned[poison_idx, col] = fm + outliers
        y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]

    else:
        logger.warning(f"Unknown attack type: {attack_type}. Defaulting to label_flipping.")
        y_poisoned[poison_idx] = 1 - y_poisoned[poison_idx]

    logger.info(f"Poisoning ({attack_type}): {n_poison} samples affected.")
    return X_poisoned, y_poisoned, demographic_info


def simple_data_poisoning(X, y, poison_rate=simulation_config.ATTACK_TYPES["data_poisoning"]["default"]):
    """Backward-compatible wrapper."""
    X_p, y_p, _ = simulate_data_poisoning(X, y, poison_rate, attack_type="label_flipping")
    return X_p, y_p


def detect_poisoning_attempt(
        X: np.ndarray,
        y: np.ndarray,
        detection_method: str = "statistical",
        contamination: float = 0.1,
        **kwargs
) -> Dict[str, Any]:
    """Detect poisoned samples. Sklearn imports moved to top of file."""
    n_samples = len(X)

    if detection_method == "statistical":
        detector = IsolationForest(contamination=contamination, random_state=42, **kwargs)
        is_outlier = detector.fit_predict(X) == -1

    elif detection_method == "clustering":
        detector = LocalOutlierFactor(contamination=contamination, novelty=False, **kwargs)
        is_outlier = detector.fit_predict(X) == -1

    elif detection_method == "model_based":
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        y_prob = cross_val_predict(model, X, y, cv=5, method="predict_proba")
        confidence = y_prob[np.arange(len(y)), y.astype(int)]
        threshold = np.percentile(confidence, contamination * 100)
        is_outlier = confidence < threshold

    else:
        raise ValueError(f"Unknown detection method: {detection_method}")

    n_detected = int(np.sum(is_outlier))
    return {
        "detection_method": detection_method,
        "n_detected": n_detected,
        "detection_rate": round(n_detected / n_samples, 4),
        "contamination_estimate": contamination,
        "outlier_indices": np.where(is_outlier)[0].tolist(),
        "is_outlier": is_outlier,
    }


def calculate_fairness_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        demographic_info: np.ndarray
) -> Dict[str, Any]:
    """Calculate comprehensive fairness metrics."""
    groups = np.unique(demographic_info)
    if len(groups) < 2:
        return {"fairness_score": 1.0, "disparity": 0.0, "group_metrics": {}}

    # Pre-compute all masks at once (performance fix)
    masks = {int(g): (demographic_info == g) for g in groups}

    group_metrics: Dict[int, Dict] = {}
    for group, mask in masks.items():
        if np.sum(mask) == 0:
            continue
        gt, gp = y_true[mask], y_pred[mask]
        tp = int(np.sum((gp == 1) & (gt == 1)))
        fp = int(np.sum((gp == 1) & (gt == 0)))
        tn = int(np.sum((gp == 0) & (gt == 0)))
        fn = int(np.sum((gp == 0) & (gt == 1)))
        group_metrics[group] = {
            "accuracy":    round(float(np.mean(gp == gt)), 4),
            "error_rate":  round(float(np.mean(gp != gt)), 4),
            "fpr":         round(fp / (fp + tn) if (fp + tn) > 0 else 0.0, 4),
            "fnr":         round(fn / (fn + tp) if (fn + tp) > 0 else 0.0, 4),
            "sample_size": int(np.sum(mask)),
        }

    positive_rates = [m["accuracy"] for m in group_metrics.values()]
    fprs = [m["fpr"] for m in group_metrics.values()]
    fnrs = [m["fnr"] for m in group_metrics.values()]
    parity_diff       = float(max(positive_rates) - min(positive_rates))
    fpr_disparity     = float(max(fprs) - min(fprs))
    fnr_disparity     = float(max(fnrs) - min(fnrs))
    eq_odds_diff      = float(max(fpr_disparity, fnr_disparity))
    fairness_score    = float(np.clip(1.0 - (parity_diff + eq_odds_diff) / 2, 0, 1))

    return {
        "fairness_score":                  round(fairness_score, 4),
        "demographic_parity_difference":   round(parity_diff, 4),
        "equalized_odds_difference":       round(eq_odds_diff, 4),
        "fpr_disparity":                   round(fpr_disparity, 4),
        "fnr_disparity":                   round(fnr_disparity, 4),
        "group_metrics":                   group_metrics,   # always present
    }


def run_simple_simulation(
        bias_types: List[str],
        bias_factor: float = 0.3,
        poison_rate: float = 0.1,
        n_samples: int = settings.DEFAULT_N_SAMPLES,
        n_features: int = 10,
        attack_type: str = "label_flipping",
        include_detailed_metrics: bool = False,
        # Feature 1: agent economy
        run_agent_economy: bool = False,
        economy_domain: str = "agrotech",
        # Feature 2: multimodal red-team
        run_multimodal_redteam: bool = False,
        redteam_domain: str = "healthcare",
        # Feature 4: governance voting
        run_governance_vote: bool = False,
        governance_policy: str = "Deploy AI triage in public hospitals",
        # Feature 5: strategic arena
        run_strategic_arena: bool = False,
        arena_rounds: int = 10,
) -> Dict[str, Any]:
    """
    End-to-end simulation of bias and attacks on healthcare/governance AI.
    Optionally activates Feature modules 1, 2, 4, and 5.

    Returns a consistent envelope:
    {
      "status": "success" | "error",
      "mode": "simple" | "detailed",
      "results": { ... core metrics ... },
      "features": { ... optional module outputs ... },
      "warnings": [ ... ],
      "metadata": { ... },
    }
    """
    warnings_list: List[str] = []

    # ── Input validation ──────────────────────────────────────────────────────
    if not (0 <= bias_factor <= 1):
        warnings_list.append(f"bias_factor={bias_factor} out of [0,1]. Clamping.")
        bias_factor = float(np.clip(bias_factor, 0, 1))
    if not (0 <= poison_rate <= 1):
        warnings_list.append(f"poison_rate={poison_rate} out of [0,1]. Clamping.")
        poison_rate = float(np.clip(poison_rate, 0, 1))

    invalid_biases = [b for b in bias_types if b not in simulation_config.BIAS_TYPES]
    if invalid_biases:
        warnings_list.append(f"Ignored unknown bias types: {invalid_biases}")
        bias_types = [b for b in bias_types if b in simulation_config.BIAS_TYPES]

    logger.info(f"Simulation start — samples={n_samples}, biases={bias_types}, "
                f"poison_rate={poison_rate}, attack={attack_type}")

    # ── 1. Data generation ────────────────────────────────────────────────────
    X, y_true, demographic_info = generate_synthetic_data(n_samples=n_samples, n_features=n_features)
    original_size = len(X)
    applied_biases: List[str] = []

    # ── 2. Bias injection ─────────────────────────────────────────────────────
    for bias_type in bias_types:
        X, y_true, demographic_info = apply_bias(X, y_true, bias_type, bias_factor, demographic_info)
        applied_biases.append(bias_type)

    # ── 3. Poisoning attack ───────────────────────────────────────────────────
    X, y_noisy, demographic_info = simulate_data_poisoning(X, y_true, poison_rate, attack_type, demographic_info)

    # ── 4. Simulated predictions ──────────────────────────────────────────────
    y_pred = y_noisy.copy()
    risk_scores = np.sum(X[:, :3], axis=1) / 3
    uncertainty = 1 / (1 + np.exp(-(risk_scores - 0.5) * 10))
    for i in range(len(y_pred)):
        if np.random.rand() < uncertainty[i] * 0.3:
            y_pred[i] = 1 - y_pred[i]

    # ── 5. Metrics ────────────────────────────────────────────────────────────
    accuracy = float(np.mean(y_pred == y_true))
    fairness_metrics = calculate_fairness_metrics(y_true, y_pred, demographic_info)

    data_distribution = {
        "original_size":             original_size,
        "final_size":                len(X),
        "class_balance_original":    round(float(np.mean(y_true)), 4),
        "class_balance_final":       round(float(np.mean(y_noisy)), 4),
        "demographic_distribution":  {
            int(g): round(float(np.mean(demographic_info == g)), 4)
            for g in np.unique(demographic_info)
        },
    }

    sim_result = SimulationResult(
        accuracy=round(accuracy, 4),
        fairness_score=fairness_metrics.get("fairness_score", 0.5),
        fairness_metrics=fairness_metrics,
        poisoned_samples=int(len(X) * poison_rate),
        applied_biases=applied_biases,
        sample_size_after_bias=len(X),
        data_distribution=data_distribution,
        performance_history=[round(accuracy, 4)],
        metadata={
            "bias_factor": bias_factor, "poison_rate": poison_rate,
            "attack_type": attack_type, "n_features": n_features,
            "simulation_version": "3.0",
        },
        warnings=warnings_list,
    )

    # ── 6. Optional feature modules ───────────────────────────────────────────
    feature_outputs: Dict[str, Any] = {}

    if run_agent_economy:
        sandbox = AgentEconomySandbox(domain=economy_domain)
        feature_outputs["agent_economy"] = sandbox.run_full_simulation()

    if run_multimodal_redteam:
        redteamer = MultimodalRedTeamer()
        feature_outputs["multimodal_redteam"] = redteamer.run_combined_attack(X, y_noisy, redteam_domain)
        # Strip arrays from output (not JSON-serialisable in default mode)
        feature_outputs["multimodal_redteam"].pop("final_X", None)
        feature_outputs["multimodal_redteam"].pop("final_y", None)

    if run_governance_vote:
        baseline = {"accuracy": 0.85, "fairness_score": 0.90, "demographic_parity": 0.02}
        current  = {"accuracy": accuracy, "fairness_score": fairness_metrics.get("fairness_score", 0.5),
                    "demographic_parity": fairness_metrics.get("demographic_parity_difference", 0.0)}
        gov = HybridGovernanceLayer()
        entry = gov.propose_and_vote(governance_policy, baseline, current)
        feature_outputs["governance"] = {
            "policy": governance_policy,
            "outcome": entry.vote_outcome.value,
            "tally": entry.vote_tally,
            "ai_flags": entry.ai_flags,
            "ledger_hash": entry.hash,
        }

    if run_strategic_arena:
        arena = StrategicReasoningArena()
        feature_outputs["strategic_arena"] = arena.run_tournament(n_rounds=arena_rounds)

    logger.info(f"Simulation complete — accuracy={accuracy:.3f}, "
                f"fairness={sim_result.fairness_score:.3f}, warnings={len(warnings_list)}")

    # ── 7. Consistent return envelope ─────────────────────────────────────────
    core_results = sim_result.__dict__ if include_detailed_metrics else {
        "accuracy":                 sim_result.accuracy,
        "fairness_score":           sim_result.fairness_score,
        "applied_biases":           applied_biases,
        "poisoned_samples":         sim_result.poisoned_samples,
        "sample_size_after_bias":   sim_result.sample_size_after_bias,
        "demographic_parity":       fairness_metrics.get("demographic_parity_difference", 0.0),
        "equalized_odds":           fairness_metrics.get("equalized_odds_difference", 0.0),
        "group_metrics":            fairness_metrics.get("group_metrics", {}),
        "data_distribution":        data_distribution,
    }

    return {
        "status":   "success",
        "mode":     "detailed" if include_detailed_metrics else "simple",
        "results":  core_results,
        "features": feature_outputs,
        "warnings": warnings_list,
        "metadata": sim_result.metadata,
    }


def simulate_bias_mitigation(
        X: np.ndarray,
        y: np.ndarray,
        demographic_info: np.ndarray,
        mitigation_strategy: str = "reweighting"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply bias mitigation strategies to the data."""
    if mitigation_strategy == "reweighting":
        weights = np.ones(len(X))
        for group in np.unique(demographic_info):
            mask = (demographic_info == group)
            weights[mask] = 1 / (np.mean(demographic_info == group) + 1e-8)
        weights = weights / weights.sum() * len(X)
        indices = np.random.choice(len(X), size=len(X), replace=True, p=weights / weights.sum())
        return X[indices], y[indices], demographic_info[indices]

    elif mitigation_strategy == "oversampling":
        groups, counts = np.unique(demographic_info, return_counts=True)
        max_count = int(np.max(counts))
        all_X, all_y, all_demo = [], [], []

        for group in groups:
            mask = (demographic_info == group)
            X_g, y_g = X[mask], y[mask]
            indices = np.random.choice(len(X_g), size=max_count, replace=True)
            all_X.append(X_g[indices])
            all_y.append(y_g[indices])
            all_demo.append(np.full(max_count, group))

        return np.vstack(all_X), np.concatenate(all_y), np.concatenate(all_demo)

    return X, y, demographic_info