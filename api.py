# api.py
"""
GAGS REST API — v1.0
====================
FastAPI-based REST endpoint that exposes the GAGS simulation engine
for programmatic use from MLOps pipelines, CI/CD fairness gates,
and institutional integrations.

Start the server
----------------
    pip install fastapi uvicorn
    uvicorn api:app --host 0.0.0.0 --port 8502 --reload

Or with gunicorn for production:
    gunicorn api:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8502

API documentation (auto-generated)
-----------------------------------
    http://localhost:8502/docs      (Swagger UI)
    http://localhost:8502/redoc     (ReDoc)

Endpoints
---------
    GET  /health                        System health check
    GET  /scenarios                     List community scenarios
    GET  /scenarios/{id}                Get a single scenario
    GET  /plugins                       List registered plugins
    POST /simulate                      Run a simulation (main endpoint)
    POST /simulate/bias-only            Run bias injection only (no model training)
    POST /simulate/fairness-check       Evaluate fairness on provided predictions
    POST /xai/explain                   Explain a model prediction (LIME-lite)
    POST /xai/counterfactual            Generate counterfactual
    POST /compliance/report             Generate compliance report
    GET  /compliance/frameworks         List supported regulatory frameworks
    POST /monitor/snapshot              Submit a monitoring snapshot
    GET  /monitor/status                Get monitoring status

Authentication
--------------
    Set GAGS_API_KEY environment variable.
    Pass as: Authorization: Bearer <key>
    If not set, API runs in open mode (development only).

Example usage
-------------
    import requests

    resp = requests.post("http://localhost:8502/simulate", json={
        "domain":           "healthcare",
        "bias_types":       ["demographic", "socioeconomic"],
        "bias_intensity":   0.3,
        "poison_rate":      0.05,
        "n_samples":        5000,
        "include_xai":      True,
        "include_compliance": True,
    })
    result = resp.json()
    print(result["fairness"]["fairness_score"])
    print(result["compliance"]["summary"]["overall_compliant"])
"""

import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

# ── FastAPI imports (graceful failure if not installed) ───────────────────────
try:
    from fastapi import FastAPI, HTTPException, Depends, Header, status
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, field_validator
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False
    # Stub classes so the module still imports in non-API environments
    class BaseModel:
        pass
    def Field(*a, **kw): return None
    FastAPI = None

# ── GAGS core imports ─────────────────────────────────────────────────────────
from components.governance_logic import (
    generate_education_data, calculate_education_equity, EDUCATION_SCENARIO_PRESETS,
    generate_financial_data, calculate_financial_fairness, FINANCIAL_SCENARIO_PRESETS,
    generate_judicial_data, calculate_judicial_fairness, JUDICIAL_SCENARIO_PRESETS,
    generate_disinformation_data, calculate_disinformation_fairness, DISINFORMATION_SCENARIO_PRESETS,

    generate_synthetic_data,
    apply_bias,
    simulate_data_poisoning,
    calculate_fairness_metrics,
    simulate_bias_mitigation,
    COMMUNITY_SCENARIOS,
    list_scenarios,
    scenario_to_simulation_config,
    plugin_registry,
    ExplainableModel,
    generate_compliance_report,
    AttackSeverity,
)
from utils.config import simulation_config, settings

# ── Auth helper ───────────────────────────────────────────────────────────────

_API_KEY = os.environ.get("GAGS_API_KEY", "")

def _check_auth(authorization: Optional[str] = Header(default=None)) -> None:
    if not _API_KEY:
        return  # Open mode — no key configured
    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Missing Authorization header. Use: Bearer <key>")
    token = authorization.removeprefix("Bearer ").strip()
    if token != _API_KEY:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN,
                            detail="Invalid API key.")

# ── Pydantic models ───────────────────────────────────────────────────────────

class SimulationRequest(BaseModel):
    domain:               str                  = Field("healthcare",
        description="healthcare | national_security | agrotech")
    bias_types:           List[str]            = Field(["demographic"],
        description="List of bias type strings")
    bias_intensity:       float                = Field(0.25, ge=0.0, le=1.0)
    poison_rate:          float                = Field(0.05, ge=0.0, le=0.5)
    n_samples:            int                  = Field(5000, ge=100, le=100_000)
    n_features:           int                  = Field(12,  ge=2,   le=50)
    attack_type:          str                  = Field("label_flipping")
    mitigation_strategy:  Optional[str]        = Field(None,
        description="reweighting | oversampling | None")
    include_xai:          bool                 = Field(False)
    include_compliance:   bool                 = Field(False)
    compliance_frameworks: Optional[List[str]] = None
    random_seed:          int                  = Field(42)
    scenario_id:          Optional[str]        = Field(None,
        description="Apply a community scenario (overrides bias_types/intensity/poison_rate)")

    if _FASTAPI_AVAILABLE:
        @field_validator("bias_types")
        @classmethod
        def validate_bias_types(cls, v):
            valid = list(simulation_config.BIAS_TYPES) + ["gender", "linguistic"]
            invalid = [b for b in v if b not in valid]
            if invalid:
                raise ValueError(f"Unknown bias types: {invalid}. Valid: {valid}")
            return v

        @field_validator("domain")
        @classmethod
        def validate_domain(cls, v):
            if v not in ("healthcare","national_security","agrotech","education","finance","judicial","disinformation","generic"):
                raise ValueError(f"Unknown domain '{v}'")
            return v


class FairnessCheckRequest(BaseModel):
    y_true:           List[int]   = Field(description="Ground-truth labels (0/1)")
    y_pred:           List[int]   = Field(description="Model predictions (0/1)")
    demographic_info: List[int]   = Field(description="Demographic group per sample")


class ExplainRequest(BaseModel):
    X_train:      List[List[float]] = Field(description="Training feature matrix")
    y_train:      List[int]         = Field(description="Training labels")
    instance:     List[float]       = Field(description="Instance to explain")
    domain:       str               = Field("generic")
    feature_names: Optional[List[str]] = None


class MonitorSnapshot(BaseModel):
    model_id:         str
    timestamp:        Optional[str] = None
    y_true:           List[int]
    y_pred:           List[int]
    demographic_info: List[int]
    metadata:         Dict[str, Any] = Field(default_factory=dict)


# ── In-memory monitoring store ────────────────────────────────────────────────
# For production: replace with a persistent store (Redis, Postgres, etc.)

_MONITOR_STORE: Dict[str, List[Dict]] = {}
_DRIFT_THRESHOLDS = {
    "fairness_score":              ("below", 0.70),
    "demographic_parity_difference": ("above", 0.10),
    "equalized_odds_difference":   ("above", 0.10),
    "accuracy":                    ("below", 0.55),
}


def _check_drift(current: Dict, history: List[Dict]) -> List[str]:
    """Compare current snapshot to threshold and trend. Returns list of alerts."""
    alerts = []
    for metric, (direction, threshold) in _DRIFT_THRESHOLDS.items():
        val = current.get(metric)
        if val is None:
            continue
        if direction == "below" and val < threshold:
            alerts.append(f"{metric} dropped to {val:.3f} (threshold: {threshold})")
        elif direction == "above" and val > threshold:
            alerts.append(f"{metric} exceeded {val:.3f} (threshold: {threshold})")

    # Trend check: compare last 3 snapshots
    if len(history) >= 3:
        for metric in ["fairness_score", "accuracy"]:
            recent = [s.get(metric, 0) for s in history[-3:]]
            if all(recent[i] > recent[i+1] for i in range(2)):
                alerts.append(f"{metric} declining over last 3 snapshots: "
                               f"{recent[0]:.3f} → {recent[1]:.3f} → {recent[2]:.3f}")

    return alerts


# ── Core simulation function (shared by API and monitoring) ───────────────────

def _run_core_simulation(req: SimulationRequest) -> Dict[str, Any]:
    """Execute one simulation run and return a structured result dict."""
    rng = np.random.RandomState(req.random_seed)

    # Apply scenario if provided
    if req.scenario_id:
        scen = COMMUNITY_SCENARIOS.get(req.scenario_id)
        if not scen:
            raise HTTPException(404, f"Scenario '{req.scenario_id}' not found.")
        cfg = scenario_to_simulation_config(scen)
        bias_types    = cfg["selected_biases"]
        bias_intensity = cfg["bias_intensity"]
        poison_rate    = cfg["poison_rate"]
    else:
        bias_types    = req.bias_types
        bias_intensity = req.bias_intensity
        poison_rate    = req.poison_rate

    # Generate data
    X, y, demo = generate_synthetic_data(
        n_samples=req.n_samples,
        n_features=req.n_features,
        decision_boundary=simulation_config.DECISION_BOUNDARY,
        random_state=req.random_seed,
    )
    X = X.astype(np.float64)

    # Bias injection
    valid_biases = [b for b in bias_types
                    if b in list(simulation_config.BIAS_TYPES) + ["gender","linguistic"]]
    for bt in valid_biases:
        X, y, demo = apply_bias(X, y, bt, bias_intensity,
                                demographic_info=demo,
                                severity=AttackSeverity.MEDIUM)

    # Poisoning
    X, y, demo = simulate_data_poisoning(
        X, y, poison_rate,
        attack_type=req.attack_type,
        demographic_info=demo,
        targeted=False,
    )

    # Mitigation (optional)
    mitigation_applied = None
    if req.mitigation_strategy:
        X, y, demo = simulate_bias_mitigation(X, y, demo, req.mitigation_strategy)
        mitigation_applied = req.mitigation_strategy

    # Train / evaluate
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te, d_tr, d_te = train_test_split(
        X_sc, y, demo, test_size=0.3, random_state=req.random_seed,
        stratify=y if len(np.unique(y)) > 1 else None,
    )

    clf = RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                  random_state=req.random_seed)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    acc  = float(accuracy_score(y_te, y_pred))
    rec  = float(recall_score(y_te, y_pred, zero_division=0))
    prec = float(precision_score(y_te, y_pred, zero_division=0))
    f1   = float(f1_score(y_te, y_pred, zero_division=0))
    fpr  = float(np.mean(y_pred[y_te==0]==1)) if (y_te==0).any() else 0.0
    fair = calculate_fairness_metrics(y_te, y_pred, d_te)

    result: Dict[str, Any] = {
        "request_id":  str(uuid.uuid4())[:12],
        "timestamp":   datetime.now().isoformat(),
        "domain":      req.domain,
        "n_samples":   req.n_samples,
        "bias_types":  valid_biases,
        "bias_intensity": bias_intensity,
        "poison_rate": poison_rate,
        "mitigation":  mitigation_applied,
        "performance": {
            "accuracy":  round(acc, 4),
            "recall":    round(rec, 4),
            "precision": round(prec, 4),
            "f1_score":  round(f1, 4),
            "fpr":       round(fpr, 4),
        },
        "fairness": {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in fair.items() if k != "group_metrics"},
        "group_metrics": {str(k): {mk: round(mv, 4) if isinstance(mv, float) else mv
                                   for mk, mv in mv_.items()}
                          for k, mv_ in fair.get("group_metrics", {}).items()},
    }

    # Custom metric plugins
    custom_metrics = plugin_registry.apply_custom_metrics(y_te, y_pred, d_te, req.domain)
    if custom_metrics:
        result["custom_metrics"] = custom_metrics

    # XAI
    if req.include_xai:
        try:
            xm = ExplainableModel(domain=req.domain)
            xm.model = clf
            xm._X_train = X_tr
            xm._is_fitted = True
            fi = xm.feature_importance(X_te, y_te, n_repeats=5)
            high_risk = np.where(y_te == 1)[0]
            result["xai"] = {
                "feature_importance": {
                    "top_features": fi.top_k,
                    "importances":  dict(zip(fi.feature_names[:10], fi.importances[:10])),
                    "method":       fi.method,
                    "narrative":    fi.narrative,
                }
            }
            if len(high_risk) > 0:
                expl = xm.explain_instance(X_te[high_risk[0]])
                result["xai"]["instance_explanation"] = {
                    "predicted_class":  expl.predicted_class,
                    "confidence":       expl.confidence,
                    "decision_path":    expl.decision_path,
                    "top_positive":     expl.top_positive,
                    "top_negative":     expl.top_negative,
                    "lime_stability":   expl.lime_stability,
                }
        except Exception as e:
            result["xai"] = {"error": str(e)}

    # Compliance report
    if req.include_compliance:
        try:
            mc = xm.model_card(result["performance"], result["fairness"],
                               domain=req.domain) if req.include_xai and "xai" in result and "error" not in result.get("xai",{}) else None
            if mc is None:
                xm2 = ExplainableModel(domain=req.domain)
                xm2.model = clf; xm2._is_fitted = True
                mc = xm2.model_card(result["performance"], result["fairness"], domain=req.domain)

            cr = generate_compliance_report(
                mc, result["fairness"],
                {"accuracy": acc},
                frameworks=req.compliance_frameworks or ["EU AI Act","ISO 42001","NIST AI RMF","NITDA","UNESCO"],
            )
            # Serialise (remove non-JSON types)
            cr_clean = json_safe(cr)
            result["compliance"] = cr_clean
        except Exception as e:
            result["compliance"] = {"error": str(e)}

    return result


def json_safe(obj):
    """Recursively make an object JSON-serialisable."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int, float, str, type(None))):
        return obj
    return str(obj)


# ── FastAPI app ───────────────────────────────────────────────────────────────

if _FASTAPI_AVAILABLE:
    import json as _json

    app = FastAPI(
        title="GAGS Resilience Framework API",
        description=(
            "REST API for the GAGS AI governance simulation engine. "
            "Run fairness simulations, generate compliance reports, "
            "and monitor AI models programmatically from MLOps pipelines."
        ),
        version="1.0.0",
        contact={"name": "GAGS Framework", "url": "https://github.com/gags-framework"},
        license_info={"name": "MIT"},
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health ────────────────────────────────────────────────────────────

    @app.get("/health", tags=["System"])
    async def health_check():
        """System health check — returns API status and version."""
        return {
            "status":  "healthy",
            "version": "1.0.0",
            "gags_version": "3.0",
            "timestamp": datetime.now().isoformat(),
            "auth_mode": "key" if _API_KEY else "open",
            "domain_count": 7,
            "scenario_count": len(COMMUNITY_SCENARIOS),
            "plugin_count": sum(
                len(plugin_registry.get(t))
                for t in ["data_loader","metric","scenario","bias"]
            ),
        }

    # ── Scenarios ─────────────────────────────────────────────────────────

    @app.get("/scenarios", tags=["Scenario Library"])
    async def get_scenarios(
        domain:   Optional[str] = None,
        priority: Optional[str] = None,
        tag:      Optional[str] = None,
        _: None = Depends(_check_auth),
    ):
        """List all community scenarios with optional filters."""
        scens = list_scenarios(
            domain=domain,
            mitigation_priority=priority,
            tags=[tag] if tag else None,
        )
        return {
            "count": len(scens),
            "scenarios": [
                {
                    "id":                    s.id,
                    "name":                  s.name,
                    "domain":                s.domain,
                    "region":                s.region,
                    "bias_types":            s.bias_types,
                    "bias_intensity":        s.bias_intensity,
                    "poison_rate":           s.poison_rate,
                    "expected_fairness_range": list(s.expected_fairness_range),
                    "mitigation_priority":   s.mitigation_priority,
                    "citation":              s.citation,
                    "contributor":           s.contributor,
                    "version":               s.version,
                    "tags":                  s.tags,
                }
                for s in scens
            ],
        }

    @app.get("/scenarios/{scenario_id}", tags=["Scenario Library"])
    async def get_scenario(scenario_id: str, _: None = Depends(_check_auth)):
        """Get a single scenario by ID."""
        s = COMMUNITY_SCENARIOS.get(scenario_id)
        if not s:
            raise HTTPException(404, f"Scenario '{scenario_id}' not found.")
        return json_safe(s.__dict__)

    # ── Plugins ───────────────────────────────────────────────────────────

    @app.get("/plugins", tags=["Plugin API"])
    async def list_plugins(_: None = Depends(_check_auth)):
        """List all registered plugins grouped by type."""
        return plugin_registry.list_all()

    # ── Simulate ──────────────────────────────────────────────────────────

    @app.post("/simulate", tags=["Simulation"], response_model=None)
    async def run_simulation(
        req: SimulationRequest,
        _: None = Depends(_check_auth),
    ):
        """
        Run a full GAGS simulation.

        Returns performance metrics, fairness metrics, optional XAI,
        and optional compliance report.

        Example MLOps fairness gate:

            result = requests.post("/simulate", json={...}).json()
            if result["fairness"]["fairness_score"] < 0.70:
                raise RuntimeError("Fairness gate failed — block deployment.")
        """
        start = time.time()
        result = _run_core_simulation(req)
        result["elapsed_ms"] = round((time.time() - start) * 1000, 1)
        return JSONResponse(content=json_safe(result))

    @app.post("/simulate/fairness-check", tags=["Simulation"])
    async def fairness_check(
        req: FairnessCheckRequest,
        _: None = Depends(_check_auth),
    ):
        """
        Evaluate fairness on already-computed predictions.
        Use this when you have your own model and just want GAGS fairness metrics.
        """
        y_true = np.array(req.y_true, dtype=int)
        y_pred = np.array(req.y_pred, dtype=int)
        demo   = np.array(req.demographic_info, dtype=int)

        if len(y_true) != len(y_pred) or len(y_true) != len(demo):
            raise HTTPException(422, "y_true, y_pred, and demographic_info must have the same length.")

        fair = calculate_fairness_metrics(y_true, y_pred, demo)
        return json_safe({k: v for k, v in fair.items() if k != "group_metrics"})

    # ── XAI ──────────────────────────────────────────────────────────────

    @app.post("/xai/explain", tags=["Explainable AI"])
    async def explain_instance(
        req: ExplainRequest,
        _: None = Depends(_check_auth),
    ):
        """
        Generate a LIME-lite explanation for a single prediction.
        Provide training data, the instance to explain, and optional feature names.
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        X_train = np.array(req.X_train, dtype=np.float64)
        y_train = np.array(req.y_train, dtype=int)
        inst    = np.array(req.instance, dtype=np.float64)

        if len(np.unique(y_train)) < 2:
            raise HTTPException(422, "Training data must contain at least 2 classes.")

        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X_train)
        inst_sc = scaler.transform(inst.reshape(1,-1))[0]

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_sc, y_train)

        xm = ExplainableModel(domain=req.domain)
        xm.model = clf
        xm.feature_names = req.feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        xm._X_train = X_sc
        xm._is_fitted = True

        expl = xm.explain_instance(inst_sc)
        return json_safe(expl.__dict__)

    @app.post("/xai/counterfactual", tags=["Explainable AI"])
    async def counterfactual(
        req: ExplainRequest,
        _: None = Depends(_check_auth),
    ):
        """Generate a DiCE-lite counterfactual explanation."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        X_train = np.array(req.X_train, dtype=np.float64)
        y_train = np.array(req.y_train, dtype=int)
        inst    = np.array(req.instance, dtype=np.float64)

        scaler  = StandardScaler()
        X_sc    = scaler.fit_transform(X_train)
        inst_sc = scaler.transform(inst.reshape(1,-1))[0]

        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_sc, y_train)

        xm = ExplainableModel(domain=req.domain)
        xm.model = clf
        xm.feature_names = req.feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]
        xm._X_train = X_sc
        xm._is_fitted = True

        cf = xm.counterfactual(inst_sc)
        return json_safe(cf.__dict__)

    # ── Compliance ────────────────────────────────────────────────────────

    @app.get("/compliance/frameworks", tags=["Compliance"])
    async def list_frameworks(_: None = Depends(_check_auth)):
        """List all supported regulatory frameworks."""
        return {
            "frameworks": [
                {"id": "EU AI Act",    "description": "EU AI Act — risk classification + Article 9"},
                {"id": "ISO 42001",    "description": "ISO 42001 — AI management system"},
                {"id": "NIST AI RMF",  "description": "NIST AI RMF — GOVERN/MAP/MEASURE/MANAGE"},
                {"id": "NITDA",        "description": "NITDA AI Policy 2023 — Nigeria"},
                {"id": "UNESCO",       "description": "UNESCO Recommendation on Ethics of AI"},
                {"id": "WHO",          "description": "WHO AI Ethics Guidelines (healthcare)"},
            ]
        }


    # ── New domain quick-simulate endpoints ───────────────────────────────────

    @app.post("/simulate/education", tags=["Simulation"])
    async def simulate_education(req: SimulationRequest, _: None = Depends(_check_auth)):
        """Run an Education Equity simulation. Domain-specific metrics: opportunity gap, gender gap."""
        req.domain = "education"
        start = time.time()
        r = _run_core_simulation(req)
        try:
            X, y, demo, feat_names, preset = generate_education_data(req.n_samples, random_state=req.random_seed)
            eq = calculate_education_equity(y, y, X, feat_names, preset)
            r["education_equity"] = json_safe(eq.__dict__)
        except Exception as e:
            r["education_equity"] = {"error": str(e)}
        r["elapsed_ms"] = round((time.time()-start)*1000, 1)
        return JSONResponse(content=json_safe(r))

    @app.post("/simulate/finance", tags=["Simulation"])
    async def simulate_finance(req: SimulationRequest, _: None = Depends(_check_auth)):
        """Run a Financial Inclusion simulation. Domain-specific metrics: disparate impact ratio."""
        req.domain = "finance"
        start = time.time()
        r = _run_core_simulation(req)
        try:
            X, y, demo, feat_names, preset = generate_financial_data(req.n_samples, random_state=req.random_seed)
            ff = calculate_financial_fairness(y, y, X, feat_names, preset)
            r["financial_fairness"] = json_safe(ff.__dict__)
        except Exception as e:
            r["financial_fairness"] = {"error": str(e)}
        r["elapsed_ms"] = round((time.time()-start)*1000, 1)
        return JSONResponse(content=json_safe(r))

    @app.post("/simulate/judicial", tags=["Simulation"])
    async def simulate_judicial(req: SimulationRequest, _: None = Depends(_check_auth)):
        """Run a Judicial Justice simulation. Domain-specific: racial FPR gap, liberty score."""
        req.domain = "judicial"
        start = time.time()
        r = _run_core_simulation(req)
        try:
            X, y, demo, feat_names, preset = generate_judicial_data(req.n_samples, random_state=req.random_seed)
            jf = calculate_judicial_fairness(y, y, X, feat_names, preset)
            r["judicial_fairness"] = json_safe(jf.__dict__)
        except Exception as e:
            r["judicial_fairness"] = {"error": str(e)}
        r["elapsed_ms"] = round((time.time()-start)*1000, 1)
        return JSONResponse(content=json_safe(r))

    @app.post("/simulate/disinformation", tags=["Simulation"])
    async def simulate_disinformation(req: SimulationRequest, _: None = Depends(_check_auth)):
        """Run a Disinformation Resilience simulation. Domain-specific: language FPR gap."""
        req.domain = "disinformation"
        start = time.time()
        r = _run_core_simulation(req)
        try:
            X, y, demo, feat_names, preset = generate_disinformation_data(req.n_samples, random_state=req.random_seed)
            df_ = calculate_disinformation_fairness(y, y, X, feat_names, preset)
            r["disinformation_fairness"] = json_safe(df_.__dict__)
        except Exception as e:
            r["disinformation_fairness"] = {"error": str(e)}
        r["elapsed_ms"] = round((time.time()-start)*1000, 1)
        return JSONResponse(content=json_safe(r))

    # ── Real-time monitoring ──────────────────────────────────────────────

    @app.post("/monitor/snapshot", tags=["Monitoring"])
    async def submit_snapshot(
        snap: MonitorSnapshot,
        _: None = Depends(_check_auth),
    ):
        """
        Submit a monitoring snapshot for a live model.

        Call this from your model-serving layer on each batch or at regular intervals.
        GAGS calculates fairness metrics, compares against thresholds, and returns
        any drift alerts.

        Recommended frequency: every 100–1000 predictions, or every 1 hour.
        """
        y_true = np.array(snap.y_true, dtype=int)
        y_pred = np.array(snap.y_pred, dtype=int)
        demo   = np.array(snap.demographic_info, dtype=int)

        fair = calculate_fairness_metrics(y_true, y_pred, demo)
        from sklearn.metrics import accuracy_score
        acc  = float(accuracy_score(y_true, y_pred))

        current_metrics = {
            "accuracy":                      round(acc, 4),
            "fairness_score":                fair.get("fairness_score", 0.5),
            "demographic_parity_difference": fair.get("demographic_parity_difference", 0),
            "equalized_odds_difference":     fair.get("equalized_odds_difference", 0),
        }

        history = _MONITOR_STORE.get(snap.model_id, [])
        alerts  = _check_drift(current_metrics, history)

        record = {
            "snapshot_id":  str(uuid.uuid4())[:8],
            "model_id":     snap.model_id,
            "timestamp":    snap.timestamp or datetime.now().isoformat(),
            "n_samples":    len(y_true),
            "metrics":      current_metrics,
            "alerts":       alerts,
            "metadata":     snap.metadata,
        }

        _MONITOR_STORE.setdefault(snap.model_id, []).append(record)
        # Keep last 100 snapshots per model
        _MONITOR_STORE[snap.model_id] = _MONITOR_STORE[snap.model_id][-100:]

        return {
            "snapshot_id": record["snapshot_id"],
            "metrics":     current_metrics,
            "alerts":      alerts,
            "alert_count": len(alerts),
            "history_length": len(_MONITOR_STORE[snap.model_id]),
        }

    @app.get("/monitor/status", tags=["Monitoring"])
    async def monitoring_status(_: None = Depends(_check_auth)):
        """Get the current monitoring status for all tracked models."""
        status_out = {}
        for model_id, snaps in _MONITOR_STORE.items():
            if not snaps:
                continue
            latest = snaps[-1]
            all_alerts = [a for s in snaps for a in s.get("alerts", [])]
            status_out[model_id] = {
                "snapshot_count":  len(snaps),
                "latest_timestamp": latest["timestamp"],
                "latest_metrics":  latest["metrics"],
                "total_alerts":    len(all_alerts),
                "recent_alerts":   latest.get("alerts", []),
            }
        return {"models": status_out, "total_models_tracked": len(status_out)}

    @app.get("/monitor/{model_id}/history", tags=["Monitoring"])
    async def model_history(model_id: str, limit: int = 20, _: None = Depends(_check_auth)):
        """Get the snapshot history for a specific model."""
        snaps = _MONITOR_STORE.get(model_id, [])
        return {
            "model_id": model_id,
            "snapshot_count": len(snaps),
            "snapshots": snaps[-limit:],
        }

else:
    # Stub app for environments without FastAPI
    class _StubApp:
        def get(self, *a, **kw): return lambda f: f
        def post(self, *a, **kw): return lambda f: f
        def add_middleware(self, *a, **kw): pass

    app = _StubApp()
    print("⚠️  FastAPI not installed. REST API disabled. "
          "Run: pip install fastapi uvicorn")


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not _FASTAPI_AVAILABLE:
        print("FastAPI not installed. Install with: pip install fastapi uvicorn")
        raise SystemExit(1)
    try:
        import uvicorn
        print("Starting GAGS REST API on http://0.0.0.0:8502")
        print("Documentation: http://localhost:8502/docs")
        print(f"Auth mode: {'key (GAGS_API_KEY set)' if _API_KEY else 'open (no key configured)'}")
        uvicorn.run("api:app", host="0.0.0.0", port=8502, reload=True)
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn")
        raise SystemExit(1)