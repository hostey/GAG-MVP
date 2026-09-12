import numpy as np
import pandas as pd
import time
import requests
import math
from typing import Optional, Dict, Any, Union
from functools import lru_cache
import streamlit as st
from fpdf import FPDF
import json
from groq import Groq


# ----------------------------------------------------------------------
# Nigerian Context Constants
# ----------------------------------------------------------------------
NATIONAL_POVERTY_LINE_NGN = 137_430  # approximate per-capita poverty line proxy

# ----------------------------------------------------------------------
# 1. Intersectional Feature Engineering (kept & slightly hardened)
# ----------------------------------------------------------------------
def get_osrm_route_distance_and_time(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    season_profile: str = "Dry Season",
    terrain_type: str = "Rural Unpaved",
) -> dict:
    """Calculates actual road distance and travel time using OSRM with seasonal penalty multipliers."""

    # Terrain & Seasonal Multipliers for travel time
    multipliers = {
        ("Dry Season", "Urban Paved"): 1.0,
        ("Dry Season", "Rural Unpaved"): 1.25,
        ("Peak Rainy Season", "Urban Paved"): 1.30,
        ("Peak Rainy Season", "Rural Unpaved"): 1.75,
        ("Peak Rainy Season", "Off-Road Terrain"): 2.20,
    }
    time_multiplier = multipliers.get((season_profile, terrain_type), 1.25)

    # OSRM Public API Call (Driving Profile)
    url = f"http://router.project-osrm.org/route/v1/driving/{origin_lon},{origin_lat};{dest_lon},{dest_lat}?overview=false"

    try:
        res = requests.get(url, timeout=3.0)
        if res.status_code == 200:
            data = res.json()
            if data.get("routes"):
                raw_dist_km = data["routes"][0]["distance"] / 1000.0
                raw_duration_min = data["routes"][0]["duration"] / 60.0
                adjusted_duration_min = raw_duration_min * time_multiplier

                return {
                    "distance_km": round(raw_dist_km, 2),
                    "travel_time_min": round(adjusted_duration_min, 1),
                    "routing_source": "OSRM Live Road Topology",
                    "applied_multiplier": time_multiplier,
                }
    except Exception:
        pass  # Fallback to Haversine if API network fails or times out

    # ── Fallback: Haversine Formula ──
    R = 6371.0
    dlat = math.radians(dest_lat - origin_lat)
    dlon = math.radians(dest_lon - origin_lon)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(origin_lat))
        * math.cos(math.radians(dest_lat))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    dist_km = R * c

    base_time = (dist_km / 30.0) * 60.0
    return {
        "distance_km": round(dist_km, 2),
        "travel_time_min": round(base_time * time_multiplier, 1),
        "routing_source": "Haversine Fallback",
        "applied_multiplier": time_multiplier,
    }


def render_mitigation_workbench():
    st.markdown("#### 🛠️ Mitigation Options")
    mitigation_strategy = st.selectbox(
        "Bias Mitigation Strategy",
        options=["None (Baseline)", "Reweighing (Pre-processing)", "Equalized Odds (Post-processing)"],
        key="cfg_health_mitigation_strategy",
        help="Select an algorithmic intervention strategy to promote fairness across demographic groups."
    )

    active_threshold = st.session_state.get("cfg_health_active_threshold", 0.50)
    if "Post-processing" in mitigation_strategy:
        active_threshold = st.slider(
            "Decision Threshold Adjustment",
            min_value=0.1,
            max_value=0.9,
            step=0.01,
            key="cfg_health_active_threshold",
            help="Adjust classification threshold dynamically for sensitive groups to equalize True Positive Rates."
        )

    strategy_code = (
        "none" if "None" in mitigation_strategy
        else "reweighing" if "Reweighing" in mitigation_strategy
        else "post_processing"
    )
    return strategy_code, active_threshold

def render_medical_compliance_panel(impact_matrix: dict, show_metrics: bool = True):
    if not impact_matrix:
        st.info("No active impact matrix calculated yet. Run a simulation cycle.")
        return

    disadv = impact_matrix.get("disadvantaged", {"FNR": 0, "FPR": 0, "TPR": 0, "PPV": 0, "Count": 1})
    priv = impact_matrix.get("privileged", {"FNR": 0, "FPR": 0, "TPR": 0, "PPV": 0, "Count": 1})
    fnr_gap = impact_matrix.get("fnr_disparity", 0.0)
    eq_opp = impact_matrix.get("equal_opportunity_ratio", 1.0)
    is_compliant = impact_matrix.get("is_ndpr_compliant", impact_matrix.get("is_fmoh_compliant", True))

    if show_metrics:
        st.subheader("📋 NDPR & FMOH Regulatory Compliance Scorecard")
        st.caption("Real-time fairness audit mapping against algorithmic disparity thresholds.")

        if is_compliant:
            st.success("✅ **Regulatory Status: COMPLIANT** — Under-diagnosis disparity is within acceptable limits (< 5%).")
        else:
            st.error("⚠️ **Regulatory Status: AUDIT WARNING** — Elevated under-diagnosis risk detected in underserved cohorts.")

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
            st.metric(label="Underserved Cohort FNR", value=f"{disadv.get('FNR', 0) * 100:.1f}%")
        with col4:
            st.metric(label="Privileged Cohort FNR", value=f"{priv.get('FNR', 0) * 100:.1f}%")
        st.markdown("---")

    st.markdown("##### 📊 Subgroup Performance Breakdown")
    comparison_df = pd.DataFrame({
        "Metric": [
            "False Negative Rate (Under-diagnosis)",
            "False Positive Rate (Over-diagnosis)",
            "True Positive Rate (Sensitivity)",
            "Positive Predictive Value (Precision)",
            "Sample Size"
        ],
        "Underserved Cohort": [
            f"{disadv.get('FNR', 0) * 100:.1f}%",
            f"{disadv.get('FPR', 0) * 100:.1f}%",
            f"{disadv.get('TPR', 0) * 100:.1f}%",
            f"{disadv.get('PPV', 0) * 100:.1f}%",
            disadv.get('Count', 0)
        ],
        "Privileged Cohort": [
            f"{priv.get('FNR', 0) * 100:.1f}%",
            f"{priv.get('FPR', 0) * 100:.1f}%",
            f"{priv.get('TPR', 0) * 100:.1f}%",
            f"{priv.get('PPV', 0) * 100:.1f}%",
            priv.get('Count', 0)
        ]
    })
    st.dataframe(comparison_df, use_container_width=True)




def get_active_groq_model(
    client: Groq, preferred_model: str = "llama-3.3-70b-versatile"
) -> str:
    """Dynamically fetches accessible models to prevent 404 decommissioning errors."""
    try:
        available_models = [m.id for m in client.models.list().data]
        if preferred_model in available_models:
            return preferred_model
        for fallback in [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
        ]:
            if fallback in available_models:
                return fallback
        return available_models[0]
    except Exception:
        return preferred_model

def run_multi_agent_triage_pipeline(
    vignette_text: str,
    facilities_df: pd.DataFrame,
    patient_lat: float,
    patient_lon: float,
    api_key: str,
    provider: str,
    season_profile: str = "Peak Rainy Season",
    terrain_type: str = "Rural Unpaved",
    top_n_candidates: int = 10,  # Limits OSRM API calls to speed up execution
) -> dict:
    """Fast multi-agent orchestration: Pre-filters top N facilities locally before OSRM routing."""

    # ── Agent 1: Clinical Triage Agent ────────────────────────────────────────
    if api_key == "DEMO" or not api_key:
        clinical_eval = {
            "urgency": "CRITICAL",
            "required_tier": "Tertiary",
            "required_capabilities": ["Emergency Surgery", "Blood Bank"],
            "summary": "High-risk clinical presentation requiring priority care.",
        }
    else:
        try:
            client = Groq(api_key=api_key)
            selected_model = get_active_groq_model(
                client, "llama-3.3-70b-versatile"
            )

            system_prompt = """
            You are Agent 1 (Clinical Triage). Return JSON with keys:
            - "urgency": ("CRITICAL", "HIGH", "MODERATE", "LOW")
            - "required_tier": ("Primary", "Secondary", "Tertiary")
            - "required_capabilities": list of strings
            - "summary": string evaluation summary
            """

            resp = client.chat.completions.create(
                model=selected_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": vignette_text},
                ],
                temperature=0.1,
            )
            clinical_eval = json.loads(resp.choices[0].message.content)
        except Exception as e:
            clinical_eval = {
                "urgency": "HIGH",
                "required_tier": "Secondary",
                "required_capabilities": ["Emergency Triage"],
                "summary": f"Fallback assessment active: {str(e)}",
            }

    # ── Agent 2: Fast Pre-Filter & OSRM Matcher Agent ────────────────────────
    matched_facilities = []

    if not facilities_df.empty:
        df_calc = facilities_df.copy()

        # 1. Fast vector straight-line pre-filtering (sub-millisecond execution)
        if "lat" in df_calc.columns and "lon" in df_calc.columns:
            df_calc["haversine_km"] = calculate_haversine_distances(
                patient_lat, patient_lon, df_calc
            )
            # Slice down to closest N candidates before making any HTTP requests
            candidate_df = df_calc.sort_values("haversine_km").head(
                top_n_candidates
            )
        else:
            candidate_df = df_calc.head(top_n_candidates)

        # 2. Run OSRM road routing ONLY on candidate subset
        for _, row in candidate_df.iterrows():
            f_lat, f_lon = float(row["lat"]), float(row["lon"])
            route_info = get_osrm_route_distance_and_time(
                patient_lat,
                patient_lon,
                f_lat,
                f_lon,
                season_profile=season_profile,
                terrain_type=terrain_type,
            )
            matched_facilities.append(
                {
                    "facility_name": row.get("facility_name", "Unknown"),
                    "facility_tier": row.get("facility_tier", "Unclassified"),
                    "distance_km": route_info["distance_km"],
                    "travel_time_min": route_info["travel_time_min"],
                    "routing_engine": route_info["routing_source"],
                }
            )

        matched_df = pd.DataFrame(matched_facilities).sort_values(
            "travel_time_min"
        )
        top_facility = (
            matched_df.iloc[0].to_dict() if not matched_df.empty else {}
        )
    else:
        matched_df = pd.DataFrame(
            columns=[
                "facility_name",
                "facility_tier",
                "distance_km",
                "travel_time_min",
                "routing_engine",
            ]
        )
        top_facility = {}

    # ── Agent 3: USSD Payload Generator ─────────────────────────────────────
    ussd_payload = (
        f"*999*DISPATCH*{clinical_eval.get('urgency', 'HIGH')}#"
        f"\nTARGET: {top_facility.get('facility_name', 'N/A')}"
        f"\nTRANSIT: {top_facility.get('travel_time_min', 'N/A')} mins ({season_profile})"
        f"\nREQS: {', '.join(clinical_eval.get('required_capabilities', ['General Care']))}"
    )

    return {
        "clinical_assessment": clinical_eval,
        "top_referral_target": top_facility,
        "matched_facilities_df": matched_df,
        "ussd_dispatch_payload": ussd_payload,
    }


def _generate_parametric_fallback_vignette(
        specialty: str,
        complexity: str,
        geospatial_setting: str,
        patient_cohort: str,
        care_point: str,
        time_window: str,
        facility_constraints: list,
        patient_vulnerabilities: list,
        transport_barriers: list,
) -> str:
    """Deterministic, high-fidelity clinical generator used when all external LLM APIs fail or run offline."""
    # Synthesize context-aware vital signs
    if "Critical" in complexity or "Resuscitative" in complexity:
        vitals = "BP 85/50 mmHg | HR 128 bpm (tachycardic) | SpO2 88% on room air | Temp 38.9°C | GCS 12/15"
    elif "High-Risk" in complexity or "Acute" in complexity:
        vitals = "BP 155/100 mmHg | HR 110 bpm | SpO2 93% on room air | Temp 37.8°C | GCS 14/15"
    else:
        vitals = "BP 120/80 mmHg | HR 82 bpm | SpO2 98% on room air | Temp 36.8°C | GCS 15/15"

    deficits_str = (
        ", ".join(facility_constraints)
        if facility_constraints
        else "Limited diagnostic equipment"
    )
    vuln_str = (
        ", ".join(patient_vulnerabilities)
        if patient_vulnerabilities
        else "Low socio-economic safety net"
    )
    barriers_str = (
        ", ".join(transport_barriers)
        if transport_barriers
        else "Impassable unpaved access roads"
    )

    return (
        f"**[CLINICAL VIGNETTE | {specialty.upper()} - {geospatial_setting.upper()}]**\n\n"
        f"**Patient Demographic & Presentation:** {patient_cohort} presented at {care_point} during {time_window}. "
        f"Initial clinical assessment indicates acute {specialty} pathology with an overall severity tier of **{complexity}**.\n\n"
        f"**Vitals & Diagnostic Status:** {vitals}.\n\n"
        f"**Facility Resource Constraints:** The presenting facility ({care_point}) currently operates under critical infrastructure deficits, including: {deficits_str}. "
        f"On-site surgical stabilization or specialty escalation is unavailable.\n\n"
        f"**Socio-Spatial & Transport Bottlenecks:** Patient background reflects {vuln_str}. Transfer to secondary/tertiary care is acutely compromised by {barriers_str}. "
        f"Immediate spatial routing and emergency transport dispatch required."
    )


def generate_llm_vignette(
        specialty: str,
        provider: str,
        api_key: str,
        complexity: str = "High-Risk / Acute",
        geospatial_setting: str = "Rural Remote LGA",
        patient_cohort: str = "Maternal / Pregnant Female",
        care_point: str = "Primary Health Centre (PHC)",
        time_window: str = "Nighttime Emergency (00:00 - 06:00)",
        facility_constraints: list = None,
        patient_vulnerabilities: list = None,
        transport_barriers: list = None,
) -> str:
    """Robust clinical vignette synthesis engine with multi-tier provider failovers and strict schema guards."""
    facility_constraints = facility_constraints or [
        "No Functional Blood Bank",
        "Lack of Specialist Medical Staff",
    ]
    patient_vulnerabilities = patient_vulnerabilities or [
        "Low Household Income / Out-of-Pocket Payment"
    ]
    transport_barriers = transport_barriers or [
        "Unpaved Dirt Tracks / Mud Degradation"
    ]

    # Mode 1: Offline / Demo Mode Direct Routing
    if (
            api_key == "DEMO"
            or not api_key
            or api_key.strip() == ""
            or "Offline" in provider
    ):
        return _generate_parametric_fallback_vignette(
            specialty,
            complexity,
            geospatial_setting,
            patient_cohort,
            care_point,
            time_window,
            facility_constraints,
            patient_vulnerabilities,
            transport_barriers,
        )

    system_prompt = """
    You are an expert global health clinician and health equity auditor. 
    Synthesize a realistic, highly detailed clinical patient vignette for spatial health accessibility testing.

    CRITICAL STRUCTURAL REQUIREMENTS:
    - Output MUST be structured into 4 distinct bold headings: 
      1. **Patient Demographic & Presentation**
      2. **Vitals & Diagnostic Status**
      3. **Facility Resource Constraints**
      4. **Socio-Spatial & Transport Bottlenecks**
    - Include realistic numerical vitals (BP, HR, SpO2, Temp, GCS).
    - Maintain an objective, academic medical tone. Do NOT include conversational intros or introspective text (e.g., "Sure, here is...").
    """

    user_prompt = f"""
    TARGET CLINICAL PROFILE:
    - Specialty: {specialty}
    - Cohort/Demographics: {patient_cohort}
    - Urgency Severity: {complexity}
    - Location Setting: {geospatial_setting}
    - Care Point: {care_point}
    - Time Window: {time_window}

    INFRASTRUCTURE & SOCIAL MODIFIERS:
    - Facility Deficits: {', '.join(facility_constraints)}
    - Patient Vulnerabilities: {', '.join(patient_vulnerabilities)}
    - Transport Barriers: {', '.join(transport_barriers)}
    """

    # Mode 2: Groq Provider Tier
    if "Groq" in provider:
        try:
            client = Groq(api_key=api_key)
            selected_model = get_active_groq_model(
                client, preferred_model="llama-3.3-70b-versatile"
            )

            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=500,
                timeout=8.0,  # Strict timeout guard
            )
            raw_content = response.choices[0].message.content.strip()

            if len(raw_content) > 100:
                return raw_content
        except Exception as e:
            st.warning(
                f"Groq API connection degraded ({str(e)}). Engaging dynamic fallback engine."
            )

    # Mode 3: OpenAI / Generic Rest Wrapper Fallback Tier
    elif "OpenAI" in provider:
        try:
            import openai

            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_tokens=500,
                timeout=8.0,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.warning(
                f"OpenAI API connection failed ({str(e)}). Switching to parametric engine."
            )

    # Mode 4: Final Safety Net (Guaranteed Execution)
    return _generate_parametric_fallback_vignette(
        specialty,
        complexity,
        geospatial_setting,
        patient_cohort,
        care_point,
        time_window,
        facility_constraints,
        patient_vulnerabilities,
        transport_barriers,
    )

def generate_clinical_recommendation(
    vignette_text: str, api_key: str, provider: str
) -> dict:
    """Evaluates clinical vignettes and returns structured JSON using verified active models."""
    if api_key == "DEMO" or not api_key:
        return {
            "summary": "Immediate referral required to Tertiary Facility with surgical capability.",
            "urgency": "CRITICAL",
            "required_tier": "Tertiary Hub",
            "risk_factors": [
                "Maternal distress risk",
                "Transit delays via unpaved roads",
            ],
            "action_plan": "Stabilize patient and initiate emergency transit.",
        }

    if "Groq" in provider:
        try:
            client = Groq(api_key=api_key)
            selected_model = get_active_groq_model(
                client, preferred_model="llama-3.3-70b-versatile"
            )

            system_prompt = (
                "You are a clinical decision support AI. Return a JSON object with keys: "
                '"summary", "urgency", "required_tier", "risk_factors", and "action_plan".'
            )

            response = client.chat.completions.create(
                model=selected_model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": vignette_text},
                ],
                temperature=0.2,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {
                "summary": f"Fallback Assessment: Triage protocol active. (Details: {str(e)})",
                "urgency": "HIGH",
                "required_tier": "Secondary / Tertiary",
                "risk_factors": ["Unverified facility capacity"],
                "action_plan": "Refer to primary health center for immediate triage.",
            }

    return {"summary": "Standard assessment complete.", "urgency": "MODERATE"}

def build_intersectional_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers intersectional demographic features adapted for Nigerian reality:
    Wealth quintiles / local poverty line + Rural status + Gender + Education + Geopolitical Region.
    """
    df = df.copy()

    # 1. Wealth / Poverty
    if "income_ngn" in df.columns:
        df["below_poverty_line"] = (df["income_ngn"] < NATIONAL_POVERTY_LINE_NGN).astype(int)
        try:
            df["wealth_quintile"] = pd.qcut(
                df["income_ngn"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
            )
        except ValueError:
            df["wealth_quintile"] = 3
    elif "income_level" in df.columns:
        df["below_poverty_line"] = (df["income_level"] < 0.20).astype(int)
        try:
            df["wealth_quintile"] = pd.qcut(
                df["income_level"], q=5, labels=[1, 2, 3, 4, 5], duplicates="drop"
            )
        except ValueError:
            df["wealth_quintile"] = 3
    else:
        df["below_poverty_line"] = np.random.binomial(1, 0.40, len(df))
        df["wealth_quintile"] = np.random.choice(
            [1, 2, 3, 4, 5], len(df), p=[0.25, 0.25, 0.20, 0.15, 0.15]
        )

    # 2. Rural / Urban
    if "is_rural" not in df.columns:
        if "location_type" in df.columns:
            df["is_rural"] = (
                df["location_type"].astype(str).str.lower().isin(["rural", "hard_to_reach"])
            ).astype(int)
        else:
            df["is_rural"] = np.random.binomial(1, 0.52, len(df))

    # 3. Gender / Education / Zone
    if "gender" not in df.columns:
        df["gender"] = np.random.choice(["Female", "Male"], len(df), p=[0.51, 0.49])
    if "education_level" not in df.columns:
        df["education_level"] = np.random.choice(
            ["None", "Primary", "Secondary", "Tertiary"],
            len(df),
            p=[0.20, 0.30, 0.35, 0.15],
        )
    if "geopolitical_zone" not in df.columns:
        df["geopolitical_zone"] = np.random.choice(
            ["North West", "North East", "North Central", "South West", "South South", "South East"],
            len(df),
            p=[0.25, 0.15, 0.15, 0.20, 0.13, 0.12],
        )

    # Composite underserved flag: (Q1/Q2 or below poverty) AND rural
    is_disadvantaged = (
        (df["wealth_quintile"].isin([1, 2]) | (df["below_poverty_line"] == 1))
        & (df["is_rural"] == 1)
    )
    df["is_underserved_cohort"] = is_disadvantaged.astype(int)

    df["intersectional_group"] = (
        df["geopolitical_zone"].astype(str)
        + "_"
        + np.where(df["is_rural"] == 1, "Rural", "Urban")
        + "_Q"
        + df["wealth_quintile"].astype(str)
    )

    return df


# ----------------------------------------------------------------------
# 2. Distance / Travel-Time Engines
# ----------------------------------------------------------------------
R_EARTH_KM = 6371.0


def calculate_haversine_distances(
        patient_lat: float,
        patient_lon: float,
        df_facilities: pd.DataFrame | np.ndarray = None,
        lon_values: np.ndarray = None,
        lat_col: str = "lat",
        lon_col: str = "lon",
) -> pd.Series:
    """
    Calculates great-circle distance (km).

    Supports two calling styles:
    1. calculate_haversine_distances(lat, lon, df)               ← preferred
    2. calculate_haversine_distances(lat, lon, lat_array, lon_array)  ← legacy
    """
    # Legacy style: two numpy arrays were passed
    if isinstance(df_facilities, np.ndarray) and lon_values is not None:
        f_lats = np.radians(df_facilities.astype(float))
        f_lons = np.radians(lon_values.astype(float))
        index = np.arange(len(f_lats))
    else:
        if df_facilities is None or (hasattr(df_facilities, "empty") and df_facilities.empty):
            return pd.Series(dtype=float)

        f_lats = np.radians(df_facilities[lat_col].to_numpy(dtype=float))
        f_lons = np.radians(df_facilities[lon_col].to_numpy(dtype=float))
        index = df_facilities.index

    p_lat = np.radians(patient_lat)
    p_lon = np.radians(patient_lon)

    dlat = f_lats - p_lat
    dlon = f_lons - p_lon

    a = (
            np.sin(dlat * 0.5) ** 2
            + np.cos(p_lat) * np.cos(f_lats) * np.sin(dlon * 0.5) ** 2
    )
    a = np.clip(a, 0.0, 1.0)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return pd.Series(6371.0 * c, index=index, name="distance_km")


def _ors_matrix(
    patient_lat: float,
    patient_lon: float,
    candidates: pd.DataFrame,
    api_key: str,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> Optional[pd.DataFrame]:
    """OpenRouteService Matrix API (best quality when key available)."""
    if not api_key or candidates.empty:
        return None

    locations = [[patient_lon, patient_lat]] + candidates[[lon_col, lat_col]].values.tolist()

    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    body = {
        "locations": locations,
        "metrics": ["distance", "duration"],
        "units": "km",
    }

    try:
        resp = requests.post(
            "https://api.openrouteservice.org/v2/matrix/driving-car",
            json=body,
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        distances = np.array(data["distances"][0][1:])          # km
        durations = np.array(data["durations"][0][1:]) / 60.0   # → minutes

        return pd.DataFrame(
            {"distance_km": distances, "duration_min": durations},
            index=candidates.index,
        )
    except Exception:
        return None


def _osrm_matrix(
    patient_lat: float,
    patient_lon: float,
    candidates: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> Optional[pd.DataFrame]:
    """Public OSRM table service (no API key required)."""
    if candidates.empty:
        return None

    coords = [f"{patient_lon},{patient_lat}"] + [
        f"{row[lon_col]},{row[lat_col]}" for _, row in candidates.iterrows()
    ]
    coords_str = ";".join(coords)

    url = f"https://router.project-osrm.org/table/v1/driving/{coords_str}"
    params = {"annotations": "distance,duration", "sources": "0"}

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if data.get("code") != "Ok":
            return None

        distances = np.array(data["distances"][0][1:]) / 1000.0  # m → km
        durations = np.array(data["durations"][0][1:]) / 60.0    # s → min

        return pd.DataFrame(
            {"distance_km": distances, "duration_min": durations},
            index=candidates.index,
        )
    except Exception:
        return None


def calculate_facility_access(
    patient_lat: float,
    patient_lon: float,
    df_facilities: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    facilities_df=None,
    method: str = "auto",                 # "auto" | "ors" | "osrm" | "haversine"
    ors_api_key: Optional[str] = None,
    max_routing_locations: int = 40,
    terrain_speed_kmh: float = 30.0,
**kwargs,) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
        distance_km, duration_min, method_used
    Prefer real road-network time; fall back to Haversine + terrain speed.
    """
    if facilities_df is None and "df_facilities" in kwargs:
        facilities_df = kwargs["df_facilities"]

    if df_facilities.empty:
        return pd.DataFrame(columns=["distance_km", "duration_min", "method_used"])

    df = df_facilities.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col])

    result = pd.DataFrame(index=df.index)
    result["distance_km"] = np.nan
    result["duration_min"] = np.nan
    result["method_used"] = "none"

    # Pre-filter with Haversine so we only route the nearest candidates
    hav = calculate_haversine_distances(patient_lat, patient_lon, df, lat_col, lon_col)
    candidates = df.loc[hav.nsmallest(min(max_routing_locations, len(df))).index]

    routing_df = None
    used = None

    if method in ("auto", "ors") and ors_api_key:
        routing_df = _ors_matrix(patient_lat, patient_lon, candidates, ors_api_key, lat_col, lon_col)
        used = "ors"

    if routing_df is None and method in ("auto", "osrm"):
        routing_df = _osrm_matrix(patient_lat, patient_lon, candidates, lat_col, lon_col)
        used = "osrm"

    if routing_df is not None and not routing_df.empty:
        result.loc[routing_df.index, "distance_km"] = routing_df["distance_km"]
        result.loc[routing_df.index, "duration_min"] = routing_df["duration_min"]
        result.loc[routing_df.index, "method_used"] = used

    # Haversine fallback for remaining rows
    missing = result["distance_km"].isna()
    if missing.any():
        hav_miss = calculate_haversine_distances(
            patient_lat, patient_lon, df.loc[missing], lat_col, lon_col
        )
        result.loc[missing, "distance_km"] = hav_miss
        result.loc[missing, "duration_min"] = (hav_miss / max(terrain_speed_kmh, 1.0)) * 60
        result.loc[missing, "method_used"] = "haversine_fallback"

    return result


def nearest_feasible_facility(
    patient_lat: float,
    patient_lon: float,
    df_facilities: pd.DataFrame,
    max_duration_min: float = 60.0,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience helper used by the Equity tab."""
    access = calculate_facility_access(patient_lat, patient_lon, df_facilities, **kwargs)

    if access.empty:
        return {
            "status": "INFEASIBLE",
            "reason": "No facilities",
            "distance_km": None,
            "duration_min": None,
            "facility_name": None,
            "method_used": None,
        }

    access = access.sort_values(["duration_min", "distance_km"])
    best = access.iloc[0]
    facility_row = df_facilities.loc[best.name]

    feasible = best["duration_min"] <= max_duration_min

    return {
        "status": "FEASIBLE" if feasible else "INFEASIBLE",
        "distance_km": float(best["distance_km"]),
        "duration_min": float(best["duration_min"]),
        "method_used": best["method_used"],
        "facility_name": str(
            facility_row.get("facility_name", facility_row.get("name", "Unknown"))
        ),
        "facility_lat": float(facility_row.get("lat", patient_lat)),
        "facility_lon": float(facility_row.get("lon", patient_lon)),
        "facility_row": facility_row,
    }


# ----------------------------------------------------------------------
# 3. AI Recommendation Audit (now uses road-network when possible)
# ----------------------------------------------------------------------
def audit_ai_recommendation(
    llm_output: str,
    p_lat: float,
    p_lon: float,
    facility_df: pd.DataFrame,
    max_duration_min: float = 60.0,
    ors_api_key: Optional[str] = None,
    terrain_speed_kmh: float = 30.0,
) -> dict:
    """Audits clinical recommendations against geographical accessibility."""
    llm_text = llm_output.lower()
    required_caps = []
    if any(k in llm_text for k in ["ct scan", "mri", "imaging"]):
        required_caps.append("Advanced Diagnostics (CT/MRI)")
    if any(k in llm_text for k in ["blood transfusion", "tpa", "thrombolysis"]):
        required_caps.append("Emergency Blood/Biologics")
    if any(k in llm_text for k in ["icu", "intensive care", "laparotomy", "surgical"]):
        required_caps.append("Tertiary Surgical / ICU")

    if facility_df.empty:
        return {
            "status": "INFEASIBLE",
            "penalty_score": 100,
            "nearest_distance_km": 0.0,
            "nearest_duration_min": 0.0,
            "nearest_facility_name": "No Facility Data Available",
            "required_capabilities": required_caps,
            "infeasible_flags": ["No local infrastructure data loaded."],
            "method_used": "none",
            "facility_lat": p_lat,
            "facility_lon": p_lon,
        }

    best = nearest_feasible_facility(
        p_lat,
        p_lon,
        facility_df,
        max_duration_min=max_duration_min,
        ors_api_key=ors_api_key,
        terrain_speed_kmh=terrain_speed_kmh,
    )

    feasible = best["status"] == "FEASIBLE"
    penalty = 0 if feasible else min(100, int(best["duration_min"] / 2) if best["duration_min"] else 50)

    return {
        "status": best["status"],
        "penalty_score": penalty,
        "nearest_distance_km": best["distance_km"],
        "nearest_duration_min": best["duration_min"],
        "nearest_facility_name": best["facility_name"],
        "required_capabilities": required_caps if required_caps else ["General Care"],
        "infeasible_flags": [] if feasible else [f"Nearest suitable facility is {best['duration_min']:.0f} min away."],
        "method_used": best["method_used"],
        "facility_lat": best.get("facility_lat", p_lat),
        "facility_lon": best.get("facility_lon", p_lon),
    }


# ----------------------------------------------------------------------
# 4. PDF Report (minor national branding kept)
# ----------------------------------------------------------------------
def generate_pdf_report(res_df, llm_provider, p_lat, p_lon, fail_rate, avg_penalty) -> bytes:
    """Generates a PDF audit report aligned with NDPR / FMOH framing."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "NDPR & FMOH Health AI Audit Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.set_font("Helvetica", "I", 10)
    pdf.cell(0, 6, "Health AI Infrastructure Equity Evaluation", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(8)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 7, "1. Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, f"- Audit Date: {time.strftime('%Y-%m-%d %H:%M')}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"- Evaluated Model/Provider: {llm_provider}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"- Infeasibility Rate: {fail_rate:.1f}%", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 6, f"- Average Equity Penalty: {avg_penalty:.1f} / 100", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    return bytes(pdf.output())











