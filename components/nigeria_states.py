"""
components/nigeria_states.py
==============================
State-level calibration data for all 36 Nigerian states + FCT.

Each state has empirical parameters drawn from:
  - NDHS 2021 (demographic & health)
  - NBS 2022/2023 (labour force, poverty)
  - WHO AFRO 2022 (health workforce)
  - World Bank Nigeria 2023 (poverty)
  - NHIA 2023 (insurance coverage)
  - UBEC 2023 (education)

These override the national-average defaults whenever a user
selects a specific state from the sidebar selector.

Usage
-----
    from components.nigeria_states import (
        get_state_names,
        get_state_params,
        state_selector,
        apply_state_to_preset,
    )

    # In sidebar
    selected_state = state_selector()

    # Get state-specific parameters
    params = get_state_params(selected_state)

    # Override a preset with state parameters
    preset = apply_state_to_preset(preset, selected_state)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
import streamlit as st

# ── State data structure ───────────────────────────────────────────────────────

@dataclass
class StateProfile:
    """Empirical parameters for one Nigerian state."""
    name:                   str
    geopolitical_zone:      str   # NW | NE | NC | SW | SE | SS
    capital:                str
    population_millions:    float
    is_northern:            bool  # True = NW/NE/NC

    # Health parameters
    maternal_mortality_ratio: float   # per 100,000 live births
    u5_mortality_rate:        float   # per 1,000 live births
    health_worker_density:    float   # per 10,000 population
    nhis_coverage_rate:       float   # fraction with formal health insurance
    oop_expenditure_pct:      float   # OOP as share of total health spending
    rural_population_pct:     float   # rural fraction of population

    # Economic parameters
    poverty_rate:             float   # proportion below poverty line
    informal_sector_pct:      float   # informal workforce fraction
    youth_unemployment_rate:  float   # youth (15-34) unemployment
    gender_wage_gap:          float   # raw gender wage gap
    bvn_coverage:             float   # Bank Verification Number coverage
    gig_worker_pct:           float   # fraction in gig economy

    # Education parameters
    literacy_rate:            float   # adult literacy rate
    net_primary_enrolment:    float   # primary school net enrolment
    urban_rural_score_gap:    float   # JAMB score gap urban vs rural

    # Context
    primary_language:         str
    notes:                    str = ""


# ── All 36 states + FCT ───────────────────────────────────────────────────────
# Parameters sourced from NDHS 2021, NBS 2023, WHO 2022, NHIA 2023

NIGERIA_STATES: Dict[str, StateProfile] = {

    # ── North-West ────────────────────────────────────────────────────────────
    "Kano": StateProfile(
        name="Kano", geopolitical_zone="NW", capital="Kano", population_millions=15.9,
        is_northern=True,
        maternal_mortality_ratio=1200, u5_mortality_rate=143, health_worker_density=1.1,
        nhis_coverage_rate=0.025, oop_expenditure_pct=0.82,
        rural_population_pct=0.52,
        poverty_rate=0.47, informal_sector_pct=0.74, youth_unemployment_rate=0.38,
        gender_wage_gap=0.31, bvn_coverage=0.48, gig_worker_pct=0.12,
        literacy_rate=0.51, net_primary_enrolment=0.61, urban_rural_score_gap=0.26,
        primary_language="Hausa",
    ),
    "Kaduna": StateProfile(
        name="Kaduna", geopolitical_zone="NW", capital="Kaduna", population_millions=9.0,
        is_northern=True,
        maternal_mortality_ratio=1050, u5_mortality_rate=131, health_worker_density=1.4,
        nhis_coverage_rate=0.032, oop_expenditure_pct=0.79,
        rural_population_pct=0.58,
        poverty_rate=0.43, informal_sector_pct=0.71, youth_unemployment_rate=0.36,
        gender_wage_gap=0.28, bvn_coverage=0.51, gig_worker_pct=0.10,
        literacy_rate=0.57, net_primary_enrolment=0.65, urban_rural_score_gap=0.24,
        primary_language="Hausa",
    ),
    "Katsina": StateProfile(
        name="Katsina", geopolitical_zone="NW", capital="Katsina", population_millions=8.8,
        is_northern=True,
        maternal_mortality_ratio=1320, u5_mortality_rate=158, health_worker_density=0.9,
        nhis_coverage_rate=0.018, oop_expenditure_pct=0.85,
        rural_population_pct=0.71,
        poverty_rate=0.52, informal_sector_pct=0.78, youth_unemployment_rate=0.41,
        gender_wage_gap=0.34, bvn_coverage=0.40, gig_worker_pct=0.07,
        literacy_rate=0.43, net_primary_enrolment=0.55, urban_rural_score_gap=0.29,
        primary_language="Hausa",
    ),
    "Sokoto": StateProfile(
        name="Sokoto", geopolitical_zone="NW", capital="Sokoto", population_millions=5.9,
        is_northern=True,
        maternal_mortality_ratio=1560, u5_mortality_rate=172, health_worker_density=0.7,
        nhis_coverage_rate=0.015, oop_expenditure_pct=0.87,
        rural_population_pct=0.76,
        poverty_rate=0.58, informal_sector_pct=0.81, youth_unemployment_rate=0.43,
        gender_wage_gap=0.36, bvn_coverage=0.35, gig_worker_pct=0.05,
        literacy_rate=0.38, net_primary_enrolment=0.49, urban_rural_score_gap=0.31,
        primary_language="Hausa",
    ),
    "Zamfara": StateProfile(
        name="Zamfara", geopolitical_zone="NW", capital="Gusau", population_millions=5.0,
        is_northern=True,
        maternal_mortality_ratio=1490, u5_mortality_rate=168, health_worker_density=0.6,
        nhis_coverage_rate=0.012, oop_expenditure_pct=0.88,
        rural_population_pct=0.78,
        poverty_rate=0.61, informal_sector_pct=0.82, youth_unemployment_rate=0.44,
        gender_wage_gap=0.37, bvn_coverage=0.32, gig_worker_pct=0.04,
        literacy_rate=0.35, net_primary_enrolment=0.46, urban_rural_score_gap=0.32,
        primary_language="Hausa",
    ),
    "Kebbi": StateProfile(
        name="Kebbi", geopolitical_zone="NW", capital="Birnin Kebbi", population_millions=4.4,
        is_northern=True,
        maternal_mortality_ratio=1380, u5_mortality_rate=161, health_worker_density=0.8,
        nhis_coverage_rate=0.016, oop_expenditure_pct=0.86,
        rural_population_pct=0.74,
        poverty_rate=0.56, informal_sector_pct=0.80, youth_unemployment_rate=0.42,
        gender_wage_gap=0.35, bvn_coverage=0.37, gig_worker_pct=0.06,
        literacy_rate=0.40, net_primary_enrolment=0.51, urban_rural_score_gap=0.30,
        primary_language="Hausa",
    ),
    "Jigawa": StateProfile(
        name="Jigawa", geopolitical_zone="NW", capital="Dutse", population_millions=5.8,
        is_northern=True,
        maternal_mortality_ratio=1180, u5_mortality_rate=148, health_worker_density=1.0,
        nhis_coverage_rate=0.020, oop_expenditure_pct=0.83,
        rural_population_pct=0.72,
        poverty_rate=0.50, informal_sector_pct=0.76, youth_unemployment_rate=0.39,
        gender_wage_gap=0.32, bvn_coverage=0.42, gig_worker_pct=0.07,
        literacy_rate=0.47, net_primary_enrolment=0.57, urban_rural_score_gap=0.27,
        primary_language="Hausa",
    ),

    # ── North-East ────────────────────────────────────────────────────────────
    "Borno": StateProfile(
        name="Borno", geopolitical_zone="NE", capital="Maiduguri", population_millions=6.1,
        is_northern=True,
        maternal_mortality_ratio=1620, u5_mortality_rate=181, health_worker_density=0.6,
        nhis_coverage_rate=0.014, oop_expenditure_pct=0.89,
        rural_population_pct=0.68,
        poverty_rate=0.63, informal_sector_pct=0.83, youth_unemployment_rate=0.46,
        gender_wage_gap=0.38, bvn_coverage=0.30, gig_worker_pct=0.04,
        literacy_rate=0.33, net_primary_enrolment=0.44, urban_rural_score_gap=0.34,
        primary_language="Kanuri/Hausa",
        notes="Conflict-affected; high humanitarian context",
    ),
    "Adamawa": StateProfile(
        name="Adamawa", geopolitical_zone="NE", capital="Yola", population_millions=4.9,
        is_northern=True,
        maternal_mortality_ratio=1100, u5_mortality_rate=136, health_worker_density=1.1,
        nhis_coverage_rate=0.022, oop_expenditure_pct=0.81,
        rural_population_pct=0.62,
        poverty_rate=0.46, informal_sector_pct=0.73, youth_unemployment_rate=0.37,
        gender_wage_gap=0.30, bvn_coverage=0.44, gig_worker_pct=0.08,
        literacy_rate=0.53, net_primary_enrolment=0.62, urban_rural_score_gap=0.25,
        primary_language="Hausa/Fulfulde",
    ),
    "Gombe": StateProfile(
        name="Gombe", geopolitical_zone="NE", capital="Gombe", population_millions=3.9,
        is_northern=True,
        maternal_mortality_ratio=1090, u5_mortality_rate=133, health_worker_density=1.2,
        nhis_coverage_rate=0.024, oop_expenditure_pct=0.80,
        rural_population_pct=0.60,
        poverty_rate=0.45, informal_sector_pct=0.72, youth_unemployment_rate=0.36,
        gender_wage_gap=0.29, bvn_coverage=0.46, gig_worker_pct=0.09,
        literacy_rate=0.55, net_primary_enrolment=0.64, urban_rural_score_gap=0.24,
        primary_language="Hausa",
    ),
    "Yobe": StateProfile(
        name="Yobe", geopolitical_zone="NE", capital="Damaturu", population_millions=3.5,
        is_northern=True,
        maternal_mortality_ratio=1400, u5_mortality_rate=165, health_worker_density=0.7,
        nhis_coverage_rate=0.013, oop_expenditure_pct=0.87,
        rural_population_pct=0.73,
        poverty_rate=0.57, informal_sector_pct=0.80, youth_unemployment_rate=0.42,
        gender_wage_gap=0.36, bvn_coverage=0.33, gig_worker_pct=0.05,
        literacy_rate=0.36, net_primary_enrolment=0.47, urban_rural_score_gap=0.31,
        primary_language="Hausa/Kanuri",
    ),
    "Bauchi": StateProfile(
        name="Bauchi", geopolitical_zone="NE", capital="Bauchi", population_millions=7.2,
        is_northern=True,
        maternal_mortality_ratio=1150, u5_mortality_rate=140, health_worker_density=1.0,
        nhis_coverage_rate=0.021, oop_expenditure_pct=0.82,
        rural_population_pct=0.65,
        poverty_rate=0.48, informal_sector_pct=0.74, youth_unemployment_rate=0.38,
        gender_wage_gap=0.31, bvn_coverage=0.43, gig_worker_pct=0.08,
        literacy_rate=0.50, net_primary_enrolment=0.59, urban_rural_score_gap=0.26,
        primary_language="Hausa",
    ),
    "Taraba": StateProfile(
        name="Taraba", geopolitical_zone="NE", capital="Jalingo", population_millions=3.7,
        is_northern=True,
        maternal_mortality_ratio=1080, u5_mortality_rate=130, health_worker_density=1.2,
        nhis_coverage_rate=0.023, oop_expenditure_pct=0.80,
        rural_population_pct=0.67,
        poverty_rate=0.44, informal_sector_pct=0.71, youth_unemployment_rate=0.35,
        gender_wage_gap=0.28, bvn_coverage=0.45, gig_worker_pct=0.08,
        literacy_rate=0.56, net_primary_enrolment=0.65, urban_rural_score_gap=0.23,
        primary_language="Hausa/Jukun",
    ),

    # ── North-Central ─────────────────────────────────────────────────────────
    "FCT (Abuja)": StateProfile(
        name="FCT (Abuja)", geopolitical_zone="NC", capital="Abuja", population_millions=3.9,
        is_northern=False,
        maternal_mortality_ratio=420, u5_mortality_rate=61, health_worker_density=4.8,
        nhis_coverage_rate=0.112, oop_expenditure_pct=0.58,
        rural_population_pct=0.25,
        poverty_rate=0.14, informal_sector_pct=0.42, youth_unemployment_rate=0.21,
        gender_wage_gap=0.18, bvn_coverage=0.78, gig_worker_pct=0.22,
        literacy_rate=0.87, net_primary_enrolment=0.91, urban_rural_score_gap=0.11,
        primary_language="English/Hausa",
        notes="Federal capital; best indicators nationally",
    ),
    "Niger": StateProfile(
        name="Niger", geopolitical_zone="NC", capital="Minna", population_millions=6.9,
        is_northern=True,
        maternal_mortality_ratio=890, u5_mortality_rate=115, health_worker_density=1.5,
        nhis_coverage_rate=0.031, oop_expenditure_pct=0.76,
        rural_population_pct=0.63,
        poverty_rate=0.40, informal_sector_pct=0.68, youth_unemployment_rate=0.33,
        gender_wage_gap=0.26, bvn_coverage=0.49, gig_worker_pct=0.09,
        literacy_rate=0.62, net_primary_enrolment=0.70, urban_rural_score_gap=0.21,
        primary_language="Hausa/Nupe",
    ),
    "Plateau": StateProfile(
        name="Plateau", geopolitical_zone="NC", capital="Jos", population_millions=4.2,
        is_northern=False,
        maternal_mortality_ratio=730, u5_mortality_rate=95, health_worker_density=2.0,
        nhis_coverage_rate=0.041, oop_expenditure_pct=0.71,
        rural_population_pct=0.55,
        poverty_rate=0.34, informal_sector_pct=0.63, youth_unemployment_rate=0.30,
        gender_wage_gap=0.23, bvn_coverage=0.54, gig_worker_pct=0.11,
        literacy_rate=0.69, net_primary_enrolment=0.75, urban_rural_score_gap=0.18,
        primary_language="English/Birom",
    ),
    "Benue": StateProfile(
        name="Benue", geopolitical_zone="NC", capital="Makurdi", population_millions=6.1,
        is_northern=False,
        maternal_mortality_ratio=810, u5_mortality_rate=104, health_worker_density=1.7,
        nhis_coverage_rate=0.035, oop_expenditure_pct=0.74,
        rural_population_pct=0.60,
        poverty_rate=0.37, informal_sector_pct=0.66, youth_unemployment_rate=0.31,
        gender_wage_gap=0.24, bvn_coverage=0.51, gig_worker_pct=0.10,
        literacy_rate=0.65, net_primary_enrolment=0.72, urban_rural_score_gap=0.20,
        primary_language="Tiv/Idoma",
    ),
    "Kogi": StateProfile(
        name="Kogi", geopolitical_zone="NC", capital="Lokoja", population_millions=4.5,
        is_northern=False,
        maternal_mortality_ratio=850, u5_mortality_rate=110, health_worker_density=1.6,
        nhis_coverage_rate=0.033, oop_expenditure_pct=0.75,
        rural_population_pct=0.58,
        poverty_rate=0.38, informal_sector_pct=0.67, youth_unemployment_rate=0.32,
        gender_wage_gap=0.25, bvn_coverage=0.52, gig_worker_pct=0.10,
        literacy_rate=0.67, net_primary_enrolment=0.73, urban_rural_score_gap=0.19,
        primary_language="Yoruba/Igbo/Nupe",
    ),
    "Kwara": StateProfile(
        name="Kwara", geopolitical_zone="NC", capital="Ilorin", population_millions=3.2,
        is_northern=False,
        maternal_mortality_ratio=680, u5_mortality_rate=87, health_worker_density=2.2,
        nhis_coverage_rate=0.048, oop_expenditure_pct=0.69,
        rural_population_pct=0.48,
        poverty_rate=0.30, informal_sector_pct=0.58, youth_unemployment_rate=0.27,
        gender_wage_gap=0.21, bvn_coverage=0.58, gig_worker_pct=0.13,
        literacy_rate=0.74, net_primary_enrolment=0.79, urban_rural_score_gap=0.16,
        primary_language="Yoruba",
    ),
    "Nasarawa": StateProfile(
        name="Nasarawa", geopolitical_zone="NC", capital="Lafia", population_millions=2.5,
        is_northern=False,
        maternal_mortality_ratio=760, u5_mortality_rate=98, health_worker_density=1.8,
        nhis_coverage_rate=0.038, oop_expenditure_pct=0.72,
        rural_population_pct=0.57,
        poverty_rate=0.35, informal_sector_pct=0.65, youth_unemployment_rate=0.30,
        gender_wage_gap=0.23, bvn_coverage=0.53, gig_worker_pct=0.11,
        literacy_rate=0.67, net_primary_enrolment=0.73, urban_rural_score_gap=0.18,
        primary_language="Eggon/Hausa",
    ),

    # ── South-West ────────────────────────────────────────────────────────────
    "Lagos": StateProfile(
        name="Lagos", geopolitical_zone="SW", capital="Ikeja", population_millions=21.3,
        is_northern=False,
        maternal_mortality_ratio=310, u5_mortality_rate=42, health_worker_density=6.1,
        nhis_coverage_rate=0.148, oop_expenditure_pct=0.51,
        rural_population_pct=0.09,
        poverty_rate=0.09, informal_sector_pct=0.51, youth_unemployment_rate=0.17,
        gender_wage_gap=0.15, bvn_coverage=0.88, gig_worker_pct=0.31,
        literacy_rate=0.92, net_primary_enrolment=0.94, urban_rural_score_gap=0.07,
        primary_language="Yoruba/English",
        notes="Commercial capital; highest indicators SW",
    ),
    "Oyo": StateProfile(
        name="Oyo", geopolitical_zone="SW", capital="Ibadan", population_millions=8.3,
        is_northern=False,
        maternal_mortality_ratio=380, u5_mortality_rate=52, health_worker_density=4.2,
        nhis_coverage_rate=0.082, oop_expenditure_pct=0.61,
        rural_population_pct=0.32,
        poverty_rate=0.21, informal_sector_pct=0.58, youth_unemployment_rate=0.22,
        gender_wage_gap=0.17, bvn_coverage=0.72, gig_worker_pct=0.20,
        literacy_rate=0.85, net_primary_enrolment=0.89, urban_rural_score_gap=0.10,
        primary_language="Yoruba",
    ),
    "Osun": StateProfile(
        name="Osun", geopolitical_zone="SW", capital="Osogbo", population_millions=4.7,
        is_northern=False,
        maternal_mortality_ratio=410, u5_mortality_rate=56, health_worker_density=3.8,
        nhis_coverage_rate=0.071, oop_expenditure_pct=0.63,
        rural_population_pct=0.38,
        poverty_rate=0.23, informal_sector_pct=0.60, youth_unemployment_rate=0.23,
        gender_wage_gap=0.18, bvn_coverage=0.69, gig_worker_pct=0.18,
        literacy_rate=0.84, net_primary_enrolment=0.88, urban_rural_score_gap=0.10,
        primary_language="Yoruba",
    ),
    "Ogun": StateProfile(
        name="Ogun", geopolitical_zone="SW", capital="Abeokuta", population_millions=5.9,
        is_northern=False,
        maternal_mortality_ratio=350, u5_mortality_rate=48, health_worker_density=4.6,
        nhis_coverage_rate=0.095, oop_expenditure_pct=0.58,
        rural_population_pct=0.28,
        poverty_rate=0.16, informal_sector_pct=0.54, youth_unemployment_rate=0.19,
        gender_wage_gap=0.16, bvn_coverage=0.79, gig_worker_pct=0.24,
        literacy_rate=0.89, net_primary_enrolment=0.92, urban_rural_score_gap=0.08,
        primary_language="Yoruba",
    ),
    "Ondo": StateProfile(
        name="Ondo", geopolitical_zone="SW", capital="Akure", population_millions=5.2,
        is_northern=False,
        maternal_mortality_ratio=440, u5_mortality_rate=59, health_worker_density=3.5,
        nhis_coverage_rate=0.062, oop_expenditure_pct=0.65,
        rural_population_pct=0.42,
        poverty_rate=0.25, informal_sector_pct=0.62, youth_unemployment_rate=0.24,
        gender_wage_gap=0.19, bvn_coverage=0.65, gig_worker_pct=0.16,
        literacy_rate=0.82, net_primary_enrolment=0.87, urban_rural_score_gap=0.11,
        primary_language="Yoruba",
    ),
    "Ekiti": StateProfile(
        name="Ekiti", geopolitical_zone="SW", capital="Ado-Ekiti", population_millions=3.3,
        is_northern=False,
        maternal_mortality_ratio=460, u5_mortality_rate=62, health_worker_density=3.3,
        nhis_coverage_rate=0.058, oop_expenditure_pct=0.66,
        rural_population_pct=0.45,
        poverty_rate=0.26, informal_sector_pct=0.63, youth_unemployment_rate=0.25,
        gender_wage_gap=0.19, bvn_coverage=0.63, gig_worker_pct=0.15,
        literacy_rate=0.81, net_primary_enrolment=0.86, urban_rural_score_gap=0.11,
        primary_language="Yoruba",
    ),

    # ── South-East ────────────────────────────────────────────────────────────
    "Anambra": StateProfile(
        name="Anambra", geopolitical_zone="SE", capital="Awka", population_millions=6.0,
        is_northern=False,
        maternal_mortality_ratio=390, u5_mortality_rate=53, health_worker_density=4.0,
        nhis_coverage_rate=0.078, oop_expenditure_pct=0.62,
        rural_population_pct=0.35,
        poverty_rate=0.18, informal_sector_pct=0.55, youth_unemployment_rate=0.20,
        gender_wage_gap=0.16, bvn_coverage=0.74, gig_worker_pct=0.22,
        literacy_rate=0.87, net_primary_enrolment=0.91, urban_rural_score_gap=0.09,
        primary_language="Igbo",
    ),
    "Imo": StateProfile(
        name="Imo", geopolitical_zone="SE", capital="Owerri", population_millions=5.4,
        is_northern=False,
        maternal_mortality_ratio=430, u5_mortality_rate=58, health_worker_density=3.6,
        nhis_coverage_rate=0.068, oop_expenditure_pct=0.64,
        rural_population_pct=0.40,
        poverty_rate=0.22, informal_sector_pct=0.58, youth_unemployment_rate=0.22,
        gender_wage_gap=0.17, bvn_coverage=0.70, gig_worker_pct=0.19,
        literacy_rate=0.85, net_primary_enrolment=0.89, urban_rural_score_gap=0.10,
        primary_language="Igbo",
    ),
    "Enugu": StateProfile(
        name="Enugu", geopolitical_zone="SE", capital="Enugu", population_millions=5.1,
        is_northern=False,
        maternal_mortality_ratio=400, u5_mortality_rate=55, health_worker_density=3.9,
        nhis_coverage_rate=0.074, oop_expenditure_pct=0.63,
        rural_population_pct=0.38,
        poverty_rate=0.20, informal_sector_pct=0.56, youth_unemployment_rate=0.21,
        gender_wage_gap=0.16, bvn_coverage=0.72, gig_worker_pct=0.21,
        literacy_rate=0.86, net_primary_enrolment=0.90, urban_rural_score_gap=0.09,
        primary_language="Igbo",
    ),
    "Abia": StateProfile(
        name="Abia", geopolitical_zone="SE", capital="Umuahia", population_millions=3.7,
        is_northern=False,
        maternal_mortality_ratio=450, u5_mortality_rate=61, health_worker_density=3.4,
        nhis_coverage_rate=0.061, oop_expenditure_pct=0.65,
        rural_population_pct=0.43,
        poverty_rate=0.24, informal_sector_pct=0.60, youth_unemployment_rate=0.23,
        gender_wage_gap=0.18, bvn_coverage=0.67, gig_worker_pct=0.17,
        literacy_rate=0.83, net_primary_enrolment=0.87, urban_rural_score_gap=0.10,
        primary_language="Igbo",
    ),
    "Ebonyi": StateProfile(
        name="Ebonyi", geopolitical_zone="SE", capital="Abakaliki", population_millions=3.1,
        is_northern=False,
        maternal_mortality_ratio=580, u5_mortality_rate=78, health_worker_density=2.5,
        nhis_coverage_rate=0.043, oop_expenditure_pct=0.70,
        rural_population_pct=0.56,
        poverty_rate=0.32, informal_sector_pct=0.66, youth_unemployment_rate=0.29,
        gender_wage_gap=0.22, bvn_coverage=0.57, gig_worker_pct=0.12,
        literacy_rate=0.73, net_primary_enrolment=0.79, urban_rural_score_gap=0.15,
        primary_language="Igbo",
    ),

    # ── South-South ───────────────────────────────────────────────────────────
    "Rivers": StateProfile(
        name="Rivers", geopolitical_zone="SS", capital="Port Harcourt", population_millions=7.9,
        is_northern=False,
        maternal_mortality_ratio=480, u5_mortality_rate=64, health_worker_density=3.2,
        nhis_coverage_rate=0.067, oop_expenditure_pct=0.65,
        rural_population_pct=0.36,
        poverty_rate=0.22, informal_sector_pct=0.57, youth_unemployment_rate=0.22,
        gender_wage_gap=0.18, bvn_coverage=0.69, gig_worker_pct=0.19,
        literacy_rate=0.83, net_primary_enrolment=0.87, urban_rural_score_gap=0.10,
        primary_language="English/Ijaw",
    ),
    "Delta": StateProfile(
        name="Delta", geopolitical_zone="SS", capital="Asaba", population_millions=5.7,
        is_northern=False,
        maternal_mortality_ratio=510, u5_mortality_rate=68, health_worker_density=2.9,
        nhis_coverage_rate=0.058, oop_expenditure_pct=0.67,
        rural_population_pct=0.40,
        poverty_rate=0.25, informal_sector_pct=0.59, youth_unemployment_rate=0.24,
        gender_wage_gap=0.19, bvn_coverage=0.66, gig_worker_pct=0.17,
        literacy_rate=0.81, net_primary_enrolment=0.86, urban_rural_score_gap=0.11,
        primary_language="Urhobo/Igbo",
    ),
    "Edo": StateProfile(
        name="Edo", geopolitical_zone="SS", capital="Benin City", population_millions=4.7,
        is_northern=False,
        maternal_mortality_ratio=490, u5_mortality_rate=66, health_worker_density=3.0,
        nhis_coverage_rate=0.062, oop_expenditure_pct=0.66,
        rural_population_pct=0.38,
        poverty_rate=0.23, informal_sector_pct=0.58, youth_unemployment_rate=0.23,
        gender_wage_gap=0.18, bvn_coverage=0.68, gig_worker_pct=0.18,
        literacy_rate=0.82, net_primary_enrolment=0.87, urban_rural_score_gap=0.10,
        primary_language="Edo/Igbo",
    ),
    "Cross River": StateProfile(
        name="Cross River", geopolitical_zone="SS", capital="Calabar", population_millions=4.1,
        is_northern=False,
        maternal_mortality_ratio=560, u5_mortality_rate=75, health_worker_density=2.6,
        nhis_coverage_rate=0.047, oop_expenditure_pct=0.69,
        rural_population_pct=0.52,
        poverty_rate=0.30, informal_sector_pct=0.64, youth_unemployment_rate=0.28,
        gender_wage_gap=0.21, bvn_coverage=0.59, gig_worker_pct=0.13,
        literacy_rate=0.75, net_primary_enrolment=0.80, urban_rural_score_gap=0.14,
        primary_language="Efik/English",
    ),
    "Akwa Ibom": StateProfile(
        name="Akwa Ibom", geopolitical_zone="SS", capital="Uyo", population_millions=5.5,
        is_northern=False,
        maternal_mortality_ratio=540, u5_mortality_rate=72, health_worker_density=2.7,
        nhis_coverage_rate=0.050, oop_expenditure_pct=0.68,
        rural_population_pct=0.49,
        poverty_rate=0.28, informal_sector_pct=0.62, youth_unemployment_rate=0.27,
        gender_wage_gap=0.20, bvn_coverage=0.61, gig_worker_pct=0.14,
        literacy_rate=0.77, net_primary_enrolment=0.82, urban_rural_score_gap=0.13,
        primary_language="Ibibio/English",
    ),
    "Bayelsa": StateProfile(
        name="Bayelsa", geopolitical_zone="SS", capital="Yenagoa", population_millions=2.3,
        is_northern=False,
        maternal_mortality_ratio=620, u5_mortality_rate=83, health_worker_density=2.3,
        nhis_coverage_rate=0.040, oop_expenditure_pct=0.71,
        rural_population_pct=0.58,
        poverty_rate=0.33, informal_sector_pct=0.66, youth_unemployment_rate=0.29,
        gender_wage_gap=0.22, bvn_coverage=0.56, gig_worker_pct=0.12,
        literacy_rate=0.72, net_primary_enrolment=0.78, urban_rural_score_gap=0.15,
        primary_language="Ijaw/English",
    ),
}

# ── National average (default when no state selected) ─────────────────────────
NATIONAL_AVERAGE = StateProfile(
    name="Nigeria (National Average)",
    geopolitical_zone="All",
    capital="Abuja",
    population_millions=218.0,
    is_northern=False,
    maternal_mortality_ratio=1047,
    u5_mortality_rate=117,
    health_worker_density=1.95,
    nhis_coverage_rate=0.045,
    oop_expenditure_pct=0.748,
    rural_population_pct=0.48,
    poverty_rate=0.387,
    informal_sector_pct=0.649,
    youth_unemployment_rate=0.333,
    gender_wage_gap=0.24,
    bvn_coverage=0.55,
    gig_worker_pct=0.15,
    literacy_rate=0.62,
    net_primary_enrolment=0.70,
    urban_rural_score_gap=0.22,
    primary_language="English/Hausa/Yoruba/Igbo",
)


# ── Helper functions ───────────────────────────────────────────────────────────

def get_state_names() -> list[str]:
    """Return sorted list of all state names for display."""
    return sorted(NIGERIA_STATES.keys())


def get_state_params(state_name: str) -> StateProfile:
    """Return StateProfile for a given state name. Falls back to national average."""
    if state_name in ("Nigeria (National Average)", "Nigeria", "", None):
        return NATIONAL_AVERAGE
    return NIGERIA_STATES.get(state_name, NATIONAL_AVERAGE)


def get_states_by_zone() -> dict[str, list[str]]:
    """Return states grouped by geopolitical zone for organised display."""
    zones: dict[str, list[str]] = {}
    for name, profile in NIGERIA_STATES.items():
        zones.setdefault(profile.geopolitical_zone, []).append(name)
    return {z: sorted(states) for z, states in sorted(zones.items())}


def apply_state_to_preset(preset, state_name: str):
    """
    Override a scenario preset's population parameters with state-level values.

    Works with both EconomicScenarioPreset and HealthFinanceScenarioPreset.
    Returns a modified copy of the preset with state-specific values.
    """
    import copy
    state = get_state_params(state_name)
    p = copy.copy(preset)

    # Fields that exist on both preset types
    shared_fields = {
        "poverty_rate":           state.poverty_rate,
        "informal_sector_pct":    state.informal_sector_pct,
        "rural_population_pct":   state.rural_population_pct,
    }
    # Health-specific fields
    health_fields = {
        "nhis_coverage_rate":      state.nhis_coverage_rate,
        "oop_expenditure_pct":     state.oop_expenditure_pct,
        "maternal_mortality_ratio": state.maternal_mortality_ratio,
        "u5_mortality_rate":       state.u5_mortality_rate,
        "health_worker_density":   state.health_worker_density,
        "north_south_literacy_gap": abs(state.literacy_rate - 0.62),
    }
    # Economic-specific fields
    economic_fields = {
        "youth_unemployment_rate": state.youth_unemployment_rate,
        "gender_wage_gap":         state.gender_wage_gap,
        "gig_worker_pct":          state.gig_worker_pct,
    }

    for field_name, value in {**shared_fields, **health_fields, **economic_fields}.items():
        if hasattr(p, field_name):
            setattr(p, field_name, value)

    return p


# ── Streamlit selector widget ──────────────────────────────────────────────────

def state_selector(
    key: str = "_selected_state",
    label: str = " State / Region",
    include_national: bool = True,
    location: str = "sidebar",
) -> str:
    """
    Render a state selector dropdown.

    Parameters
    ----------
    key              : Streamlit session_state key (unique per page)
    label            : Display label above the dropdown
    include_national : Whether to include "Nigeria (National Average)" option
    location         : "sidebar" | "main"

    Returns
    -------
    Selected state name string
    """
    options = []
    if include_national:
        options.append("🇳🇬 Nigeria (National Average)")

    zones = get_states_by_zone()
    zone_labels = {
        "NC": "North-Central", "NE": "North-East", "NW": "North-West",
        "SE": "South-East",    "SS": "South-South", "SW": "South-West",
    }

    # Flat list grouped by zone
    for zone_code in ["NW", "NE", "NC", "SW", "SE", "SS"]:
        zone_name = zone_labels.get(zone_code, zone_code)
        for state in zones.get(zone_code, []):
            profile = NIGERIA_STATES[state]
            options.append(f"  {state} ({zone_name})")

    container = st.sidebar if location == "sidebar" else st

    container.markdown(
        f"<p style='font-size:.78rem;font-weight:600;color:#0891b2;"
        f"margin-bottom:2px;'>{label}</p>",
        unsafe_allow_html=True,
    )

    selected_display = container.selectbox(
        label,
        options,
        index=0,
        key=key,
        label_visibility="collapsed",
        help="Select a Nigerian state to calibrate simulation parameters to "
             "that state's demographic and health data.",
    )

    # Extract clean state name
    clean = selected_display.strip()
    if clean.startswith("🇳🇬"):
        return "Nigeria (National Average)"
    # Remove zone suffix like "(North-West)"
    clean = clean.split(" (")[0].strip()
    return clean


def state_info_card(state_name: str) -> None:
    """
    Render a compact info card showing key indicators for the selected state.
    Call after state_selector() to give users context before running.
    """
    profile = get_state_params(state_name)

    is_national = state_name in ("Nigeria (National Average)", "Nigeria", "", None)
    title = "🇳🇬 National Average" if is_national else f"📍 {profile.name}"
    zone_full = {
        "NW": "North-West", "NE": "North-East", "NC": "North-Central",
        "SW": "South-West", "SE": "South-East",  "SS": "South-South",
        "All": "All Zones",
    }.get(profile.geopolitical_zone, profile.geopolitical_zone)

    # Colour code zone
    zone_colour = {
        "NW": "#dc2626", "NE": "#ea580c", "NC": "#d97706",
        "SW": "#16a34a", "SE": "#0891b2", "SS": "#7c3aed", "All": "#334155",
    }.get(profile.geopolitical_zone, "#334155")

    st.markdown(
        f"""<div style="background:#f8fafc;border:1px solid #e2e8f0;
        border-left:4px solid {zone_colour};border-radius:8px;
        padding:10px 14px;margin:8px 0 4px;">
        <p style="margin:0 0 4px;font-weight:700;font-size:.92rem;
        color:#0f172a;">{title}</p>
        <p style="margin:0 0 6px;font-size:.75rem;color:{zone_colour};
        font-weight:600;">{zone_full} · {profile.primary_language}</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:2px 12px;
        font-size:.72rem;color:#475569;">
        <span>👩‍⚕️ MMR: <b>{profile.maternal_mortality_ratio:,}/100k</b></span>
        <span>🏥 HW density: <b>{profile.health_worker_density}</b>/10k</span>
        <span>💳 NHIS cover: <b>{profile.nhis_coverage_rate:.1%}</b></span>
        <span>💸 OOP burden: <b>{profile.oop_expenditure_pct:.1%}</b></span>
        <span>📉 Poverty: <b>{profile.poverty_rate:.1%}</b></span>
        <span>🏙️ Rural pop: <b>{profile.rural_population_pct:.1%}</b></span>
        <span>👷 Informal: <b>{profile.informal_sector_pct:.1%}</b></span>
        <span>📚 Literacy: <b>{profile.literacy_rate:.1%}</b></span>
        </div>
        {f'<p style="margin:6px 0 0;font-size:.68rem;color:#94a3b8;font-style:italic;">{profile.notes}</p>' if profile.notes else ''}
        """,
        unsafe_allow_html=True,
    )