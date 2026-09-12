"""
components/nigeria_states.py
==============================
State-level calibration data for all 36 Nigerian states + FCT.

Each state has empirical parameters drawn from:
  - NDHS (demographic & health)
  - NBS (labour force, poverty)
  - WHO AFRO (health workforce)
  - World Bank Nigeria (poverty & live macro metrics)
  - NHIA (insurance coverage)
  - UBEC (education)

These override the national-average defaults whenever a user
selects a specific state from the sidebar selector.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Any
import copy
import streamlit as st


# ── State data structure ───────────────────────────────────────────────────────

@dataclass
class StateProfile:
    """Empirical parameters for one Nigerian state."""
    name: str
    geopolitical_zone: str  # NW | NE | NC | SW | SE | SS
    capital: str
    population_millions: float
    is_northern: bool  # True = NW/NE/NC

    # Health parameters
    maternal_mortality_ratio: float  # per 100,000 live births (MMR)
    u5_mortality_rate: float  # per 1,000 live births (U5MR)
    health_worker_density: float  # per 10,000 population
    nhis_coverage_rate: float  # fraction with formal health insurance (0.0 - 1.0)
    oop_expenditure_pct: float  # OOP as share of total health spending (0.0 - 1.0)
    rural_population_pct: float  # rural fraction of population (0.0 - 1.0)

    # Economic parameters
    poverty_rate: float  # proportion below poverty line (0.0 - 1.0)
    informal_sector_pct: float  # informal workforce fraction (0.0 - 1.0)
    youth_unemployment_rate: float  # youth (15-34) unemployment (0.0 - 1.0)
    gender_wage_gap: float  # raw gender wage gap (0.0 - 1.0)
    bvn_coverage: float  # Bank Verification Number coverage (0.0 - 1.0)
    gig_worker_pct: float  # fraction in gig economy (0.0 - 1.0)

    # Education parameters
    literacy_rate: float  # adult literacy rate (0.0 - 1.0)
    net_primary_enrolment: float  # primary school net enrolment (0.0 - 1.0)
    urban_rural_score_gap: float  # JAMB score gap urban vs rural

    # Context
    primary_language: str
    notes: str = ""

    def to_metrics_dict(self) -> Dict[str, Any]:
        """Convert state profile parameters into standard key-value dictionary."""
        return {
            "u5mr": self.u5_mortality_rate,
            "mmr": self.maternal_mortality_ratio,
            "oop": round(self.oop_expenditure_pct * 100, 1),
            "hw_density": self.health_worker_density,
            "insurance_cov": round(self.nhis_coverage_rate * 100, 1),
            "poverty_rate": round(self.poverty_rate * 100, 1),
            "informal_pct": round(self.informal_sector_pct * 100, 1),
            "literacy_rate": round(self.literacy_rate * 100, 1),
            "rural_pct": round(self.rural_population_pct * 100, 1),
            "primary_language": self.primary_language,
            "zone": self.geopolitical_zone
        }


# ── All 36 states + FCT Baseline Profiles ─────────────────────────────────────

NIGERIA_STATES: Dict[str, StateProfile] = {

    # ── North-West ────────────────────────────────────────────────────────────
    "Kano": StateProfile(
        name="Kano", geopolitical_zone="NW", capital="Kano", population_millions=15.9,
        is_northern=True,
        maternal_mortality_ratio=1200, u5_mortality_rate=143, health_worker_density=1.1,
        nhis_coverage_rate=0.025, oop_expenditure_pct=0.82, rural_population_pct=0.52,
        poverty_rate=0.47, informal_sector_pct=0.74, youth_unemployment_rate=0.38,
        gender_wage_gap=0.31, bvn_coverage=0.48, gig_worker_pct=0.12,
        literacy_rate=0.51, net_primary_enrolment=0.61, urban_rural_score_gap=0.26,
        primary_language="Hausa",
    ),
    "Kaduna": StateProfile(
        name="Kaduna", geopolitical_zone="NW", capital="Kaduna", population_millions=9.0,
        is_northern=True,
        maternal_mortality_ratio=1050, u5_mortality_rate=131, health_worker_density=1.4,
        nhis_coverage_rate=0.032, oop_expenditure_pct=0.79, rural_population_pct=0.58,
        poverty_rate=0.43, informal_sector_pct=0.71, youth_unemployment_rate=0.36,
        gender_wage_gap=0.28, bvn_coverage=0.51, gig_worker_pct=0.10,
        literacy_rate=0.57, net_primary_enrolment=0.65, urban_rural_score_gap=0.24,
        primary_language="Hausa",
    ),
    "Katsina": StateProfile(
        name="Katsina", geopolitical_zone="NW", capital="Katsina", population_millions=8.8,
        is_northern=True,
        maternal_mortality_ratio=1320, u5_mortality_rate=158, health_worker_density=0.9,
        nhis_coverage_rate=0.018, oop_expenditure_pct=0.85, rural_population_pct=0.71,
        poverty_rate=0.52, informal_sector_pct=0.78, youth_unemployment_rate=0.41,
        gender_wage_gap=0.34, bvn_coverage=0.40, gig_worker_pct=0.07,
        literacy_rate=0.43, net_primary_enrolment=0.55, urban_rural_score_gap=0.29,
        primary_language="Hausa",
    ),
    "Sokoto": StateProfile(
        name="Sokoto", geopolitical_zone="NW", capital="Sokoto", population_millions=5.9,
        is_northern=True,
        maternal_mortality_ratio=1560, u5_mortality_rate=172, health_worker_density=0.7,
        nhis_coverage_rate=0.015, oop_expenditure_pct=0.87, rural_population_pct=0.76,
        poverty_rate=0.58, informal_sector_pct=0.81, youth_unemployment_rate=0.43,
        gender_wage_gap=0.36, bvn_coverage=0.35, gig_worker_pct=0.05,
        literacy_rate=0.38, net_primary_enrolment=0.49, urban_rural_score_gap=0.31,
        primary_language="Hausa",
    ),
    "Zamfara": StateProfile(
        name="Zamfara", geopolitical_zone="NW", capital="Gusau", population_millions=5.0,
        is_northern=True,
        maternal_mortality_ratio=1490, u5_mortality_rate=168, health_worker_density=0.6,
        nhis_coverage_rate=0.012, oop_expenditure_pct=0.88, rural_population_pct=0.78,
        poverty_rate=0.61, informal_sector_pct=0.82, youth_unemployment_rate=0.44,
        gender_wage_gap=0.37, bvn_coverage=0.32, gig_worker_pct=0.04,
        literacy_rate=0.35, net_primary_enrolment=0.46, urban_rural_score_gap=0.32,
        primary_language="Hausa",
    ),
    "Kebbi": StateProfile(
        name="Kebbi", geopolitical_zone="NW", capital="Birnin Kebbi", population_millions=4.4,
        is_northern=True,
        maternal_mortality_ratio=1380, u5_mortality_rate=161, health_worker_density=0.8,
        nhis_coverage_rate=0.016, oop_expenditure_pct=0.86, rural_population_pct=0.74,
        poverty_rate=0.56, informal_sector_pct=0.80, youth_unemployment_rate=0.42,
        gender_wage_gap=0.35, bvn_coverage=0.37, gig_worker_pct=0.06,
        literacy_rate=0.40, net_primary_enrolment=0.51, urban_rural_score_gap=0.30,
        primary_language="Hausa",
    ),
    "Jigawa": StateProfile(
        name="Jigawa", geopolitical_zone="NW", capital="Dutse", population_millions=5.8,
        is_northern=True,
        maternal_mortality_ratio=1180, u5_mortality_rate=148, health_worker_density=1.0,
        nhis_coverage_rate=0.020, oop_expenditure_pct=0.83, rural_population_pct=0.72,
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
        nhis_coverage_rate=0.014, oop_expenditure_pct=0.89, rural_population_pct=0.68,
        poverty_rate=0.63, informal_sector_pct=0.83, youth_unemployment_rate=0.46,
        gender_wage_gap=0.38, bvn_coverage=0.30, gig_worker_pct=0.04,
        literacy_rate=0.33, net_primary_enrolment=0.44, urban_rural_score_gap=0.34,
        primary_language="Kanuri/Hausa", notes="Conflict-affected; high humanitarian context",
    ),
    "Adamawa": StateProfile(
        name="Adamawa", geopolitical_zone="NE", capital="Yola", population_millions=4.9,
        is_northern=True,
        maternal_mortality_ratio=1100, u5_mortality_rate=136, health_worker_density=1.1,
        nhis_coverage_rate=0.022, oop_expenditure_pct=0.81, rural_population_pct=0.62,
        poverty_rate=0.46, informal_sector_pct=0.73, youth_unemployment_rate=0.37,
        gender_wage_gap=0.30, bvn_coverage=0.44, gig_worker_pct=0.08,
        literacy_rate=0.53, net_primary_enrolment=0.62, urban_rural_score_gap=0.25,
        primary_language="Hausa/Fulfulde",
    ),
    "Gombe": StateProfile(
        name="Gombe", geopolitical_zone="NE", capital="Gombe", population_millions=3.9,
        is_northern=True,
        maternal_mortality_ratio=1090, u5_mortality_rate=133, health_worker_density=1.2,
        nhis_coverage_rate=0.024, oop_expenditure_pct=0.80, rural_population_pct=0.60,
        poverty_rate=0.45, informal_sector_pct=0.72, youth_unemployment_rate=0.36,
        gender_wage_gap=0.29, bvn_coverage=0.46, gig_worker_pct=0.09,
        literacy_rate=0.55, net_primary_enrolment=0.64, urban_rural_score_gap=0.24,
        primary_language="Hausa",
    ),
    "Yobe": StateProfile(
        name="Yobe", geopolitical_zone="NE", capital="Damaturu", population_millions=3.5,
        is_northern=True,
        maternal_mortality_ratio=1400, u5_mortality_rate=165, health_worker_density=0.7,
        nhis_coverage_rate=0.013, oop_expenditure_pct=0.87, rural_population_pct=0.73,
        poverty_rate=0.57, informal_sector_pct=0.80, youth_unemployment_rate=0.42,
        gender_wage_gap=0.36, bvn_coverage=0.33, gig_worker_pct=0.05,
        literacy_rate=0.36, net_primary_enrolment=0.47, urban_rural_score_gap=0.31,
        primary_language="Hausa/Kanuri",
    ),
    "Bauchi": StateProfile(
        name="Bauchi", geopolitical_zone="NE", capital="Bauchi", population_millions=7.2,
        is_northern=True,
        maternal_mortality_ratio=1150, u5_mortality_rate=140, health_worker_density=1.0,
        nhis_coverage_rate=0.021, oop_expenditure_pct=0.82, rural_population_pct=0.65,
        poverty_rate=0.48, informal_sector_pct=0.74, youth_unemployment_rate=0.38,
        gender_wage_gap=0.31, bvn_coverage=0.43, gig_worker_pct=0.08,
        literacy_rate=0.50, net_primary_enrolment=0.59, urban_rural_score_gap=0.26,
        primary_language="Hausa",
    ),
    "Taraba": StateProfile(
        name="Taraba", geopolitical_zone="NE", capital="Jalingo", population_millions=3.7,
        is_northern=True,
        maternal_mortality_ratio=1080, u5_mortality_rate=130, health_worker_density=1.2,
        nhis_coverage_rate=0.023, oop_expenditure_pct=0.80, rural_population_pct=0.67,
        poverty_rate=0.44, informal_sector_pct=0.71, youth_unemployment_rate=0.35,
        gender_wage_gap=0.28, bvn_coverage=0.45, gig_worker_pct=0.08,
        literacy_rate=0.56, net_primary_enrolment=0.65, urban_rural_score_gap=0.23,
        primary_language="Hausa/Jukun",
    ),

    # ── North-Central ─────────────────────────────────────────────────────────
    "FCT": StateProfile(
        name="FCT", geopolitical_zone="NC", capital="Abuja", population_millions=3.9,
        is_northern=False,
        maternal_mortality_ratio=380, u5_mortality_rate=58, health_worker_density=18.5,
        nhis_coverage_rate=0.35, oop_expenditure_pct=0.542, rural_population_pct=0.25,
        poverty_rate=0.14, informal_sector_pct=0.42, youth_unemployment_rate=0.21,
        gender_wage_gap=0.18, bvn_coverage=0.78, gig_worker_pct=0.22,
        literacy_rate=0.87, net_primary_enrolment=0.91, urban_rural_score_gap=0.11,
        primary_language="English/Hausa", notes="Federal Capital Territory; high urban capacity",
    ),
    "Niger": StateProfile(
        name="Niger", geopolitical_zone="NC", capital="Minna", population_millions=6.9,
        is_northern=True,
        maternal_mortality_ratio=890, u5_mortality_rate=115, health_worker_density=1.5,
        nhis_coverage_rate=0.031, oop_expenditure_pct=0.76, rural_population_pct=0.63,
        poverty_rate=0.40, informal_sector_pct=0.68, youth_unemployment_rate=0.33,
        gender_wage_gap=0.26, bvn_coverage=0.49, gig_worker_pct=0.09,
        literacy_rate=0.62, net_primary_enrolment=0.70, urban_rural_score_gap=0.21,
        primary_language="Hausa/Nupe",
    ),
    "Plateau": StateProfile(
        name="Plateau", geopolitical_zone="NC", capital="Jos", population_millions=4.2,
        is_northern=False,
        maternal_mortality_ratio=730, u5_mortality_rate=95, health_worker_density=2.0,
        nhis_coverage_rate=0.041, oop_expenditure_pct=0.71, rural_population_pct=0.55,
        poverty_rate=0.34, informal_sector_pct=0.63, youth_unemployment_rate=0.30,
        gender_wage_gap=0.23, bvn_coverage=0.54, gig_worker_pct=0.11,
        literacy_rate=0.69, net_primary_enrolment=0.75, urban_rural_score_gap=0.18,
        primary_language="English/Birom",
    ),
    "Benue": StateProfile(
        name="Benue", geopolitical_zone="NC", capital="Makurdi", population_millions=6.1,
        is_northern=False,
        maternal_mortality_ratio=810, u5_mortality_rate=104, health_worker_density=1.7,
        nhis_coverage_rate=0.035, oop_expenditure_pct=0.74, rural_population_pct=0.60,
        poverty_rate=0.37, informal_sector_pct=0.66, youth_unemployment_rate=0.31,
        gender_wage_gap=0.24, bvn_coverage=0.51, gig_worker_pct=0.10,
        literacy_rate=0.65, net_primary_enrolment=0.72, urban_rural_score_gap=0.20,
        primary_language="Tiv/Idoma",
    ),
    "Kogi": StateProfile(
        name="Kogi", geopolitical_zone="NC", capital="Lokoja", population_millions=4.5,
        is_northern=False,
        maternal_mortality_ratio=850, u5_mortality_rate=110, health_worker_density=1.6,
        nhis_coverage_rate=0.033, oop_expenditure_pct=0.75, rural_population_pct=0.58,
        poverty_rate=0.38, informal_sector_pct=0.67, youth_unemployment_rate=0.32,
        gender_wage_gap=0.25, bvn_coverage=0.52, gig_worker_pct=0.10,
        literacy_rate=0.67, net_primary_enrolment=0.73, urban_rural_score_gap=0.19,
        primary_language="Yoruba/Igbo/Nupe",
    ),
    "Kwara": StateProfile(
        name="Kwara", geopolitical_zone="NC", capital="Ilorin", population_millions=3.2,
        is_northern=False,
        maternal_mortality_ratio=680, u5_mortality_rate=87, health_worker_density=2.2,
        nhis_coverage_rate=0.048, oop_expenditure_pct=0.69, rural_population_pct=0.48,
        poverty_rate=0.30, informal_sector_pct=0.58, youth_unemployment_rate=0.27,
        gender_wage_gap=0.21, bvn_coverage=0.58, gig_worker_pct=0.13,
        literacy_rate=0.74, net_primary_enrolment=0.79, urban_rural_score_gap=0.16,
        primary_language="Yoruba",
    ),
    "Nasarawa": StateProfile(
        name="Nasarawa", geopolitical_zone="NC", capital="Lafia", population_millions=2.5,
        is_northern=False,
        maternal_mortality_ratio=760, u5_mortality_rate=98, health_worker_density=1.8,
        nhis_coverage_rate=0.038, oop_expenditure_pct=0.72, rural_population_pct=0.57,
        poverty_rate=0.35, informal_sector_pct=0.65, youth_unemployment_rate=0.30,
        gender_wage_gap=0.23, bvn_coverage=0.53, gig_worker_pct=0.11,
        literacy_rate=0.67, net_primary_enrolment=0.73, urban_rural_score_gap=0.18,
        primary_language="Eggon/Hausa",
    ),

    # ── South-West ────────────────────────────────────────────────────────────
    "Lagos": StateProfile(
        name="Lagos", geopolitical_zone="SW", capital="Ikeja", population_millions=21.3,
        is_northern=False,
        maternal_mortality_ratio=320, u5_mortality_rate=45, health_worker_density=22.1,
        nhis_coverage_rate=0.42, oop_expenditure_pct=0.61, rural_population_pct=0.09,
        poverty_rate=0.09, informal_sector_pct=0.51, youth_unemployment_rate=0.17,
        gender_wage_gap=0.15, bvn_coverage=0.88, gig_worker_pct=0.31,
        literacy_rate=0.92, net_primary_enrolment=0.94, urban_rural_score_gap=0.07,
        primary_language="Yoruba/English", notes="Commercial hub; highest healthcare density",
    ),
    "Oyo": StateProfile(
        name="Oyo", geopolitical_zone="SW", capital="Ibadan", population_millions=8.3,
        is_northern=False,
        maternal_mortality_ratio=380, u5_mortality_rate=52, health_worker_density=4.2,
        nhis_coverage_rate=0.082, oop_expenditure_pct=0.61, rural_population_pct=0.32,
        poverty_rate=0.21, informal_sector_pct=0.58, youth_unemployment_rate=0.22,
        gender_wage_gap=0.17, bvn_coverage=0.72, gig_worker_pct=0.20,
        literacy_rate=0.85, net_primary_enrolment=0.89, urban_rural_score_gap=0.10,
        primary_language="Yoruba",
    ),
    "Osun": StateProfile(
        name="Osun", geopolitical_zone="SW", capital="Osogbo", population_millions=4.7,
        is_northern=False,
        maternal_mortality_ratio=410, u5_mortality_rate=56, health_worker_density=3.8,
        nhis_coverage_rate=0.071, oop_expenditure_pct=0.63, rural_population_pct=0.38,
        poverty_rate=0.23, informal_sector_pct=0.60, youth_unemployment_rate=0.23,
        gender_wage_gap=0.18, bvn_coverage=0.69, gig_worker_pct=0.18,
        literacy_rate=0.84, net_primary_enrolment=0.88, urban_rural_score_gap=0.10,
        primary_language="Yoruba",
    ),
    "Ogun": StateProfile(
        name="Ogun", geopolitical_zone="SW", capital="Abeokuta", population_millions=5.9,
        is_northern=False,
        maternal_mortality_ratio=350, u5_mortality_rate=48, health_worker_density=4.6,
        nhis_coverage_rate=0.095, oop_expenditure_pct=0.58, rural_population_pct=0.28,
        poverty_rate=0.16, informal_sector_pct=0.54, youth_unemployment_rate=0.19,
        gender_wage_gap=0.16, bvn_coverage=0.79, gig_worker_pct=0.24,
        literacy_rate=0.89, net_primary_enrolment=0.92, urban_rural_score_gap=0.08,
        primary_language="Yoruba",
    ),
    "Ondo": StateProfile(
        name="Ondo", geopolitical_zone="SW", capital="Akure", population_millions=5.2,
        is_northern=False,
        maternal_mortality_ratio=440, u5_mortality_rate=59, health_worker_density=3.5,
        nhis_coverage_rate=0.062, oop_expenditure_pct=0.65, rural_population_pct=0.42,
        poverty_rate=0.25, informal_sector_pct=0.62, youth_unemployment_rate=0.24,
        gender_wage_gap=0.19, bvn_coverage=0.65, gig_worker_pct=0.16,
        literacy_rate=0.82, net_primary_enrolment=0.87, urban_rural_score_gap=0.11,
        primary_language="Yoruba",
    ),
    "Ekiti": StateProfile(
        name="Ekiti", geopolitical_zone="SW", capital="Ado-Ekiti", population_millions=3.3,
        is_northern=False,
        maternal_mortality_ratio=460, u5_mortality_rate=62, health_worker_density=3.3,
        nhis_coverage_rate=0.058, oop_expenditure_pct=0.66, rural_population_pct=0.45,
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
        nhis_coverage_rate=0.078, oop_expenditure_pct=0.62, rural_population_pct=0.35,
        poverty_rate=0.18, informal_sector_pct=0.55, youth_unemployment_rate=0.20,
        gender_wage_gap=0.16, bvn_coverage=0.74, gig_worker_pct=0.22,
        literacy_rate=0.87, net_primary_enrolment=0.91, urban_rural_score_gap=0.09,
        primary_language="Igbo",
    ),
    "Imo": StateProfile(
        name="Imo", geopolitical_zone="SE", capital="Owerri", population_millions=5.4,
        is_northern=False,
        maternal_mortality_ratio=430, u5_mortality_rate=58, health_worker_density=3.6,
        nhis_coverage_rate=0.068, oop_expenditure_pct=0.64, rural_population_pct=0.40,
        poverty_rate=0.22, informal_sector_pct=0.58, youth_unemployment_rate=0.22,
        gender_wage_gap=0.17, bvn_coverage=0.70, gig_worker_pct=0.19,
        literacy_rate=0.85, net_primary_enrolment=0.89, urban_rural_score_gap=0.10,
        primary_language="Igbo",
    ),
    "Enugu": StateProfile(
        name="Enugu", geopolitical_zone="SE", capital="Enugu", population_millions=5.1,
        is_northern=False,
        maternal_mortality_ratio=400, u5_mortality_rate=55, health_worker_density=3.9,
        nhis_coverage_rate=0.074, oop_expenditure_pct=0.63, rural_population_pct=0.38,
        poverty_rate=0.20, informal_sector_pct=0.56, youth_unemployment_rate=0.21,
        gender_wage_gap=0.16, bvn_coverage=0.72, gig_worker_pct=0.21,
        literacy_rate=0.86, net_primary_enrolment=0.90, urban_rural_score_gap=0.09,
        primary_language="Igbo",
    ),
    "Abia": StateProfile(
        name="Abia", geopolitical_zone="SE", capital="Umuahia", population_millions=3.7,
        is_northern=False,
        maternal_mortality_ratio=450, u5_mortality_rate=61, health_worker_density=3.4,
        nhis_coverage_rate=0.061, oop_expenditure_pct=0.65, rural_population_pct=0.43,
        poverty_rate=0.24, informal_sector_pct=0.60, youth_unemployment_rate=0.23,
        gender_wage_gap=0.18, bvn_coverage=0.67, gig_worker_pct=0.17,
        literacy_rate=0.83, net_primary_enrolment=0.87, urban_rural_score_gap=0.10,
        primary_language="Igbo",
    ),
    "Ebonyi": StateProfile(
        name="Ebonyi", geopolitical_zone="SE", capital="Abakaliki", population_millions=3.1,
        is_northern=False,
        maternal_mortality_ratio=580, u5_mortality_rate=78, health_worker_density=2.5,
        nhis_coverage_rate=0.043, oop_expenditure_pct=0.70, rural_population_pct=0.56,
        poverty_rate=0.32, informal_sector_pct=0.66, youth_unemployment_rate=0.29,
        gender_wage_gap=0.22, bvn_coverage=0.57, gig_worker_pct=0.12,
        literacy_rate=0.73, net_primary_enrolment=0.79, urban_rural_score_gap=0.15,
        primary_language="Igbo",
    ),

    # ── South-South ───────────────────────────────────────────────────────────
    "Rivers": StateProfile(
        name="Rivers", geopolitical_zone="SS", capital="Port Harcourt", population_millions=7.9,
        is_northern=False,
        maternal_mortality_ratio=410, u5_mortality_rate=62, health_worker_density=14.2,
        nhis_coverage_rate=0.20, oop_expenditure_pct=0.65, rural_population_pct=0.36,
        poverty_rate=0.22, informal_sector_pct=0.57, youth_unemployment_rate=0.22,
        gender_wage_gap=0.18, bvn_coverage=0.69, gig_worker_pct=0.19,
        literacy_rate=0.83, net_primary_enrolment=0.87, urban_rural_score_gap=0.10,
        primary_language="English/Ijaw",
    ),
    "Delta": StateProfile(
        name="Delta", geopolitical_zone="SS", capital="Asaba", population_millions=5.7,
        is_northern=False,
        maternal_mortality_ratio=510, u5_mortality_rate=68, health_worker_density=2.9,
        nhis_coverage_rate=0.058, oop_expenditure_pct=0.67, rural_population_pct=0.40,
        poverty_rate=0.25, informal_sector_pct=0.59, youth_unemployment_rate=0.24,
        gender_wage_gap=0.19, bvn_coverage=0.66, gig_worker_pct=0.17,
        literacy_rate=0.81, net_primary_enrolment=0.86, urban_rural_score_gap=0.11,
        primary_language="Urhobo/Igbo",
    ),
    "Edo": StateProfile(
        name="Edo", geopolitical_zone="SS", capital="Benin City", population_millions=4.7,
        is_northern=False,
        maternal_mortality_ratio=490, u5_mortality_rate=66, health_worker_density=3.0,
        nhis_coverage_rate=0.062, oop_expenditure_pct=0.66, rural_population_pct=0.38,
        poverty_rate=0.23, informal_sector_pct=0.58, youth_unemployment_rate=0.23,
        gender_wage_gap=0.18, bvn_coverage=0.68, gig_worker_pct=0.18,
        literacy_rate=0.82, net_primary_enrolment=0.87, urban_rural_score_gap=0.10,
        primary_language="Edo/Igbo",
    ),
    "Cross River": StateProfile(
        name="Cross River", geopolitical_zone="SS", capital="Calabar", population_millions=4.1,
        is_northern=False,
        maternal_mortality_ratio=560, u5_mortality_rate=75, health_worker_density=2.6,
        nhis_coverage_rate=0.047, oop_expenditure_pct=0.69, rural_population_pct=0.52,
        poverty_rate=0.30, informal_sector_pct=0.64, youth_unemployment_rate=0.28,
        gender_wage_gap=0.21, bvn_coverage=0.59, gig_worker_pct=0.13,
        literacy_rate=0.75, net_primary_enrolment=0.80, urban_rural_score_gap=0.14,
        primary_language="Efik/English",
    ),
    "Akwa Ibom": StateProfile(
        name="Akwa Ibom", geopolitical_zone="SS", capital="Uyo", population_millions=5.5,
        is_northern=False,
        maternal_mortality_ratio=540, u5_mortality_rate=72, health_worker_density=2.7,
        nhis_coverage_rate=0.050, oop_expenditure_pct=0.68, rural_population_pct=0.49,
        poverty_rate=0.28, informal_sector_pct=0.62, youth_unemployment_rate=0.27,
        gender_wage_gap=0.20, bvn_coverage=0.61, gig_worker_pct=0.14,
        literacy_rate=0.77, net_primary_enrolment=0.82, urban_rural_score_gap=0.13,
        primary_language="Ibibio/English",
    ),
    "Bayelsa": StateProfile(
        name="Bayelsa", geopolitical_zone="SS", capital="Yenagoa", population_millions=2.3,
        is_northern=False,
        maternal_mortality_ratio=620, u5_mortality_rate=83, health_worker_density=2.3,
        nhis_coverage_rate=0.040, oop_expenditure_pct=0.71, rural_population_pct=0.58,
        poverty_rate=0.33, informal_sector_pct=0.66, youth_unemployment_rate=0.29,
        gender_wage_gap=0.22, bvn_coverage=0.56, gig_worker_pct=0.12,
        literacy_rate=0.72, net_primary_enrolment=0.78, urban_rural_score_gap=0.15,
        primary_language="Ijaw/English",
    ),
}

# Aliases to fix naming mismatches across codebase
NIGERIA_STATES["FCT (Abuja)"] = NIGERIA_STATES["FCT"]

# ── National Average (Default Fallback) ────────────────────────────────────────

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


# ── State Retrieval Helpers ───────────────────────────────────────────────────

def get_state_names() -> List[str]:
    """Return sorted list of all unique state names."""
    return sorted(list(set(NIGERIA_STATES.keys()) - {"FCT (Abuja)"}))


def get_state_params(state_name: str) -> StateProfile:
    """Return StateProfile object for a state. Defaults to National Average."""
    clean_name = state_name.replace("🇳🇬", "").strip()
    if clean_name in ("Nigeria (National Average)", "Nigeria", "", None):
        return NATIONAL_AVERAGE
    return NIGERIA_STATES.get(clean_name, NATIONAL_AVERAGE)


def get_state_metrics_dict(state_name: str) -> Dict[str, Any]:
    """Returns a key-value dictionary of state indicators for render functions."""
    profile = get_state_params(state_name)
    return profile.to_metrics_dict()


def get_states_by_zone() -> Dict[str, List[str]]:
    """Return states grouped by geopolitical zone."""
    zones: Dict[str, List[str]] = {}
    for name, profile in NIGERIA_STATES.items():
        if name != "FCT (Abuja)":
            zones.setdefault(profile.geopolitical_zone, []).append(name)
    return {z: sorted(list(set(states))) for z, states in sorted(zones.items())}


# ── Live World Bank Data Scaling ──────────────────────────────────────────────

def apply_live_data_overrides(state_profile: StateProfile, live_nat_data: Dict[str, Any]) -> StateProfile:
    """
    Dynamically scales static state profiles using recent live World Bank national indicators.
    """
    if not live_nat_data or not live_nat_data.get("_any_live", False):
        return state_profile

    scaled = copy.copy(state_profile)

    # Scale U5MR if live national figure exists
    live_u5mr = live_nat_data.get("u5mr", {}).get("value") if isinstance(live_nat_data.get("u5mr"), dict) else None
    if live_u5mr and NATIONAL_AVERAGE.u5_mortality_rate > 0:
        ratio = live_u5mr / NATIONAL_AVERAGE.u5_mortality_rate
        scaled.u5_mortality_rate = round(scaled.u5_mortality_rate * ratio, 1)

    # Scale Out-of-Pocket Expenditure if live figure exists
    live_oop = live_nat_data.get("oop", {}).get("value") if isinstance(live_nat_data.get("oop"), dict) else None
    if live_oop and NATIONAL_AVERAGE.oop_expenditure_pct > 0:
        ratio = (live_oop / 100.0) / NATIONAL_AVERAGE.oop_expenditure_pct
        scaled.oop_expenditure_pct = min(0.95, round(scaled.oop_expenditure_pct * ratio, 3))

    return scaled


def get_state_historical_trends(state_name: str, start_year: int = 2018, end_year: int = 2025) -> Dict[
    str, List[Dict[str, Any]]]:
    """Generates multi-year historical trend series scaled for a specific state."""
    profile = get_state_params(state_name)

    # Base national trajectory slopes
    u5mr_base = profile.u5_mortality_rate
    oop_base = profile.oop_expenditure_pct * 100

    u5mr_series = []
    oop_series = []

    years = list(range(start_year, end_year + 1))
    n_years = len(years)

    for idx, yr in enumerate(years):
        # Progressively model slight historical declines/changes
        year_offset = (n_years - 1 - idx)
        u5mr_val = round(u5mr_base + (year_offset * 1.8), 1)
        oop_val = round(max(40.0, min(90.0, oop_base + (year_offset * 0.4))), 1)

        u5mr_series.append({"year": yr, "value": u5mr_val})
        oop_series.append({"year": yr, "value": oop_val})

    return {
        "u5mr_series": u5mr_series,
        "oop_series": oop_series
    }


def apply_state_to_preset(preset: Any, state_name: str) -> Any:
    """Override a scenario preset's parameters with state-level values."""
    state = get_state_params(state_name)
    p = copy.copy(preset)

    fields_map = {
        "poverty_rate": state.poverty_rate,
        "informal_sector_pct": state.informal_sector_pct,
        "rural_population_pct": state.rural_population_pct,
        "nhis_coverage_rate": state.nhis_coverage_rate,
        "oop_expenditure_pct": state.oop_expenditure_pct,
        "maternal_mortality_ratio": state.maternal_mortality_ratio,
        "u5_mortality_rate": state.u5_mortality_rate,
        "health_worker_density": state.health_worker_density,
        "youth_unemployment_rate": state.youth_unemployment_rate,
        "gender_wage_gap": state.gender_wage_gap,
        "gig_worker_pct": state.gig_worker_pct,
    }

    for field_name, value in fields_map.items():
        if hasattr(p, field_name):
            setattr(p, field_name, value)

    return p


# ── Streamlit Widgets ─────────────────────────────────────────────────────────

def state_selector(
        key: str = "_selected_state",
        label: str = "📍 Select State / Region",
        include_national: bool = True,
        location: str = "sidebar",
) -> str:
    """Render a state selector dropdown in Streamlit sidebar or main page."""
    options = []
    if include_national:
        options.append("🇳🇬 Nigeria (National Average)")

    zones = get_states_by_zone()
    zone_labels = {
        "NC": "North-Central", "NE": "North-East", "NW": "North-West",
        "SE": "South-East", "SS": "South-South", "SW": "South-West",
    }

    for zone_code in ["NW", "NE", "NC", "SW", "SE", "SS"]:
        zone_name = zone_labels.get(zone_code, zone_code)
        for st_name in zones.get(zone_code, []):
            options.append(f"  {st_name} ({zone_name})")

    container = st.sidebar if location == "sidebar" else st

    container.markdown(
        f"<p style='font-size:.78rem;font-weight:600;color:#0891b2;margin-bottom:2px;'>{label}</p>",
        unsafe_allow_html=True,
    )

    selected_display = container.selectbox(
        label,
        options,
        index=0,
        key=key,
        label_visibility="collapsed",
        help="Select a Nigerian state to calibrate parameters.",
    )

    clean = selected_display.strip()
    if clean.startswith("🇳🇬"):
        return "Nigeria (National Average)"
    clean = clean.split(" (")[0].strip()
    return clean


# Updated components/nigeria_states.py

METRIC_DESCRIPTIONS = {
    "mmr": {
        "label": "MMR",
        "year": "2024",
        "source": "NDHS",
        "desc": "Maternal Mortality Ratio: Deaths per 100,000 live births."
    },
    "hw_density": {
        "label": "HW Density",
        "year": "2023",
        "source": "MoH",
        "desc": "Core health workers (doctors, nurses, midwives) per 10,000 residents."
    },
    "insurance_cov": {
        "label": "Insurance Coverage",
        "year": "2024",
        "source": "NHIA",
        "desc": "Share of population covered by formal health insurance."
    },
    "oop": {
        "label": "Out-of-Pocket Spend",
        "year": "2023",
        "source": "NBS/WHO",
        "desc": "Percentage of total medical costs paid directly by individuals."
    },
    "poverty_rate": {
        "label": "Poverty Rate",
        "year": "2022",
        "source": "NBS MPI",
        "desc": "Share of residents living below the national multidimensional poverty line."
    },
    "rural_pct": {
        "label": "Rural Population",
        "year": "2023",
        "source": "NPC",
        "desc": "Percentage of residents living in rural areas vs. urban centers."
    },
    "informal_pct": {
        "label": "Informal Sector",
        "year": "2023",
        "source": "NBS Labor",
        "desc": "Share of workforce in non-salaried, unregulated, or self-employed jobs."
    },
    "literacy_rate": {
        "label": "Literacy Rate",
        "year": "2022",
        "source": "NBS/UNESCO",
        "desc": "Percentage of adults (15+) who can read and write with understanding."
    },
}


def state_info_card(state_name: str) -> None:
    """Render state info card with metric descriptions and source years."""
    profile = get_state_params(state_name)
    is_national = state_name in ("Nigeria (National Average)", "Nigeria", "", None)
    title = "🇳🇬 National Average" if is_national else f"📍 {profile.name}"

    zone_colour = {
        "NW": "#dc2626", "NE": "#ea580c", "NC": "#d97706",
        "SW": "#16a34a", "SE": "#0891b2", "SS": "#7c3aed", "All": "#334155",
    }.get(profile.geopolitical_zone, "#334155")

    zone_full = {
        "NW": "North-West", "NE": "North-East", "NC": "North-Central",
        "SW": "South-West", "SE": "South-East", "SS": "South-South", "All": "All Zones",
    }.get(profile.geopolitical_zone, profile.geopolitical_zone)

    # Render card
    st.markdown(
        f"""<div style="background:#f8fafc;border:1px solid #e2e8f0;
        border-left:4px solid {zone_colour};border-radius:8px;
        padding:10px 14px;margin:8px 0 4px;">
        <p style="margin:0 0 4px;font-weight:700;font-size:.92rem;color:#0f172a;">{title}</p>
        <p style="margin:0 0 6px;font-size:.75rem;color:{zone_colour};font-weight:600;">{zone_full} · {profile.primary_language}</p>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px 12px;font-size:.72rem;color:#475569;">
        <span title="{METRIC_DESCRIPTIONS['mmr']['desc']} ({METRIC_DESCRIPTIONS['mmr']['source']} {METRIC_DESCRIPTIONS['mmr']['year']})">👩‍⚕️ MMR: <b>{profile.maternal_mortality_ratio:,}/10k</b></span>
        <span title="{METRIC_DESCRIPTIONS['hw_density']['desc']} ({METRIC_DESCRIPTIONS['hw_density']['source']} {METRIC_DESCRIPTIONS['hw_density']['year']})">🏥 HW density: <b>{profile.health_worker_density}</b>/10k</span>
        <span title="{METRIC_DESCRIPTIONS['insurance_cov']['desc']} ({METRIC_DESCRIPTIONS['insurance_cov']['source']} {METRIC_DESCRIPTIONS['insurance_cov']['year']})">💳 Insurance: <b>{profile.nhis_coverage_rate:.1%}</b></span>
        <span title="{METRIC_DESCRIPTIONS['oop']['desc']} ({METRIC_DESCRIPTIONS['oop']['source']} {METRIC_DESCRIPTIONS['oop']['year']})">💸 OOP spend: <b>{profile.oop_expenditure_pct:.1%}</b></span>
        <span title="{METRIC_DESCRIPTIONS['poverty_rate']['desc']} ({METRIC_DESCRIPTIONS['poverty_rate']['source']} {METRIC_DESCRIPTIONS['poverty_rate']['year']})">📉 Poverty: <b>{profile.poverty_rate:.1%}</b></span>
        <span title="{METRIC_DESCRIPTIONS['rural_pct']['desc']} ({METRIC_DESCRIPTIONS['rural_pct']['source']} {METRIC_DESCRIPTIONS['rural_pct']['year']})">🏙️ Rural pop: <b>{profile.rural_population_pct:.1%}</b></span>
        <span title="{METRIC_DESCRIPTIONS['informal_pct']['desc']} ({METRIC_DESCRIPTIONS['informal_pct']['source']} {METRIC_DESCRIPTIONS['informal_pct']['year']})">👷 Informal: <b>{profile.informal_sector_pct:.1%}</b></span>
        <span title="{METRIC_DESCRIPTIONS['literacy_rate']['desc']} ({METRIC_DESCRIPTIONS['literacy_rate']['source']} {METRIC_DESCRIPTIONS['literacy_rate']['year']})">📚 Literacy: <b>{profile.literacy_rate:.1%}</b></span>
        </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Expandable definition table with sources & years
    with st.expander("📖 Data Sources & Indicator Definitions"):
        for _, info in METRIC_DESCRIPTIONS.items():
            st.markdown(
                f"- **{info['label']}** *({info['source']} {info['year']})*: {info['desc']}"
            )