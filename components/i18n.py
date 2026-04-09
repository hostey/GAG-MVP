# components/i18n.py
"""
GAGS Internationalisation (i18n) Module — v1.0
================================================
Hausa · Yoruba · Igbo · English

Full UI translation for all four Nigerian languages used in GAGS.
Covers all labels, tooltips, metric names, alerts, and recommendations
that appear in the three simulation pages.

Usage
-----
    from components.i18n import t, language_switcher, get_lang

    # In the sidebar (before any other controls):
    language_switcher()

    # Anywhere a string appears:
    st.markdown(f"## {t('dashboard_title')}")
    st.metric(t('metric_accuracy'), f"{avg_acc:.1%}")

Architecture
------------
All strings are stored in TRANSLATIONS dict keyed by language code:
  en  — English (default)
  ha  — Hausa
  yo  — Yoruba
  ig  — Igbo

The active language is stored in st.session_state["_gags_lang"].
t(key) always returns a string — falls back to English if key missing
in non-English locale, then to the key itself as last resort.
"""

from typing import Dict, Optional
import streamlit as st

# ── Language metadata ─────────────────────────────────────────────────────────

LANGUAGES: Dict[str, Dict[str, str]] = {
    "en": {"name": "English",  "native": "English",   "flag": "🇬🇧", "dir": "ltr"},
    "ha": {"name": "Hausa",    "native": "Hausa",      "flag": "🇳🇬", "dir": "ltr"},
    "yo": {"name": "Yoruba",   "native": "Yorùbá",     "flag": "🇳🇬", "dir": "ltr"},
    "ig": {"name": "Igbo",     "native": "Igbo",       "flag": "🇳🇬", "dir": "ltr"},
}

# ── Translation table ─────────────────────────────────────────────────────────
# Format: TRANSLATIONS[key][lang_code] = translated string
# English is canonical; other languages fall back to English if key missing.

TRANSLATIONS: Dict[str, Dict[str, str]] = {

    # ── Navigation & general ──────────────────────────────────────────────
    "app_name": {
        "en": "GAGS Resilience Framework",
        "ha": "Tsarin GAGS na Juriya",
        "yo": "Eto GAGS ti Alafia",
        "ig": "Usoro GAGS nke Ike",
    },
    "welcome": {
        "en": "Welcome",
        "ha": "Maraba",
        "yo": "Ẹ káàbọ̀",
        "ig": "Nnọọ",
    },
    "run_simulation": {
        "en": "Run Simulation",
        "ha": "Gudanar da Kwaikwayo",
        "yo": "Ṣe Àfarawé",
        "ig": "Bido Nnwale",
    },
    "reset": {
        "en": "Reset",
        "ha": "Sake Saita",
        "yo": "Tún ṣe",
        "ig": "Tọgharia",
    },
    "loading": {
        "en": "Loading…",
        "ha": "Ana ɗaukar…",
        "yo": "Ń gbéru…",
        "ig": "Na-ebuga…",
    },
    "complete": {
        "en": "Complete ✓",
        "ha": "An gama ✓",
        "yo": "Parí ✓",
        "ig": "Mechara ✓",
    },
    "download_csv": {
        "en": "Download Results (CSV)",
        "ha": "Saukar da Sakamakon (CSV)",
        "yo": "Gba àwọn ìyọrísí (CSV)",
        "ig": "Budata Nsonaazụ (CSV)",
    },
    "export_config": {
        "en": "Export Configuration (JSON)",
        "ha": "Fitar da Tsarin (JSON)",
        "yo": "Ṣe ìpínlẹ̀ (JSON)",
        "ig": "Ñefu Nhazi (JSON)",
    },
    "settings": {
        "en": "Configuration",
        "ha": "Saituna",
        "yo": "Ìtò",
        "ig": "Nhazi",
    },
    "language": {
        "en": "Language",
        "ha": "Harshe",
        "yo": "Èdè",
        "ig": "Asụsụ",
    },
    "quick_start": {
        "en": "Quick-start presets",
        "ha": "Saitunan Farawa Cikin Sauri",
        "yo": "Àwọn ìmúrasílẹ̀ ìbẹ̀rẹ̀ iyara",
        "ig": "Usoro ịmalite ngwa ngwa",
    },

    # ── Dashboard titles ──────────────────────────────────────────────────
    "dashboard_title_health": {
        "en": "Healthcare Equity Dashboard",
        "ha": "Allon Daidaito na Kiwon Lafiya",
        "yo": "Pàtàkì Ìlera Dáadáa",
        "ig": "Dị Oke Mkpa nke Ahụike",
    },
    "dashboard_title_security": {
        "en": "Security Assessment Dashboard",
        "ha": "Allon Kimantawa na Tsaro",
        "yo": "Ìṣirò Ààbò",
        "ig": "Ọnụnọ Nlebara Anya nke Nchekwa",
    },
    "dashboard_title_agrotech": {
        "en": "Agrotech Equity Dashboard",
        "ha": "Allon Daidaito na Aikin Gona",
        "yo": "Pàtàkì Ìmọ̀-Àgbẹ̀ Dáadáa",
        "ig": "Dị Oke Mkpa nke Ugbo-ọrụ",
    },

    # ── Sidebar sections ──────────────────────────────────────────────────
    "data_source": {
        "en": "Data Source",
        "ha": "Tushen Bayanan",
        "yo": "Orísun Dátà",
        "ig": "Isi Ihe Ọmụmụ",
    },
    "bias_config": {
        "en": "Bias Configuration",
        "ha": "Tsarin Nuna Bambanci",
        "yo": "Ìtò Ìlòdì",
        "ig": "Nhazi Mmegharị",
    },
    "bias_types": {
        "en": "Bias Types",
        "ha": "Nau'in Nuna Bambanci",
        "yo": "Irú Ìlòdì",
        "ig": "Ụdị Mmegharị",
    },
    "bias_intensity": {
        "en": "Bias Intensity",
        "ha": "Ƙarfin Nuna Bambanci",
        "yo": "Agbára Ìlòdì",
        "ig": "Ike Mmegharị",
    },
    "attack_config": {
        "en": "Adversarial Attacks",
        "ha": "Hare-haren Makiya",
        "yo": "Ìkọlù Àwọn Ọ̀tá",
        "ig": "Mwakpo Ndị Iro",
    },
    "poison_rate": {
        "en": "Poisoning Rate",
        "ha": "Ƙimar Gurɓataccen Bayanan",
        "yo": "Ìwọn Majèlé",
        "ig": "Ọnụ Ọgụgụ Ihe Ọjọọ",
    },
    "simulation_params": {
        "en": "Simulation Parameters",
        "ha": "Sigogin Kwaikwayo",
        "yo": "Àwọn Ìpíndọ̀ Àfarawé",
        "ig": "Ihe Nchọpụta Nnwale",
    },
    "sample_size": {
        "en": "Sample Size",
        "ha": "Girman Samfuri",
        "yo": "Ìwọn Àpẹẹrẹ",
        "ig": "Ogo Nlere",
    },
    "num_runs": {
        "en": "Simulation Runs",
        "ha": "Gudu na Kwaikwayo",
        "yo": "Ìsẹ̀jẹ̀ Àfarawé",
        "ig": "Ọnụ Ọgụgụ Nnwale",
    },
    "feature_modules": {
        "en": "Feature Modules",
        "ha": "Sassan Abubuwa",
        "yo": "Àwọn Ẹ̀ka Ìpìlẹ̀",
        "ig": "Ụzọ Ihe Ọmụmụ",
    },
    "include_baseline": {
        "en": "Include Baseline (no bias/attack)",
        "ha": "Haɗa Layin Tushe (ba bambanci/hare-hare)",
        "yo": "Fi Ìpìlẹ̀ sí (láìsí ìlòdì/ìkọlù)",
        "ig": "Tinye Nnọchite Anya (enweghị mmegharị/mwakpo)",
    },
    "view_as": {
        "en": "View as",
        "ha": "Duba a matsayin",
        "yo": "Wo gẹ́gẹ́ bí",
        "ig": "Hụ dị ka",
    },

    # ── Core metrics ──────────────────────────────────────────────────────
    "metric_accuracy": {
        "en": "Prediction Accuracy",
        "ha": "Daidaiton Hasashe",
        "yo": "Ìtọ́sẹ̀ Àsọtẹ́lẹ̀",
        "ig": "Ụzọ Amụma",
    },
    "metric_fairness": {
        "en": "Fairness Score",
        "ha": "Maki na Adalci",
        "yo": "Ìdíyelé Ìdájọ́",
        "ig": "Ogo Ezi Omume",
    },
    "metric_equity": {
        "en": "Algorithmic Equity",
        "ha": "Daidaito na Algorithm",
        "yo": "Ìdọ́gba Algorithm",
        "ig": "Ịha Anya Algorithm",
    },
    "metric_sensitivity": {
        "en": "Sensitivity (Recall)",
        "ha": "Hankali (Tunawa)",
        "yo": "Ìfaragbára (Rírántí)",
        "ig": "Nwee Mmetụta (Ncheta)",
    },
    "metric_specificity": {
        "en": "Specificity",
        "ha": "Takaici",
        "yo": "Pàtó",
        "ig": "Nkenke",
    },
    "metric_fpr": {
        "en": "False Positive Rate",
        "ha": "Ƙimar Ƙaramar Tabbatarwa",
        "yo": "Ìwọn Ìjẹ́rìí Àṣìṣe",
        "ig": "Ọnụ Ọgụgụ Ihe Ọ Bụghị Nke Ọma",
    },
    "metric_fnr": {
        "en": "False Negative Rate",
        "ha": "Ƙimar Ƙaramar Ƙin Gane",
        "yo": "Ìwọn Ìparẹ́ Àṣìṣe",
        "ig": "Ọnụ Ọgụgụ Ihe Ọ Bụghị Nke Njọ",
    },
    "metric_parity": {
        "en": "Demographic Parity Gap",
        "ha": "Giɓin Daidaito na Jama'a",
        "yo": "Àáǹtítọ́ Ìdọ́gba Àwọn Ènìyàn",
        "ig": "Ọdịiche Ịhakwa Ndị Mmadụ",
    },
    "metric_liberty": {
        "en": "Liberty Score",
        "ha": "Maki na Yanci",
        "yo": "Ìdíyelé Òmìnira",
        "ig": "Ogo Nnwere Onwe",
    },
    "metric_detection": {
        "en": "Threat Detection Rate",
        "ha": "Ƙimar Gano Barazana",
        "yo": "Ìwọn Ìṣèwádìí Ewu",
        "ig": "Ọnụ Ọgụgụ Ịchọpụta Ihe Egwu",
    },
    "metric_gender_gap": {
        "en": "Gender Gap",
        "ha": "Bambancin Jinsi",
        "yo": "Iyàtọ̀ Ìbálò",
        "ig": "Ọdịiche Mmadụ nke Nwoke na Nwanyị",
    },
    "metric_digital_inclusion": {
        "en": "Digital Inclusion Score",
        "ha": "Maki na Shigar da Dijital",
        "yo": "Ìdíyelé Ìfowọpọ Ẹ̀rọ",
        "ig": "Ogo Itinye na Ihe Dijital",
    },
    "metric_permeability": {
        "en": "Permeability Score",
        "ha": "Maki na Shiga-shiga",
        "yo": "Ìdíyelé Ìbọ̀ wọlé",
        "ig": "Ogo Mgbanwe",
    },

    # ── Alerts and verdicts ───────────────────────────────────────────────
    "alert_high_fpr": {
        "en": "⚠️ High false alarm rate detected. Immediate action required.",
        "ha": "⚠️ An gano yawan alarar ƙarya. Ana buƙatar aiki nan da nan.",
        "yo": "⚠️ Ìwọn ìkìlọ̀ àṣìṣe gíga ni a rí. Ìgbésẹ̀ lẹ́sẹ̀kẹsẹ̀ ni a nílò.",
        "ig": "⚠️ Achọpụtara ọnụ ọgụgụ ọchịchọ nke ụgha dị elu. Achọrọ mmemme ozugbo.",
    },
    "alert_low_fairness": {
        "en": "⚠️ Fairness score below acceptable threshold. Bias mitigation required.",
        "ha": "⚠️ Makin adalci ya faɗi ƙasa da iyakar da ake yarda da ita. Ana buƙatar rage nuna bambanci.",
        "yo": "⚠️ Ìdíyelé ìdájọ́ wà ní ìsàlẹ̀ ìwọ̀n tí a gbà. A nílò dídin ìlòdì.",
        "ig": "⚠️ Ogo ezi omume dị n'okpuru ókè ọ nabatara. Achọrọ ibelata mmegharị.",
    },
    "alert_good_balance": {
        "en": "✅ Good balance achieved. Continue monitoring.",
        "ha": "✅ An sami daidaito mai kyau. Ci gaba da sa ido.",
        "yo": "✅ Àáǹtítọ́ dáadáa ni a wà. Jẹ́ kí a máa ṣàmójútó.",
        "ig": "✅ Enwetara nhazi ọma. Gaa n'ihu na-eleba anya.",
    },
    "compliant": {
        "en": "COMPLIANT ✅",
        "ha": "NA DAIDAITO ✅",
        "yo": "ÌBÁRADÉ ✅",
        "ig": "DABARA ✅",
    },
    "non_compliant": {
        "en": "NON-COMPLIANT ❌",
        "ha": "BA NA DAIDAITO ❌",
        "yo": "KO ÌBÁRADÉ ❌",
        "ig": "Ọ DABARAGHỊ ❌",
    },

    # ── Healthcare-specific ───────────────────────────────────────────────
    "healthcare_setting": {
        "en": "Healthcare Setting",
        "ha": "Yanayin Kiwon Lafiya",
        "yo": "Àyíká Ìlera",
        "ig": "Ọnọdụ Ahụike",
    },
    "prediction_task": {
        "en": "Prediction Task",
        "ha": "Ayyukan Hasashe",
        "yo": "Iṣẹ́ Àsọtẹ́lẹ̀",
        "ig": "Ọrụ Amụma",
    },
    "patient_demographics": {
        "en": "Patient Demographics",
        "ha": "Bayanin Marasa Lafiya",
        "yo": "Alaye Àwọn Aláìsàn",
        "ig": "Ọnọdụ Ndị Ọrịa",
    },
    "access_inequality": {
        "en": "Access Inequality",
        "ha": "Rashin Daidaito na Samun Kiwon Lafiya",
        "yo": "Àìdọ́gba Ìráàyèsí",
        "ig": "Enweghị Ịhakwa n'Ịnweta",
    },
    "sensitivity": {
        "en": "Sensitivity",
        "ha": "Hankali",
        "yo": "Ìfaragbára",
        "ig": "Nwee Mmetụta",
    },
    "specificity": {
        "en": "Specificity",
        "ha": "Takaici",
        "yo": "Pàtó",
        "ig": "Nkenke",
    },
    "clinical_impact": {
        "en": "Clinical Impact",
        "ha": "Tasirin Asibiti",
        "yo": "Ipa Ìlera",
        "ig": "Mmetụta Ọgwụ",
    },

    # ── Agrotech-specific ─────────────────────────────────────────────────
    "crop_scenario": {
        "en": "Crop Scenario",
        "ha": "Yanayin Amfanin Gona",
        "yo": "Ìṣẹlẹ̀ Iresi/Àgbàdo",
        "ig": "Ọnọdụ Ihe Ọkụkụ",
    },
    "crop_failure_risk": {
        "en": "Crop Failure Risk",
        "ha": "Haɗarin Rashin Amfanin Gona",
        "yo": "Ewu Ìkùnà Irúgbìn",
        "ig": "Ihe Egwu Ọkụkụ ọ Daa",
    },
    "market_access": {
        "en": "Market Access",
        "ha": "Samun Kasuwa",
        "yo": "Ìráàyèsí Ọjà",
        "ig": "Ịnweta Ahịa",
    },
    "fertilizer_allocation": {
        "en": "Fertilizer Allocation",
        "ha": "Rabawa na Taki",
        "yo": "Ìpínrọ̀ Ìdásílẹ̀ Ilẹ̀",
        "ig": "Ịkewa Nri Ala",
    },
    "climate_risk": {
        "en": "Climate Risk Assessment",
        "ha": "Kimantawa Haɗarin Yanayi",
        "yo": "Ìṣirò Ewu Ojú Ọjọ́",
        "ig": "Nyocha Ihe Egwu Ihu Ọnọdụ",
    },
    "female_farmers": {
        "en": "Female Farmers",
        "ha": "Manoman Mata",
        "yo": "Àwọn Àgbẹ̀ Obìnrin",
        "ig": "Ndị Ọrụ Ubi Ụmụ Nwanyị",
    },
    "gender_equity_audit": {
        "en": "Gender Equity Audit",
        "ha": "Duba Daidaito na Jinsi",
        "yo": "Ìṣirò Ìdọ́gba Ìbálò",
        "ig": "Nyocha Ịhakwa nke Ụdị Mmadụ",
    },
    "agent_economy": {
        "en": "Agent Economy",
        "ha": "Tattalin Arzikin Wakili",
        "yo": "Ọrọ̀-Ajé Olùṣàkóso",
        "ig": "Akụnụba Onye Ọrụ",
    },
    "irrigation_access": {
        "en": "Irrigation Access",
        "ha": "Samun Ban Ruwa",
        "yo": "Ìráàyèsí Irigésòn",
        "ig": "Ịnweta Ịgba Ihe Mmiri",
    },

    # ── Security-specific ─────────────────────────────────────────────────
    "threat_scenario": {
        "en": "Threat Scenario",
        "ha": "Yanayin Barazana",
        "yo": "Ìṣẹlẹ̀ Ewu",
        "ig": "Ọnọdụ Ihe Egwu",
    },
    "surveillance_intensity": {
        "en": "Surveillance Intensity",
        "ha": "Ƙarfin Sa Ido",
        "yo": "Agbára Ìwòran",
        "ig": "Ike Nchekwa Anya",
    },
    "oversight_mechanism": {
        "en": "Oversight Mechanism",
        "ha": "Hanyar Kula da Aiki",
        "yo": "Ọ̀nà Àbójútó",
        "ig": "Usoro Nlekọta",
    },
    "data_retention": {
        "en": "Data Retention (days)",
        "ha": "Adana Bayanan (kwanaki)",
        "yo": "Ìpamọ́ Dátà (ọjọ́)",
        "ig": "Ọdọ Nchekwa Ihe Ọmụmụ (ụbọchị)",
    },
    "threat_level": {
        "en": "Perceived Threat Level",
        "ha": "Mataki na Barazana da Ake Ji",
        "yo": "Ìwọ̀n Ewu Tí a Fojú Inú Rí",
        "ig": "Ọkwa Ihe Egwu Echere",
    },

    # ── Regulatory ────────────────────────────────────────────────────────
    "compliance_report": {
        "en": "Regulatory Compliance Report",
        "ha": "Rahoto na Bin Doka",
        "yo": "Ìjábọ̀ Ìgbọràn Òfin",
        "ig": "Akụkọ Ịdabara Iwu",
    },
    "nigeria_regulatory": {
        "en": "Nigeria Regulatory Frameworks",
        "ha": "Dokokin Ƙasa na Najeriya",
        "yo": "Àwọn Ìlànà Ìṣèlú Nàìjíríà",
        "ig": "Usoro Iwu Nigeria",
    },
    "nitda_policy": {
        "en": "NITDA AI Policy",
        "ha": "Manufar NITDA AI",
        "yo": "Ìlànà AI ti NITDA",
        "ig": "Iwu AI nke NITDA",
    },
    "ndpr": {
        "en": "NDPR Data Protection",
        "ha": "Kariyar Bayanai NDPR",
        "yo": "Ààbò Dátà NDPR",
        "ig": "Nchekwa Ihe Ọmụmụ NDPR",
    },

    # ── XAI ───────────────────────────────────────────────────────────────
    "xai_title": {
        "en": "Explainable AI",
        "ha": "AI da Ake Iya Bayyanawa",
        "yo": "AI Tí a Lè Ṣàlàyé",
        "ig": "AI Nke Enwere Ike Ikọwa",
    },
    "feature_importance": {
        "en": "Feature Importance",
        "ha": "Muhimmancin Fasali",
        "yo": "Pàtàkì Àwọn Ẹ̀yà",
        "ig": "Ọkachamara Atụmatụ",
    },
    "prediction_explanation": {
        "en": "Prediction Explanation",
        "ha": "Bayani kan Hasashe",
        "yo": "Ìtumọ̀ Àsọtẹ́lẹ̀",
        "ig": "Nkọwa Amụma",
    },
    "counterfactual": {
        "en": "Counterfactual",
        "ha": "Abin Da Zai Canza Hukunci",
        "yo": "Ohun Tí Yóò Yí Ìdájọ́ Padà",
        "ig": "Ihe Ga-agbanwe Mkpebi",
    },
    "what_would_change": {
        "en": "What would change the prediction?",
        "ha": "Mene ne zai canza hasashen?",
        "yo": "Kí ni yóò yí àsọtẹ́lẹ̀ padà?",
        "ig": "Gịnị ga-agbanwe amụma?",
    },

    # ── Recommendations ───────────────────────────────────────────────────
    "recommendations": {
        "en": "Recommendations",
        "ha": "Shawarwari",
        "yo": "Àwọn Ìmọ̀ràn",
        "ig": "Ndụmọdụ",
    },
    "rec_bias_mitigation": {
        "en": "Apply bias mitigation before deployment.",
        "ha": "Yi amfani da rage nuna bambanci kafin tura shi.",
        "yo": "Lo ìdínkù ìlòdì ṣájú fífiná rẹ̀ ṣiṣẹ́.",
        "ig": "Tinye ibelata mmegharị tupu ịtọ ya n'ọrụ.",
    },
    "rec_gender_audit": {
        "en": "Conduct gender equity audit before deployment.",
        "ha": "Yi duba daidaiton jinsi kafin tura shi.",
        "yo": "Ṣe ìṣirò ìdọ́gba ìbálò ṣájú fífiná rẹ̀ ṣiṣẹ́.",
        "ig": "Mee nyocha ịhakwa ụdị mmadụ tupu ịtọ ya n'ọrụ.",
    },
    "rec_ussd_fallback": {
        "en": "Deploy USSD/SMS fallback for low-connectivity users.",
        "ha": "Samar da madadin USSD/SMS ga masu iyakacin hanyar sadarwa.",
        "yo": "Fi USSD/SMS náà sí fún àwọn tí kò ní àǹfààní ìnáájọ.",
        "ig": "Tọọ USSD/SMS maka ndị enweghị njikọ.",
    },
    "rec_community_validation": {
        "en": "Validate with community members before launch.",
        "ha": "Yi amincewa tare da membobin al'umma kafin ƙaddamarwa.",
        "yo": "Jẹ́rìísí pẹ̀lú àwọn ará àdúgbò ṣájú ìfilọ́lẹ̀.",
        "ig": "Kwenye na ndị obodo tupu ịmalite.",
    },

    # ── USSD Simulator ────────────────────────────────────────────────────
    "ussd_title": {
        "en": "USSD/SMS Simulation Mode",
        "ha": "Yanayin Kwaikwayar USSD/SMS",
        "yo": "Ipo Àfarawé USSD/SMS",
        "ig": "Ọnọdụ Nnwale USSD/SMS",
    },
    "ussd_description": {
        "en": "Simulate how this AI system would work for farmers using basic phones with no internet.",
        "ha": "Kwaikwayon yadda wannan tsarin AI zai yi aiki ga manoma masu amfani da wayoyin yau da kullun ba tare da intanet ba.",
        "yo": "Ṣe àfarawé bí ètò AI yìí ṣe máa ṣiṣẹ́ fún àwọn àgbẹ̀ tó ń lò fóònù àtijọ́ láìsí íntánẹ́ẹ̀tì.",
        "ig": "Mee nnwale otu usoro AI a ga-arụ ọrụ maka ndị ọrụ ubi na-eji ekwentị dị mfe enweghị ịntanetị.",
    },
    "ussd_enter_number": {
        "en": "Enter your USSD code (e.g. *384*1#)",
        "ha": "Shigar da lambar USSD ɗinka (misali *384*1#)",
        "yo": "Tẹ kóòdù USSD rẹ (ìpẹẹrẹ *384*1#)",
        "ig": "Tinye koodu USSD gị (dịka *384*1#)",
    },
    "ussd_menu_title": {
        "en": "GAGS AI Advisory Service",
        "ha": "Sabis na Shawarwari na GAGS AI",
        "yo": "Iṣẹ́ Ìmọ̀ràn GAGS AI",
        "ig": "Ọrụ Ndụmọdụ GAGS AI",
    },
    "ussd_select_option": {
        "en": "Select option:",
        "ha": "Zaɓi zaɓi:",
        "yo": "Yan àṣàyàn:",
        "ig": "Họrọ nhọrọ:",
    },
    "ussd_result_high_risk": {
        "en": "Result: HIGH RISK. Contact extension officer.",
        "ha": "Sakamakon: HAƊARI BABBA. Tuntubi jami'in ƙasar gona.",
        "yo": "Ìyọrísí: EWUU GÍGA. Kan sí olùkọ́ni àgbẹ̀.",
        "ig": "Nsonaazụ: IHE EGWU DỊ ELU. Kpọtụrụ onye ndụmọdụ ugbo.",
    },
    "ussd_result_low_risk": {
        "en": "Result: LOW RISK. Continue current practices.",
        "ha": "Sakamakon: HAƊARI ƘARAMI. Ci gaba da ayyukan yanzu.",
        "yo": "Ìyọrísí: EWUU KẸ́KẸ́. Tẹ̀síwájú àwọn àṣà lọ́wọ́lọ́wọ́.",
        "ig": "Nsonaazụ: IHE EGWU DỊ NTAKỊRỊ. Gaa n'ihu n'omume ugbu a.",
    },

    # ── Tabs ──────────────────────────────────────────────────────────────
    "tab_performance": {
        "en": "📈 Performance",
        "ha": "📈 Aiki",
        "yo": "📈 Iṣẹ́",
        "ig": "📈 Ọrụ",
    },
    "tab_equity": {
        "en": "⚖️ Equity",
        "ha": "⚖️ Daidaito",
        "yo": "⚖️ Ìdọ́gba",
        "ig": "⚖️ Ịhakwa",
    },
    "tab_compliance": {
        "en": "📋 Compliance",
        "ha": "📋 Bi da Doka",
        "yo": "📋 Ìgbọràn Òfin",
        "ig": "📋 Ịdabara Iwu",
    },
    "tab_nigeria_reg": {
        "en": "🇳🇬 Nigeria Regulatory",
        "ha": "🇳🇬 Dokokin Najeriya",
        "yo": "🇳🇬 Òfin Nàìjíríà",
        "ig": "🇳🇬 Iwu Nigeria",
    },
    "tab_xai": {
        "en": "🧠 Explainable AI",
        "ha": "🧠 AI da Bayani",
        "yo": "🧠 AI Ìmọ̀ràn",
        "ig": "🧠 AI Nkọwa",
    },
    "tab_longitudinal": {
        "en": "🔁 Longitudinal",
        "ha": "🔁 Tsawo Lokaci",
        "yo": "🔁 Àkókò gígùn",
        "ig": "🔁 Ogologo Oge",
    },
    "tab_federated": {
        "en": "🌐 Federated",
        "ha": "🌐 Haɗin Kai",
        "yo": "🌐 Ìsopọ̀",
        "ig": "🌐 Njikọ",
    },
    "tab_raw_results": {
        "en": "📋 Raw Results",
        "ha": "📋 Sakamakon Asali",
        "yo": "📋 Àwọn Ìyọrísí Àkọ́kọ́",
        "ig": "📋 Nsonaazụ Ọzọ",
    },

    # ── Sidebar section headers ───────────────────────────────────────────
    "data_source_header":  {"en":"🏥 Data Source","ha":"🏥 Tushen Bayanan","yo":"🏥 Orísun Dátà","ig":"🏥 Isi Ihe Ọmụmụ"},
    "demographic_header":  {"en":"👥 Patient Demographics","ha":"👥 Halittar Masu Ciwon","yo":"👥 Àwọn Aláìsàn","ig":"👥 Ndị Ọrịa"},
    "bias_config_header":  {"en":"🎭 Bias Configuration","ha":"🎭 Tsarin Nuna Bambanci","yo":"🎭 Ìtò Ìlòdì","ig":"🎭 Nhazi Mmegharị"},
    "attack_header":       {"en":"⚠️ Adversarial Attacks","ha":"⚠️ Hare-hare","yo":"⚠️ Àwọn Ìkọlù","ig":"⚠️ Mwakpo"},
    "sim_params_header":   {"en":"📊 Simulation Parameters","ha":"📊 Sigogi na Gwaji","yo":"📊 Àwọn Ẹ̀ka Àfarawé","ig":"📊 Ihe Nlele"},
    "modules_header":      {"en":"🔬 Feature Modules","ha":"🔬 Ƙarin Fasalulluka","yo":"🔬 Àwọn Ẹ̀ka Iṣẹ́","ig":"🔬 Ụdị Njirimara"},
    "view_mode_header":    {"en":"📊 View Mode","ha":"📊 Yanayin Kallewa","yo":"📊 Ọ̀nà Ìwò","ig":"📊 Ụzọ Nlele"},
    # ── Widget labels ──────────────────────────────────────────────────────
    "perspective":         {"en":"Perspective","ha":"Hangen Nesa","yo":"Ìríran","ig":"Nlele"},
    "industry_view":       {"en":"Industry","ha":"Masana'antu","yo":"Iléeṣẹ́","ig":"Ọrụ Ọchịchọ"},
    "research_view":       {"en":"Research","ha":"Bincike","yo":"Ìwádìí","ig":"Nyocha"},
    "bias_intensity_lbl":  {"en":"Bias Intensity","ha":"Ƙarfin Bambanci","yo":"Agbára Ìlòdì","ig":"Ike Mmegharị"},
    "poisoning_rate_lbl":  {"en":"Poisoning Rate","ha":"Adadin Guba","yo":"Ìwọ̀n Olóró","ig":"Ọnụ Ọgụgụ Ọjọọ"},
    "sample_size_lbl":     {"en":"Sample Size","ha":"Girman Samfuri","yo":"Ìwọ̀n Àyẹ̀wò","ig":"Ogo Nlele"},
    "sim_runs_lbl":        {"en":"Simulation Runs","ha":"Adadin Gwaje-gwaje","yo":"Iye Àfarawé","ig":"Ọnụ Ọgụgụ Nnwale"},
    "enable_xai_lbl":      {"en":"Explainable AI","ha":"Bayyana AI","yo":"Ìmọ̀ AI","ig":"AI Nkọwa"},
    "enable_gov_lbl":      {"en":"Governance Layer","ha":"Matakin Gudanarwa","yo":"Ìpele Ìṣàkóso","ig":"Ọkwa Ọchịchị"},
    "run_btn":             {"en":"▶ Run","ha":"▶ Gudanar","yo":"▶ Ṣe","ig":"▶ Bido"},
    "reset_btn":           {"en":"🔄 Reset","ha":"🔄 Sake Saita","yo":"🔄 Tún ṣe","ig":"🔄 Tọgharia"},
    # ── Tab labels ─────────────────────────────────────────────────────────
    "tab_performance":     {"en":"📈 Performance","ha":"📈 Aiki","yo":"📈 Ìṣe","ig":"📈 Ọrụ"},
    "tab_equity":          {"en":"⚖️ Equity Gaps","ha":"⚖️ Giɓin Daidaito","yo":"⚖️ Àwọn Ààlà","ig":"⚖️ Ọdịiche Nhata"},
    "tab_xai":             {"en":"🧠 Explainable AI","ha":"🧠 Bayyana AI","yo":"🧠 Ìtumọ̀ AI","ig":"🧠 AI Nkọwa"},
    "tab_compliance":      {"en":"📋 Compliance","ha":"📋 Bin Ka'ida","yo":"📋 Ìfaramọ́","ig":"📋 Itinye Isi"},
    "tab_longitudinal":    {"en":"🔁 Longitudinal","ha":"🔁 Ci gaba","yo":"🔁 Gígùn","ig":"🔁 Ogologo"},
    "tab_federated":       {"en":"🌐 Federated","ha":"🌐 Haɗaɗɗe","yo":"🌐 Ìṣọ̀kan","ig":"🌐 Jikọọ"},
    "tab_nigeria":         {"en":"🏛️ Nigeria Regulatory","ha":"🏛️ Dokokin Najeriya","yo":"🏛️ Òfin Nàìjíríà","ig":"🏛️ Iwu Naịjirịa"},
    "tab_raw":             {"en":"📋 Raw Results","ha":"📋 Sakamakon Gwaji","yo":"📋 Àwọn Ìyọrísí","ig":"📋 Nsonaazụ"},
    # ── KPI labels ─────────────────────────────────────────────────────────
    "kpi_accuracy":        {"en":"Overall Accuracy","ha":"Daidaiton Gabaɗaya","yo":"Pípéye Gbogbo","ig":"Izi Ọnụ Ọgụgụ"},
    "kpi_fairness":        {"en":"Fairness Score","ha":"Maki Adalci","yo":"Ìkún Ìdọ́gba","ig":"Akara Izi Ọcha"},
    "kpi_inclusion":       {"en":"Inclusion Score","ha":"Maki Haɗawa","yo":"Ìkún Ìfọwọ́sí","ig":"Akara Itinye Onye"},
    "kpi_detection":       {"en":"Detection Rate","ha":"Adadin Gano","yo":"Ìwọ̀n Ìṣàwárí","ig":"Ọnụ Ọgụgụ Ịchọpụta"},
    "kpi_liberty":         {"en":"Liberty Score","ha":"Maki Yanci","yo":"Ìkún Òmìnira","ig":"Akara Onwe"},
    "kpi_fpr":             {"en":"False Positive Rate","ha":"Adadin Kuskure","yo":"Ìwọ̀n Àṣìṣe","ig":"Ọnụ Ọgụgụ Ọjọọ"},
    # ── Alerts and messages ────────────────────────────────────────────────
    "run_to_see_results":  {"en":"Configure settings and click Run to begin.","ha":"Saita sigogi kuma danna Run don fara.","yo":"Ṣètò àwọn ìpèsè kí o sì tẹ Ṣe.","ig":"Hazie ntọala wee pịa Bido iji malite."},
    "no_data_yet":         {"en":"No simulation results yet.","ha":"Babu sakamakon gwaji tukuna.","yo":"Kò sí àwọn ìyọrísí títí di isisiyi.","ig":"Enweghị nsonaazụ ka ugbu a."},
    "xai_enable_prompt":   {"en":"Enable XAI in sidebar and run simulation.","ha":"Kunna XAI kuma gudanar da gwajin.","yo":"Mú XAI ṣiṣẹ́ kí o sì ṣe àfarawé.","ig":"Gbaa XAI wee mee nnwale."},
    "download_results":    {"en":"📥 Download CSV","ha":"📥 Saukar da CSV","yo":"📥 Gba àbájáde","ig":"📥 Budata CSV"},
    "export_config_btn":   {"en":"📋 Export Config","ha":"📋 Fitar da Tsarin","yo":"📋 Ṣe ìpínlẹ̀","ig":"📋 Ñefu Nhazi"},
    "pdf_report_btn":      {"en":"📄 Download PDF Report","ha":"📄 Saukar da Rahoton PDF","yo":"📄 Gba Ìròyìn PDF","ig":"📄 Budata Akwụkwọ PDF"},
    "nigeria_context":     {"en":"Nigeria Regulatory Context","ha":"Yanayin Doka na Najeriya","yo":"Ìsọmọbí Òfin Nàìjíríà","ig":"Ọnọdụ Iwu Naịjirịa"},
    "benchmark_compare":   {"en":"Compare against published study:","ha":"Kwatanta da binciken da aka buga:","yo":"Fiwéra pẹ̀lú ìwádìí tó ti jáde:","ig":"Tụnyere ọmụmụ e si n'ụlọ nkwupụta:"},
    "real_world_bm":       {"en":"📚 Real-World Benchmark Comparison","ha":"📚 Kwatancen Gwajin Ainihi","yo":"📚 Ìfiwéra Ìṣeéṣe Aṣà","ig":"📚 Ntụnyere Nnwale Ụwa Nke Ọ Bụ"},


    # ── Economic Justice specific ─────────────────────────────────────────
    "bias_header":           {"en":"🎭 Bias Configuration","ha":"🎭 Tsarin Nuna Bambanci","yo":"🎭 Ìtò Ìlòdì","ig":"🎭 Nhazi Mmegharị"},
    "scenario_header":       {"en":"💼 Economic Scenario","ha":"💼 Yanayin Tattalin Arziki","yo":"💼 Ipò Ọrọ-ajé","ig":"💼 Ọnọdụ Akụ na Ụba"},
    "sim_header":            {"en":"📊 Simulation","ha":"📊 Gwaji","yo":"📊 Àfarawé","ig":"📊 Nnwale"},
    "view_mode_lbl":         {"en":"Perspective","ha":"Hangen Nesa","yo":"Ìríran","ig":"Nlele"},
    "initialising":          {"en":"Initialising…","ha":"Ana saita…","yo":"Ń bẹ̀rẹ̀…","ig":"Na amalite…"},
    # ── Tab labels ─────────────────────────────────────────────────────────
    "tab_overview":          {"en":"📊 Overview","ha":"📊 Taƙaitawa","yo":"📊 Àkópọ̀","ig":"📊 Nchoputa"},
    "tab_fairness":          {"en":"⚖️ Fairness Deep-Dive","ha":"⚖️ Binciken Adalci","yo":"⚖️ Ìwádìí Ìdọ́gba","ig":"⚖️ Nyocha Izi Ọcha"},
    "tab_benchmarks":        {"en":"📚 Real-World Benchmarks","ha":"📚 Gwajin Ainihi","yo":"📚 Ìwọ̀n Àgbáyé","ig":"📚 Ụkpụrụ Ụwa"},
    "tab_impact":            {"en":"🌍 Economic Impact","ha":"🌍 Tasirin Tattalin Arziki","yo":"🌍 Ipa Ọrọ-ajé","ig":"🌍 Mmetụta Akụ"},
    "tab_export":            {"en":"📤 Export","ha":"📤 Fitar da","yo":"📤 Ìpínlẹ̀","ig":"📤 Ñefu"},
    "tab_stats":             {"en":"📈 Statistical Distribution","ha":"📈 Rarrabar Kididdiga","yo":"📈 Ìpínyà Ìṣirò","ig":"📈 Nkesa Ọnụ Ọgụgụ"},
    # ── Chart titles ────────────────────────────────────────────────────────
    "chart_animated":        {"en":"Animated: Bias Drift Across Retraining Cycles","ha":"Hoton: Bambanci a cikin Zagayowar Horarwa","yo":"Fídíò: Ìyípadà Ìlòdì nínú Àwọn Yíká Ìkọ́","ig":"Foto: Mmegharị n'oge Nkuzi"},
    "chart_change_needed":   {"en":"Changes Needed for a Positive Outcome","ha":"Canje-canje da ake bukata","yo":"Àwọn Ìyípadà Tó Nílò","ig":"Mgbanwe Dị Mkpa"},
    "chart_feat_importance":  {"en":"Feature Importance (What Drives Decisions?)","ha":"Mahimmancin Abubuwan (Menene ke Jagorantar??)","yo":"Pàtàkì Àwọn Ẹ̀ka (Kí Ni Ń Darí?)","ig":"Ihe Dị Mkpa (Gịnị Na-akọwa?)"},
    "chart_fed_clients":     {"en":"Federated Learning Results by Client","ha":"Sakamakon Koyon Haɗaɗɗe Ta Abokin Cinikin","yo":"Àwọn Ìyọrísí Ẹ̀kọ́ Ìṣọ̀kan Nípasẹ̀ Alabára","ig":"Nsonaazụ Nkuzi Jikọọ Site n'aka Onye Ọrụ"},
    "chart_group_diverge":   {"en":"Group Outcome Divergence Over Time","ha":"Bambancin Ƙungiya a Cikin Lokaci","yo":"Àyàtọ̀ Ẹgbẹ́ Lára Àkókò","ig":"Ọdịiche Ụzọ Otu N'oge"},
    "chart_groups_bias":     {"en":"Fairness Metrics Across Groups","ha":"Ma'aunin Adalci a Ƙungiyoyi","yo":"Àwọn Ìwọ̀n Ìdọ́gba Láàárín Àwọn Ẹgbẹ́","ig":"Ụkpụrụ Izi Ọcha N'otu Ndi"},
    "chart_heatmap":         {"en":"Fairness Heatmap (Run × Metric)","ha":"Taswira Adalci (Gwaji × Ma'auni)","yo":"Maapu Ooru Ìdọ́gba","ig":"Ihe Ngosi Izi Ọcha"},
    "chart_individual_wf":   {"en":"Why Was This Worker Affected?","ha":"Me Ya Sa An shafi Wannan Ma'aikaci?","yo":"Kí Nìdí Tí Wọ́n Kan Òṣìṣẹ́ yìí?","ig":"Gịnị Mere Ha Ji Metụ Onye Ọrụ a?"},
    "chart_outcome_rate":    {"en":"Outcome Rates by Group","ha":"Ƙididdiga Ta Ƙungiya","yo":"Ìwọ̀n Ìyọrísí Nípasẹ̀ Ẹgbẹ́","ig":"Ọnụ Ọgụgụ Nsonaazụ N'otu Ndi"},
    "chart_radar":           {"en":"Equity Radar — All Gap Dimensions","ha":"Radar Adalci — Dukkan Girman Gibi","yo":"Rédà Ìdọ́gba — Gbogbo Ìwọ̀n","ig":"Radar Nhata — Akụkụ Nile"},
    "chart_trilemma":        {"en":"Economic Fairness Trilemma: Accuracy vs Fairness vs Inclusion","ha":"Matsalar Uku: Daidaito vs Adalci vs Haɗawa","yo":"Ìṣòro Mẹ́ta: Pípéye vs Ìdọ́gba vs Ìfọwọ́sí","ig":"Nsogbu Atọ: Izi vs Nhata vs Itinye"},
    "chart_waterfall":       {"en":"SHAP Waterfall: Feature Contributions","ha":"Ruwan Tsafta na SHAP: Gudummawar Abubuwan","yo":"SHAP Ìṣàn-omi: Àwọn Àánú Ẹ̀ka","ig":"SHAP: Onyinye Njirimara"},
    # ── Section headings ────────────────────────────────────────────────────
    "heading_benchmarks":    {"en":"📚 Real-World Benchmark Comparison","ha":"📚 Kwatancen Gwajin Ainihi","yo":"📚 Ìfiwéra Ìṣeéṣe Àgbáyé","ig":"📚 Ntụnyere Ụkpụrụ Ụwa"},
    "heading_compliance":    {"en":"📋 Compliance Report","ha":"📋 Rahoton Kamun Ka'ida","yo":"📋 Ìròyìn Ìfaramọ́","ig":"📋 Akụkọ Itinye Isi"},
    "heading_counterfactual":{"en":"What Would Change This Worker's Outcome?","ha":"Menene zai canza sakamakon wannan ma'aikaci?","yo":"Kí Ni Ìyípadà Ìyọrísí Òṣìṣẹ́ yìí?","ig":"Gịnị ga-agbanwe nsonaazụ onye ọrụ a?"},
    "heading_export":        {"en":"📤 Export Results","ha":"📤 Fitar da Sakamakon","yo":"📤 Ṣe ìpínlẹ̀ Àwọn Ìyọrísí","ig":"📤 Ñefu Nsonaazụ"},
    "heading_intersectional":{"en":"Intersectional Fairness Analysis","ha":"Nazarin Adalci Mabanbanta","yo":"Ìwádìí Ìdọ́gba Alárọ̀pọ̀","ig":"Nyocha Nhata Nkwụghachi"},
    "heading_nigeria_macro":  {"en":"🇳🇬 Nigeria Macro-Economic Context","ha":"🇳🇬 Makro-tattalin Arzikin Najeriya","yo":"🇳🇬 Àyíká Ọrọ-ajé Nàìjíríà","ig":"🇳🇬 Ihe Nọ Gburugburu Akụ na Ụba Naịjirịa"},
    "heading_stats":         {"en":"📈 Statistical Distribution Across Runs","ha":"📈 Rarrabar Kididdiga Ta Gwaje-gwaje","yo":"📈 Ìpínyà Ìṣirò Lára Àwọn Ìṣe","ig":"📈 Nkesa Ọnụ Ọgụgụ N'oge Nnwale"},
    "heading_why_worker":    {"en":"Why Was This Worker Affected?","ha":"Me Ya Sa An shafi Wannan Ma'aikaci?","yo":"Kí Nìdí Tí Wọ́n Kan Òṣìṣẹ́ yìí?","ig":"Gịnị Mere Ha Ji Metụ Onye Ọrụ a?"},
    # ── Messages ────────────────────────────────────────────────────────────
    "run_to_see_fed":        {"en":"Run a simulation to see federated results.","ha":"Gudanar da gwaji don ganin sakamakon haɗaɗɗe.","yo":"Ṣe àfarawé láti rí àwọn ìyọrísí ìṣọ̀kan.","ig":"Mee nnwale iji hụ nsonaazụ jikọọ."},
    "run_to_see_lng":        {"en":"Run a simulation to see longitudinal analysis.","ha":"Gudanar da gwaji don ganin nazarin lokaci.","yo":"Ṣe àfarawé láti rí ìwádìí gígùn.","ig":"Mee nnwale iji hụ nyocha ogologo."},
    "xai_compliance_prompt": {"en":"Enable XAI and run a simulation to see the compliance report.","ha":"Kunna XAI kuma gudanar da gwaji don ganin rahoton.","yo":"Mú XAI ṣiṣẹ́ kí o sì ṣe àfarawé.","ig":"Gbaa XAI wee mee nnwale iji hụ akụkọ itinye."},
    "download_json":         {"en":"📋 Export Config (JSON)","ha":"📋 Fitar da Tsarin (JSON)","yo":"📋 Ṣe ìpínlẹ̀ (JSON)","ig":"📋 Ñefu Nhazi (JSON)"},
    "download_pdf":          {"en":"📄 Download PDF Report","ha":"📄 Saukar da Rahoton PDF","yo":"📄 Gba Ìròyìn PDF","ig":"📄 Budata Akwụkwọ PDF"},


    # ── Health Financing & Development Economics ──────────────────────────
    "module_economic":       {"en":"💼 Economic Justice","ha":"💼 Adalcin Tattalin Arziki","yo":"💼 Ìdọ́gba Ọrọ-ajé","ig":"💼 Ikpe Akụ na Ụba"},
    "module_health":         {"en":"🏥 Health Financing & Development Economics","ha":"🏥 Kuɗin Kiwon Lafiya","yo":"🏥 Ìṣúná Ìlera","ig":"🏥 Ego Ahụike"},
    "tab_health_overview":   {"en":"📊 Overview","ha":"📊 Taƙaitawa","yo":"📊 Àkópọ̀","ig":"📊 Nchoputa"},
    "tab_health_equity":     {"en":"🏥 Health Equity","ha":"🏥 Daidaiton Kiwon Lafiya","yo":"🏥 Ìdọ́gba Ìlera","ig":"🏥 Nhata Ahụike"},
    "tab_health_financing":  {"en":"💊 Health Financing","ha":"💊 Kuɗin Lafiya","yo":"💊 Ìṣúná Ìlera","ig":"💊 Ego Ọgwụ"},
    "tab_uhc_dev":           {"en":"🌍 UHC & Dev Economics","ha":"🌍 UHC & Tattalin Arzikin Ci Gaba","yo":"🌍 UHC & Ọrọ-ajé Ìdàgbàsókè","ig":"🌍 UHC & Akụ Mmepe"},
    "tab_xai_health":        {"en":"🔍 XAI & Counterfactuals","ha":"🔍 Bayyana AI","yo":"🔍 XAI & Ìtumọ̀","ig":"🔍 XAI & Nkọwa"},
    "tab_benchmarks_health": {"en":"📚 Benchmarks","ha":"📚 Ma'auni","yo":"📚 Ìwọ̀n","ig":"📚 Ụkpụrụ"},
    "tab_longitudinal_h":    {"en":"🔁 Longitudinal","ha":"🔁 Ci gaba","yo":"🔁 Gígùn","ig":"🔁 Ogologo"},
    "tab_federated_h":       {"en":"🌐 Federated","ha":"🌐 Haɗaɗɗe","yo":"🌐 Ìṣọ̀kan","ig":"🌐 Jikọọ"},
    "tab_compliance_h":      {"en":"📋 Compliance","ha":"📋 Bin Ka'ida","yo":"📋 Ìfaramọ́","ig":"📋 Itinye Isi"},
    "tab_export_h":          {"en":"📤 Export","ha":"📤 Fitar da","yo":"📤 Ìpínlẹ̀","ig":"📤 Ñefu"},
    "tab_stats_h":           {"en":"📈 Statistical Distribution","ha":"📈 Rarrabar Kididdiga","yo":"📈 Ìpínyà Ìṣirò","ig":"📈 Nkesa Ọnụ Ọgụgụ"},
    # Health KPI labels
    "kpi_health_equity":     {"en":"Health Equity Score","ha":"Maki Daidaiton Kiwon Lafiya","yo":"Ìkún Ìdọ́gba Ìlera","ig":"Akara Nhata Ahụike"},
    "kpi_wealth_gap":        {"en":"Wealth Quintile Gap","ha":"Giɓin Kuɗin ","yo":"Ààlà Ọrọ","ig":"Ọdịiche Akụ"},
    "kpi_uhc_gap":           {"en":"UHC Coverage Gap","ha":"Giɓin Rufe UHC","yo":"Ààlà UHC","ig":"Ọdịiche UHC"},
    "kpi_cat_exp":           {"en":"Catastrophic OOP Risk","ha":"Haɗarin Kashe Kuɗi OOP","yo":"Ewu OOP","ig":"Ihe Egwu OOP"},
    "kpi_ins_denial":        {"en":"Insurance Denial Gap","ha":"Giɓin Ƙin Inshora","yo":"Ààlà Kíkọ Ìṣúná","ig":"Ọdịiche Ajụ Nchekwa"},
    "kpi_maternal_gap":      {"en":"Maternal Access Gap","ha":"Giɓin Damar Uwa","yo":"Ààlà Ìlera Ìyá","ig":"Ọdịiche Nwunye Ọrịa"},
    # Health scenario labels
    "scenario_nhia":         {"en":"NHIA Insurance Enrolment AI","ha":"Shigar da AI na NHIA","yo":"AI Ìforúkọsílẹ̀ NHIA","ig":"AI Ndebanye NHIA"},
    "scenario_oop":          {"en":"Hospital OOP Triage AI","ha":"AI Triage na Asibiti","yo":"AI Triage Àsìlù","ig":"AI Triage Ụlọ Ọgwụ"},
    "scenario_maternal":     {"en":"Maternal Health AI — Nigeria","ha":"AI Lafiyar Uwa — Nigeria","yo":"AI Ìlera Ìyá — Nigeria","ig":"AI Ahụike Nwunye — Nigeria"},
    "scenario_workforce":    {"en":"Health Workforce Allocation AI","ha":"AI Rarrabawa Masu Kiwon Lafiya","yo":"AI Ìpín Òṣìṣẹ́ Ìlera","ig":"AI Nkewa Ndị Ọrụ Ahụike"},
    "scenario_pharma":       {"en":"Pharmaceutical Access AI","ha":"AI Damar Magunguna","yo":"AI Ìráwọ Oogun","ig":"AI Nnweta Ọgwụ"},
    "scenario_devaid":       {"en":"Social Investment Targeting AI","ha":"AI Niyya na Saka Hannun Jari","yo":"AI Ìfọkànsí Ìdókówò Àwùjọ","ig":"AI Ntinye Ego Mmekọ"},
    # Messages
    "run_to_see_health":     {"en":"Configure a health scenario and click Run to begin.","ha":"Saita yanayin lafiya kuma danna Run.","yo":"Ṣètò ìpò ìlera kí o sì tẹ Ṣe.","ig":"Hazie ọnọdụ ahụike wee pịa Bido."},
    "health_equity_prompt":  {"en":"Enable XAI and run a health simulation to see equity analysis.","ha":"Kunna XAI kuma gudanar da gwajin lafiya.","yo":"Mú XAI ṣiṣẹ́ kí o sì ṣe àfarawé ìlera.","ig":"Gbaa XAI wee mee nnwale ahụike."},
    "nhia_note":             {"en":"Nigeria NHIS coverage: 4.5% — 195M+ uninsured","ha":"Rufe NHIS na Najeriya: 4.5% — 195M+ ba tare da inshora","yo":"Ìdárí NHIS Nàìjíríà: 4.5% — 195M+ láìní ìṣúná","ig":"Mkpuchi NHIS Naịjirịa: 4.5% — 195M+ na-enweghị nchekwa"},
    "uhc_target":            {"en":"SDG 3.8 UHC Target: 80/100 by 2030","ha":"SDG 3.8 Manufar UHC: 80/100 kafin 2030","yo":"SDG 3.8 Àfojúsùn UHC: 80/100 ní 2030","ig":"SDG 3.8 Ebumnuche UHC: 80/100 Ka 2030"},
    "robodebt_ref":          {"en":"Robodebt Royal Commission (2023): 32% wrongful welfare AI notices","ha":"Hukumar Robodebt (2023): Kurakurai 32%","yo":"Igbimọ̀ Robodebt (2023): Àṣìṣe 32%","ig":"Ọgbakọ Robodebt (2023): Njehie 32%"},

    # ── Additional UI keys ─────────────────────────────────────────────────
    "algorithm": {
        "en": "Algorithm",
        "ha": "Ƙididdigar lissafi",
        "yo": "Àlgọ́ríìdìmù",
        "ig": "Algọridim",
    },
    "algorithm_performance_summary": {
        "en": "🤖 Algorithm Performance Summary",
        "ha": "🤖 Taƙaitaccen Aikin Ƙididdigar lissafi",
        "yo": "🤖 Àkọ́sílẹ̀ Ìṣẹ́ Àlgọ́ríìdìmù",
        "ig": "🤖 Nchịkọta Ọrụ Algọridim",
    },
    "all_available_benchmarks": {
        "en": "📋 All Available Benchmarks",
        "ha": "📋 Duk Ma'aunai da Ake Da Su",
        "yo": "📋 Gbogbo Àwọn Ìwọ̀n Tí Ó Wà",
        "ig": "📋 Ihe Niile Dị Maka Ntụle",
    },
    "attack_strength": {
        "en": "Attack Strength",
        "ha": "Ƙarfin Harin",
        "yo": "Agbára Ìkọlù",
        "ig": "Ike Mwakpo",
    },
    "attack_type": {
        "en": "Attack Type",
        "ha": "Nau'in Harin",
        "yo": "Irú Ìkọlù",
        "ig": "Ụdị Mwakpo",
    },
    "basic_statistics": {
        "en": "Basic Statistics",
        "ha": "Ƙididdiga ta Asali",
        "yo": "Àwọn Ìṣirò Ìpìlẹ̀",
        "ig": "Ihe Ọmụmụ Nke Bụ Isi",
    },
    "bias_intensity_a": {
        "en": "Bias Intensity A",
        "ha": "Ƙarfin Nuna Bambanci A",
        "yo": "Ìgbéga Ìdènà A",
        "ig": "Arọ Mwakwụ A",
    },
    "bias_intensity_b": {
        "en": "Bias Intensity B",
        "ha": "Ƙarfin Nuna Bambanci B",
        "yo": "Ìgbéga Ìdènà B",
        "ig": "Arọ Mwakwụ B",
    },
    "biases_a": {
        "en": "Biases A",
        "ha": "Nuna Bambanci A",
        "yo": "Ìdènà A",
        "ig": "Mwakwụ A",
    },
    "biases_b": {
        "en": "Biases B",
        "ha": "Nuna Bambanci B",
        "yo": "Ìdènà B",
        "ig": "Mwakwụ B",
    },
    "compare_against_a_published_study": {
        "en": "Compare against a published study:",
        "ha": "Yi kwatankwacin binciken da aka buga:",
        "yo": "Ẹ ṣe àfiwé pẹ̀lú ìwádìí tí a tẹ̀jáde:",
        "ig": "Tụọ ya na ọmụmụ e bipụtara:",
    },
    "compare_against_published_study": {
        "en": "Compare against published study:",
        "ha": "Yi kwatankwacin binciken da aka buga:",
        "yo": "Ẹ ṣe àfiwé pẹ̀lú ìwádìí tí a tẹ̀jáde:",
        "ig": "Tụọ ya na ọmụmụ e bipụtara:",
    },
    "country_income_level": {
        "en": "Country Income Level",
        "ha": "Matakin Kudin Shiga na Kasa",
        "yo": "Ìpele Owó Orílẹ̀-èdè",
        "ig": "Ọkwa Ego Obodo",
    },
    "credit_scoring": {
        "en": "💳 Credit Scoring",
        "ha": "💳 Bada Maki na Bashi",
        "yo": "💳 Ìtọ́ka Gbèsè",
        "ig": "💳 Ntụle Kredit",
    },
    "customer_records": {
        "en": "Customer Records",
        "ha": "Bayanan Abokan Ciniki",
        "yo": "Àwọn Àkọsílẹ̀ Onírà",
        "ig": "Ndekọ Ndị Ahịa",
    },
    "data_preview": {
        "en": "Data Preview",
        "ha": "Ɗan Kallo na Bayanai",
        "yo": "Ìwòye Àwọn Dátà",
        "ig": "Nlele Ọdịnihu Nke Data",
    },
    "data_retention_days": {
        "en": "Data Retention (days)",
        "ha": "Adana Bayanai (kwanaki)",
        "yo": "Ìpamọ́ Dátà (ọjọ́)",
        "ig": "Ichekwa Data (ụbọchị)",
    },
    "dataset": {
        "en": "Dataset",
        "ha": "Tarin Bayanai",
        "yo": "Ìkójọ Dátà",
        "ig": "Nchịkọta Data",
    },
    "dataset_choice": {
        "en": "Dataset Choice",
        "ha": "Zaɓin Tarin Bayanai",
        "yo": "Yíyàn Ìkójọ Dátà",
        "ig": "Nhọrọ Nchịkọta Data",
    },
    "delay_between_snapshots_sec": {
        "en": "Delay between snapshots (sec)",
        "ha": "Jinkiri tsakanin hotuna (sec)",
        "yo": "Ìdádúró Láàárín Àwọn Gbàlejò (sec)",
        "ig": "Nchekwa Oge N'etiti Ihe Oyiyi (nkp)",
    },
    "demographics": {
        "en": "👥 Demographics",
        "ha": "👥 Ƙididdigan Jama'a",
        "yo": "👥 Àwọn Ìṣirò Ènìyàn",
        "ig": "👥 Mmadụ Na Ndị Ọha",
    },
    "disinformation_scenario": {
        "en": "📡 Disinformation Scenario",
        "ha": "📡 Yanayin Misinformation",
        "yo": "📡 Ìṣẹ̀lẹ̀ Àtọ́ Ìwífún",
        "ig": "📡 Ọnọdụ Ozi Na-Azị",
    },
    "domain": {
        "en": "Domain",
        "ha": "Fanni",
        "yo": "Àgbègbè",
        "ig": "Ngalaba",
    },
    "domain_a": {
        "en": "Domain A",
        "ha": "Fanni A",
        "yo": "Àgbègbè A",
        "ig": "Ngalaba A",
    },
    "domain_b": {
        "en": "Domain B",
        "ha": "Fanni B",
        "yo": "Àgbègbè B",
        "ig": "Ngalaba B",
    },
    "drift_scenario": {
        "en": "Drift Scenario",
        "ha": "Yanayin Sauya Kai",
        "yo": "Ìṣẹ̀lẹ̀ Ìrékọjá",
        "ig": "Ọnọdụ Ịgafe",
    },
    "education_scenario": {
        "en": "🎓 Education Scenario",
        "ha": "🎓 Yanayin Ilimi",
        "yo": "🎓 Ìṣẹ̀lẹ̀ Ẹ̀kọ́",
        "ig": "🎓 Ọnọdụ Agụmakwụkwọ",
    },
    "explainable_ai": {
        "en": "Explainable AI",
        "ha": "AI Mai Bayani",
        "yo": "AI Tí A Lè Ṣàlàyé",
        "ig": "AI Nke Enwere Ike Kọọ",
    },
    "farmer_records": {
        "en": "Farmer Records",
        "ha": "Bayanan Manoma",
        "yo": "Àwọn Àkọsílẹ̀ Àgbẹ̀",
        "ig": "Ndekọ Ndị Ọrụ Ubi",
    },
    "filter_to_this_domain_only": {
        "en": "Filter to this domain only",
        "ha": "Tace zuwa wannan fanni kawai",
        "yo": "Ṣe àkóso sí àgbègbè yìí péré",
        "ig": "Họpụta naanị ngalaba a",
    },
    "financial_context": {
        "en": "🏦 Financial Context",
        "ha": "🏦 Yanayin Kudi",
        "yo": "🏦 Ìjẹrìísí Owó",
        "ig": "🏦 Ọnọdụ Ego",
    },
    "financial_scenario": {
        "en": "Financial Scenario",
        "ha": "Yanayin Kudi",
        "yo": "Ìṣẹ̀lẹ̀ Owó",
        "ig": "Ọnọdụ Ego",
    },
    "gags_scenario": {
        "en": "💰 GAGS Scenario",
        "ha": "💰 Yanayin GAGS",
        "yo": "💰 Ìṣẹ̀lẹ̀ GAGS",
        "ig": "💰 Ọnọdụ GAGS",
    },
    "gender_audit": {
        "en": "Gender Audit",
        "ha": "Duba Jinsi",
        "yo": "Àyẹwò Ìbálòpọ̀",
        "ig": "Nnwale Okike",
    },
    "governance_layer": {
        "en": "Governance Layer",
        "ha": "Layin Gudanarwa",
        "yo": "Ìpele Ìṣàkóso",
        "ig": "Ọkwa Ọchịchị",
    },
    "governance_policy": {
        "en": "Governance Policy",
        "ha": "Manufar Gudanarwa",
        "yo": "Ìlànà Ìṣàkóso",
        "ig": "Iwu Ọchịchị",
    },
    "gtd_key_fields": {
        "en": "GTD Key Fields",
        "ha": "Filayen Mahimman GTD",
        "yo": "Àwọn Ìpílẹ̀ GTD Pàtàkì",
        "ig": "Oghere Isi Nke GTD",
    },
    "how_to_use": {
        "en": "📖 How to Use",
        "ha": "📖 Yadda Ake Amfani",
        "yo": "📖 Bí A Ṣe Ń Lò",
        "ig": "📖 Otu Esi Eji Ya",
    },
    "how_to_push_real_model_predictions_to_th": {
        "en": "How to push real model predictions to this monitor",
        "ha": "Yadda ake aika hasashen model na gaske zuwa wannan mai sa ido",
        "yo": "Bí a ṣe fi àsọtẹ́lẹ̀ àwòrán gidi ránṣẹ́ sí atẹlẹwọn yìí",
        "ig": "Otu esi eziga amụma nke ụdị ezigbo ya na ihe nlele a",
    },
    "include_baseline_no_bias_attack": {
        "en": "Include Baseline (no bias/attack)",
        "ha": "Haɗa Ma'auni na Asali (babu nuna bambanci/harin)",
        "yo": "Ìfiwéra Ìpìlẹ̀ (láìsí ìdènà/ìkọlù)",
        "ig": "Tinye Ntọala (enweghị mwakwụ/mwakpo)",
    },
    "institution_type": {
        "en": "Institution Type",
        "ha": "Nau'in Cibiya",
        "yo": "Irú Ilé-iṣẹ́",
        "ig": "Ụdị Ọrụ Ụlọ",
    },
    "judicial_scenario": {
        "en": "⚖️ Judicial Scenario",
        "ha": "⚖️ Yanayin Shari'a",
        "yo": "⚖️ Ìṣẹ̀lẹ̀ Ìdájọ́",
        "ig": "⚖️ Ọnọdụ Ikpe",
    },
    "low_income_customers": {
        "en": "Low-Income Customers",
        "ha": "Abokan Ciniki Masu Ƙaramin Kudin Shiga",
        "yo": "Àwọn Onírà Tí Owó Díẹ̀ Wà Lọ́wọ́",
        "ig": "Ndị Ahịa Nwere Obere Ego",
    },
    "low_income_patients": {
        "en": "Low-Income Patients",
        "ha": "Marasa Lafiya Masu Ƙaramin Kudin Shiga",
        "yo": "Àwọn Aláìsàn Tí Owó Díẹ̀ Wà Lọ́wọ́",
        "ig": "Ndị Ọrịa Nwere Obere Ego",
    },
    "minority_representation": {
        "en": "Minority Representation",
        "ha": "Wakilcin Masu Rinjaye",
        "yo": "Àṣojú Àwọn Ẹgbẹ́ Kékeré",
        "ig": "Nnọchiteanya Ndị Obere",
    },
    "ml_algorithm": {
        "en": "🤖 ML Algorithm",
        "ha": "🤖 Ƙididdigar ML",
        "yo": "🤖 Àlgọ́ríìdìmù ML",
        "ig": "🤖 Algọridim ML",
    },
    "module": {
        "en": "Module",
        "ha": "Sashe",
        "yo": "Àpò",
        "ig": "Akụkụ",
    },
    "multimodal_red_team": {
        "en": "Multimodal Red Team",
        "ha": "Ƙungiyar Ja ta Nau'i Da Nau'i",
        "yo": "Ẹgbẹ́ Pupa Onírúurú",
        "ig": "Otu Ọbara Ọbara Ọtụtụ",
    },
    "nigeria_judicial_context": {
        "en": "🇳🇬 Nigeria Judicial Context",
        "ha": "🇳🇬 Yanayin Shari'a na Najeriya",
        "yo": "🇳🇬 Ìjẹrìísí Ìdájọ́ Nàìjíríà",
        "ig": "🇳🇬 Ọnọdụ Ikpe Naịjirịa",
    },
    "perceived_threat_level": {
        "en": "Perceived Threat Level",
        "ha": "Matakin Barazanar da Aka Fahimta",
        "yo": "Ìpele Ewu Tí A Rí",
        "ig": "Ọkwa Ihe Iyi Egwu A Hụrụ",
    },
    "perspective": {
        "en": "Perspective",
        "ha": "Ra'ayi",
        "yo": "Ojú-ìwòye",
        "ig": "Anya Nlele",
    },
    "poison_rate_a": {
        "en": "Poison rate A",
        "ha": "Ƙimar Gurɓatawa A",
        "yo": "Oṣùwọ̀n Ìbàjẹ́ A",
        "ig": "Ọnụego Mmekọrịta Ọjọọ A",
    },
    "poison_rate_b": {
        "en": "Poison rate B",
        "ha": "Ƙimar Gurɓatawa B",
        "yo": "Oṣùwọ̀n Ìbàjẹ́ B",
        "ig": "Ọnụego Mmekọrịta Ọjọọ B",
    },
    "poisoning_rate": {
        "en": "Poisoning Rate",
        "ha": "Ƙimar Gurɓatawa",
        "yo": "Oṣùwọ̀n Ìbàjẹ́",
        "ig": "Ọnụego Mmekọrịta Ọjọọ",
    },
    "records": {
        "en": "Records",
        "ha": "Bayanan",
        "yo": "Àkọsílẹ̀",
        "ig": "Ndekọ",
    },
    "region": {
        "en": "Region",
        "ha": "Yanki",
        "yo": "Àgbègbè",
        "ig": "Mpaghara",
    },
    "regulatory_compliance": {
        "en": "Regulatory Compliance",
        "ha": "Bin Doka",
        "yo": "Ìfọwọ́sí Òfin",
        "ig": "Ịrụ Ụlọ Iwu",
    },
    "research_industry_features": {
        "en": "📖 Research + Industry Features",
        "ha": "📖 Fasalullukan Bincike + Masana'antu",
        "yo": "📖 Àwọn Ẹ̀yà Ìwádìí + Iléeṣẹ́",
        "ig": "📖 Atụmatu Nchọcha + Ihe Mmepụta",
    },
    "runs": {
        "en": "Runs",
        "ha": "Gwaje-gwaje",
        "yo": "Ìgbéṣẹ̀",
        "ig": "Ọsọ",
    },
    "samples_a": {
        "en": "Samples A",
        "ha": "Samfurai A",
        "yo": "Àwọn Àpẹẹrẹ A",
        "ig": "Ọnụọgụ Nsámple A",
    },
    "samples_b": {
        "en": "Samples B",
        "ha": "Samfurai B",
        "yo": "Àwọn Àpẹẹrẹ B",
        "ig": "Ọnụọgụ Nsámple B",
    },
    "scenario": {
        "en": "Scenario",
        "ha": "Yanayi",
        "yo": "Ìṣẹ̀lẹ̀",
        "ig": "Ọnọdụ",
    },
    "scenario_a": {
        "en": "Scenario A",
        "ha": "Yanayi A",
        "yo": "Ìṣẹ̀lẹ̀ A",
        "ig": "Ọnọdụ A",
    },
    "scenario_b": {
        "en": "Scenario B",
        "ha": "Yanayi B",
        "yo": "Ìṣẹ̀lẹ̀ B",
        "ig": "Ọnọdụ B",
    },
    "search": {
        "en": "🔍 Search",
        "ha": "🔍 Bincika",
        "yo": "🔍 Ṣàwárí",
        "ig": "🔍 Chọọ",
    },
    "severity": {
        "en": "Severity",
        "ha": "Tsananin Lamari",
        "yo": "Ìṣòro",
        "ig": "Ịdị Njọ",
    },
    "simulation_runs": {
        "en": "Simulation Runs",
        "ha": "Gwaje-gwaje na Kwaikwayo",
        "yo": "Àwọn Ìgbéṣẹ̀ Àfarawé",
        "ig": "Ọsọ Nnwale",
    },
    "snapshots_to_simulate": {
        "en": "Snapshots to simulate",
        "ha": "Hotuna don kwaikwayo",
        "yo": "Àwọn Gbàlejò láti ṣe àfarawé",
        "ig": "Ihe Oyiyi Iji Nwale",
    },
    "sort": {
        "en": "Sort",
        "ha": "Shirya",
        "yo": "Tò",
        "ig": "Nhazi",
    },
    "strategic_arena": {
        "en": "Strategic Arena",
        "ha": "Filin Dabarun",
        "yo": "Ààyè Ìlànà",
        "ig": "Ogige Atụmatụ",
    },
    "surveillance_configuration": {
        "en": "🔒 Surveillance Configuration",
        "ha": "🔒 Tsarin Kulawa",
        "yo": "🔒 Ètò Ìmójútó",
        "ig": "🔒 Nhazi Nlele",
    },
    "traditional_data_weight": {
        "en": "Traditional Data Weight",
        "ha": "Nauyin Bayanai na Gargajiya",
        "yo": "Ìwọ̀n Dátà Ìbílẹ̀",
        "ig": "Ibu Data Ọdịnala",
    },
    "type": {
        "en": "Type",
        "ha": "Nau'i",
        "yo": "Irú",
        "ig": "Ụdị",
    },
    "uninsured_patients": {
        "en": "Uninsured Patients",
        "ha": "Marasa Lafiya Mara Insura",
        "yo": "Àwọn Aláìsàn Tí Kò Ní Ìmúrasílẹ̀",
        "ig": "Ndị Ọrịa Enweghị Nchedo",
    },
    "view": {
        "en": "View",
        "ha": "Duba",
        "yo": "Wo",
        "ig": "Lee",
    },
    "view_full_details_for_scenario": {
        "en": "View full details for scenario:",
        "ha": "Duba cikakken bayanin yanayi:",
        "yo": "Wo àwọn àlàyé pípé fún ìṣẹ̀lẹ̀:",
        "ig": "Lee nkọwa ọzọzọ maka ọnọdụ:",
    },
    "group_performances": {
        "en": "Group Performances",
        "ha": "Aikin Ƙungiyoyi",
        "yo": "Ìṣẹ́ Àwọn Ẹgbẹ́",
        "ig": "Ọrụ Otu",
    },
    "self_reinforcing": {
        "en": "Self-Reinforcing Bias",
        "ha": "Nuna Bambanci Mai Ƙarfafa Kansa",
        "yo": "Ìdènà Tó Ń Fara Mọ̀ Ara Rẹ̀",
        "ig": "Mwakwụ Na-egosi Onwe Ya",
    },
    "bm_key": {
        "en": "Benchmark Key",
        "ha": "Maɓalli na Ma'auni",
        "yo": "Bọ́ọ̀lù Ìwọ̀n",
        "ig": "Igodo Ntụle",
    },
    "vr_scenario": {
        "en": "VR Scenario",
        "ha": "Yanayin VR",
        "yo": "Ìṣẹ̀lẹ̀ VR",
        "ig": "Ọnọdụ VR",
    },
    "is_africa": {
        "en": "Africa Focus",
        "ha": "Mayar da Hankali Afirka",
        "yo": "Ìbójú Áfríkà",
        "ig": "Ntụle Afrịka",
    },
    "attack_header": {
        "en": "⚠️ Adversarial Attacks",
        "ha": "⚠️ Hare-hare",
        "yo": "⚠️ Àwọn Ìkọlù",
        "ig": "⚠️ Mwakpo",
    },
    "sim_params_header": {
        "en": "📊 Simulation Parameters",
        "ha": "📊 Sigogin Kwaikwayo",
        "yo": "📊 Àwọn Ìpàrọ̀ Àfarawé",
        "ig": "📊 Nkwado Nnwale",
    },
    "view_mode_header": {
        "en": "👁 View Mode",
        "ha": "👁 Yanayin Kallo",
        "yo": "👁 Irú Wiwo",
        "ig": "👁 Ụdị Nlele",
    },
    "modules_header": {
        "en": "🔬 Feature Modules",
        "ha": "🔬 Sassan Fasali",
        "yo": "🔬 Àwọn Ìpín Ẹ̀yà",
        "ig": "🔬 Akụkụ Atụmatụ",
    },
    "bias_config_header": {
        "en": "🎭 Bias Configuration",
        "ha": "🎭 Tsarin Nuna Bambanci",
        "yo": "🎭 Ètò Ìdènà",
        "ig": "🔎 Nhazi Mwakwụ",
    },


    # ── Home page keys ───────────────────────────────────────────────────────
    "healthcare_equity": {
        "en": "Healthcare Equity",
        "ha": "Adalcin Kiwon Lafiya",
        "yo": "Ìdọ́gba Ìlera",
        "ig": "Ikpe Ọdịmma",
    },
    "national_security": {
        "en": "National Security",
        "ha": "Tsaron Ƙasa",
        "yo": "Ààbò Orílẹ̀-èdè",
        "ig": "Nchedo Mba",
    },
    "education_equity": {
        "en": "Education Equity",
        "ha": "Adalcin Ilimi",
        "yo": "Ìdọ́gba Ẹ̀kọ́",
        "ig": "Ikpe Agụmakwụkwọ",
    },
    "financial_inclusion": {
        "en": "Financial Inclusion",
        "ha": "Shigar Kudi",
        "yo": "Ìfiwéra Owó",
        "ig": "Ntinye Ego",
    },
    "judicial_justice": {
        "en": "Judicial Justice",
        "ha": "Adalcin Shari'a",
        "yo": "Ìdájọ́ Òdodo",
        "ig": "Ikpe Ezi Omume",
    },
    "disinformation_module": {
        "en": "Disinformation",
        "ha": "Labaran Ƙarya",
        "yo": "Àtọ́ Ìwífún",
        "ig": "Ozi Na-azị",
    },
    "economic_justice": {
        "en": "Economic Justice",
        "ha": "Adalcin Tattalin Arziki",
        "yo": "Ìdájọ́ Ọrọ̀-ajé",
        "ig": "Ikpe Akụ Na Ụba",
    },
    "comparison_mode": {
        "en": "Comparison Mode",
        "ha": "Yanayin Kwatantawa",
        "yo": "Irú Ìfiwéra",
        "ig": "Ụdị Ntụgharị",
    },
    "scenario_library": {
        "en": "Scenario Library",
        "ha": "Laburaren Yanayi",
        "yo": "Ibi Ìpamọ́ Ìṣẹ̀lẹ̀",
        "ig": "Ọba Ọnọdụ",
    },
    "live_monitoring": {
        "en": "Live Monitoring",
        "ha": "Kulawa Mai Rai",
        "yo": "Ìmójútó Alàáyè",
        "ig": "Nlele Ọnwụọnwụ",
    },
    "federated_learning": {
        "en": "Federated Learning",
        "ha": "Ilmantarwa Haɗaɗɗiya",
        "yo": "Ẹ̀kọ́ Àpapọ̀",
        "ig": "Mmụta Jikọọtụ",
    },
    "longitudinal_analysis": {
        "en": "Longitudinal Analysis",
        "ha": "Nazari na Tsawon Lokaci",
        "yo": "Ìtúpalẹ̀ Àkókò",
        "ig": "Nyocha Ogologo Oge",
    },
    "multilingual": {
        "en": "Multilingual",
        "ha": "Harsuna Da Yawa",
        "yo": "Àwọn Èdè Púpọ̀",
        "ig": "Ọtụtụ Asụsụ",
    },
    "export": {
        "en": "Export",
        "ha": "Fitar da",
        "yo": "Gbé Jáde",
        "ig": "Mepụta",
    },
    "governance": {
        "en": "Governance",
        "ha": "Gudanarwa",
        "yo": "Ìṣàkóso",
        "ig": "Ọchịchị",
    },
    "fairness_metrics": {
        "en": "Fairness Metrics",
        "ha": "Ma'aunin Adalci",
        "yo": "Àwọn Ìwọ̀n Ìdọ́gba",
        "ig": "Ọnụọgụ Ikpe",
    },
    "compliance_reports": {
        "en": "Compliance Reports",
        "ha": "Rahotannin Bin Doka",
        "yo": "Àwọn Ìjábọ̀ Ìfọwọ́sí",
        "ig": "Ọkọlọtọ Ịrụ Ụlọ Iwu",
    },
    "explainability": {
        "en": "Explainability",
        "ha": "Iya Bayyanawa",
        "yo": "Agbára Àlàyé",
        "ig": "Ike Ikọọ",
    },
    "drift_detection": {
        "en": "Drift Detection",
        "ha": "Ganin Sauya Kai",
        "yo": "Ìwárí Ìrékọjá",
        "ig": "Ịchọpụta Ịgafe",
    },
    "agrotech": {
        "en": "Agrotech",
        "ha": "Fasahar Noma",
        "yo": "Iṣẹ́-àgbẹ̀ Ìmọ̀-ẹrọ",
        "ig": "Teknọlọjị Ọrụ Ubi",
    },
    "live_stream": {
        "en": "Live Stream",
        "ha": "Watsa Kai Tsaye",
        "yo": "Ìtújáde Alàáyè",
        "ig": "Mmepụta Ndụ",
    },
    "plugin_api": {
        "en": "Plugin API",
        "ha": "API na Plugin",
        "yo": "API Plugin",
        "ig": "API Plugin",
    },
    "healthcare": {
        "en": "Healthcare",
        "ha": "Kiwon Lafiya",
        "yo": "Ìtọ́jú Ìlera",
        "ig": "Ahụike",
    },
    "education": {
        "en": "Education",
        "ha": "Ilimi",
        "yo": "Ẹ̀kọ́",
        "ig": "Agụmakwụkwọ",
    },
    "financial": {
        "en": "Financial",
        "ha": "Tattalin Arziki",
        "yo": "Owó",
        "ig": "Ego",
    },
    "judicial": {
        "en": "Judicial",
        "ha": "Shari'a",
        "yo": "Ìdájọ́",
        "ig": "Ikpe",
    },
    "contribute": {
        "en": "Contribute",
        "ha": "Ba da Gudummawa",
        "yo": "Ṣe Àfikún",
        "ig": "Nyefee Onyinye",
    },
    "rest_api": {
        "en": "REST API",
        "ha": "REST API",
        "yo": "REST API",
        "ig": "REST API",
    },
    "ussd": {
        "en": "USSD",
        "ha": "USSD",
        "yo": "USSD",
        "ig": "USSD",
    },


    # ── Section heading keys ────────────────────────────────────────────────
    "most_relevant_to_your_scenario": {
        "en": "#### Most Relevant to Your Scenario",
        "ha": "Most Relevant to Your Scenario",
        "yo": "Most Relevant to Your Scenario",
        "ig": "Most Relevant to Your Scenario",
    },
    "side_by_side_metric_comparison": {
        "en": "#### Side-by-Side Metric Comparison",
        "ha": "Kwatancen Ma'auni Gefen Gefen",
        "yo": "Ìfiwéra Ìwọ̀n Ẹgbẹ́ sí Ẹgbẹ́",
        "ig": "Itụle Ọnụọgụ N'akụkụ",
    },
    "bias_amplification_across_retraining_cycles": {
        "en": "### 🔁 Bias Amplification Across Retraining Cycles",
        "ha": "Nuna Bambanci",
        "yo": "Ìdènà",
        "ig": "Mwakwụ",
    },
    "federated_learning_bias_across_employers_platforms": {
        "en": "### 🌐 Federated Learning — Bias Across Employers / Platforms",
        "ha": "Ilmantarwa Haɗaɗɗiya",
        "yo": "Ẹ̀kọ́ Àpapọ̀",
        "ig": "Mmụta Jikọọtụ",
    },
    "nigeria_regulatory_panel": {
        "en": "#### 🇳🇬 Nigeria Regulatory Panel",
        "ha": "Ƙa'idojin Najeriya",
        "yo": "Àwọn Òfin Nàìjíríà",
        "ig": "Iwu Naịjirịa",
    },
    "health_equity_heatmap_across_runs": {
        "en": "#### Health Equity Heatmap Across Runs",
        "ha": "Adalci",
        "yo": "Ìdọ́gba",
        "ig": "Ikpe",
    },
    "economic_justice_recommendations": {
        "en": "## 💡 Economic Justice Recommendations",
        "ha": "Adalcin Tattalin Arziki",
        "yo": "Ìdájọ́ Ọrọ̀-ajé",
        "ig": "Ikpe Akụ Na Ụba",
    },
    "economic_justice_simulation_research_industry_edit": {
        "en": "## 💼 Economic Justice Simulation — Research + Industry Edition",
        "ha": "Adalcin Tattalin Arziki",
        "yo": "Ìdájọ́ Ọrọ̀-ajé",
        "ig": "Ikpe Akụ Na Ụba",
    },
    "embedded_real_world_benchmark_database": {
        "en": "### 📚 Embedded Real-World Benchmark Database",
        "ha": "Ma'aunin Duniyar Gaske",
        "yo": "Ìwọ̀n Ayé Gidi",
        "ig": "Ntụle Ụwa Eziokwu",
    },
    "disinformation_resilience_dashboard": {
        "en": "## 📊 Disinformation Resilience Dashboard",
        "ha": "Allo",
        "yo": "Pánẹ̀lì",
        "ig": "Dashbọọdụ",
    },
    "model_performance_across_runs": {
        "en": "### Model Performance Across Runs",
        "ha": "Aikin Samfurin",
        "yo": "Ìṣẹ́ Àwòrán",
        "ig": "Ọrụ Ụdị",
    },
    "equity_gap_matrix": {
        "en": "### Equity Gap Matrix",
        "ha": "Adalci",
        "yo": "Ìdọ́gba",
        "ig": "Ikpe",
    },
    "equity_breakdown": {
        "en": "### Equity Breakdown",
        "ha": "Adalci",
        "yo": "Ìdọ́gba",
        "ig": "Ikpe",
    },
    "scenario_context": {
        "en": "#### 🎓 Scenario Context",
        "ha": "Yanayin Yanayi",
        "yo": "Ìjẹrìísí Ìṣẹ̀lẹ̀",
        "ig": "Ọnọdụ Ọnọdụ",
    },
    "student_prediction_explanation": {
        "en": "#### Student Prediction Explanation",
        "ha": "Student Prediction Explanation",
        "yo": "Student Prediction Explanation",
        "ig": "Student Prediction Explanation",
    },
    "what_would_change_the_prediction": {
        "en": "#### What would change the prediction?",
        "ha": "Menene zai canza hasashe",
        "yo": "Kíni yóò yí àsọtẹ́lẹ̀ padà",
        "ig": "Gịnị ga-agbanwe amụma",
    },
    "intersectional_fairness_gender_urban": {
        "en": "#### Intersectional Fairness (Gender × Urban)",
        "ha": "Adalcin Haɗuwar Matsayi",
        "yo": "Ìdọ́gba Àpapọ̀ Ìdánimọ̀",
        "ig": "Ikpe Njikọta Ọnọdụ",
    },
    "longitudinal_bias_analysis": {
        "en": "### 🔁 Longitudinal Bias Analysis",
        "ha": "Nazarin Nuna Bambanci na Tsawon Lokaci",
        "yo": "Ìtúpalẹ̀ Ìdènà Àkókò",
        "ig": "Nyocha Mwakwụ Ogologo Oge",
    },
    "federated_learning_simulation": {
        "en": "### 🌐 Federated Learning Simulation",
        "ha": "Ilmantarwa Haɗaɗɗiya",
        "yo": "Ẹ̀kọ́ Àpapọ̀",
        "ig": "Mmụta Jikọọtụ",
    },
    "nigeria_regulatory_compliance": {
        "en": "### 🇳🇬 Nigeria Regulatory Compliance",
        "ha": "Ƙa'idojin Najeriya",
        "yo": "Àwọn Òfin Nàìjíríà",
        "ig": "Iwu Naịjirịa",
    },
    "real_world_benchmark_comparison": {
        "en": "#### 📚 Real-World Benchmark Comparison",
        "ha": "Ma'aunin Duniyar Gaske",
        "yo": "Ìwọ̀n Ayé Gidi",
        "ig": "Ntụle Ụwa Eziokwu",
    },
    "raw_simulation_results": {
        "en": "### 📋 Raw Simulation Results",
        "ha": "Sakamakon Kwaikwayo na Asali",
        "yo": "Àwọn Ìbísí Àfarawé Ìpìlẹ̀",
        "ig": "Nsonaazụ Nnwale Ọdịdị",
    },
    "disinformation_content_moderation_policy": {
        "en": "## 💡 Disinformation & Content Moderation Policy",
        "ha": "Misinformation",
        "yo": "Àtọ́ Ìwífún",
        "ig": "Ozi Na-azị",
    },
    "explainable_ai_defendant_rights": {
        "en": "### 🔍 Explainable AI — Defendant Rights",
        "ha": "AI Mai Bayani",
        "yo": "AI Tí A Lè Ṣàlàyé",
        "ig": "AI Nke Enwere Ike Kọọ",
    },
    "why_was_this_defendant_flagged": {
        "en": "#### 🔎 Why was this defendant flagged?",
        "ha": "Me ya sa aka nuna wanda ake zargi",
        "yo": "Kí nìdí tí wọ́n fi ṣàmì ọ̀daran yìí",
        "ig": "Gịnị mere ha chọpụta onye a ekwuoro",
    },
    "what_would_have_changed_this_outcome": {
        "en": "#### 🔄 What would have changed this outcome?",
        "ha": "Menene zai canza wannan sakamakon",
        "yo": "Kíni yóò ti yí àbájáde yìí padà",
        "ig": "Gịnị ga-agbanwe nsonaazụ a",
    },
    "intersectional_fairness_race_poverty": {
        "en": "#### 🔗 Intersectional Fairness (Race × Poverty)",
        "ha": "Adalcin Haɗuwar Matsayi",
        "yo": "Ìdọ́gba Àpapọ̀ Ìdánimọ̀",
        "ig": "Ikpe Njikọta Ọnọdụ",
    },
    "real_world_benchmark_compas_vs_your_simulation": {
        "en": "### 📚 Real-World Benchmark: COMPAS vs Your Simulation",
        "ha": "Ma'aunin Duniyar Gaske",
        "yo": "Ìwọ̀n Ayé Gidi",
        "ig": "Ntụle Ụwa Eziokwu",
    },
    "historical_ai_bias_incidents_in_criminal_justice": {
        "en": "### 🌍 Historical AI Bias Incidents in Criminal Justice",
        "ha": "Nuna Bambanci",
        "yo": "Ìdènà",
        "ig": "Mwakwụ",
    },
    "nigeria_judicial_system_context": {
        "en": "#### 🇳🇬 Nigeria Judicial System Context",
        "ha": "Shari'a",
        "yo": "Ìdájọ́",
        "ig": "Ikpe",
    },
    "bias_self_reinforcement_across_retraining": {
        "en": "### 🔁 Bias Self-Reinforcement Across Retraining",
        "ha": "Nuna Bambanci",
        "yo": "Ìdènà",
        "ig": "Mwakwụ",
    },
    "federated_learning_bias_across_jurisdictions": {
        "en": "### 🌐 Federated Learning — Bias Across Jurisdictions",
        "ha": "Ilmantarwa Haɗaɗɗiya",
        "yo": "Ẹ̀kọ́ Àpapọ̀",
        "ig": "Mmụta Jikọọtụ",
    },
    "multi_framework_compliance_report": {
        "en": "### 📋 Multi-Framework Compliance Report",
        "ha": "Rahoto na Bin Doka",
        "yo": "Ìjábọ̀ Ìfọwọ́sí",
        "ig": "Ọkọlọtọ Ịrụ Ụlọ Iwu",
    },
    "nigeria_regulatory_alignment": {
        "en": "#### 🇳🇬 Nigeria Regulatory Alignment",
        "ha": "Ƙa'idojin Najeriya",
        "yo": "Àwọn Òfin Nàìjíríà",
        "ig": "Iwu Naịjirịa",
    },
    "export_results": {
        "en": "### 📤 Export Results",
        "ha": "Fitar da Sakamakon",
        "yo": "Gbé Àwọn Ìbísí Jáde",
        "ig": "Mepụta Nsonaazụ",
    },
    "statistical_distribution_across_runs": {
        "en": "### 📈 Statistical Distribution Across Runs",
        "ha": "Rarrabawa ta Ƙididdiga",
        "yo": "Ìpínkúpa Ìṣirò",
        "ig": "Nkesa Ọnụọgụ",
    },
    "judicial_ai_policy_recommendations": {
        "en": "## 💡 Judicial AI Policy Recommendations",
        "ha": "Shawarwarin Manufa",
        "yo": "Àwọn Ìmọ̀ràn Ìlànà",
        "ig": "Ndụmọdụ Iwu",
    },
    "education_equity_dashboard": {
        "en": "## 📊 Education Equity Dashboard",
        "ha": "Allo",
        "yo": "Pánẹ̀lì",
        "ig": "Dashbọọdụ",
    },
    "education_ai_policy_recommendations": {
        "en": "## 💡 Education AI Policy Recommendations",
        "ha": "Shawarwarin Manufa",
        "yo": "Àwọn Ìmọ̀ràn Ìlànà",
        "ig": "Ndụmọdụ Iwu",
    },
    "why_was_this_customer_denied": {
        "en": "#### Why was this customer denied?",
        "ha": "Me ya sa an hana abokin ciniki",
        "yo": "Kí nìdí tí wọ́n fi kọ onírà yìí",
        "ig": "Gịnị mere ha jụọ ahịa a",
    },
    "what_would_get_this_customer_approved": {
        "en": "#### What would get this customer approved?",
        "ha": "Menene zai sa abokin ciniki ya samu amincewa",
        "yo": "Kíni yóò jẹ́ kí onírà yìí fọwọ́ sí",
        "ig": "Gịnị ga-eme ka ahịa a ng촉kọ",
    },
    "financial_inclusion_policy_recommendations": {
        "en": "## 💡 Financial Inclusion Policy Recommendations",
        "ha": "Shawarwarin Manufa",
        "yo": "Àwọn Ìmọ̀ràn Ìlànà",
        "ig": "Ndụmọdụ Iwu",
    },
    "healthcare_equity_dashboard": {
        "en": "## 📊 Healthcare Equity Dashboard",
        "ha": "Allo",
        "yo": "Pánẹ̀lì",
        "ig": "Dashbọọdụ",
    },
    "health_equity_gap_analysis": {
        "en": "### ⚖️ Health Equity Gap Analysis",
        "ha": "Adalci",
        "yo": "Ìdọ́gba",
        "ig": "Ikpe",
    },
    "gender_equity_audit_feature_3_unesco_women4ethical": {
        "en": "#### 🌍 Gender Equity Audit (Feature 3 — UNESCO Women4EthicalAI)",
        "ha": "Adalci",
        "yo": "Ìdọ́gba",
        "ig": "Ikpe",
    },
    "clinical_impact_by_patient_group": {
        "en": "### 🏥 Clinical Impact by Patient Group",
        "ha": "Tasirin Asibiti",
        "yo": "Ipa Ilé-ìwòsàn",
        "ig": "Mmetụta Ụlọ Ọgwụ",
    },
    "healthcare_resource_access": {
        "en": "#### 🏥 Healthcare Resource Access",
        "ha": "🏥 Healthcare Resource Access",
        "yo": "🏥 Healthcare Resource Access",
        "ig": "🏥 Healthcare Resource Access",
    },
    "advanced_feature_module_results": {
        "en": "### 🔬 Advanced Feature Module Results",
        "ha": "🔬 Advanced Feature Module Results",
        "yo": "🔬 Advanced Feature Module Results",
        "ig": "🔬 Advanced Feature Module Results",
    },
    "feature_2_multimodal_red_teaming": {
        "en": "#### Feature 2 — Multimodal Red Teaming",
        "ha": "Feature 2 — Multimodal Red Teaming",
        "yo": "Feature 2 — Multimodal Red Teaming",
        "ig": "Feature 2 — Multimodal Red Teaming",
    },
    "feature_1_ai_agent_economy_healthcare_resources": {
        "en": "#### Feature 1 — AI Agent Economy (Healthcare Resources)",
        "ha": "Feature 1 — AI Agent Economy (Healthcare Resources)",
        "yo": "Feature 1 — AI Agent Economy (Healthcare Resources)",
        "ig": "Feature 1 — AI Agent Economy (Healthcare Resources)",
    },
    "feature_4_hybrid_governance_ledger_blockchain_styl": {
        "en": "#### Feature 4 — Hybrid Governance Ledger (Blockchain-style)",
        "ha": "Feature 4 — Hybrid Governance Ledger (Blockchain-style)",
        "yo": "Feature 4 — Hybrid Governance Ledger (Blockchain-style)",
        "ig": "Feature 4 — Hybrid Governance Ledger (Blockchain-style)",
    },
    "feature_5_strategic_social_reasoning_arena": {
        "en": "#### Feature 5 — Strategic Social Reasoning Arena",
        "ha": "Feature 5 — Strategic Social Reasoning Arena",
        "yo": "Feature 5 — Strategic Social Reasoning Arena",
        "ig": "Feature 5 — Strategic Social Reasoning Arena",
    },
    "data_quality_source_analysis": {
        "en": "### 📊 Data Quality & Source Analysis",
        "ha": "Nazarin Ingancin Bayanai da Tushe",
        "yo": "Ìtúpalẹ̀ Didara Dátà àti Orísun",
        "ig": "Nyocha Àgwà Data na Isi Iyi",
    },
    "instance_explanation": {
        "en": "#### Instance Explanation",
        "ha": "Bayani na Misali",
        "yo": "Àlàyé Àpẹẹrẹ",
        "ig": "Nkọwa Ihe Atụ",
    },
    "intersectional_fairness": {
        "en": "#### Intersectional Fairness",
        "ha": "Adalcin Haɗuwar Matsayi",
        "yo": "Ìdọ́gba Àpapọ̀ Ìdánimọ̀",
        "ig": "Ikpe Njikọta Ọnọdụ",
    },
    "regulatory_compliance_report": {
        "en": "### 📋 Regulatory Compliance Report",
        "ha": "Rahoto na Bin Doka",
        "yo": "Ìjábọ̀ Ìfọwọ́sí",
        "ig": "Ọkọlọtọ Ịrụ Ụlọ Iwu",
    },
    "policy_recommendations": {
        "en": "## 💡 Policy Recommendations",
        "ha": "Shawarwarin Manufa",
        "yo": "Àwọn Ìmọ̀ràn Ìlànà",
        "ig": "Ndụmọdụ Iwu",
    },
    "welcome_to_healthcare_equity_simulation": {
        "en": "## 🏥 Welcome to Healthcare Equity Simulation",
        "ha": "Adalci",
        "yo": "Ìdọ́gba",
        "ig": "Ikpe",
    },
    "security_assessment_dashboard": {
        "en": "## 📊 Security Assessment Dashboard",
        "ha": "Allo",
        "yo": "Pánẹ̀lì",
        "ig": "Dashbọọdụ",
    },
    "security_vs_liberty_trade_off": {
        "en": "### ⚖️ Security vs Liberty Trade-off",
        "ha": "Tsaro",
        "yo": "Ààbò",
        "ig": "Nchedo",
    },
    "fairness_vs_security": {
        "en": "### 📊 Fairness vs Security",
        "ha": "Tsaro",
        "yo": "Ààbò",
        "ig": "Nchedo",
    },
    "bias_impact_assessment": {
        "en": "### 🎭 Bias Impact Assessment",
        "ha": "Nuna Bambanci",
        "yo": "Ìdènà",
        "ig": "Mwakwụ",
    },
    "feature_1_ai_agent_economy_security_resources": {
        "en": "#### Feature 1 — AI Agent Economy (Security Resources)",
        "ha": "Tsaro",
        "yo": "Ààbò",
        "ig": "Nchedo",
    },
    "feature_4_governance_ledger_blockchain_style": {
        "en": "#### Feature 4 — Governance Ledger (Blockchain-style)",
        "ha": "Feature 4 — Governance Ledger (Blockchain-style)",
        "yo": "Feature 4 — Governance Ledger (Blockchain-style)",
        "ig": "Feature 4 — Governance Ledger (Blockchain-style)",
    },
    "data_explorer": {
        "en": "### 🔍 Data Explorer",
        "ha": "Mai Binciken Bayanai",
        "yo": "Olùṣàwárí Dátà",
        "ig": "Onye Nchọọ Data",
    },
    "security_ethics_recommendations": {
        "en": "## 💡 Security & Ethics Recommendations",
        "ha": "Tsaro",
        "yo": "Ààbò",
        "ig": "Nchedo",
    },
    "welcome_to_national_security_simulation": {
        "en": "## 🎯 Welcome to National Security Simulation",
        "ha": "Tsaro",
        "yo": "Ààbò",
        "ig": "Nchedo",
    },
    "agrotech_equity_dashboard": {
        "en": "## 📊 Agrotech Equity Dashboard",
        "ha": "Allo",
        "yo": "Pánẹ̀lì",
        "ig": "Dashbọọdụ",
    },
    "fairness_gender_equity_analysis": {
        "en": "### ⚖️ Fairness & Gender Equity Analysis",
        "ha": "Adalci",
        "yo": "Ìdọ́gba",
        "ig": "Ikpe",
    },
    "gender_equity_audit_unesco_women4ethicalai": {
        "en": "#### 🌍 Gender Equity Audit (UNESCO Women4EthicalAI)",
        "ha": "Adalci",
        "yo": "Ìdọ́gba",
        "ig": "Ikpe",
    },
    "intersectional_fairness_gender_income": {
        "en": "#### 🔀 Intersectional Fairness (Gender × Income)",
        "ha": "Adalcin Haɗuwar Matsayi",
        "yo": "Ìdọ́gba Àpapọ̀ Ìdánimọ̀",
        "ig": "Ikpe Njikọta Ọnọdụ",
    },
    "explainable_ai_understanding_model_decisions": {
        "en": "### 🧠 Explainable AI — Understanding Model Decisions",
        "ha": "AI Mai Bayani",
        "yo": "AI Tí A Lè Ṣàlàyé",
        "ig": "AI Nke Enwere Ike Kọọ",
    },
    "feature_importance_permutation_based": {
        "en": "#### Feature Importance (Permutation-based)",
        "ha": "Muhimmancin Fasali",
        "yo": "Ìjẹwọ́ Ìmọ̀",
        "ig": "Mkpa Atụmatu",
    },
    "instance_explanation_lime_lite": {
        "en": "#### Instance Explanation (LIME-lite)",
        "ha": "Bayani na Misali",
        "yo": "Àlàyé Àpẹẹrẹ",
        "ig": "Nkọwa Ihe Atụ",
    },
    "counterfactual_explanation": {
        "en": "#### Counterfactual Explanation",
        "ha": "Idan Ba",
        "yo": "Ohun Tí Ì Bá Yí padà",
        "ig": "Ihe Ọ Bụrụ Na",
    },
    "auto_generated_model_card": {
        "en": "#### 📄 Auto-Generated Model Card",
        "ha": "Katunan Samfurin da AI ta Yi",
        "yo": "Kárùn Àwòrán Tí AI Ṣe",
        "ig": "Kaadị Ụdị Emere Onwe Ya",
    },
    "feature_5_strategic_arena": {
        "en": "#### Feature 5 — Strategic Arena",
        "ha": "Feature 5 — Strategic Arena",
        "yo": "Feature 5 — Strategic Arena",
        "ig": "Feature 5 — Strategic Arena",
    },
    "ai_agent_economy_agricultural_resource_allocation": {
        "en": "### 💰 AI Agent Economy — Agricultural Resource Allocation",
        "ha": "💰 AI Agent Economy — Agricultural Resource Allocation",
        "yo": "💰 AI Agent Economy — Agricultural Resource Allocation",
        "ig": "💰 AI Agent Economy — Agricultural Resource Allocation",
    },
    "ussd_sms_accessibility_analysis": {
        "en": "#### 📱 USSD/SMS Accessibility Analysis",
        "ha": "📱 USSD/SMS Accessibility Analysis",
        "yo": "📱 USSD/SMS Accessibility Analysis",
        "ig": "📱 USSD/SMS Accessibility Analysis",
    },
    "ussd_interface_simulator": {
        "en": "#### 📱 USSD Interface Simulator",
        "ha": "📱 USSD Interface Simulator",
        "yo": "📱 USSD Interface Simulator",
        "ig": "📱 USSD Interface Simulator",
    },
    "agent_performance": {
        "en": "#### Agent Performance",
        "ha": "Aikin Wakili",
        "yo": "Ìṣẹ́ Olùṣojú",
        "ig": "Ọrụ Onye Ọrụ",
    },
    "resource_auction_results": {
        "en": "#### Resource Auction Results",
        "ha": "Sakamakon Gwanjon Albarkatun",
        "yo": "Àwọn Ìbísí Àpéjọpọ̀ Àwọn Ohun Àmúlò",
        "ig": "Nsonaazụ Ọchọ Ihe",
    },
    "agrotech_ai_recommendations": {
        "en": "## 💡 Agrotech AI Recommendations",
        "ha": "Fasahar Noma",
        "yo": "Iṣẹ́-àgbẹ̀ Ìmọ̀-ẹrọ",
        "ig": "Teknọlọjị Ọrụ Ubi",
    },
    "welcome_to_agrotech_equity_simulation": {
        "en": "## 🌾 Welcome to Agrotech Equity Simulation",
        "ha": "Adalci",
        "yo": "Ìdọ́gba",
        "ig": "Ikpe",
    },
    "monitor_configuration": {
        "en": "## 📡 Monitor Configuration",
        "ha": "Tsarin Mai Sa Ido",
        "yo": "Ètò Atẹlẹwọn",
        "ig": "Nhazi Ihe Nlele",
    },
    "alert_thresholds": {
        "en": "### Alert Thresholds",
        "ha": "Matakan Gargadi",
        "yo": "Àwọn Ìpele Ìkìlọ̀",
        "ig": "Ọkwa Ịkpọ Ihe",
    },
    "connect_to_a_live_model_via_rest_api": {
        "en": "### 🔌 Connect to a Live Model via REST API",
        "ha": "🔌 Connect to a Live Model via REST API",
        "yo": "🔌 Connect to a Live Model via REST API",
        "ig": "🔌 Connect to a Live Model via REST API",
    },
    "welcome_to_real_time_monitoring": {
        "en": "## 📡 Welcome to Real-Time Monitoring",
        "ha": "Kulawa",
        "yo": "Ìmójútó",
        "ig": "Nlele",
    },


    # ── Comparison mode keys ────────────────────────────────────────────────
    "bias_intensity_a": {
        "en": "Bias intensity A",
        "ha": "Bias intensity A",
        "yo": "Bias intensity A",
        "ig": "Bias intensity A",
    },
    "bias_intensity_b": {
        "en": "Bias intensity B",
        "ha": "Bias intensity B",
        "yo": "Bias intensity B",
        "ig": "Bias intensity B",
    },


}


# ── Core functions ────────────────────────────────────────────────────────────

def get_lang() -> str:
    """Return the currently active language code (en/ha/yo/ig)."""
    return st.session_state.get("_gags_lang", "en")


def t(key: str, lang: Optional[str] = None) -> str:
    """
    Translate a key to the active language.
    Falls back to English, then to the key itself.

    Parameters
    ----------
    key  : translation key (e.g. "metric_accuracy")
    lang : override language code; uses active language if None

    Returns
    -------
    str : translated string
    """
    lang = lang or get_lang()
    entry = TRANSLATIONS.get(key, {})
    return entry.get(lang) or entry.get("en") or key


def t_list(keys: list, lang: Optional[str] = None) -> list:
    """Translate a list of keys."""
    return [t(k, lang) for k in keys]


def language_switcher(location: str = "sidebar") -> None:
    """
    Render a compact language selector.

    Parameters
    ----------
    location : "sidebar" | "main" | "header"
    """
    lang_key = "_gags_lang"
    current  = st.session_state.get(lang_key, "en")

    options  = list(LANGUAGES.keys())
    labels   = {code: f"{meta['flag']} {meta['native']}"
                for code, meta in LANGUAGES.items()}

    if location == "sidebar":
        st.markdown(
            f"<p style='font-size:.78rem;font-weight:500;"
            f"color:var(--color-text-secondary);margin-bottom:3px;'>"
            f"{t('language')}</p>",
            unsafe_allow_html=True,
        )
        def _on_lang_change():
            st.session_state[lang_key] = st.session_state["_lang_select_sidebar"]

        st.selectbox(
            "Language",
            options,
            index=options.index(current),
            format_func=lambda c: labels[c],
            key="_lang_select_sidebar",
            label_visibility="collapsed",
            on_change=_on_lang_change,
        )
        # Sync: if widget value differs from stored lang, update and rerun
        _widget_val = st.session_state.get("_lang_select_sidebar", current)
        if _widget_val != current:
            st.session_state[lang_key] = _widget_val
            st.rerun()
    else:
        # Inline horizontal for main/header
        cols = st.columns(len(options))
        selected = current
        for col, code in zip(cols, options):
            if col.button(
                labels[code],
                key=f"_lang_btn_{code}",
                type="primary" if code == current else "secondary",
                use_container_width=True,
            ):
                selected = code
        if selected != current:
            st.session_state[lang_key] = selected
            st.rerun()


def language_badge() -> None:
    """
    Render a small language indicator badge (no switcher).
    Useful for confirming active language in headers.
    """
    lang = get_lang()
    meta = LANGUAGES.get(lang, LANGUAGES["en"])
    if lang != "en":
        st.markdown(
            f"<span style='background:var(--color-background-info);"
            f"color:var(--color-text-info);font-size:.75rem;font-weight:600;"
            f"padding:2px 8px;border-radius:4px;'>"
            f"{meta['flag']} {meta['native']}</span>",
            unsafe_allow_html=True,
        )
