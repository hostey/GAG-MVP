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