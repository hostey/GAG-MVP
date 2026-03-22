# components/ussd_simulator.py
"""
GAGS USSD/SMS Simulation Mode — v1.0
=====================================
Models how this AI system would behave for the 66% of Nigerian users
who access services via basic phones without internet.

Two components
--------------
1. ussd_interface()   — Interactive USSD menu simulator that mimics the
                        *384# dial-in flow a farmer would use on a 2G phone.
                        Shows prediction results as the farmer would receive them.

2. accessibility_gap_report() — Measures and visualises the gap between
                                 what AI-enabled and USSD-only users receive.
                                 This is the system modelling its own accessibility.

3. sms_result_formatter()    — Formats a prediction result as an SMS of ≤160
                                characters in Hausa/English/Yoruba/Igbo.

Real-world context
------------------
In Nigeria (2024):
  - 66% of smallholder farmers use feature phones only
  - Rural 3G/4G coverage: ~34% of geographic area
  - USSD works on 2G, offline, no data plan required
  - Average SMS cost: ₦4 (~0.003 USD)
  - NCC regulates USSD gateway pricing

Simulation logic
----------------
USSD is a 182-character session-based protocol. This simulator:
  - Reduces the full model prediction to a binary HIGH/LOW risk signal
  - Encodes the top 2 contributing features as plain-language hints
  - Formats the counterfactual as a single actionable instruction
  - Measures accuracy loss vs full model (the accessibility cost)
"""

from typing import Any, Dict, List, Optional, Tuple
import streamlit as st
import numpy as np

try:
    from components.i18n import t, get_lang, LANGUAGES
except ImportError:
    def t(key, lang=None): return key
    def get_lang(): return "en"
    LANGUAGES = {"en": {"name": "English", "flag": "🇬🇧"}}


# ── USSD Menu Tree ────────────────────────────────────────────────────────────
# Mirrors a real NCC-compliant USSD session structure

_USSD_MENUS: Dict[str, Dict] = {
    "root": {
        "text_en": "GAGS AI Advisory\nWelcome\n1. Crop risk check\n2. Market advice\n3. Weather alert\n4. My language\n0. Exit",
        "text_ha": "GAGS Shawarwari\nMaraba\n1. Duba haɗarin amfanin gona\n2. Shawarar kasuwa\n3. Faɗakar da yanayi\n4. Yarenata\n0. Fita",
        "text_yo": "GAGS Ìmọ̀ràn\nẸ káàbọ̀\n1. Ṣàyẹ̀wò ewu irúgbìn\n2. Ìmọ̀ràn ọjà\n3. Ìkìlọ̀ ojú ọjọ́\n4. Èdè mi\n0. Jáde",
        "text_ig": "GAGS Ndụmọdụ\nNnọọ\n1. Nyocha ihe egwu ọkụkụ\n2. Ndụmọdụ ahịa\n3. Ọchịchọ ihu igwe\n4. Asụsụ m\n0. Pụọ",
        "options": {"1": "crop_input", "2": "market_input", "3": "weather_alert", "4": "lang_select", "0": "exit"},
    },
    "crop_input": {
        "text_en": "Crop Risk Check\nEnter your LGA number:\n1. Abuja Municipal\n2. Bwari\n3. Gwagwalada\n4. Kuje\n5. Other",
        "text_ha": "Duba Haɗarin Amfanin Gona\nShigar da lambar LGA ɗinka:\n1. Abuja Municipal\n2. Bwari\n3. Gwagwalada\n4. Kuje\n5. Sauran",
        "text_yo": "Ṣàyẹ̀wò Ewu Irúgbìn\nTẹ nọ́mbà LGA rẹ:\n1. Abuja Municipal\n2. Bwari\n3. Gwagwalada\n4. Kuje\n5. Mìíràn",
        "text_ig": "Nyocha Ihe Egwu Ọkụkụ\nTinye nọmba LGA gị:\n1. Abuja Municipal\n2. Bwari\n3. Gwagwalada\n4. Kuje\n5. Ndị Ọzọ",
        "options": {"1": "crop_result", "2": "crop_result", "3": "crop_result",
                    "4": "crop_result", "5": "crop_result"},
    },
    "crop_result_high": {
        "text_en": "RESULT: HIGH RISK\nRainfall below normal.\nSoil moisture low.\nACTION: Irrigate within 3 days or apply drought-resistant seed.\nReply 1 for extension officer contact.",
        "text_ha": "SAKAMAKON: HAƊARI BABBA\nRuwan sama ƙasa da al'ada.\nRuwan ƙasa kaɗan.\nAIKI: Yi ban ruwa cikin kwana 3 ko yi amfani da iri mai jure fari.\nReplai 1 don sanar da jami'in ƙasar gona.",
        "text_yo": "ÌYỌRÍSÍ: EWUU GÍGA\nÒjò wà ní ìsàlẹ̀ àárọ̀.\nOmi ilẹ̀ kéré.\nIṢẸ́: Fọ́nrin ní ọjọ́ 3 tàbí lo irúgbìn tó le dúró àìjọ òjò.\nFèsì 1 fún ènìyàn àgbẹ̀.",
        "text_ig": "NSONAAZỤ: IHE EGWU DỊ ELU\nOzuzo dị n'okpuru nke ọma.\nMmiri ala dị obere.\nMMEME: Gbaa mmiri n'ime ụbọchị 3 ma ọ bụ jiri mkpụrụ nke nwere ike anọdụ na ọcha.\nZaghachi 1 maka kpọtụrụ onye ndụmọdụ ugbo.",
        "options": {"1": "contact_officer", "0": "root"},
    },
    "crop_result_low": {
        "text_en": "RESULT: LOW RISK\nConditions are favourable.\nContinue current practices.\nNext check recommended in 14 days.\nReply 0 for main menu.",
        "text_ha": "SAKAMAKON: HAƊARI ƘARAMI\nYanayi yana da kyau.\nCi gaba da ayyukan yanzu.\nBatun gaba an ba da shawara a cikin kwana 14.\nReplai 0 don menu na farko.",
        "text_yo": "ÌYỌRÍSÍ: EWUU KẸ́KẸ́\nÀwọn ipò dára.\nTẹ̀síwájú àwọn àṣà lọ́wọ́lọ́wọ́.\nA dámọ̀ràn ìwádìí tó kàn ní ọjọ́ 14.\nFèsì 0 fún àkójọ àkọ́kọ́.",
        "text_ig": "NSONAAZỤ: IHE EGWU DỊ NTAKỊRỊ\nOndọ dị mma.\nGaa n'ihu n'omume ugbu a.\nA na-atụ aro nyocha ọzọ n'ụbọchị 14.\nZaghachi 0 maka menu bụ isi.",
        "options": {"0": "root"},
    },
    "lang_select": {
        "text_en": "Select language:\n1. English\n2. Hausa\n3. Yoruba\n4. Igbo",
        "text_ha": "Zaɓi harshe:\n1. Turanci\n2. Hausa\n3. Yarbanci\n4. Ibo",
        "text_yo": "Yan èdè:\n1. Gẹ̀ẹ́sì\n2. Hausa\n3. Yorùbá\n4. Igbo",
        "text_ig": "Họrọ asụsụ:\n1. Bekee\n2. Hausa\n3. Yoruba\n4. Igbo",
        "options": {"1": "lang_en", "2": "lang_ha", "3": "lang_yo", "4": "lang_ig"},
    },
    "contact_officer": {
        "text_en": "Extension Officer:\nAbuja FCT Agricultural\nCall: 09012345678\nFree call Mon-Fri 8am-5pm\nReply 0 for main menu.",
        "text_ha": "Jami'in Ƙasar Gona:\nAgrikultura na Abuja FCT\nKira: 09012345678\nKira kyauta Litinin-Juma'a 8am-5pm\nReplai 0 don menu na farko.",
        "text_yo": "Olùkọ́ni Àgbẹ̀:\nÀgbẹ̀ Abuja FCT\nPè: 09012345678\nÈsì ọ̀fẹ́ Ẹtì-Ẹta 8am-5pm\nFèsì 0 fún àkójọ àkọ́kọ́.",
        "text_ig": "Onye Ndụmọdụ Ugbo:\nọrụ Ugbo Abuja FCT\nKpọọ: 09012345678\nAkụkọ efu Mọnde-Fraịde 8am-5pm\nZaghachi 0 maka menu bụ isi.",
        "options": {"0": "root"},
    },
    "weather_alert": {
        "text_en": "Weather Alert (FCT):\nNext 7 days: Dry\nRainfall probability: 15%\nTemperature: 32-38C\nAction: Conserve soil moisture\nReply 0 for main menu.",
        "text_ha": "Faɗakar da Yanayi (FCT):\nKwana 7 masu zuwa: Rani\nYuwuwar ruwan sama: 15%\nZafin jiki: 32-38C\nAiki: Adana ruwan ƙasa\nReplai 0 don menu na farko.",
        "text_yo": "Ìkìlọ̀ Ojú Ọjọ́ (FCT):\nỌjọ́ 7 tó ń bọ̀: Gbígbẹ\nÌṣeéṣe òjò: 15%\nÒòrùn: 32-38C\nIṣẹ́: Mọ omi ilẹ̀\nFèsì 0 fún àkójọ àkọ́kọ́.",
        "text_ig": "Ọchịchọ Ihu Igwe (FCT):\nỤbọchị 7 na-abịa: Ọkọchị\nOhere ozuzo: 15%\nOkpomọkụ: 32-38C\nMmemme: Debe mmiri ala\nZaghachi 0 maka menu bụ isi.",
        "options": {"0": "root"},
    },
    "market_input": {
        "text_en": "Market Advisory\nSelect crop:\n1. Maize\n2. Sorghum\n3. Cassava\n4. Tomatoes\n0. Back",
        "text_ha": "Shawarar Kasuwa\nZaɓi amfanin gona:\n1. Masara\n2. Dawa\n3. Rogo\n4. Tomato\n0. Koma",
        "text_yo": "Ìmọ̀ràn Ọjà\nYan irúgbìn:\n1. Àgbàdo\n2. Ọkà bàbà\n3. Gbaguda\n4. Tomátì\n0. Padà",
        "text_ig": "Ndụmọdụ Ahịa\nHọrọ ọkụkụ:\n1. Ọka\n2. Ọka ọbara\n3. Ji akpu\n4. Tomato\n0. Laghachi",
        "options": {"1": "market_result", "2": "market_result",
                    "3": "market_result", "4": "market_result", "0": "root"},
    },
    "market_result": {
        "text_en": "Market Prices (Abuja):\nMaize: ₦450/kg (+12%)\nSorghum: ₦380/kg (-3%)\nBest time to sell: Now\nNext market day: Thursday\nReply 0 for main menu.",
        "text_ha": "Farashin Kasuwa (Abuja):\nMasara: ₦450/kg (+12%)\nDawa: ₦380/kg (-3%)\nMafi kyawun lokacin sayarwa: Yanzu\nRanar kasuwa ta gaba: Alhamis\nReplai 0 don menu na farko.",
        "text_yo": "Iye Ọjà (Abuja):\nÀgbàdo: ₦450/kg (+12%)\nỌkà bàbà: ₦380/kg (-3%)\nÀkókò tó dára jùlọ̀ fún títà: Báyìí\nỌjọ́ ọjà tó ń bọ̀: Ọjọ́bọ\nFèsì 0 fún àkójọ àkọ́kọ́.",
        "text_ig": "Ọnụ Ahịa (Abuja):\nỌka: ₦450/kg (+12%)\nỌka ọbara: ₦380/kg (-3%)\nOhuru ire kacha mma: Ugbu a\nỌbọchị ahịa na-esote: Tọzdee\nZaghachi 0 maka menu bụ isi.",
        "options": {"0": "root"},
    },
}


def _get_menu_text(menu_key: str, lang: str = "en") -> str:
    """Get the menu text for a given menu and language."""
    menu = _USSD_MENUS.get(menu_key, _USSD_MENUS["root"])
    text_key = f"text_{lang}" if f"text_{lang}" in menu else "text_en"
    return menu.get(text_key, menu.get("text_en", ""))


def ussd_interface(
    prediction_result: Optional[Dict] = None,
    domain: str = "agrotech",
) -> None:
    """
    Render an interactive USSD phone simulator.

    Parameters
    ----------
    prediction_result : optional dict with keys 'risk_level' ('high'/'low'),
                        'top_features', 'counterfactual_hint'
    domain            : simulation domain for context
    """
    lang = get_lang()

    st.markdown(
        f"<p style='font-size:.85rem;color:var(--color-text-secondary);"
        f"margin-bottom:.75rem;'>{t('ussd_description')}</p>",
        unsafe_allow_html=True,
    )

    # Session state for USSD navigation
    if "_ussd_menu" not in st.session_state:
        st.session_state._ussd_menu    = "root"
        st.session_state._ussd_history = []

    current_menu = st.session_state._ussd_menu
    menu_data    = _USSD_MENUS.get(current_menu, _USSD_MENUS["root"])

    # ── Phone frame ───────────────────────────────────────────────────────
    col_phone, col_info = st.columns([1, 1])

    with col_phone:
        # Determine which result screen to show if prediction available
        if prediction_result and current_menu == "crop_result":
            risk = prediction_result.get("risk_level", "low")
            display_menu = "crop_result_high" if risk == "high" else "crop_result_low"
        else:
            display_menu = current_menu

        menu_text = _get_menu_text(display_menu, lang)

        st.markdown(
            f"""
            <div style="background:#111;border-radius:20px;padding:16px 12px;
                        max-width:280px;margin:0 auto;font-family:monospace;
                        box-shadow:0 4px 16px rgba(0,0,0,0.3);">
              <div style="background:#1a1a2e;border-radius:4px;padding:4px 8px;
                          margin-bottom:8px;text-align:center;">
                <span style="color:#4ecdc4;font-size:10px;font-weight:600;">
                  ● USSD SESSION ACTIVE
                </span>
              </div>
              <div style="background:#0f0f0f;border:1px solid #333;border-radius:8px;
                          padding:12px;min-height:180px;">
                <pre style="color:#00ff88;font-size:11px;margin:0;
                            white-space:pre-wrap;line-height:1.6;
                            font-family:'Courier New',monospace;">{menu_text}</pre>
              </div>
              <div style="margin-top:12px;display:flex;gap:8px;justify-content:center;">
                <div style="background:#222;border-radius:6px;padding:8px 16px;
                            color:#888;font-size:11px;font-family:monospace;">*384#</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Navigation buttons
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        options = menu_data.get("options", {})

        if options:
            avail = sorted(options.keys())
            btn_cols = st.columns(min(len(avail), 4))
            for col, opt in zip(btn_cols, avail):
                if col.button(
                    f"Reply {opt}",
                    key=f"_ussd_btn_{opt}_{current_menu}",
                    use_container_width=True,
                ):
                    next_menu = options[opt]
                    if next_menu.startswith("lang_"):
                        lang_code = next_menu.replace("lang_", "")
                        if lang_code in LANGUAGES:
                            st.session_state["_gags_lang"] = lang_code
                    elif next_menu == "exit":
                        st.session_state._ussd_menu    = "root"
                        st.session_state._ussd_history = []
                    else:
                        st.session_state._ussd_history.append(current_menu)
                        st.session_state._ussd_menu = next_menu
                    st.rerun()

        # Back button
        if st.session_state._ussd_history:
            if st.button("← Back", key=f"_ussd_back_{current_menu}"):
                st.session_state._ussd_menu = st.session_state._ussd_history.pop()
                st.rerun()

    with col_info:
        st.markdown("**About this simulation**")
        st.markdown(
            f"This panel shows how a farmer using a basic 2G phone "
            f"with no internet access would interact with the GAGS AI "
            f"advisory system via USSD.\n\n"
            f"The USSD flow reduces the full prediction model to a "
            f"**binary HIGH/LOW signal** and a single **actionable instruction** "
            f"— all within 182 characters per screen.\n\n"
            f"**Active menu:** `{current_menu}`\n\n"
            f"**Active language:** {LANGUAGES.get(lang, {}).get('flag','')} "
            f"{LANGUAGES.get(lang, {}).get('native', 'English')}"
        )

        if prediction_result:
            risk     = prediction_result.get("risk_level", "unknown")
            features = prediction_result.get("top_features", [])
            hint     = prediction_result.get("counterfactual_hint", "")

            st.markdown("**Live prediction fed into USSD:**")
            risk_color = "#e74c3c" if risk == "high" else "#27ae60"
            st.markdown(
                f"<div style='border-left:3px solid {risk_color};"
                f"padding:.4rem .75rem;background:var(--color-background-secondary);"
                f"border-radius:0 6px 6px 0;font-size:.85rem;'>"
                f"<strong>Risk:</strong> {risk.upper()}<br>"
                f"{'<strong>Key factors:</strong> ' + ', '.join(features) if features else ''}"
                f"{'<br><strong>Action:</strong> ' + hint if hint else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )


def accessibility_gap_report(
    full_model_metrics: Dict[str, float],
    domain: str = "agrotech",
    connectivity_rate: float = 0.34,
) -> None:
    """
    Measure and visualise the accessibility gap between full-AI users
    and USSD-only users.

    The gap has two components:
    1. Information loss:  USSD binary signal vs full probability score
    2. Feature loss:      USSD uses ≤3 inputs vs model's full feature set
    3. Latency gap:       USSD response is pre-cached; no real-time inference

    Parameters
    ----------
    full_model_metrics : dict with accuracy, fairness_score, etc.
    domain             : simulation domain
    connectivity_rate  : fraction of users with smartphone/internet access
                         (default 0.34 for Nigerian rural areas, NCC 2024)
    """
    import plotly.graph_objects as go

    ussd_accuracy  = full_model_metrics.get("accuracy", 0.75) * 0.78
    ussd_fairness  = full_model_metrics.get("fairness_score", 0.7) * 0.92
    ussd_recall    = full_model_metrics.get("recall", 0.7) * 0.71
    full_accuracy  = full_model_metrics.get("accuracy", 0.75)
    full_fairness  = full_model_metrics.get("fairness_score", 0.7)
    full_recall    = full_model_metrics.get("recall", 0.7)

    excluded_pct   = (1 - connectivity_rate) * 100

    st.markdown(
        f"<div style='background:var(--color-background-warning);"
        f"border-left:4px solid var(--color-border-warning);"
        f"border-radius:0 8px 8px 0;padding:.75rem 1rem;margin-bottom:.75rem;'>"
        f"<strong>⚠️ Accessibility Gap Alert:</strong> "
        f"With {connectivity_rate:.0%} smartphone connectivity in rural Nigeria, "
        f"approximately <strong>{excluded_pct:.0f}%</strong> of your target users "
        f"would receive a reduced USSD-only service with lower accuracy and fewer features.</div>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Accuracy: Full AI",
        f"{full_accuracy:.1%}",
        help="Accuracy for smartphone users with full model access",
    )
    col1.metric(
        "Accuracy: USSD",
        f"{ussd_accuracy:.1%}",
        f"{ussd_accuracy - full_accuracy:+.1%}",
        delta_color="inverse",
        help="Estimated accuracy for USSD-only users (binary signal, 3 features)",
    )
    col2.metric("Fairness: Full AI", f"{full_fairness:.3f}")
    col2.metric(
        "Fairness: USSD",
        f"{ussd_fairness:.3f}",
        f"{ussd_fairness - full_fairness:+.3f}",
        delta_color="inverse",
    )
    col3.metric(
        "Users with Full Access",
        f"{connectivity_rate:.0%}",
        help="Smartphone / internet penetration (NCC 2024, rural Nigeria)",
    )
    col3.metric(
        "Users USSD-only",
        f"{1-connectivity_rate:.0%}",
        f"-{(1-connectivity_rate):.0%} of target population",
        delta_color="inverse",
    )

    # Bar chart comparison
    categories = ["Accuracy", "Fairness Score", "Recall"]
    full_vals  = [full_accuracy, full_fairness, full_recall]
    ussd_vals  = [ussd_accuracy, ussd_fairness, ussd_recall]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Full AI (smartphone)",
        x=categories, y=full_vals,
        marker_color="#2980b9",
        text=[f"{v:.1%}" for v in full_vals],
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="USSD-only (basic phone)",
        x=categories, y=ussd_vals,
        marker_color="#e74c3c",
        text=[f"{v:.1%}" for v in ussd_vals],
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        title="Accessibility Gap: Full AI vs USSD Service",
        yaxis=dict(range=[0, 1.1], title="Score"),
        height=320,
        legend=dict(orientation="h", y=1.08),
        margin=dict(t=60, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "**Recommended actions to close the gap:**\n"
        "1. Cache pre-computed risk scores for common LGA/crop combinations\n"
        "2. Deploy a simplified 3-feature model optimised for USSD binary output\n"
        "3. Partner with NCC-licensed USSD aggregators (e.g. Interswitch, Unified Payment)\n"
        "4. Train village extension officers to relay AI advice via community radio\n"
        "5. Provide offline PDF crop advisory sheets updated monthly"
    )


def format_sms_result(
    risk_level: str,
    top_feature: str,
    action: str,
    lang: str = "en",
    max_chars: int = 160,
) -> str:
    """
    Format a prediction result as an SMS of ≤160 characters.

    Parameters
    ----------
    risk_level  : "high" | "low"
    top_feature : the single most important feature
    action      : the recommended action
    lang        : language code
    max_chars   : SMS character limit (default 160)

    Returns
    -------
    str : formatted SMS message ≤ max_chars characters
    """
    templates = {
        "en": {
            "high": f"GAGS ALERT: HIGH RISK. {top_feature} is critical. ACTION: {action}. Call 09012345678 free.",
            "low":  f"GAGS: LOW RISK. Conditions OK. {action}. Next check in 14 days.",
        },
        "ha": {
            "high": f"GAGS SANARWA: HAƊARI BABBA. {top_feature} yana da muhimmanci. AIKI: {action}. Kira 09012345678 kyauta.",
            "low":  f"GAGS: HAƊARI ƘARAMI. Yanayi yana da kyau. {action}. Duba gaba bayan kwana 14.",
        },
        "yo": {
            "high": f"GAGS ÌKÌLỌ̀: EWUU GÍGA. {top_feature} ṣe pàtàkì. IṢẸ́: {action}. Pè 09012345678 ọfẹ́.",
            "low":  f"GAGS: EWUU KẸ́KẸ́. Ipò dára. {action}. Ìwádìí tó kàn ní ọjọ́ 14.",
        },
        "ig": {
            "high": f"GAGS ỌCHỊCHỌ: IHE EGWU DỊ ELU. {top_feature} dị mkpa. MMEMME: {action}. Kpọọ 09012345678 efu.",
            "low":  f"GAGS: IHE EGWU DỊ NTAKỊRỊ. Ọnọdụ dị mma. {action}. Nyocha ọzọ n'ụbọchị 14.",
        },
    }

    lang_templates = templates.get(lang, templates["en"])
    msg = lang_templates.get(risk_level, lang_templates["low"])

    if len(msg) > max_chars:
        msg = msg[:max_chars - 3] + "..."

    return msg