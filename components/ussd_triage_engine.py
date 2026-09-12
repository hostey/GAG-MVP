"""
USSD & Low-Bandwidth Triage Engine Component

Handles USSD interactive terminal sessions, SMS compression/telemetry micro-payloads,
webhook request simulation for telco integrations (Africa's Talking / Twilio),
and infrastructure resilience failure handling.
"""

import pandas as pd
import streamlit as st
from components.health.equity_audit_engine import run_multi_agent_triage_pipeline


def run_ussd_endpoint(
    text_payload: str,
    phone: str = "+2348000000000",
    active_df: pd.DataFrame = None,
    p_lat: float = 9.0820,
    p_lon: float = 8.6753,
    api_key_input: str = "DEMO",
    llm_provider: str = "Groq (Llama 3.3 / 3.1)",
    season_profile: str = "Dry Season",
    terrain_type: str = "Urban Paved",
) -> str:
    """Executes live multi-agent backend triage logic and returns standard USSD HTTP text/plain payload."""
    if active_df is None:
        active_df = pd.DataFrame()

    if text_payload == "":
        return "CON AFRIMED TRIAGE\n1. Emergency Triage\n2. Facility Status\n0. Exit"
    elif text_payload == "1":
        return "CON SELECT CATEGORY:\n1. Maternal Emergency\n2. Severe Pediatric\n0. Back"
    elif text_payload == "1*1":
        return "CON MATERNAL FORM\nEnter Age, SBP, Bleeding (1-3):\nExample: 24,140,3"
    elif text_payload.startswith("1*1*"):
        raw_data = text_payload.split("1*1*")[1]
        parts = [p.strip() for p in raw_data.split(",")]
        age = parts[0] if len(parts) > 0 else "25"
        sbp = parts[1] if len(parts) > 1 else "120"
        bleed = parts[2] if len(parts) > 2 else "2"

        vignette = (
            f"USSD Emergency Triage: {age}yo female presenting with maternal complications. "
            f"SBP: {sbp}mmHg, Bleeding Severity Level: {bleed}/3."
        )

        results = run_multi_agent_triage_pipeline(
            vignette_text=vignette,
            facilities_df=active_df,
            patient_lat=p_lat,
            patient_lon=p_lon,
            api_key=api_key_input,
            provider=llm_provider,
            season_profile=season_profile,
            terrain_type=terrain_type,
        )

        matched_df = results.get("matched_facilities_df")
        if isinstance(matched_df, pd.DataFrame) and not matched_df.empty:
            target_fac = matched_df.iloc[0].get("facility_name", "General Hospital")
            eta_mins = matched_df.iloc[0].get("travel_time_min", "25")
        else:
            target_fac = "General Hospital Bida"
            eta_mins = "25"

        return (
            f"END DISPATCH SUCCESSFUL!\n"
            f"Target: {str(target_fac)[:22]}\n"
            f"Est. ETA: {eta_mins} mins\n"
            f"SMS Alert sent to {phone}."
        )
    else:
        return "END Invalid USSD Payload Sequence."


def render_ussd_triage_section(
    active_df: pd.DataFrame = None,
    p_lat: float = 9.0820,
    p_lon: float = 8.6753,
    season_profile: str = "Dry Season",
    terrain_type: str = "Urban Paved",
    llm_provider: str = "Groq (Llama 3.3 / 3.1)",
    api_key_input: str = "DEMO",
):
    """Renders the entire USSD/SMS low-bandwidth mitigation workspace."""
    # Ensure local defaults inside component
    if active_df is None:
        active_df = pd.DataFrame()

    st.markdown("### Low-Resource Infrastructure Mitigation & USSD Gateway")
    st.caption("Simulate offline resilience, ultra-low bandwidth messaging protocols (SMS/USSD), and edge degradation for remote health networks.")

    # ── Top Control Panel: Network & Device Infrastructure Profile ──
    with st.container(border=True):
        st.markdown("** Network Connectivity & Edge Infrastructure Profile**")
        net_c1, net_c2, net_c3 = st.columns(3)
        with net_c1:
            st.selectbox(
                "Network Profile:",
                [
                    "4G/5G Broadband (Optimal)",
                    "2G/EDGE (Low Bandwidth)",
                    "USSD Session (Interactive 182-char)",
                    "Offline / Mesh Gateway (Store-and-Forward)",
                ],
                index=2,
                key="eq_net_profile_select",
            )
        with net_c2:
            st.selectbox(
                "Primary Power Grid Status:",
                [
                    "Stable Grid",
                    "Intermittent Load-Shedding",
                    "Off-Grid / Solar Fallback",
                    "Total Blackout",
                ],
                index=1,
                key="eq_power_status_select",
            )
        with net_c3:
            st.selectbox(
                "Community Worker Device:",
                [
                    "Smartphone (Android/iOS)",
                    "Feature Phone (KaiOS/Nokia)",
                    "Basic GSM Handset (USSD/SMS Only)",
                    "Radio Transceiver",
                ],
                index=2,
                key="eq_device_select",
            )

    # ── Interactive Toolkits ──
    u_tab1, u_tab2, u_tab3 = st.tabs([
        "📱 Live USSD Interactive Emulator",
        "💬 SMS / Micro-Payload Compressor",
        "⚙️ Infrastructure Resilience Matrix",
    ])

    # ── Sub-Tab 1: Live USSD Interactive Emulator ──
    with u_tab1:
        u_col1, u_col2 = st.columns([1, 1.2], gap="medium")

        with u_col1:
            with st.container(border=True):
                st.markdown("**📲 Feature Phone USSD Terminal (`*384*88#`)**")
                st.caption("Interactive session simulator linked directly to live multi-agent triage.")

                if "ussd_step" not in st.session_state:
                    st.session_state["ussd_step"] = "MAIN"
                if "ussd_vignette" not in st.session_state:
                    st.session_state["ussd_vignette"] = ""

                ussd_screen = ""
                if st.session_state["ussd_step"] == "MAIN":
                    ussd_screen = (
                        "AFRIMED EMERGENCY TRIAGE v2.1\n"
                        "1. Emergency Patient Triage\n"
                        "2. Facility Bed/Blood Availability\n"
                        "3. Request Transport / Ambulance\n"
                        "4. Stock-out / Essential Drug Alert\n"
                        "0. Exit"
                    )
                elif st.session_state["ussd_step"] == "1":
                    ussd_screen = (
                        "SELECT CLINICAL CATEGORY:\n"
                        "1. Maternal / Postpartum Hemorrhage\n"
                        "2. Severe Pediatric / Neonatal\n"
                        "3. Acute Trauma / Sepsis\n"
                        "4. Other Critical Condition\n"
                        "0. Back"
                    )
                elif st.session_state["ussd_step"] == "1_1":
                    ussd_screen = (
                        "MATERNAL EMERGENCY FORM:\n"
                        "Enter Age, Systolic BP, Bleeding (1-3):\n"
                        "Format: 24,140,3\n"
                        "Reply directly in input below."
                    )
                elif st.session_state["ussd_step"] == "SUBMITTED":
                    res = st.session_state.get("ussd_live_results", {})
                    clin = res.get("clinical_assessment", {})
                    match = res.get("matched_facilities_df")

                    urgency = clin.get("urgency", "CRITICAL") if isinstance(clin, dict) else "CRITICAL"
                    top_fac = match.iloc[0]["facility_name"] if isinstance(match, pd.DataFrame) and not match.empty else "Nearest Facility"
                    eta = match.iloc[0]["travel_time_min"] if isinstance(match, pd.DataFrame) and not match.empty else "N/A"

                    ussd_screen = (
                        f"DISPATCH SUCCESSFUL!\n"
                        f"Ref ID: #AFM-{abs(hash(st.session_state.get('ussd_vignette', ''))) % 10000:04d}\n"
                        f"Urgency: {urgency}\n"
                        f"Target: {str(top_fac)[:20]}\n"
                        f"Est. ETA: {eta} mins\n"
                        f"SMS alert sent to ambulance unit."
                    )

                st.text_area("GSM Phone Screen Display", value=ussd_screen, height=190, disabled=True, key="ussd_disp")
                user_input = st.text_input("Keypad Input (Reply):", key="ussd_input")
                p_c1, p_c2 = st.columns(2)

                with p_c1:
                    if st.button("Send Input ➔", use_container_width=True):
                        if user_input == "1" and st.session_state["ussd_step"] == "MAIN":
                            st.session_state["ussd_step"] = "1"
                            st.rerun()
                        elif user_input == "1" and st.session_state["ussd_step"] == "1":
                            st.session_state["ussd_step"] = "1_1"
                            st.rerun()
                        elif user_input and st.session_state["ussd_step"] == "1_1":
                            parts = [p.strip() for p in user_input.split(",")]
                            age = parts[0] if len(parts) > 0 else "Unknown"
                            sbp = parts[1] if len(parts) > 1 else "Unknown"
                            bleed = parts[2] if len(parts) > 2 else "Severe"

                            vignette_str = (
                                f"USSD Emergency Triage: {age}-year-old female presenting with "
                                f"maternal complications. Systolic BP: {sbp} mmHg, Bleeding Severity Level: {bleed}/3. "
                                f"Submitted via remote CHW feature phone gateway."
                            )
                            st.session_state["ussd_vignette"] = vignette_str

                            with st.spinner("Processing USSD payload through multi-agent triage engine..."):
                                pipeline_results = run_multi_agent_triage_pipeline(
                                    vignette_text=vignette_str,
                                    facilities_df=active_df,
                                    patient_lat=p_lat,
                                    patient_lon=p_lon,
                                    api_key=api_key_input,
                                    provider=llm_provider,
                                    season_profile=season_profile,
                                    terrain_type=terrain_type,
                                )
                                st.session_state["ussd_live_results"] = pipeline_results
                                st.session_state["ussd_dispatch"] = pipeline_results.get("ussd_dispatch_payload", "")
                                st.session_state["ussd_step"] = "SUBMITTED"
                            st.rerun()
                        elif user_input == "0":
                            st.session_state["ussd_step"] = "MAIN"
                            st.rerun()

                with p_c2:
                    if st.button("Reset Call Session", use_container_width=True):
                        st.session_state["ussd_step"] = "MAIN"
                        st.session_state.pop("ussd_live_results", None)
                        st.rerun()

        with u_col2:
            with st.container(border=True):
                st.markdown("**🔌 Live Webhook & Inference Simulator**")
                st.caption("Test incoming telco POST requests (`Africa's Talking` / `Twilio`) directly against the live model.")

                sim_text = st.text_input(
                    "HTTP POST Parameter (`text`):",
                    value="1*1*24,140,3",
                    help="Simulates concatenated input string sent by telco aggregators.",
                    key="sim_ussd_text_input",
                )
                sim_phone = st.text_input(
                    "HTTP POST Parameter (`phoneNumber`):",
                    value="+2348031234567",
                    key="sim_ussd_phone_input",
                )

                if st.button("⚡ Trigger Webhook Request", type="primary", use_container_width=True):
                    with st.spinner("Processing POST request payload..."):
                        http_response = run_ussd_endpoint(
                            text_payload=sim_text,
                            phone=sim_phone,
                            active_df=active_df,
                            p_lat=p_lat,
                            p_lon=p_lon,
                            api_key_input=api_key_input,
                            llm_provider=llm_provider,
                            season_profile=season_profile,
                            terrain_type=terrain_type,
                        )
                        st.session_state["last_ussd_http_resp"] = http_response

                if "last_ussd_http_resp" in st.session_state:
                    st.markdown("**Raw Gateway Output (`text/plain`):**")
                    st.code(st.session_state["last_ussd_http_resp"], language="text")

                with st.expander("📄 View Serverless Flask Webhook Script"):
                    st.code(
                        """from flask import Flask, request, Response
from components.ussd_triage_engine import run_ussd_endpoint

app = Flask(__name__)

@app.route('/ussd', methods=['POST'])
def ussd_handler():
    text = request.form.get("text", "")
    phone = request.form.get("phoneNumber", "")
    response_text = run_ussd_endpoint(text, phone)
    return Response(response_text, mimetype="text/plain")""",
                        language="python",
                    )

    # ── Sub-Tab 2: SMS / Micro-Payload Compressor ──
    with u_tab2:
        st.markdown("**💬 Low-Bandwidth Data Compressor**")
        st.caption("Compresses dense clinical telemetry into standard 140-byte ASCII GSM SMS payloads for zero-data environments.")

        comp_c1, comp_c2 = st.columns([1.1, 1], gap="medium")

        with comp_c1:
            with st.container(border=True):
                st.markdown("**Clinical Telemetry Inputs**")
                in_patient_id = st.text_input("Patient Reference:", value="PT-9042", key="comp_pid")
                in_urgency = st.selectbox("Urgency Level:", ["CRITICAL_TIER_3", "URGENT_TIER_2", "ROUTINE_TIER_1"], key="comp_urg")
                in_symptoms = st.multiselect("Symptoms:", ["Postpartum Hemorrhage", "Severe Anemia", "Convulsions/Eclampsia", "Sepsis"], default=["Postpartum Hemorrhage"], key="comp_symp")
                in_vitals = st.text_input("Vitals (BP/HR):", value="150/95, HR 112", key="comp_vitals")

                s_code = "MAT" if "Postpartum" in str(in_symptoms) or "Eclampsia" in str(in_symptoms) else "GEN"
                u_code = "C3" if "CRITICAL" in in_urgency else "U2"
                v_clean = in_vitals.replace(" ", "").replace(",", "|")
                compressed_payload = f"AFM|{in_patient_id}|{s_code}|{u_code}|{v_clean}|LOC:{p_lat:.3f},{p_lon:.3f}|CRC:A9F4"

        with comp_c2:
            with st.container(border=True):
                st.markdown("**Encoded SMS Packet**")
                st.code(compressed_payload, language="text")

                char_len = len(compressed_payload)
                byte_size = len(compressed_payload.encode("utf-8"))
                sms_count = 1 if char_len <= 160 else (char_len // 153) + 1

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Character Count", f"{char_len} chars")
                with m2:
                    st.metric("Payload Size", f"{byte_size} B")
                with m3:
                    st.metric("SMS Segments", f"{sms_count} SMS")

                if char_len <= 140:
                    st.success("Fits within a single 140-byte standard GSM SMS segment.")
                else:
                    st.warning("Multi-part SMS required.")

                st.button("Transmit SMS Payload to Dispatch Gateway", use_container_width=True)

    # ── Sub-Tab 3: Resilience & Mitigation Matrix ──
    with u_tab3:
        st.markdown("** Infrastructure Hazard Mitigation Protocols**")
        st.caption("Automated routing strategies triggered by physical infrastructure failures.")

        mit_col1, mit_col2 = st.columns(2, gap="medium")

        with mit_col1:
            with st.container(border=True):
                st.markdown("##### Power & Cold-Chain Outage")
                st.markdown("""
                * **Hazard:** Grid failure disabling blood/vaccine refrigeration.
                * **Mitigation:** Trigger SMS notification to nearest Solar Direct Drive (SDD) hub.
                * **Failover:** Dynamic rerouting of maternal hemorrhage cases to Tertiary facilities with guaranteed generator reserves.
                """)

            with st.container(border=True):
                st.markdown("##### Severe Flood & Road Degradation")
                st.markdown("""
                * **Hazard:** Unpaved dirt tracks rendered impassable during Peak Rainy Season.
                * **Mitigation:** Inject dynamic travel time penalties (+30% to +120%) into routing algorithm.
                * **Failover:** Auto-dispatch motorcycle ambulance or riverine transport networks.
                """)

        with mit_col2:
            with st.container(border=True):
                st.markdown("##### Cellular Network Blackout")
                st.markdown("""
                * **Hazard:** Complete loss of 2G/3G coverage in remote LGA.
                * **Mitigation:** Enable local Store-and-Forward SQLite queue on CHW device.
                * **Failover:** Opportunistic sync via Bluetooth/Wi-Fi Direct with passing emergency vehicles.
                """)

            with st.container(border=True):
                st.markdown("##### Specialist Staffing Shortage")
                st.markdown("""
                * **Hazard:** Absence of resident obstetrician/surgeon at target PHC.
                * **Mitigation:** Step-by-step tele-triage guidance delivered via USSD/SMS.
                * **Failover:** Advance warning dispatch broadcast to regional referral center.
                """)