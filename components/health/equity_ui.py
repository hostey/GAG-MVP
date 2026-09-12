import streamlit as st


# ── 2. Equity Tab (World‑Class UX Refactor) ──────────────────────────────
with T["⚖️ Equity"]:
    # ---------------------------------------------------------------------
    #   Global Styles – Cohesive design tokens for the entire equity module
    # ---------------------------------------------------------------------
    st.markdown(
        """
        <style>
        /* Base font & background */
        .eq-container {
            font-family: 'Inter', 'Segoe UI', sans-serif;
            color: #1e293b;
        }
        /* Hero header */
        .eq-hero {
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            border: 1px solid #334155;
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.5rem;
            color: #f8fafc;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        }
        .eq-badge {
            background: #0891b2;
            color: #ffffff;
            font-size: 0.72rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 20px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-block;
        }
        .eq-metric-box {
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 0.75rem;
            text-align: center;
        }
        /* Section headers */
        .eq-section-title {
            font-size: 1.2rem;
            font-weight: 700;
            color: #0f172a;
            margin-bottom: 0.5rem;
        }
        /* Card containers */
        .eq-card {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.25rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
            margin-bottom: 1rem;
        }
        /* Buttons (override Streamlit primary) */
        .stButton > button[kind="primary"] {
            background-color: #0891b2;
            color: white;
            border-radius: 8px;
            font-weight: 600;
            border: none;
            transition: all 0.2s;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #0e7490;
            box-shadow: 0 2px 8px rgba(8,145,178,0.3);
        }
        /* File uploader & inputs */
        .stFileUploader > div > div {
            border-radius: 8px;
        }
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px 8px 0 0;
            padding: 0.6rem 1.2rem;
            font-weight: 500;
            background: #f1f5f9;
        }
        .stTabs [aria-selected="true"] {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-bottom-color: transparent;
            color: #0891b2;
        }
        /* Dataframe styling */
        .stDataFrame {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid #e2e8f0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


    # ---------------------------------------------------------------------
    #   Helper Functions (kept outside to maintain readability)
    # ---------------------------------------------------------------------
    def render_hero_header():
        st.markdown(
            """
            <div class="eq-hero">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <span class="eq-badge">Spatial & Access Intelligence Engine</span>
                    <span style="font-size: 0.8rem; color: #94a3b8;">WHO / OSRM Integrated Framework</span>
                </div>
                <h2 style="margin: 0; font-size: 1.7rem; font-weight: 800; color: #ffffff;">Healthcare Equity & Spatial Infrastructure Audit</h2>
                <p style="margin: 0.4rem 0 0 0; font-size: 0.9rem; color: #cbd5e1; max-width: 850px;">
                    Simulate real‑world transfer barriers, severe weather penalties, and multi‑agent clinical triage feasibility across Sub‑Saharan infrastructure grids.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


    def render_global_controls():
        st.markdown('<div class="eq-card">', unsafe_allow_html=True)
        st.markdown("**⚙️ Global Audit & Transport Calibration**")
        cfg_c1, cfg_c2, cfg_c3, cfg_c4 = st.columns([1.1, 1, 1, 1.1], gap="medium")

        with cfg_c1:
            location_mode = st.radio(
                "Patient Origin Selector:",
                ["Default Coordinates", "Select from Uploaded Data"],
                horizontal=True,
                key="location_mode",
            )
            p_lat, p_lon = 9.0820, 8.6753

        with cfg_c2:
            season_profile = st.selectbox(
                "🌦️ Weather Profile:",
                ["Dry Season", "Peak Rainy Season"],
                help="Rainy season adds dynamic travel time penalties (+30% to +120%) due to unpaved road degradation.",
                key="season_profile",
            )

        with cfg_c3:
            terrain_type = st.selectbox(
                "🛣️ Terrain Profile:",
                ["Urban Paved", "Rural Unpaved", "Off-Road Terrain"],
                key="terrain_type",
            )
            avg_speed_kmh = (
                50.0 if terrain_type == "Urban Paved"
                else 30.0 if terrain_type == "Rural Unpaved"
                else 15.0
            )

        with cfg_c4:
            llm_provider = st.selectbox(
                "LLM Inference Engine:",
                ["Groq (Llama 3.3 / 3.1)", "Hugging Face (Mistral)", "OpenAI (GPT-4o-mini)"],
                key="llm_provider",
            )
            api_key_input = st.text_input(
                "API Key (or leave empty for Demo):",
                value="DEMO",
                type="password",
                key="api_key_input",
            )
        st.markdown('</div>', unsafe_allow_html=True)
        return location_mode, p_lat, p_lon, season_profile, terrain_type, avg_speed_kmh, llm_provider, api_key_input


    def render_dataset_spatial_tab(active_df, location_mode, p_lat, p_lon):
        st.markdown("**📂 Step 1: Upload & Schema Alignment**")
        with st.container(border=True):
            up_col1, up_col2 = st.columns(2)
            with up_col1:
                facility_file = st.file_uploader(
                    "Upload Health Facilities (.xlsx, .csv)",
                    type=["xlsx", "xls", "csv"],
                    key="custom_fac_upload",
                )
                raw_fac_df = load_user_dataset(facility_file)
            with up_col2:
                scenario_file = st.file_uploader(
                    "Upload Scenarios/AfriMed-QA (.csv, .xlsx)",
                    type=["csv", "xlsx", "xls"],
                    key="custom_scenario_upload",
                )
                raw_scenario_df = load_user_dataset(scenario_file)

            fac_mapping = {}
            if not raw_fac_df.empty:
                fac_mapping = detect_or_select_columns(raw_fac_df, key_prefix="fac_schema")
                active_df = raw_fac_df.copy()
                active_df["lat"] = pd.to_numeric(active_df[fac_mapping["lat"]], errors="coerce")
                active_df["lon"] = pd.to_numeric(active_df[fac_mapping["lon"]], errors="coerce")
                active_df["facility_name"] = active_df[fac_mapping["name"]].astype(str)
                active_df["facility_tier"] = (
                    active_df[fac_mapping["tier"]].astype(str) if fac_mapping.get("tier") else "Unclassified"
                )
                active_df["state"] = (
                    active_df[fac_mapping["state"]].astype(str) if fac_mapping.get("state") else "Unknown"
                )
                active_df["lga"] = (
                    active_df[fac_mapping["lga"]].astype(str) if fac_mapping.get("lga") else "Unknown"
                )
                active_df = active_df.dropna(subset=["lat", "lon"])
            else:
                active_df = pd.DataFrame()

        # Dynamic origin selection
        if location_mode == "Select from Uploaded Data" and not active_df.empty:
            sel_fac = st.selectbox(
                "Select Origin Facility Anchor:",
                sorted(active_df["facility_name"].unique()),
                key="origin_facility_anchor",
            )
            match_row = active_df[active_df["facility_name"] == sel_fac].iloc[0]
            p_lat, p_lon = float(match_row["lat"]), float(match_row["lon"])

        st.markdown("**🗺️ Step 2: Geospatial Map & Infrastructure Filters**")
        if not active_df.empty:
            with st.container(border=True):
                filter_c1, filter_c2, filter_c3 = st.columns(3, gap="medium")
                filtered_df = active_df.copy()
                with filter_c1:
                    states_list = sorted(active_df["state"].unique()) if "state" in active_df.columns else []
                    selected_states = st.multiselect(
                        "Filter by State/Region:",
                        states_list,
                        key="filter_state_select",
                    )
                    if selected_states:
                        filtered_df = filtered_df[filtered_df["state"].isin(selected_states)]
                with filter_c2:
                    tiers_list = sorted(
                        filtered_df["facility_tier"].unique()) if "facility_tier" in filtered_df.columns else []
                    selected_tiers = st.multiselect(
                        "Filter by Facility Tier:",
                        tiers_list,
                        key="filter_tier_select",
                    )
                    if selected_tiers:
                        filtered_df = filtered_df[filtered_df["facility_tier"].isin(selected_tiers)]
                with filter_c3:
                    if "lga" in filtered_df.columns:
                        selected_lgas = st.multiselect(
                            "Filter by LGA:",
                            sorted(filtered_df["lga"].unique()),
                            key="filter_lga_select",
                        )
                        if selected_lgas:
                            filtered_df = filtered_df[filtered_df["lga"].isin(selected_lgas)]

                fig_map = px.scatter_mapbox(
                    filtered_df,
                    lat="lat",
                    lon="lon",
                    hover_name="facility_name",
                    color="facility_tier",
                    zoom=5,
                    mapbox_style="carto-positron",
                    opacity=0.8,
                    height=420,
                )
                fig_map.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
                st.plotly_chart(fig_map, use_container_width=True)
        else:
            st.info(
                "💡 Upload health facility spatial datasets above to unlock interactive geospatial filtering and GIS mapping.")
        return active_df, p_lat, p_lon


    def render_interactive_audit_tab(active_df, p_lat, p_lon, season_profile, terrain_type, llm_provider, api_key_input,
                                     selected_scenario, active_vignette_text):
        tab1, tab2, tab3 = st.tabs([
            "🩺 Single-Patient Live Audit",
            "📊 Batch Benchmark Audit",
            "🛡️ Infrastructure Mitigation & USSD",
        ])

        with tab1:
            col1, col2 = st.columns([1.1, 1], gap="medium")

            with col1:
                with st.container(border=True):
                    st.markdown("**📋 Vignette & Clinical Profile Configuration**")

                    scenario_mode = st.radio(
                        "Select Scenario Source:",
                        ["⚡ Preset Vignettes", "🤖 AI Synthesizer", "🛠️ Custom Manual Override"],
                        horizontal=True,
                        key="health_scenario_source_radio",
                    )

                    if scenario_mode == "⚡ Preset Vignettes":
                        sel_preset = st.selectbox(
                            "Select Preset Template:",
                            list(CLINICAL_VIGNETTES_PRESETS.keys()),
                            key="preset_template_select",
                        )
                        selected_scenario = sel_preset
                        if st.session_state.get(
                                "last_preset") != sel_preset or "active_vignette_text" not in st.session_state:
                            st.session_state["active_vignette_text"] = CLINICAL_VIGNETTES_PRESETS[sel_preset]["text"]
                            st.session_state["last_preset"] = sel_preset

                    elif scenario_mode == "🤖 AI Synthesizer":
                        p_col1, p_col2, p_col3 = st.columns(3)
                        with p_col1:
                            g_spec = st.selectbox("Target Specialty:", CLINICAL_SPECIALTIES, key="syn_spec_select")
                        with p_col2:
                            g_cohort = st.selectbox(
                                "Patient Cohort & Gender:",
                                ["Maternal / Pregnant Female", "Pediatric (Under 5)", "Neonatal Emergency",
                                 "Adult Female (Reproductive Age)", "Adult Male (Trauma/Acute)",
                                 "Geriatric Patient (>65 yrs)"],
                                key="syn_cohort_select",
                            )
                        with p_col3:
                            g_complexity = st.selectbox("Clinical Urgency:", ["Standard Care", "High-Risk / Acute",
                                                                              "Critical / Resuscitative"],
                                                        key="syn_comp_select")

                        p_col4, p_col5, p_col6 = st.columns(3)
                        with p_col4:
                            g_setting = st.selectbox("Geospatial Context:",
                                                     ["Rural Remote LGA", "Peri-Urban Informal Settlement",
                                                      "Riverine / Island Territory", "Border / Conflict-Affected Zone"],
                                                     key="syn_setting_select")
                        with p_col5:
                            g_care_point = st.selectbox("Initial Point of Care:",
                                                        ["Primary Health Centre (PHC)", "Community Health Post",
                                                         "Faith-Based / Mission Clinic",
                                                         "Traditional Birth Attendant (TBA) Home",
                                                         "Private Patent Medicine Vendor (PPMV)"],
                                                        key="syn_care_point_select")
                        with p_col6:
                            g_time_window = st.selectbox("Presentation Window:",
                                                         ["Daytime Regular Hours (08:00 - 16:00)",
                                                          "Evening Shift (16:00 - 00:00)",
                                                          "Nighttime Emergency (00:00 - 06:00)",
                                                          "Peak Weekend / Holiday"], key="syn_time_window_select")

                        with st.expander("🛠️ Facility Deficits & Modifiers", expanded=False):
                            adv_col1, adv_col2 = st.columns(2)
                            with adv_col1:
                                g_constraints = st.multiselect("Facility Deficits:", ["No Functional Blood Bank",
                                                                                      "No Emergency Surgical Theater",
                                                                                      "No On-Site Oxygen Concentrators",
                                                                                      "Lack of Specialist Medical Staff",
                                                                                      "Intermittent Solar Power Only",
                                                                                      "Out-of-Stock Essential Medications"],
                                                               default=["No Functional Blood Bank"],
                                                               key="syn_deficits_select")
                                g_transport = st.multiselect("Transport Barriers:",
                                                             ["Unpaved Dirt Tracks / Mud Degradation",
                                                              "Seasonal River Flooding / Boat Required",
                                                              "No Local Ambulance Service Available",
                                                              "Curfew / Nighttime Travel Insecurity"],
                                                             default=["Unpaved Dirt Tracks / Mud Degradation"],
                                                             key="syn_transport_select")
                            with adv_col2:
                                g_vulnerabilities = st.multiselect("Patient Vulnerabilities:",
                                                                   ["Low Household Income / Out-of-Pocket Payment",
                                                                    "Uninsured / Non-NHIA Enrollee",
                                                                    "Severe Pre-existing Anemia / Malnutrition",
                                                                    "Language / Ethnic Minority Barrier",
                                                                    "No Male Escort (Sociocultural Constraint)"],
                                                                   default=[
                                                                       "Low Household Income / Out-of-Pocket Payment"],
                                                                   key="syn_vuln_select")

                        selected_scenario = f"AI Vignette: {g_spec} ({g_cohort} at {g_setting})"

                        if st.button("✨ Synthesize Dynamic Vignette", use_container_width=True):
                            with st.spinner(f"Synthesizing clinical case via {llm_provider}..."):
                                generated_text = generate_llm_vignette(
                                    specialty=g_spec,
                                    provider=llm_provider,
                                    api_key=api_key_input,
                                    complexity=g_complexity,
                                    geospatial_setting=g_setting,
                                    patient_cohort=g_cohort,
                                    care_point=g_care_point,
                                    time_window=g_time_window,
                                    facility_constraints=g_constraints,
                                    patient_vulnerabilities=g_vulnerabilities,
                                    transport_barriers=g_transport,
                                )
                                st.session_state["active_vignette_text"] = generated_text
                                st.success("Case presentation synthesized!")

                    else:  # Custom Manual Override
                        selected_scenario = st.text_input("Custom Scenario Name:", value="Custom Healthcare Scenario",
                                                          key="custom_scenario_name_input")

                    st.session_state["selected_scenario"] = selected_scenario

                    active_vignette = st.text_area(
                        "Active Vignette Narrative:",
                        value=st.session_state.get("active_vignette_text", ""),
                        height=130,
                        key="active_vignette_text_area",
                    )
                    st.session_state["active_vignette_text"] = active_vignette

                    btn_col1, btn_col2 = st.columns(2, gap="small")
                    with btn_col1:
                        run_agent_audit = st.button("🚀 Multi-Agent Audit", type="primary", use_container_width=True,
                                                    key="run_agent_audit_btn")
                    with btn_col2:
                        run_spatial_audit = st.button("⚡ Spatial & Clinical Audit", use_container_width=True,
                                                      key="run_spatial_audit_btn")

                    if run_agent_audit:
                        if not active_vignette.strip():
                            st.warning("Please enter or synthesize a vignette first.")
                        else:
                            with st.spinner(f"Running multi-agent pipeline ({season_profile} / {terrain_type})..."):
                                pipeline_results = run_multi_agent_triage_pipeline(
                                    vignette_text=active_vignette,
                                    facilities_df=active_df,
                                    patient_lat=p_lat,
                                    patient_lon=p_lon,
                                    api_key=api_key_input,
                                    provider=llm_provider,
                                    season_profile=season_profile,
                                    terrain_type=terrain_type,
                                )
                                st.session_state["active_llm_out"] = pipeline_results["clinical_assessment"]
                                st.session_state["active_nearest_facs"] = pipeline_results[
                                    "matched_facilities_df"].head(5)
                                st.session_state["ussd_dispatch"] = pipeline_results["ussd_dispatch_payload"]
                                st.session_state["health_ds_report"] = pipeline_results
                            st.success("Audit complete!")

                    if run_spatial_audit:
                        if not active_vignette.strip():
                            st.warning("Please enter or synthesize a vignette first.")
                        else:
                            with st.spinner("Analyzing spatial routing & clinical pathway..."):
                                llm_out = generate_clinical_recommendation(
                                    vignette_text=active_vignette,
                                    api_key=api_key_input,
                                    provider=llm_provider,
                                )
                                st.session_state["active_llm_out"] = llm_out

                                if not active_df.empty:
                                    pipeline_results = run_multi_agent_triage_pipeline(
                                        vignette_text=active_vignette,
                                        facilities_df=active_df,
                                        patient_lat=p_lat,
                                        patient_lon=p_lon,
                                        api_key=api_key_input,
                                        provider=llm_provider,
                                        season_profile=season_profile,
                                        terrain_type=terrain_type,
                                    )
                                    st.session_state["active_nearest_facs"] = pipeline_results[
                                        "matched_facilities_df"].head(5)

                                st.session_state["health_ds_report"] = {
                                    "scenario": selected_scenario,
                                    "vignette": active_vignette,
                                    "llm_recommendation": llm_out,
                                }
                            st.success("Spatial audit completed!")

            # Results Column
            with col2:
                st.markdown("**📊 Audit Intelligence & Diagnostic Feasibility**")
                llm_out_data = st.session_state.get("active_llm_out")

                if llm_out_data:
                    with st.container(border=True):
                        if isinstance(llm_out_data, dict):
                            m_c1, m_c2 = st.columns(2)
                            with m_c1:
                                urgency_val = llm_out_data.get('urgency', 'CRITICAL')
                                st.metric("Clinical Urgency", urgency_val)
                            with m_c2:
                                st.metric("Target Facility Tier", llm_out_data.get('required_tier', 'Tertiary'))

                            st.markdown("---")
                            st.markdown(f"**Summary:** {llm_out_data.get('summary', '')}")

                            if llm_out_data.get("risk_factors"):
                                st.warning("**Risk Factors:** " + ", ".join(llm_out_data.get("risk_factors", [])))
                            if llm_out_data.get("action_plan"):
                                st.success(f"**Action Plan:** {llm_out_data.get('action_plan')}")
                        else:
                            st.info(f"**Recommendation:**\n\n{llm_out_data}")

                nearest_df = st.session_state.get("active_nearest_facs")
                if isinstance(nearest_df, pd.DataFrame) and not nearest_df.empty:
                    with st.container(border=True):
                        st.markdown("**🏥 Matched Referral Facilities & Access Metrics**")
                        target_cols = ["facility_name", "facility_tier", "distance_km", "travel_time_min"]
                        available_cols = [c for c in target_cols if c in nearest_df.columns]
                        st.dataframe(
                            nearest_df[available_cols] if available_cols else nearest_df,
                            use_container_width=True,
                            height=200,
                        )
                elif nearest_df is not None:
                    st.info("ℹ️ Load health facilities to calculate live road network travel times.")

                if st.session_state.get("ussd_dispatch"):
                    with st.container(border=True):
                        st.markdown("**📱 USSD / SMS Low-Bandwidth Dispatch Payload**")
                        st.code(st.session_state["ussd_dispatch"], language="text")

        with tab2:
            st.markdown("#### 📊 Batch Benchmark Analysis")
            st.caption("Batch evaluates datasets for spatial and demographic under-diagnosis risks.")
            if "df_afrimed" in locals() and not df_afrimed.empty:
                st.dataframe(df_afrimed.head(10), use_container_width=True)

        with tab3:
            st.markdown("#### 🛡️ Low-Resource & USSD Fallback Options")
            st.caption("Simulate feature degradation on non-smartphone & offline healthcare delivery channels.")
            ussd_interface()


    # ---------------------------------------------------------------------
    #   Main Equity Tab Flow
    # ---------------------------------------------------------------------
    render_hero_header()

    # Get global controls & patient origin
    location_mode, p_lat, p_lon, season_profile, terrain_type, avg_speed_kmh, llm_provider, api_key_input = render_global_controls()

    # Tabs – Studio Workspace
    equity_studio_tab, dataset_spatial_tab = st.tabs([
        "🩺 Interactive Audit Workbench",
        "📂 Infrastructure Data & Spatial Map",
    ])

    # Dataset & Spatial Map Tab
    with dataset_spatial_tab:
        active_df, p_lat, p_lon = render_dataset_spatial_tab(
            active_df=pd.DataFrame() if not 'active_df' in locals() else active_df,
            location_mode=location_mode,
            p_lat=p_lat,
            p_lon=p_lon,
        )

    # Interactive Audit Workbench Tab
    with equity_studio_tab:
        render_interactive_audit_tab(
            active_df=active_df,
            p_lat=p_lat,
            p_lon=p_lon,
            season_profile=season_profile,
            terrain_type=terrain_type,
            llm_provider=llm_provider,
            api_key_input=api_key_input,
            selected_scenario=st.session_state.get("selected_scenario", ""),
            active_vignette_text=st.session_state.get("active_vignette_text", ""),
        )

    # Downstream Components & Report Generators
    st.divider()
    lifecycle_report = st.session_state.get("health_lifecycle_report", {})
    if isinstance(lifecycle_report, str):
        try:
            lifecycle_report = json.loads(lifecycle_report)
        except Exception:
            lifecycle_report = {"error": lifecycle_report}

    latest_results = st.session_state.get("health_ds_report", {})
    if isinstance(latest_results, str):
        try:
            latest_results = json.loads(latest_results)
        except Exception:
            latest_results = {"summary": latest_results}

    render_lifecycle_tab("healthcare", lifecycle_report)
    domain_challenge_panel("health", latest_results)