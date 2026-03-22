# pages/05_📚_Scenario_Library.py
"""
GAGS Scenario Library v4.0
===========================
Browse, search, filter, and apply all simulation scenarios across 7 domains.
Covers community scenarios (§19) + domain presets (§21-§25).
Light-mode design. Research-grade metadata with citations.
"""
import re
import streamlit as st
import pandas as pd

from components.governance_logic import (
    COMMUNITY_SCENARIOS,
    EDUCATION_SCENARIO_PRESETS,
    FINANCIAL_SCENARIO_PRESETS,
    JUDICIAL_SCENARIO_PRESETS,
    DISINFORMATION_SCENARIO_PRESETS,
    ECONOMIC_SCENARIO_PRESETS,
)

st.set_page_config(
    page_title="Scenario Library • GAGS", page_icon="📚", layout="wide",
)

# ── Auto-dismiss stale guided tour banners from other pages ───────────────────
for _tk in ["_tour_dismissed_agrotech","_tour_dismissed_health","_tour_dismissed_security"]:
    if _tk not in st.session_state:
        st.session_state[_tk] = True

# ── Design system ──────────────────────────────────────────────────────────────
try:
    from components.gags_design import inject_css, DOMAIN_ACCENTS, plotly_theme as _pt
    inject_css("scenario")
    ACCENT = DOMAIN_ACCENTS["scenario"]
except ImportError:
    ACCENT = "#6d28d9"

# ── Domain config ──────────────────────────────────────────────────────────────
DOMAIN_META = {
    "healthcare":        {"icon":"🏥","label":"Healthcare","color":"#0891b2","page":"02_Healthcare_Equity"},
    "national_security": {"icon":"🛡️","label":"Security",  "color":"#dc2626","page":"2_National_Security"},
    "agrotech":          {"icon":"🌾","label":"Agrotech",  "color":"#16a34a","page":"3_Sustainable_Agrotech"},
    "education":         {"icon":"🎓","label":"Education", "color":"#d97706","page":"Education_equity"},
    "finance":           {"icon":"💰","label":"Finance",   "color":"#0d9488","page":"Financial_Inclusion"},
    "judicial":          {"icon":"⚖️","label":"Judicial",  "color":"#7c3aed","page":"Judicial_system"},
    "disinformation":    {"icon":"📡","label":"Disinfo",   "color":"#ea580c","page":"Disinformation_Misinformation"},
    "economic":          {"icon":"💼","label":"Economic",  "color":"#b45309","page":"11_Economic_Justice"},
}

SEV_COLOR = {
    "critical": "#ef4444", "high": "#f59e0b",
    "medium":   "#3b82f6", "low":  "#22c55e",
}
SEV_BG = {
    "critical": "#fef2f2", "high": "#fffbeb",
    "medium":   "#eff6ff", "low":  "#f0fdf4",
}

# ── Build unified scenario catalogue ──────────────────────────────────────────
_ALL_SCENARIOS = []

# §19 Community scenarios
for key, sc in COMMUNITY_SCENARIOS.items():
    _ALL_SCENARIOS.append({
        "key":       key,
        "id":        getattr(sc, "id", key),
        "name":      sc.name,
        "domain":    sc.domain,
        "type":      "Community",
        "region":    getattr(sc, "region", "Global"),
        "description": getattr(sc, "description", ""),
        "severity":  getattr(sc, "severity", "medium"),
        "tags":      getattr(sc, "tags", []),
        "citation":  getattr(sc, "citation", ""),
        "lesson":    getattr(sc, "lesson", ""),
        "scenario_key": key,
        "page":      DOMAIN_META.get(sc.domain, {}).get("page",""),
        "mitigation_priority": getattr(sc, "mitigation_priority", ""),
        "real_world_analogue": getattr(sc, "real_world_analogue", ""),
    })

# §21-§25 Domain presets
_PRESET_DOMAINS = [
    ("education",      EDUCATION_SCENARIO_PRESETS,     "🎓 Education"),
    ("finance",        FINANCIAL_SCENARIO_PRESETS,      "💰 Finance"),
    ("judicial",       JUDICIAL_SCENARIO_PRESETS,       "⚖️ Judicial"),
    ("disinformation", DISINFORMATION_SCENARIO_PRESETS, "📡 Disinformation"),
    ("economic",       ECONOMIC_SCENARIO_PRESETS,       "💼 Economic"),
]

for domain, presets, _ in _PRESET_DOMAINS:
    for key, sc in presets.items():
        desc = getattr(sc, "description", "")
        # Infer severity from description keywords
        sev = ("critical" if any(w in desc.lower() for w in ["critical","wrongful","harm","death","unlawful"])
               else "high" if any(w in desc.lower() for w in ["bias","discrimin","exclud","gap","inequit"])
               else "medium")
        _ALL_SCENARIOS.append({
            "key":       key,
            "id":        key.upper().replace("_","-")[:12],
            "name":      getattr(sc, "name", key.replace("_"," ").title()),
            "domain":    domain,
            "type":      "Domain Preset",
            "region":    "Nigeria" if any(w in desc.lower() for w in ["nigeria","abuja","lagos","cbn"])
                         else "Global",
            "description": desc,
            "severity":  sev,
            "tags":      [domain, "preset"],
            "citation":  getattr(sc, "citation", ""),
            "lesson":    "",
            "scenario_key": key,
            "page":      DOMAIN_META.get(domain, {}).get("page",""),
            "mitigation_priority": sev,
            "real_world_analogue": "",
        })

df_all = pd.DataFrame(_ALL_SCENARIOS)

# ── Header ─────────────────────────────────────────────────────────────────────
_badge_html = "".join(
    f'<span class="badge" style="color:{d["color"]};border-color:{d["color"]}44;'
    f'background:{d["color"]}11">{d["icon"]} {d["label"]}</span>'
    for d in DOMAIN_META.values()
)

st.markdown(f"""
<div class="page-header fade-in">
  <p class="eyebrow">GAGS · Scenario Library · {len(_ALL_SCENARIOS)} Scenarios · 8 Domains</p>
  <h1>📚 Simulation Scenario Library</h1>
  <p>Browse all community-contributed and domain preset scenarios across 8 AI governance domains.
     Filter by domain, severity, and region. Click any scenario to navigate to its simulation page.</p>
  <div style="margin-top:.65rem">
    {_badge_html}
  </div>
</div>""", unsafe_allow_html=True)

# ── Stats strip ────────────────────────────────────────────────────────────────
comm_count  = sum(1 for s in _ALL_SCENARIOS if s["type"]=="Community")
preset_count= sum(1 for s in _ALL_SCENARIOS if s["type"]=="Domain Preset")
crit_count  = sum(1 for s in _ALL_SCENARIOS if s["severity"]=="critical")
ng_count    = sum(1 for s in _ALL_SCENARIOS if "Nigeria" in s["region"])

s1,s2,s3,s4,s5 = st.columns(5)
for col, label, value, color in [
    (s1, "Total Scenarios",     str(len(_ALL_SCENARIOS)), ACCENT),
    (s2, "Community Scenarios", str(comm_count),          "#0891b2"),
    (s3, "Domain Presets",      str(preset_count),        "#16a34a"),
    (s4, "Critical Severity",   str(crit_count),          "#ef4444"),
    (s5, "Nigeria-Specific",    str(ng_count),            "#d97706"),
]:
    col.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
  padding:.9rem 1.1rem;box-shadow:0 1px 3px rgba(0,0,0,.05);
  border-bottom:3px solid {color}">
  <p style="font-family:'DM Mono',monospace;font-size:.65rem;letter-spacing:.08em;
     text-transform:uppercase;color:#94a3b8;margin:0 0 .1rem">{label}</p>
  <p style="font-family:'Syne',sans-serif;font-size:1.65rem;font-weight:700;
     color:{color};margin:0">{value}</p>
</div>""", unsafe_allow_html=True)

st.write("")

# ── Filters ────────────────────────────────────────────────────────────────────
st.markdown('<p class="section-label">Filter & Search</p>', unsafe_allow_html=True)

fc1, fc2, fc3, fc4 = st.columns([2, 1.2, 1.2, 1.2])
with fc1:
    search = st.text_input("🔍 Search", placeholder="Search by name, domain, keyword…",
                           label_visibility="collapsed")
with fc2:
    domain_filter = st.multiselect("Domain", options=sorted(df_all["domain"].unique()),
        format_func=lambda d: f"{DOMAIN_META.get(d,{}).get('icon','📊')} {d.replace('_',' ').title()}",
        placeholder="All domains")
with fc3:
    sev_filter = st.multiselect("Severity", options=["critical","high","medium","low"],
        format_func=lambda s: f"{'🔴' if s=='critical' else '🟡' if s=='high' else '🔵' if s=='medium' else '🟢'} {s.title()}",
        placeholder="All severities")
with fc4:
    type_filter = st.multiselect("Type", options=["Community","Domain Preset"],
        placeholder="All types")

# Apply filters
filtered = df_all.copy()
if search:
    mask = filtered.apply(
        lambda r: search.lower() in str(r["name"]).lower()
               or search.lower() in str(r["description"]).lower()
               or search.lower() in str(r["domain"]).lower()
               or search.lower() in str(r["tags"]).lower()
               or search.lower() in str(r["region"]).lower(), axis=1)
    filtered = filtered[mask]
if domain_filter:
    filtered = filtered[filtered["domain"].isin(domain_filter)]
if sev_filter:
    filtered = filtered[filtered["severity"].isin(sev_filter)]
if type_filter:
    filtered = filtered[filtered["type"].isin(type_filter)]

# ── View toggle ────────────────────────────────────────────────────────────────
view_col, sort_col = st.columns([3,1])
with view_col:
    view_mode = st.radio("View", ["Cards","Table","Domain Overview"],
                         horizontal=True, label_visibility="collapsed")
with sort_col:
    sort_by = st.selectbox("Sort", ["Domain","Severity","Name","Type","Region"],
                           label_visibility="collapsed")

sort_map = {"Domain":"domain","Severity":"severity","Name":"name",
            "Type":"type","Region":"region"}
filtered = filtered.sort_values(sort_map[sort_by]).reset_index(drop=True)

st.markdown(f'<p style="font-family:\'DM Mono\',monospace;font-size:.72rem;color:#94a3b8;margin:.5rem 0">'
            f'Showing {len(filtered)} of {len(_ALL_SCENARIOS)} scenarios</p>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CARD VIEW
# ═══════════════════════════════════════════════════════════════════════════════
if view_mode == "Cards":
    # Group by domain
    for domain in sorted(filtered["domain"].unique()):
        dm = DOMAIN_META.get(domain, {"icon":"📊","label":domain,"color":"#6d28d9"})
        domain_rows = filtered[filtered["domain"]==domain]

        st.markdown(f"""
<div style="display:flex;align-items:center;gap:.75rem;margin:1.5rem 0 .75rem">
  <span style="font-size:1.2rem">{dm['icon']}</span>
  <span style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
    color:{dm['color']}">{dm['label'].replace('_',' ').title()}</span>
  <span style="background:{dm['color']}15;border:1px solid {dm['color']}30;
    border-radius:99px;padding:1px 10px;font-family:'DM Mono',monospace;
    font-size:.67rem;color:{dm['color']}">{len(domain_rows)} scenarios</span>
  <span style="flex:1;height:1px;background:#e2e8f0"></span>
</div>""", unsafe_allow_html=True)

        cols = st.columns(3)
        for ci, (_, row) in enumerate(domain_rows.iterrows()):
            with cols[ci % 3]:
                sev = row["severity"]
                sev_col = SEV_COLOR.get(sev,"#3b82f6")
                sev_bg  = SEV_BG.get(sev,"#eff6ff")
                tags_html = "".join(
                    f'<span style="background:#f1f5f9;border:1px solid #e2e8f0;'
                    f'border-radius:3px;padding:1px 7px;font-size:.65rem;'
                    f'color:#64748b;margin:2px 2px 0 0;display:inline-block">{t}</span>'
                    for t in (row["tags"] if isinstance(row["tags"],list) else [])[:4])

                _desc = str(row["description"])
                _desc_html = (_desc[:160] + "…") if len(_desc) > 160 else _desc
                _rwa = str(row.get("real_world_analogue","")).strip()
                _rwa_html = (
                    f'<p style="font-size:.74rem;color:#64748b;font-style:italic;'
                    f'margin:0 0 .45rem;border-left:2px solid {dm["color"]}33;'
                    f'padding-left:.5rem">{_rwa[:120]}…</p>'
                ) if _rwa else ""
                _cit = str(row.get("citation","")).strip()
                _cit_html = (
                    f'<p style="font-size:.7rem;color:#94a3b8;margin:0">📚 {_cit[:90]}…</p>'
                ) if _cit else ""

                st.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:10px;
  padding:1rem 1.15rem;margin-bottom:.75rem;box-shadow:0 1px 3px rgba(0,0,0,.05);
  height:100%;position:relative;overflow:hidden;
  border-top:3px solid {dm['color']}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:.5rem;margin-bottom:.4rem">
    <span style="font-family:'DM Mono',monospace;font-size:.65rem;
      color:#94a3b8">{row['id']}</span>
    <span style="background:{sev_bg};color:{sev_col};font-family:'DM Mono',monospace;
      font-size:.63rem;font-weight:600;padding:2px 8px;border-radius:99px;
      border:1px solid {sev_col}40;text-transform:uppercase;white-space:nowrap">
      {sev}</span>
  </div>
  <p style="font-family:'Syne',sans-serif;font-size:.88rem;font-weight:700;
     color:#0f172a;margin:0 0 .2rem;line-height:1.25">{row['name']}</p>
  <p style="font-family:'DM Mono',monospace;font-size:.67rem;color:#94a3b8;
     margin:0 0 .4rem">🌍 {row['region']} · {row['type']}</p>
  <p style="font-size:.78rem;color:#475569;line-height:1.45;margin:0 0 .5rem">
    {_desc_html}</p>
  {_rwa_html}
  <div style="margin-bottom:.45rem">{tags_html}</div>
  {_cit_html}
</div>""", unsafe_allow_html=True)

                if row["page"]:
                    st.page_link(
                        f'pages/{row["page"]}.py',
                        label=f"→ Open {dm['label']} Simulation",
                        help=f"Run this scenario in the {dm['label']} module",
                    )
                st.write("")

# ═══════════════════════════════════════════════════════════════════════════════
# TABLE VIEW
# ═══════════════════════════════════════════════════════════════════════════════
elif view_mode == "Table":
    display_cols = ["id","name","domain","type","region","severity","mitigation_priority"]
    display_df = filtered[display_cols].copy()
    display_df.columns = ["ID","Name","Domain","Type","Region","Severity","Priority"]
    display_df["Domain"] = display_df["Domain"].apply(
        lambda d: f"{DOMAIN_META.get(d,{}).get('icon','📊')} {d.replace('_',' ').title()}")

    def _color_sev(val):
        colors = {"critical":"background:#fef2f2;color:#991b1b",
                  "high":"background:#fffbeb;color:#92400e",
                  "medium":"background:#eff6ff;color:#1e40af",
                  "low":"background:#f0fdf4;color:#166534"}
        return colors.get(val.lower(),"") if isinstance(val,str) else ""

    styled = display_df.style.applymap(_color_sev, subset=["Severity","Priority"])
    st.dataframe(styled, use_container_width=True, hide_index=True, height=480)

    # Detail expander for selected scenario
    selected_id = st.selectbox("View full details for scenario:",
        ["— Select —"] + list(filtered["name"]))
    if selected_id != "— Select —":
        row = filtered[filtered["name"]==selected_id].iloc[0]
        dm  = DOMAIN_META.get(row["domain"],{"icon":"📊","color":"#6d28d9"})
        sev_col = SEV_COLOR.get(row["severity"],"#3b82f6")
        with st.expander(f"{dm['icon']} {row['name']}", expanded=True):
            c1, c2 = st.columns([2,1])
            with c1:
                st.markdown(f"**Description:** {row['description']}")
                if row.get("real_world_analogue"):
                    st.markdown(f"**Real-World Analogue:** _{row['real_world_analogue']}_")
                if row.get("lesson"):
                    st.markdown(f"**Key Lesson:** _{row['lesson']}_")
                if row.get("citation"):
                    st.markdown(f"**Citation:** {row['citation']}")
            with c2:
                st.markdown(f"""
- **ID:** `{row['id']}`
- **Domain:** {dm['icon']} {row['domain'].replace('_',' ').title()}
- **Type:** {row['type']}
- **Region:** {row['region']}
- **Severity:** <span style="color:{sev_col};font-weight:600">{row['severity'].upper()}</span>
- **Priority:** {row.get('mitigation_priority','—')}
""", unsafe_allow_html=True)
            if row["page"]:
                st.page_link(f'pages/{row["page"]}.py',
                             label=f"→ Open {row['domain'].replace('_',' ').title()} Simulation")

# ═══════════════════════════════════════════════════════════════════════════════
# DOMAIN OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<p class="section-label">Domain Overview</p>', unsafe_allow_html=True)

    for domain, dm in DOMAIN_META.items():
        domain_rows = df_all[df_all["domain"]==domain]
        if len(domain_rows) == 0:
            continue

        crit = sum(1 for _,r in domain_rows.iterrows() if r["severity"]=="critical")
        high = sum(1 for _,r in domain_rows.iterrows() if r["severity"]=="high")
        ng   = sum(1 for _,r in domain_rows.iterrows() if "Nigeria" in str(r["region"]))

        with st.expander(
            f"{dm['icon']} {dm['label']} — {len(domain_rows)} scenarios"
            f"{'  🔴 '+str(crit)+' critical' if crit else ''}"
            f"{'  🟡 '+str(high)+' high' if high else ''}"
            f"{'  🇳🇬 '+str(ng)+' Nigeria' if ng else ''}",
            expanded=(domain in ("healthcare","economic","judicial"))
        ):
            c1, c2 = st.columns([3,1])
            with c1:
                for _, row in domain_rows.iterrows():
                    sev_col = SEV_COLOR.get(row["severity"],"#3b82f6")
                    sev_bg  = SEV_BG.get(row["severity"],"#eff6ff")
                    st.markdown(f"""
<div style="background:#fff;border:1px solid #e2e8f0;border-radius:8px;
  padding:.75rem 1rem;margin-bottom:.5rem;display:flex;gap:1rem;align-items:flex-start">
  <div style="flex:0 0 auto">
    <span style="background:{sev_bg};color:{sev_col};font-family:'DM Mono',monospace;
      font-size:.62rem;font-weight:600;padding:2px 7px;border-radius:99px;
      border:1px solid {sev_col}40;text-transform:uppercase">{row['severity']}</span>
  </div>
  <div style="flex:1">
    <p style="font-family:'Syne',sans-serif;font-size:.85rem;font-weight:700;
       color:#0f172a;margin:0 0 .15rem">{row['name']}</p>
    <p style="font-size:.76rem;color:#64748b;margin:0">
      🌍 {row['region']} · {row['type']} · <code style="background:#f1f5f9;
      padding:1px 5px;border-radius:3px;font-size:.7rem">{row['id']}</code>
    </p>
    {f'<p style="font-size:.75rem;color:#475569;margin:.25rem 0 0;line-height:1.4">{str(row["description"])[:140]}…</p>'
     if str(row.get("description","")).strip() else ''}
  </div>
</div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
<div style="background:{dm['color']}0d;border:1px solid {dm['color']}25;
  border-radius:10px;padding:1rem;text-align:center">
  <p style="font-size:2rem;margin:0 0 .3rem">{dm['icon']}</p>
  <p style="font-family:'Syne',sans-serif;font-size:.9rem;font-weight:700;
     color:{dm['color']};margin:0 0 .75rem">{dm['label']}</p>
  <p style="font-family:'DM Mono',monospace;font-size:.72rem;color:#64748b;margin:0 0 .35rem">
    {len(domain_rows)} total scenarios</p>
  <p style="font-family:'DM Mono',monospace;font-size:.72rem;color:#ef4444;margin:0 0 .35rem">
    {crit} critical severity</p>
  <p style="font-family:'DM Mono',monospace;font-size:.72rem;color:#d97706;margin:0 0 .75rem">
    {ng} Nigeria-specific</p>
</div>""", unsafe_allow_html=True)
                if dm["page"]:
                    st.page_link(f'pages/{dm["page"]}.py',
                                 label=f"→ Open {dm['label']}",
                                 help=f"Go to the {dm['label']} simulation page")

# ── Download all scenarios ─────────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-label">Export</p>', unsafe_allow_html=True)
ec1, ec2 = st.columns(2)
with ec1:
    export_df = filtered[["id","name","domain","type","region","severity","description","citation"]].copy()
    st.download_button(
        "📥 Download Filtered Scenarios (CSV)",
        export_df.to_csv(index=False).encode(),
        f"gags_scenarios_{len(filtered)}.csv", "text/csv",
        use_container_width=True)
with ec2:
    import json
    st.download_button(
        "📋 Download All Scenarios (JSON)",
        json.dumps([{k:v for k,v in s.items() if k not in ["tags"]}
                    for s in _ALL_SCENARIOS], indent=2, default=str).encode(),
        "gags_all_scenarios.json", "application/json",
        use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    f"<div style='text-align:center;color:#94a3b8;font-family:DM Mono,monospace;"
    f"font-size:.68rem;padding:.5rem 0'>"
    f"📚 GAGS Scenario Library v4.0 · {len(_ALL_SCENARIOS)} Scenarios · "
    f"8 Domains · Community + Domain Presets · Nigeria-Calibrated</div>",
    unsafe_allow_html=True)