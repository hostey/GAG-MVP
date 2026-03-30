"""
components/live_data.py - GAGS Live Data Layer v2.0
====================================================
Pulls the most current publicly available Nigeria data from 4 free APIs:

  World Bank Open Data   -> GDP 2023, inflation 2024, U5MR 2022, OOP 2022
  WHO Global Health Obs  -> MMR 2020, health workers 2020
  UNICEF CME Warehouse   -> Under-5 mortality 2023 estimate
  Frankfurter/ECB API    -> NGN/USD exchange rate (LIVE daily, 2025/2026)

HONEST NOTE: No public API has Nigerian STATE-LEVEL data after NDHS 2021.
The next NDHS is planned for 2026/27. State figures in this module are
the most current subnational data that exists anywhere.
"""
from __future__ import annotations
import json, ssl, urllib.request
from datetime import datetime, timezone
from typing import Any, Optional
import streamlit as st


# ── HTTP helper ────────────────────────────────────────────────────────────────
def _get(url: str, timeout: int = 7) -> Optional[Any]:
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={
            "User-Agent": "GAGS-Framework/4.0 Nigeria-AI-Fairness",
            "Accept": "application/json",
        })
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


# ── Source fetchers ────────────────────────────────────────────────────────────
def _wb(indicator: str, mrv: int = 2) -> dict:
    """World Bank Open Data - Nigeria national indicator."""
    url = (f"https://api.worldbank.org/v2/country/NGA/indicator/{indicator}"
           f"?format=json&mrv={mrv}&per_page=5")
    data = _get(url)
    if data and isinstance(data, list) and len(data) > 1:
        for rec in (data[1] or []):
            if rec and rec.get("value") is not None:
                return {"value": float(rec["value"]),
                        "date": str(rec.get("date", "?")),
                        "source": "World Bank", "is_live": True}
    return {"value": None, "date": "N/A", "source": "World Bank", "is_live": False}


def _who(indicator: str) -> dict:
    """WHO Global Health Observatory - Nigeria national indicator."""
    url = (f"https://ghoapi.azureedge.net/api/{indicator}"
           f"?%24filter=SpatialDim%20eq%20%27NGA%27"
           f"&%24orderby=TimeDim%20desc&%24top=5")
    data = _get(url)
    if data and isinstance(data, dict):
        for rec in data.get("value", []):
            if rec.get("NumericValue") is not None:
                return {"value": float(rec["NumericValue"]),
                        "date": str(rec.get("TimeDim", "?")),
                        "source": "WHO GHO", "is_live": True}
    return {"value": None, "date": "N/A", "source": "WHO GHO", "is_live": False}


def _unicef_u5mr() -> dict:
    """UNICEF CME - Under-5 mortality estimate for Nigeria (most current)."""
    url = ("https://sdmx.data.unicef.org/ws/public/sdmxapi/rest/data/"
           "UNICEF,CME_DF,1.0/NGA.CME_TMY0T4..?format=jsondata"
           "&lastNObservations=3&startPeriod=2021")
    data = _get(url)
    try:
        series  = data["data"]["dataSets"][0]["series"]
        periods = data["data"]["structure"]["dimensions"]["observation"][0]["values"]
        key     = list(series.keys())[0]
        obs     = series[key]["observations"]
        for idx in sorted(obs.keys(), key=int, reverse=True):
            val = obs[idx][0]
            if val is not None:
                return {"value": float(val),
                        "date": str(periods[int(idx)]["id"]),
                        "source": "UNICEF CME", "is_live": True}
    except Exception:
        pass
    return {"value": None, "date": "N/A", "source": "UNICEF", "is_live": False}


def _exchange_rate_ngn() -> dict:
    """Frankfurter/ECB - Live daily NGN/USD rate. Genuinely 2025/2026 current."""
    data = _get("https://api.frankfurter.app/latest?from=USD&to=NGN")
    if data and data.get("rates", {}).get("NGN"):
        return {"value": float(data["rates"]["NGN"]),
                "date": data.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                "source": "Frankfurter/ECB", "is_live": True}
    return {"value": None, "date": "N/A", "source": "ECB", "is_live": False}


# ── Master cached fetch ────────────────────────────────────────────────────────
@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_nigeria_live() -> dict:
    """
    Fetch all live national indicators for Nigeria. Cached 24 hours.
    No API key required. Fails gracefully if any source is unreachable.

    Returns
    -------
    dict with keys: mmr, u5mr, u5mr_unicef, oop, hw_density,
                    gdp_pc, population, inflation, exchange_rate,
                    _fetched_at, _any_live
    """
    results = {
        "mmr":            _who("WHOSIS_000001"),           # WHO MMR latest=2020
        "u5mr":           _wb("SH.DYN.MORT", mrv=2),      # WB U5MR latest=2022
        "u5mr_unicef":    _unicef_u5mr(),                  # UNICEF U5MR latest=2023
        "oop":            _wb("SH.XPD.OOPC.CH.ZS", mrv=2),# OOP latest=2022
        "hw_density":     _who("HRH_17"),                  # WHO health workers=2020
        "gdp_pc":         _wb("NY.GDP.PCAP.CD", mrv=1),   # GDP/capita latest=2023
        "population":     _wb("SP.POP.TOTL", mrv=1),      # Population latest=2023
        "inflation":      _wb("FP.CPI.TOTL.ZG", mrv=2),   # CPI inflation latest=2024
        "exchange_rate":  _exchange_rate_ngn(),             # Live daily
        "_fetched_at":    datetime.now(timezone.utc).isoformat(),
    }
    results["_any_live"] = any(
        v.get("is_live") for k, v in results.items()
        if isinstance(v, dict) and not k.startswith("_")
    )
    return results


# ── Card renderer ──────────────────────────────────────────────────────────────
def live_state_info_card(state_name: str, show_live_badge: bool = True) -> None:
    """
    Render an enriched state info card with three data layers:
      1. State estimates  (NDHS 2021 / NBS 2023 — most current available)
      2. National live    (World Bank 2022-2024, WHO 2020-2022)
      3. Economic context (exchange rate daily, inflation 2024)

    The card clearly labels each figure's source and year so users
    understand exactly how current each number is.
    """
    from components.nigeria_states import get_state_params
    profile     = get_state_params(state_name)
    is_national = state_name in ("Nigeria (National Average)", "Nigeria", "", None)

    live     = fetch_nigeria_live()
    any_live = live.get("_any_live", False)

    zone_colour = {"NW":"#dc2626","NE":"#ea580c","NC":"#d97706",
                   "SW":"#16a34a","SE":"#0891b2","SS":"#7c3aed",
                   "All":"#334155"}.get(profile.geopolitical_zone, "#334155")

    zone_full = {"NW":"North-West","NE":"North-East","NC":"North-Central",
                 "SW":"South-West","SE":"South-East","SS":"South-South",
                 "All":"All Zones"}.get(profile.geopolitical_zone, "")

    title = "🇳🇬 Nigeria — National" if is_national else ("📍 " + profile.name + " State")
    badge_html = ""
    if show_live_badge:
        if any_live:
            badge_html = ("<span style='background:#dcfce7;color:#166534;"
                          "padding:2px 8px;border-radius:8px;font-size:.66rem;"
                          "font-weight:700;'>🔴 LIVE</span>")
        else:
            badge_html = ("<span style='background:#fef9c3;color:#854d0e;"
                          "padding:2px 8px;border-radius:8px;font-size:.66rem;"
                          "font-weight:700;'>📚 STATIC</span>")

    def _v(rec, fmt, fb="N/A"):
        try: return fmt.format(rec["value"]) if rec.get("value") is not None else fb
        except: return fb

    def _dot(rec):
        return "🔴" if rec.get("is_live") else "📚"

    def _arrow(sv, nat_rec, lower_better=True):
        if is_national: return ""
        try:
            nv  = nat_rec.get("value")
            if sv is None or nv is None: return ""
            pct = abs((float(sv) - float(nv)) / float(nv)) * 100
            if pct < 4: return ""
            worse = (float(sv) > float(nv)) == lower_better
            col   = "#dc2626" if worse else "#16a34a"
            sym   = "▲" if worse else "▼"
            return ("<span style='color:" + col + ";font-size:.66rem;'>"
                    + sym + "{:.0f}%".format(pct) + "</span>")
        except: return ""

    def row(icon, label, sv, sfmt, sunit, ssrc, nat, nfmt, lb=True):
        return (
            "<tr>"
            "<td style='padding:4px 6px 4px 0;font-size:.71rem;color:#374151;"
            "white-space:nowrap;vertical-align:top;'>" + icon + " " + label + "</td>"
            "<td style='padding:4px 8px;vertical-align:top;'>"
            "<b style='font-size:.78rem;color:#0f172a;'>" +
            _v({"value": sv}, sfmt) + " " + sunit + "</b> " + _arrow(sv, nat, lb) +
            "<br><span style='font-size:.61rem;color:#9ca3af;'>" + ssrc + "</span></td>"
            "<td style='padding:4px 0;text-align:right;vertical-align:top;'>"
            "<span style='font-size:.75rem;'>" + _v(nat, nfmt) + "</span>"
            "<br><span style='font-size:.61rem;color:#9ca3af;'>" +
            _dot(nat) + " " + nat.get("source","") +
            "<br>" + nat.get("date","?") + "</span></td></tr>"
        )

    # Best U5MR (UNICEF preferred as more current)
    u5 = live.get("u5mr_unicef",{}) if live.get("u5mr_unicef",{}).get("is_live") \
         else live.get("u5mr",{})

    # Economic indicators
    xr  = live.get("exchange_rate", {})
    inf = live.get("inflation", {})
    gdp = live.get("gdp_pc", {})
    pop = live.get("population", {})

    xr_str  = ("N{:,.0f}/$".format(xr["value"]) if xr.get("value") else "N/A")
    inf_str = ("{:.1f}%".format(inf["value"]) if inf.get("value") else "N/A")
    gdp_str = ("${:,.0f}".format(gdp["value"]) if gdp.get("value") else "~$2,184")
    pop_str = ("{:.1f}M".format(pop["value"]/1e6) if pop.get("value") else "~218M")

    econ_row = (
        "<tr style='background:#f0fdf4;'>"
        "<td style='padding:4px 6px 4px 0;font-size:.71rem;color:#374151;'>💰 GDP/capita</td>"
        "<td colspan='2' style='padding:4px 0;'>"
        "<b style='font-size:.78rem;'>" + gdp_str + " USD</b>"
        " &nbsp;·&nbsp; Pop <b>" + pop_str + "</b>"
        "<br><span style='font-size:.61rem;color:#9ca3af;'>"
        + _dot(gdp) + " World Bank " + gdp.get("date","?") + "</span>"
        "</td></tr>"
        "<tr>"
        "<td style='padding:4px 6px 4px 0;font-size:.71rem;color:#374151;'>💱 Exchange</td>"
        "<td colspan='2' style='padding:4px 0;'>"
        "<b style='font-size:.78rem;color:#16a34a;'>" + xr_str + "</b>"
        " &nbsp;·&nbsp; Inflation <b>" + inf_str + "</b>"
        "<br><span style='font-size:.61rem;color:#6b7280;'>"
        + _dot(xr) + " Frankfurter/ECB " + xr.get("date","")
        + " &nbsp;·&nbsp; WB " + inf.get("date","?") + "</span>"
        "</td></tr>"
    )

    rows_html = (
        row("👩‍⚕️","Maternal Mortality",profile.maternal_mortality_ratio,
            "{:,.0f}","/100k","NDHS 2021",live.get("mmr",{}),"{:.0f}") +
        row("👶","Under-5 Mortality",profile.u5_mortality_rate,
            "{:.1f}","/1k","NDHS 2021",u5,"{:.1f}") +
        row("🏥","Health Workers",profile.health_worker_density,
            "{:.2f}","/10k","WHO 2022",live.get("hw_density",{}),"{:.2f}",lb=False) +
        row("💳","NHIS Coverage",profile.nhis_coverage_rate*100,
            "{:.1f}","%","NHIA 2023",
            {"value":4.5,"date":"2023","source":"NHIA","is_live":False},"{:.1f}%",lb=False) +
        row("💸","OOP Expenditure",profile.oop_expenditure_pct*100,
            "{:.1f}","%","WHO 2023",live.get("oop",{}),"{:.1f}%") +
        row("📉","Poverty Rate",profile.poverty_rate*100,
            "{:.1f}","%","NBS 2023",
            {"value":38.7,"date":"2022","source":"WB","is_live":False},"{:.1f}%") +
        row("👷","Informal Sector",profile.informal_sector_pct*100,
            "{:.1f}","%","NBS 2023",
            {"value":64.9,"date":"2023","source":"NBS","is_live":False},"{:.1f}%") +
        row("📚","Literacy Rate",profile.literacy_rate*100,
            "{:.1f}","%","UNESCO 2023",
            {"value":62.0,"date":"2021","source":"UNESCO","is_live":False},"{:.1f}%",lb=False) +
        row("🏙️","Rural Population",profile.rural_population_pct*100,
            "{:.1f}","%","NBS 2023",
            {"value":48.0,"date":"2023","source":"WB","is_live":False},"{:.1f}%") +
        econ_row
    )

    note = ""
    if not is_national:
        note = ("<p style='margin:6px 0 0;font-size:.63rem;color:#9ca3af;"
                "font-style:italic;'>State column: NDHS 2021/NBS 2023 subnational "
                "estimates — the most current state-level data available anywhere. "
                "No public API has post-2021 Nigerian state data.</p>")
    if profile.notes:
        note += ("<p style='margin:4px 0 0;font-size:.65rem;color:#9ca3af;"
                 "font-style:italic;'>ℹ️ " + profile.notes + "</p>")

    html = (
        "<div style='background:#fff;border:1px solid #e2e8f0;"
        "border-left:5px solid " + zone_colour + ";"
        "border-radius:10px;padding:12px 16px;margin:8px 0 6px;"
        "font-family:system-ui,sans-serif;'>"
        "<div style='display:flex;justify-content:space-between;"
        "align-items:center;margin-bottom:3px;'>"
        "<span style='font-weight:700;font-size:.95rem;color:#0f172a;'>" + title + "</span>"
        + badge_html +
        "</div>"
        "<p style='margin:0 0 8px;font-size:.73rem;color:" + zone_colour + ";"
        "font-weight:600;'>" + zone_full + " · " + profile.primary_language +
        " · Pop: {:.1f}M</p>".format(profile.population_millions) +
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr>"
        "<th style='text-align:left;font-size:.60rem;color:#9ca3af;"
        "padding:0 6px 5px 0;font-weight:500;border-bottom:1px solid #f1f5f9;'>INDICATOR</th>"
        "<th style='text-align:left;font-size:.60rem;color:#9ca3af;"
        "padding:0 8px 5px;font-weight:500;border-bottom:1px solid #f1f5f9;'>STATE EST.</th>"
        "<th style='text-align:right;font-size:.60rem;color:#9ca3af;"
        "padding:0 0 5px;font-weight:500;border-bottom:1px solid #f1f5f9;'>NATIONAL LIVE</th>"
        "</tr></thead>"
        "<tbody>" + rows_html + "</tbody></table>"
        + note +
        "<p style='margin:6px 0 0;font-size:.62rem;color:#9ca3af;'>"
        "🔴 Live API &nbsp;|&nbsp; 📚 Static survey "
        "&nbsp;|&nbsp; ▲▼ state vs national average</p>"
        "</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def national_live_banner() -> None:
    """Compact top-of-page live data banner. Silently skipped if offline."""
    try:
        live = fetch_nigeria_live()
    except Exception:
        return
    if not live.get("_any_live"):
        return

    xr  = live.get("exchange_rate", {})
    inf = live.get("inflation", {})
    gdp = live.get("gdp_pc", {})
    u5  = live.get("u5mr_unicef") or live.get("u5mr", {})

    parts = []
    if xr.get("value"):  parts.append("💱 <b>&#8358;{:,.0f}/$</b> ({})".format(xr["value"], xr["date"]))
    if inf.get("value"): parts.append("📈 Inflation <b>{:.1f}%</b> ({})".format(inf["value"], inf["date"]))
    if gdp.get("value"): parts.append("💰 GDP/cap <b>${:,.0f}</b> ({})".format(gdp["value"], gdp["date"]))
    if u5.get("value"):  parts.append("👶 U5MR <b>{:.1f}</b>/1k ({})".format(u5["value"], u5["date"]))

    if not parts:
        return

    st.markdown(
        "<div style='background:#f0fdf4;border:1px solid #bbf7d0;"
        "border-radius:6px;padding:5px 14px;margin-bottom:10px;"
        "font-size:.72rem;'>"
        "<span style='color:#16a34a;font-weight:700;'>🔴 LIVE Nigeria&nbsp;&nbsp;</span>"
        + "&nbsp;·&nbsp;".join(parts) +
        "<span style='color:#9ca3af;margin-left:10px;'>"
        "World Bank / UNICEF / Frankfurter ECB</span></div>",
        unsafe_allow_html=True,
    )