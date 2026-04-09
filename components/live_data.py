"""
components/live_data.py  —  GAGS Live Data Layer v3.0
======================================================
Clean, sidebar-optimised Nigeria national data card.
Sources: World Bank, WHO, UNICEF, Frankfurter/ECB.
"""
from __future__ import annotations
import json, ssl, urllib.request
from datetime import datetime, timezone
from typing import Any, Optional
import streamlit as st


def _get(url: str, timeout: int = 7) -> Optional[Any]:
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={
            "User-Agent": "GAGS-Framework/4.0", "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _wb(indicator: str, mrv: int = 2) -> dict:
    url = (f"https://api.worldbank.org/v2/country/NGA/indicator/{indicator}"
           f"?format=json&mrv={mrv}&per_page=5")
    data = _get(url)
    if data and isinstance(data, list) and len(data) > 1:
        for rec in (data[1] or []):
            if rec and rec.get("value") is not None:
                return {"value": float(rec["value"]), "date": str(rec.get("date","?")),
                        "source": "World Bank", "is_live": True}
    return {"value": None, "date": "N/A", "source": "World Bank", "is_live": False}


def _who(indicator: str) -> dict:
    url = (f"https://ghoapi.azureedge.net/api/{indicator}"
           f"?%24filter=SpatialDim%20eq%20%27NGA%27"
           f"&%24orderby=TimeDim%20desc&%24top=5")
    data = _get(url)
    if data and isinstance(data, dict):
        for rec in data.get("value", []):
            if rec.get("NumericValue") is not None:
                return {"value": float(rec["NumericValue"]), "date": str(rec.get("TimeDim","?")),
                        "source": "WHO GHO", "is_live": True}
    return {"value": None, "date": "N/A", "source": "WHO GHO", "is_live": False}


def _exchange_rate_ngn() -> dict:
    data = _get("https://api.frankfurter.app/latest?from=USD&to=NGN")
    if data and data.get("rates", {}).get("NGN"):
        return {"value": float(data["rates"]["NGN"]),
                "date": data.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d")),
                "source": "ECB / Frankfurter", "is_live": True}
    return {"value": None, "date": "N/A", "source": "ECB", "is_live": False}


@st.cache_data(ttl=86_400, show_spinner=False)
def fetch_nigeria_live() -> dict:
    """Fetch all live national indicators. Cached 24h. No API key required."""
    results = {
        "mmr":           _who("WHOSIS_000001"),
        "u5mr":          _wb("SH.DYN.MORT", mrv=2),
        "oop":           _wb("SH.XPD.OOPC.CH.ZS", mrv=2),
        "hw_density":    _who("HRH_17"),
        "gdp_pc":        _wb("NY.GDP.PCAP.CD", mrv=1),
        "population":    _wb("SP.POP.TOTL", mrv=1),
        "inflation":     _wb("FP.CPI.TOTL.ZG", mrv=2),
        "exchange_rate": _exchange_rate_ngn(),
        "poverty":       _wb("SI.POV.DDAY", mrv=3),
        "_fetched_at":   datetime.now(timezone.utc).isoformat(),
    }
    results["_any_live"] = any(
        v.get("is_live") for k, v in results.items()
        if isinstance(v, dict) and not k.startswith("_")
    )
    return results


def live_state_info_card(state_name: str, show_live_badge: bool = True) -> None:
    """Sidebar-optimised state info card with live national data layer."""
    from components.nigeria_states import get_state_params
    profile     = get_state_params(state_name)
    is_national = state_name in ("Nigeria (National Average)", "Nigeria", "", None)
    live        = fetch_nigeria_live()
    any_live    = live.get("_any_live", False)

    zone_colour = {
        "NW":"#dc2626","NE":"#ea580c","NC":"#d97706",
        "SW":"#16a34a","SE":"#0891b2","SS":"#7c3aed","All":"#334155",
    }.get(profile.geopolitical_zone, "#334155")

    zone_full = {
        "NW":"North-West","NE":"North-East","NC":"North-Central",
        "SW":"South-West","SE":"South-East","SS":"South-South","All":"All Zones",
    }.get(profile.geopolitical_zone, "")

    title = "🇳🇬 Nigeria — National" if is_national else ("📍 " + profile.name)

    if show_live_badge and any_live:
        badge = "<span style='background:#dcfce7;color:#166534;padding:1px 7px;border-radius:10px;font-size:.61rem;font-weight:700;'>🔴 LIVE</span>"
    elif show_live_badge:
        badge = "<span style='background:#fef9c3;color:#854d0e;padding:1px 7px;border-radius:10px;font-size:.61rem;font-weight:700;'>📚 STATIC</span>"
    else:
        badge = ""

    def fv(val, fmt, fb="—"):
        try: return fmt.format(val) if val is not None else fb
        except: return fb

    def lv(rec, fmt):
        v, d = rec.get("value"), rec.get("date","")
        dot  = "🔴" if rec.get("is_live") else "📚"
        return fv(v, fmt), d, dot

    def metric_row(icon, label, sv, sfmt, sunit, nat, nfmt, lb=True):
        sv_s      = fv(sv, sfmt)
        nv_s, nd, ndot = lv(nat, nfmt)
        arrow = ""
        nv = nat.get("value")
        if not is_national and sv is not None and nv is not None:
            try:
                pct   = abs((float(sv)-float(nv))/max(float(nv),0.001))*100
                if pct >= 5:
                    col = "#dc2626" if (float(sv)>float(nv))==lb else "#16a34a"
                    sym = "▲" if (float(sv)>float(nv))==lb else "▼"
                    arrow = "<span style='color:"+col+";font-size:.6rem;font-weight:600;'>"+sym+"{:.0f}%</span>".format(pct)
            except: pass
        return (
            "<div style='padding:5px 0;border-bottom:1px solid #f1f5f9;'>"
            "<div style='font-size:.67rem;color:#6b7280;'>"+icon+" "+label+"</div>"
            "<div style='display:flex;justify-content:space-between;align-items:baseline;'>"
            "<b style='font-size:.8rem;color:#0f172a;'>"+sv_s+" "+sunit+"</b>"
            +arrow+
            "</div>"
            "<div style='font-size:.59rem;color:#9ca3af;'>"+ndot+" "+nv_s+" ("+nd+")</div>"
            "</div>"
        )

    xr   = live.get("exchange_rate",{})
    gdp  = live.get("gdp_pc",{})
    inf  = live.get("inflation",{})
    pop  = live.get("population",{})

    xr_s  = fv(xr.get("value"),  "₦{:,.0f}/$")
    gdp_s = fv(gdp.get("value"), "${:,.0f}")
    inf_s = fv(inf.get("value"), "{:.1f}%")
    pop_s = ("{:.1f}M".format(pop["value"]/1e6) if pop.get("value")
             else "{:.1f}M".format(profile.population_millions))
    xr_dot  = "🔴" if xr.get("is_live") else "📚"
    gdp_dot = "🔴" if gdp.get("is_live") else "📚"

    econ = (
        "<div style='padding:5px 0;'>"
        "<div style='font-size:.67rem;color:#6b7280;'>💰 Economic indicators</div>"
        "<div style='display:flex;gap:8px;flex-wrap:wrap;margin-top:2px;'>"
        "<span style='font-size:.71rem;'>" + xr_dot + " <b>" + xr_s + "</b> <span style='color:#9ca3af;font-size:.58rem;'>" + xr.get("date","") + "</span></span>"
        "<span style='font-size:.71rem;'>" + gdp_dot + " GDP <b>" + gdp_s + "</b></span>"
        "<span style='font-size:.71rem;'>📈 CPI <b>" + inf_s + "</b></span>"
        "</div></div>"
    )

    rows = (
        metric_row("👩‍⚕️","Maternal Mortality",
                   profile.maternal_mortality_ratio,"{:,.0f}","/100k",live.get("mmr",{}),"{:.0f}") +
        metric_row("👶","Under-5 Mortality",
                   profile.u5_mortality_rate,"{:.1f}","/1k",live.get("u5mr",{}),"{:.1f}") +
        metric_row("🏥","Health Workers",
                   profile.health_worker_density,"{:.2f}","/10k",
                   live.get("hw_density",{}),"{:.2f}",lb=False) +
        metric_row("💸","OOP Expenditure",
                   profile.oop_expenditure_pct*100,"{:.1f}","%",live.get("oop",{}),"{:.1f}%") +
        metric_row("💳","NHIS Coverage",
                   profile.nhis_coverage_rate*100,"{:.1f}","%",
                   {"value":4.5,"date":"2023","source":"NHIA","is_live":False},"{:.1f}%",lb=False) +
        metric_row("📉","Poverty Rate",
                   profile.poverty_rate*100,"{:.1f}","%",live.get("poverty",{}),"{:.1f}%") +
        metric_row("👷","Informal Sector",
                   profile.informal_sector_pct*100,"{:.1f}","%",
                   {"value":64.9,"date":"2023","source":"NBS","is_live":False},"{:.1f}%") +
        metric_row("📚","Literacy Rate",
                   profile.literacy_rate*100,"{:.1f}","%",
                   {"value":62.0,"date":"2021","source":"UNESCO","is_live":False},"{:.1f}%",lb=False) +
        econ
    )

    note = ""
    if not is_national:
        note = ("<p style='margin:5px 0 0;font-size:.59rem;color:#9ca3af;font-style:italic;'>"
                "State: NDHS 2021/NBS 2023. National: World Bank/WHO APIs.</p>")
    if profile.notes:
        note += "<p style='margin:2px 0 0;font-size:.59rem;color:#9ca3af;'>ℹ️ "+profile.notes+"</p>"

    html = (
        "<div style='background:#fff;border:1px solid #e2e8f0;"
        "border-left:4px solid "+zone_colour+";border-radius:8px;"
        "padding:10px 12px;margin:4px 0;font-family:system-ui,sans-serif;'>"
        "<div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:3px;'>"
        "<span style='font-weight:700;font-size:.86rem;color:#0f172a;'>"+title+"</span>"
        +badge.replace("'","'")+
        "</div>"
        "<p style='margin:0 0 6px;font-size:.67rem;color:"+zone_colour+";font-weight:600;'>"
        +zone_full
        +((" · "+profile.primary_language) if not is_national else "")
        +" · "+("🔴" if pop.get("is_live") else "📚")+" "+pop_s+"</p>"
        +rows+note+
        "<p style='margin:5px 0 0;font-size:.58rem;color:#9ca3af;'>🔴 Live API · 📚 Estimated · ▲▼ vs national</p>"
        "</div>"
    )

    st.markdown(html, unsafe_allow_html=True)


def national_live_banner() -> None:
    """Compact live data bar shown at top of page."""
    try:
        live = fetch_nigeria_live()
    except Exception:
        return
    if not live.get("_any_live"):
        return

    xr  = live.get("exchange_rate",{})
    inf = live.get("inflation",{})
    gdp = live.get("gdp_pc",{})
    u5  = live.get("u5mr",{})

    parts = []
    if xr.get("value"):  parts.append("💱 <b>₦{:,.0f}/$</b>".format(xr["value"]))
    if inf.get("value"): parts.append("📈 CPI <b>{:.1f}%</b>".format(inf["value"]))
    if gdp.get("value"): parts.append("💰 GDP/cap <b>${:,.0f}</b>".format(gdp["value"]))
    if u5.get("value"):  parts.append("👶 U5MR <b>{:.1f}</b>/1k".format(u5["value"]))

    if not parts:
        return

    st.markdown(
        "<div style='background:#f0fdf4;border:1px solid #bbf7d0;"
        "border-radius:6px;padding:5px 14px;margin-bottom:12px;font-size:.72rem;'>"
        "<span style='color:#16a34a;font-weight:700;'>🔴 Live Nigeria&nbsp;&nbsp;</span>"
        +"&nbsp;·&nbsp;".join(parts)
        +"<span style='color:#9ca3af;font-size:.62rem;margin-left:8px;'>World Bank / WHO / ECB</span></div>",
        unsafe_allow_html=True,
    )
