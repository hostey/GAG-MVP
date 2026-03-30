"""
GAGS Design System v5.0 — Clean Light Mode
============================================
Professional, editorial light mode. White backgrounds, coloured accents,
crisp typography. Syne + DM Mono + Inter.
"""
from typing import List, Optional
import streamlit as st

DOMAIN_ACCENTS = {
    "home":           "#2563eb",
    "health":         "#0891b2",
    "security":       "#dc2626",
    "agrotech":       "#16a34a",
    "education":      "#d97706",
    "finance":        "#0d9488",
    "judicial":       "#7c3aed",
    "disinformation": "#ea580c",
    "monitor":        "#0284c7",
    "scenario":       "#6d28d9",
    "economic":       "#b45309",
}


def _rgb(h: str) -> str:
    h = h.lstrip("#")
    return f"{int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)}"


def build_css(accent: str = "#2563eb") -> str:
    acr = _rgb(accent)
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=DM+Mono:wght@400;500&family=Inter:wght@300;400;500;600&display=swap');

:root {{
  --ac:{accent}; --acr:{acr};
  --bg0:#ffffff; --bg1:#f8fafc; --bg2:#f1f5f9; --bg3:#e2e8f0;
  --t0:#0f172a;  --t1:#334155; --t2:#64748b;  --t3:#94a3b8;
  --bdr:rgba({acr},.20); --glow:0 4px 20px rgba({acr},.15);
  --ff-d:'Syne',sans-serif; --ff-m:'DM Mono',monospace; --ff-b:'Inter',sans-serif;
}}

/* ── Base ── */
html,body,.stApp{{background:var(--bg0)!important;color:var(--t0)!important;font-family:var(--ff-b)!important}}

/* ── Sidebar ── */
[data-testid="stSidebar"]{{background:var(--bg1)!important;border-right:1px solid var(--bg3)!important}}
[data-testid="stSidebar"] label,[data-testid="stSidebar"] p{{color:var(--t1)!important;font-family:var(--ff-b)!important}}
[data-testid="stSidebar"] h2{{color:var(--ac)!important;font-family:var(--ff-d)!important;font-weight:700!important}}
[data-testid="stSidebarNav"]{{background:var(--bg0)!important}}

/* ── Buttons ── */
.stButton>button{{background:var(--ac)!important;color:#fff!important;border:none!important;
  font-family:var(--ff-b)!important;font-size:.85rem!important;font-weight:600!important;
  border-radius:6px!important;padding:.5rem 1.4rem!important;transition:all .18s!important;
  box-shadow:0 1px 3px rgba(0,0,0,.12)!important}}
.stButton>button:hover{{filter:brightness(1.07)!important;box-shadow:var(--glow)!important}}
.stButton>button[kind="secondary"]{{background:var(--bg2)!important;color:var(--t1)!important;
  border:1px solid var(--bg3)!important}}

/* ── Inputs ── */
[data-testid="stSelectbox"]>div,[data-testid="stMultiSelect"]>div,
[data-testid="stNumberInput"]>div{{background:var(--bg0)!important;
  border:1px solid var(--bg3)!important;border-radius:6px!important;color:var(--t0)!important}}
.stSlider [role="slider"]{{background:var(--ac)!important}}

/* ── Metric cards ── */
[data-testid="metric-container"]{{background:var(--bg0)!important;border:1px solid var(--bg3)!important;
  border-radius:10px!important;padding:1rem 1.25rem!important;box-shadow:0 1px 3px rgba(0,0,0,.06)!important}}
[data-testid="metric-container"] [data-testid="stMetricValue"]{{color:var(--ac)!important;
  font-family:var(--ff-d)!important;font-size:1.85rem!important;font-weight:700!important}}
[data-testid="metric-container"] [data-testid="stMetricLabel"]{{color:var(--t2)!important;
  font-size:.72rem!important;text-transform:uppercase!important;letter-spacing:.07em!important;font-family:var(--ff-m)!important}}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"]{{border-bottom:2px solid var(--bg3)!important;background:transparent!important}}
[data-testid="stTabs"] [role="tab"]{{color:var(--t2)!important;font-family:var(--ff-m)!important;
  font-size:.73rem!important;letter-spacing:.04em!important;text-transform:uppercase!important;
  border-bottom:2px solid transparent!important;margin-bottom:-2px!important;
  transition:color .15s,border-color .15s!important;padding:.5rem 1rem!important}}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{{color:var(--ac)!important;
  border-bottom-color:var(--ac)!important;font-weight:600!important}}
[data-testid="stTabs"] [role="tab"]:hover{{color:var(--t0)!important}}

/* ── Expanders ── */
[data-testid="stExpander"]{{background:var(--bg1)!important;border:1px solid var(--bg3)!important;border-radius:8px!important}}
[data-testid="stExpander"] summary{{color:var(--t1)!important;font-family:var(--ff-m)!important;font-size:.82rem!important}}

/* ── Dataframes ── */
[data-testid="stDataFrame"]{{background:var(--bg0)!important;border:1px solid var(--bg3)!important;
  border-radius:8px!important;box-shadow:0 1px 3px rgba(0,0,0,.05)!important}}
[data-testid="stDataFrame"] th{{background:var(--bg1)!important;color:var(--t2)!important;
  font-family:var(--ff-m)!important;font-size:.7rem!important;letter-spacing:.06em!important;
  text-transform:uppercase!important;border-bottom:1px solid var(--bg3)!important}}
[data-testid="stDataFrame"] td{{color:var(--t1)!important;font-family:var(--ff-m)!important;font-size:.82rem!important}}

/* ── Progress / dividers / scrollbar ── */
[data-testid="stProgress"]>div>div{{background:var(--ac)!important}}
hr{{border-color:var(--bg3)!important;margin:1.5rem 0!important}}
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:var(--bg1)}}
::-webkit-scrollbar-thumb{{background:var(--bg3);border-radius:3px}}

/* ── Animations ── */
@keyframes fadeUp{{from{{opacity:0;transform:translateY(8px)}}to{{opacity:1;transform:translateY(0)}}}}
.fade-in  {{animation:fadeUp .35s ease both}}
.fade-in-2{{animation:fadeUp .35s .07s ease both}}
.fade-in-3{{animation:fadeUp .35s .14s ease both}}
.fade-in-4{{animation:fadeUp .35s .21s ease both}}

/* ── Page header ── */
.page-header{{
  background:linear-gradient(135deg,rgba({acr},.06) 0%,rgba({acr},.02) 100%);
  border:1px solid rgba({acr},.18);border-left:4px solid var(--ac);
  border-radius:12px;padding:1.75rem 2rem;margin-bottom:1.5rem;
  position:relative;overflow:hidden;
}}
.page-header::before{{content:'';position:absolute;top:-60px;right:-60px;
  width:180px;height:180px;
  background:radial-gradient(circle,rgba({acr},.07) 0%,transparent 70%);border-radius:50%}}
.page-header h1{{font-family:var(--ff-d)!important;font-size:1.9rem!important;
  font-weight:800!important;letter-spacing:-.03em!important;color:var(--t0)!important;
  margin:0 0 .3rem!important;line-height:1.1!important}}
.page-header p{{color:var(--t1)!important;font-size:.9rem!important;
  line-height:1.65!important;margin:0!important;max-width:680px}}
.eyebrow{{font-family:var(--ff-m)!important;font-size:.65rem!important;
  letter-spacing:.16em!important;text-transform:uppercase!important;
  color:var(--ac)!important;margin:0 0 .45rem!important;
  display:flex;align-items:center;gap:.4rem}}
.eyebrow::before{{content:'';display:inline-block;width:14px;height:1.5px;background:var(--ac)}}

/* ── Badge ── */
.badge{{display:inline-flex;align-items:center;padding:.18rem .65rem;border-radius:99px;
  font-family:var(--ff-m);font-size:.64rem;font-weight:500;letter-spacing:.04em;
  text-transform:uppercase;border:1px solid rgba({acr},.25);
  color:var(--ac);background:rgba({acr},.08);margin:.15rem .1rem 0 0}}

/* ── Custom metric card ── */
.mc{{background:var(--bg0);border:1px solid var(--bg3);border-radius:10px;
  padding:1rem 1.2rem;box-shadow:0 1px 3px rgba(0,0,0,.06);
  position:relative;overflow:hidden}}
.mc::after{{content:'';position:absolute;bottom:0;left:0;right:0;
  height:3px;background:var(--ac);opacity:.7}}
.mc.ok::after  {{background:#16a34a}}
.mc.warn::after{{background:#d97706}}
.mc.crit::after{{background:#dc2626}}
.mc-val{{font-family:var(--ff-d);font-size:1.8rem;font-weight:700;
  color:var(--ac);margin:0;line-height:1.1}}
.mc-label{{font-family:var(--ff-m);font-size:.67rem;letter-spacing:.08em;
  text-transform:uppercase;color:var(--t3);margin:0 0 .15rem}}

/* ── Alert variants ── */
.alert{{border-radius:8px;padding:.8rem 1rem;font-size:.84rem;line-height:1.55;margin:.65rem 0;border-left:3px solid}}
.alert-danger {{background:#fef2f2;border-color:#ef4444;color:#991b1b}}
.alert-warning{{background:#fffbeb;border-color:#f59e0b;color:#92400e}}
.alert-success{{background:#f0fdf4;border-color:#22c55e;color:#166534}}
.alert-info   {{background:#eff6ff;border-color:#3b82f6;color:#1e40af}}

/* ── Section label ── */
.section-label{{font-family:var(--ff-m);font-size:.65rem;letter-spacing:.14em;
  text-transform:uppercase;color:var(--t3);display:flex;align-items:center;
  gap:.65rem;margin:1.5rem 0 .85rem}}
.section-label::after{{content:'';flex:1;height:1px;background:var(--bg3)}}

/* ── Live badge ── */
.live-badge{{display:inline-flex;align-items:center;gap:.35rem;
  background:rgba({acr},.08);border:1px solid rgba({acr},.2);border-radius:99px;
  padding:.2rem .65rem;font-family:var(--ff-m);font-size:.67rem;
  letter-spacing:.05em;text-transform:uppercase;color:var(--ac)}}
@keyframes livePulse{{0%,100%{{opacity:1}}50%{{opacity:.4}}}}
.live-dot{{width:7px;height:7px;border-radius:50%;background:var(--ac);animation:livePulse 1.8s ease infinite}}

/* ── Terminal block ── */
.terminal{{background:#0f172a;border:1px solid var(--bg3);border-radius:8px;
  padding:1rem 1.25rem;font-family:var(--ff-m);font-size:.8rem;color:#4ade80;line-height:1.7}}
.terminal .prompt{{color:#475569}}

/* ── Feature strip ── */
.feat-strip{{background:var(--bg1);border:1px solid var(--bg3);border-radius:10px;padding:1.3rem 1.6rem}}
.feat-item{{display:flex;align-items:flex-start;gap:.8rem;margin-bottom:.85rem}}
.feat-item:last-child{{margin-bottom:0}}
.feat-icon{{width:32px;height:32px;border-radius:7px;background:rgba({acr},.1);
  display:flex;align-items:center;justify-content:center;font-size:.95rem;flex-shrink:0;
  border:1px solid rgba({acr},.2)}}
.feat-text .ft{{font-size:.85rem;font-weight:600;color:var(--t0);margin:0}}
.feat-text .fs{{font-size:.76rem;color:var(--t2);margin:.06rem 0 0;line-height:1.4}}

/* ── Domain card (homepage) ── */
.domain-card{{background:var(--bg0);border:1px solid var(--bg3);border-radius:12px;
  padding:1.25rem 1.4rem;height:100%;position:relative;overflow:hidden;
  transition:border-color .2s,transform .18s,box-shadow .2s;
  box-shadow:0 1px 3px rgba(0,0,0,.05)}}
.domain-card:hover{{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.1);border-color:rgba({acr},.3)}}
.domain-card .dca{{position:absolute;top:0;left:0;right:0;height:3px;opacity:.85}}
.domain-card .dci{{font-size:1.7rem;margin:.3rem 0 .5rem;display:block}}
.domain-card .dct{{font-family:var(--ff-d);font-size:.92rem;font-weight:700;
  color:var(--t0);margin:0 0 .28rem;letter-spacing:-.01em}}
.domain-card .dcd{{font-size:.77rem;color:var(--t2);line-height:1.5;margin:0 0 .65rem}}
.dc-pill{{font-family:var(--ff-m);font-size:.6rem;letter-spacing:.04em;padding:2px 7px;
  border-radius:3px;background:var(--bg2);color:var(--t2);border:1px solid var(--bg3);
  margin:2px 2px 0 0;display:inline-block}}

/* ── Narrative box ── */
.nbox{{background:var(--bg1);border:1px solid var(--bg3);border-left:3px solid var(--ac);
  border-radius:0 8px 8px 0;padding:.85rem 1.1rem;font-size:.84rem;
  color:var(--t1);line-height:1.65;margin:.65rem 0}}
.flag-crit{{background:#fef2f2;border-left:3px solid #ef4444;border-radius:0 8px 8px 0;
  padding:.75rem 1rem;margin:.45rem 0;font-size:.82rem;color:#991b1b;line-height:1.5}}
.flag-warn{{background:#fffbeb;border-left:3px solid #f59e0b;border-radius:0 8px 8px 0;
  padding:.75rem 1rem;margin:.45rem 0;font-size:.82rem;color:#92400e;line-height:1.5}}



/* ── Slot cards (comparison mode) ── */
.slot-a{{background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;
  padding:.5rem .85rem;margin:.5rem 0;font-family:var(--ff-m);font-size:.78rem;color:#1d4ed8}}
.slot-b{{background:#fef2f2;border:1px solid #fecaca;border-radius:8px;
  padding:.5rem .85rem;margin:.5rem 0;font-family:var(--ff-m);font-size:.78rem;color:#b91c1c}}
</style>"""


def inject_css(domain: str = "home") -> None:
    st.markdown(build_css(DOMAIN_ACCENTS.get(domain, "#2563eb")), unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, subtitle: str,
                badges: Optional[List[str]] = None) -> None:
    badges_html = "".join(f'<span class="badge">{b}</span>' for b in (badges or []))
    st.markdown(f"""
<div class="page-header fade-in">
  <p class="eyebrow">{eyebrow}</p>
  <h1>{title}</h1>
  <p>{subtitle}</p>
  <div style="margin-top:.65rem">{badges_html}</div>
</div>""", unsafe_allow_html=True)


def metric_card(label: str, value: str, sub: str = "",
                delta: str = "", delta_up: bool = True,
                status: str = "") -> str:
    arrow = "↑" if delta_up else "↓"
    col   = "#16a34a" if delta_up else "#dc2626"
    d = (f'<p style="font-family:var(--ff-m);font-size:.72rem;color:{col};margin:.12rem 0 0">'
         f'{arrow} {delta}</p>') if delta else ""
    s = (f'<p style="font-family:var(--ff-m);font-size:.72rem;'
         f'color:var(--t3);margin:.1rem 0 0">{sub}</p>') if sub else ""
    return (f'<div class="mc {status}">'
            f'<p class="mc-label">{label}</p>'
            f'<p class="mc-val">{value}</p>{s}{d}</div>')


def section_label(text: str) -> None:
    st.markdown(f'<p class="section-label">{text}</p>', unsafe_allow_html=True)


def plotly_theme(accent: str = "#2563eb") -> dict:
    """Light-mode Plotly layout defaults."""
    return dict(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8fafc",
        font=dict(family="DM Mono, monospace", color="#334155", size=11),
        title_font=dict(family="Syne, sans-serif", color="#0f172a", size=13),
        colorway=[accent,"#dc2626","#16a34a","#d97706","#7c3aed",
                  "#0d9488","#ea580c","#0284c7","#6d28d9"],
        xaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0",
                   tickcolor="#94a3b8", tickfont=dict(color="#64748b", size=10)),
        yaxis=dict(gridcolor="#e2e8f0", linecolor="#e2e8f0",
                   tickcolor="#94a3b8", tickfont=dict(color="#64748b", size=10)),
        legend=dict(bgcolor="rgba(255,255,255,.9)", bordercolor="#e2e8f0",
                    borderwidth=1, font=dict(size=10)),
        margin=dict(t=42, b=26, l=26, r=14),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#e2e8f0",
                        font=dict(family="DM Mono, monospace", size=11)),
    )