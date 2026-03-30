"""
components/translate.py — GAGS Real-Time Translation Layer v2.0
================================================================
Wraps every Streamlit output call so ALL text on every page is
automatically translated using Google Translate (free, no API key).

Works for:
  • Every st.markdown(), st.write(), st.info(), st.warning(), st.error(),
    st.success(), st.caption(), st.metric(), st.subheader(), st.header()
  • Dynamic f-string outputs (narratives, compliance text, XAI explanations)
  • Plotly chart titles, axis labels, and annotations
  • Alert messages and policy recommendations from governance_logic

Architecture
------------
1. User picks language from the sidebar switcher (en / ha / yo / ig)
2. active_lang() returns the current language code
3. tx(text) translates any string on the fly via Google Translate
   — Results cached in st.session_state to avoid repeated API calls
   — HTML tags are preserved (only text nodes translated)
   — Empty strings, numbers, and error messages pass through untouched
4. Monkey-patched wrappers replace st.markdown, st.info, etc. so
   existing page code needs ZERO changes after importing this module

Usage
-----
Add ONE line at the top of each page, before any other st. calls:

    from components.translate import install_auto_translate
    install_auto_translate()

That's it. Every st.markdown("...") on that page will automatically
translate its content before rendering.

For manual use:
    from components.translate import tx
    st.metric(tx("Fairness Score"), f"{score:.2%}")

Supported languages
-------------------
  en — English  (no translation, passthrough)
  ha — Hausa    (Google code: 'ha')
  yo — Yoruba   (Google code: 'yo')
  ig — Igbo     (Google code: 'ig')

Requirements (add to requirements.txt)
---------------------------------------
  requests>=2.31.0   (already in most Streamlit deployments)

Offline fallback
----------------
If Google Translate is unreachable (offline dev, rate-limited):
  — Falls back to the i18n.py static dictionary if the key exists
  — Returns the original English string unchanged otherwise
  — Logs a single warning, then goes silent for the session

Notes on Plotly translation
----------------------------
Call tx_plotly(fig) after building any Plotly figure to translate
all text nodes in the figure layout and traces. Example:

    fig = px.bar(df, ...)
    fig = tx_plotly(fig)
    st.plotly_chart(fig)
"""

from __future__ import annotations
import re
import time
import hashlib
import functools
import threading
from typing import Optional

import streamlit as st

# ── Language configuration ─────────────────────────────────────────────────────
LANG_KEY        = "_gags_lang"          # session_state key
CACHE_KEY       = "_tx_cache"           # translation cache in session_state
RATE_LIMIT_KEY  = "_tx_rate_warned"     # flag: have we warned about rate-limiting?

GOOGLE_LANG_MAP = {
    "en": None,   # passthrough
    "ha": "ha",
    "yo": "yo",
    "ig": "ig",
}

# Strings under this length (chars) are not worth translating
MIN_TX_LENGTH = 3

# Google Translate free endpoint — no API key required
GOOGLE_URL = "https://translate.googleapis.com/translate_a/single"

# Thread lock for cache writes
_cache_lock = threading.Lock()


# ── File-based persistent translation cache ────────────────────────────────────
# Survives page refreshes. Stored in .streamlit/tx_cache.json on Render disk.

import json
import pathlib

_TX_FILE_CACHE: dict | None = None
_TX_CACHE_PATH = pathlib.Path(".streamlit/tx_cache.json")


def _load_file_cache() -> dict:
    global _TX_FILE_CACHE
    if _TX_FILE_CACHE is not None:
        return _TX_FILE_CACHE
    try:
        _TX_CACHE_PATH.parent.mkdir(exist_ok=True)
        if _TX_CACHE_PATH.exists():
            _TX_FILE_CACHE = json.loads(_TX_CACHE_PATH.read_text(encoding="utf-8"))
        else:
            _TX_FILE_CACHE = {}
    except Exception:
        _TX_FILE_CACHE = {}
    return _TX_FILE_CACHE


def _save_file_cache(key: str, value: str) -> None:
    cache = _load_file_cache()
    cache[key] = value
    try:
        _TX_CACHE_PATH.write_text(
            json.dumps(cache, ensure_ascii=False, indent=None),
            encoding="utf-8"
        )
    except Exception:
        pass  # disk write failure is non-fatal

# ── Core translation function ──────────────────────────────────────────────────

def active_lang() -> str:
    """Return the currently selected language code."""
    return st.session_state.get(LANG_KEY, "en")


def _cache_get(key: str) -> Optional[str]:
    # Check session cache first (fastest)
    cache = st.session_state.get(CACHE_KEY, {})
    if key in cache:
        return cache[key]
    # Check file cache (survives refreshes)
    file_cache = _load_file_cache()
    if key in file_cache:
        # Promote to session cache
        _cache_set(key, file_cache[key])
        return file_cache[key]
    return None


def _cache_set(key: str, value: str) -> None:
    with _cache_lock:
        if CACHE_KEY not in st.session_state:
            st.session_state[CACHE_KEY] = {}
        st.session_state[CACHE_KEY][key] = value
    # Also persist to file cache asynchronously
    _save_file_cache(key, value)


def _google_translate(text: str, target_lang: str) -> Optional[str]:
    """
    Call the free Google Translate API.
    Returns translated string or None on failure.
    """
    try:
        import requests
        params = {
            "client": "gtx",
            "sl":     "en",
            "tl":     target_lang,
            "dt":     "t",
            "q":      text,
        }
        resp = requests.get(GOOGLE_URL, params=params, timeout=4)
        resp.raise_for_status()
        data = resp.json()
        # Response format: [[["translated","original",...], ...], ...]
        parts = []
        for item in data[0]:
            if item and item[0]:
                parts.append(item[0])
        return "".join(parts) if parts else None
    except Exception:
        return None


def _i18n_fallback(text: str, lang: str) -> Optional[str]:
    """Try static i18n.py dictionary as fallback."""
    try:
        from components.i18n import TRANSLATIONS
        # Look for exact match in any key
        for key, langs in TRANSLATIONS.items():
            if langs.get("en") == text:
                return langs.get(lang) or langs.get("en")
    except ImportError:
        pass
    return None


def tx(text: str, lang: Optional[str] = None) -> str:
    """
    Translate `text` to the active language.

    Parameters
    ----------
    text : The string to translate (English).
    lang : Override language. Uses active_lang() if None.

    Returns
    -------
    Translated string. Falls back to original English on any error.
    """
    if not text or not isinstance(text, str):
        return text

    lang = lang or active_lang()

    # Passthrough for English or very short strings
    if lang == "en" or len(text.strip()) < MIN_TX_LENGTH:
        return text

    # Passthrough for strings that are purely numeric or code
    stripped = text.strip()
    if re.match(r'^[\d\s%.,±\-+*/=:()[\]{}]+$', stripped):
        return text

    google_lang = GOOGLE_LANG_MAP.get(lang)
    if not google_lang:
        return text

    # Check cache first
    cache_key = hashlib.md5(f"{lang}:{text}".encode()).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    # Try Google Translate
    result = _google_translate(text, google_lang)

    if result:
        _cache_set(cache_key, result)
        return result

    # Fallback 1: i18n static dictionary
    static = _i18n_fallback(text, lang)
    if static:
        _cache_set(cache_key, static)
        return static

    # Fallback 2: warn once, return original
    if not st.session_state.get(RATE_LIMIT_KEY):
        st.session_state[RATE_LIMIT_KEY] = True
        # Warn silently in console, not in UI
        import warnings
        warnings.warn(
            f"[GAGS translate] Google Translate unreachable. "
            f"Falling back to English for lang='{lang}'.",
            RuntimeWarning, stacklevel=2
        )
    _cache_set(cache_key, text)   # cache the original to avoid retrying
    return text


def tx_html(html: str, lang: Optional[str] = None) -> str:
    """
    Translate visible text inside an HTML string while preserving all tags.

    HTML attributes, class names, hrefs, and tag names are untouched.
    Only the text content between tags is translated.

    Example:
        tx_html('<b>Fairness Score</b>: 0.72')
        → '<b>Ma\'aunin Adalci</b>: 0.72'
    """
    lang = lang or active_lang()
    if lang == "en":
        return html

    # Extract and translate text nodes only
    def translate_node(m: re.Match) -> str:
        node = m.group(0)
        # Skip if it's a tag
        if node.startswith("<"):
            return node
        translated = tx(node, lang)
        return translated

    # Split on HTML tags, translate text segments
    parts = re.split(r'(<[^>]+>)', html)
    translated_parts = []
    for part in parts:
        if part.startswith("<"):
            translated_parts.append(part)  # keep tags verbatim
        elif part.strip():
            translated_parts.append(tx(part, lang))
        else:
            translated_parts.append(part)  # keep whitespace

    return "".join(translated_parts)


def tx_batch(texts: list[str], lang: Optional[str] = None) -> list[str]:
    """
    Translate a list of strings efficiently.
    Uses a single API call for all uncached strings when possible.
    """
    lang = lang or active_lang()
    if lang == "en":
        return texts

    results = []
    uncached_indices = []
    uncached_texts   = []

    for i, text in enumerate(texts):
        cache_key = hashlib.md5(f"{lang}:{text}".encode()).hexdigest()
        cached = _cache_get(cache_key)
        if cached is not None:
            results.append(cached)
        else:
            results.append(None)       # placeholder
            uncached_indices.append(i)
            uncached_texts.append(text)

    # Translate uncached in bulk (join with separator)
    if uncached_texts:
        SEPARATOR = " ||| "
        joined = SEPARATOR.join(uncached_texts)
        google_lang = GOOGLE_LANG_MAP.get(lang)
        if google_lang:
            translated_joined = _google_translate(joined, google_lang)
            if translated_joined:
                translated_parts = translated_joined.split(SEPARATOR)
                for i, (orig_idx, orig_text, trans) in enumerate(
                    zip(uncached_indices, uncached_texts, translated_parts)
                ):
                    trans = trans.strip()
                    cache_key = hashlib.md5(f"{lang}:{orig_text}".encode()).hexdigest()
                    _cache_set(cache_key, trans)
                    results[orig_idx] = trans
            else:
                # fallback: fill with originals
                for orig_idx, orig_text in zip(uncached_indices, uncached_texts):
                    results[orig_idx] = orig_text

    return [r or t for r, t in zip(results, texts)]


def tx_plotly(fig):
    """
    Translate all text in a Plotly figure.

    Translates:
      - fig.layout.title.text
      - fig.layout.xaxis.title.text
      - fig.layout.yaxis.title.text
      - All annotation texts
      - All trace names
      - All category axis tick labels (if they are strings)

    Returns the modified figure (in-place + returned for chaining).
    """
    lang = active_lang()
    if lang == "en":
        return fig

    try:
        # Title
        if hasattr(fig.layout, 'title') and fig.layout.title.text:
            fig.layout.title.text = tx(fig.layout.title.text, lang)

        # Axis labels
        for axis in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
            ax = getattr(fig.layout, axis, None)
            if ax and hasattr(ax, 'title') and ax.title.text:
                ax.title.text = tx(ax.title.text, lang)

        # Annotations
        if fig.layout.annotations:
            for ann in fig.layout.annotations:
                if ann.text:
                    ann.text = tx_html(ann.text, lang)

        # Trace names and text labels
        for trace in fig.data:
            if hasattr(trace, 'name') and trace.name:
                trace.name = tx(trace.name, lang)
            # Text labels on bars/scatter
            if hasattr(trace, 'text') and trace.text:
                if isinstance(trace.text, str):
                    trace.text = tx(trace.text, lang)
                elif isinstance(trace.text, (list, tuple)):
                    trace.text = [tx(t, lang) if isinstance(t, str) else t
                                  for t in trace.text]
            # Hover template (translate label-like parts only)
            # We skip hovertemplate as it has complex format strings

        # Legend title
        if hasattr(fig.layout, 'legend') and hasattr(fig.layout.legend, 'title'):
            if fig.layout.legend.title.text:
                fig.layout.legend.title.text = tx(fig.layout.legend.title.text, lang)

    except Exception:
        pass  # never break a chart due to translation failure

    return fig


# ── Streamlit monkey-patching ──────────────────────────────────────────────────

_ST_PATCHED = False   # guard: only patch once per process


def install_auto_translate() -> None:
    """
    Monkey-patch Streamlit output functions so all text is automatically
    translated to the active language without changing any page code.

    Call this ONCE at the top of each page file:

        from components.translate import install_auto_translate
        install_auto_translate()

    Guarded: only re-patches when the language actually changes.
    Safe to call on every render — negligible overhead when no change.
    """
    lang = active_lang()

    # Only re-patch if language changed since last patch
    last_patched_lang = st.session_state.get("_tx_patched_lang")
    if lang == last_patched_lang:
        return   # ← FAST PATH: nothing to do

    _patch_streamlit(lang)
    st.session_state["_tx_patched_lang"] = lang


def _patch_streamlit(lang: str) -> None:
    """Apply translation wrappers to Streamlit functions."""

    if lang == "en":
        # Restore originals if we previously patched
        _restore_originals()
        return

    # Save originals once
    _save_originals()

    # ── st.markdown ────────────────────────────────────────────────────────
    _orig_markdown = st._original_markdown if hasattr(st, '_original_markdown') else st.markdown

    def _tx_markdown(body, *args, unsafe_allow_html: bool = False, **kwargs):
        if isinstance(body, str):
            body = tx_html(body, lang) if unsafe_allow_html else tx(body, lang)
        return _orig_markdown(body, *args, unsafe_allow_html=unsafe_allow_html, **kwargs)
    st.markdown = _tx_markdown

    # ── st.write ──────────────────────────────────────────────────────────
    _orig_write = st._original_write if hasattr(st, '_original_write') else st.write

    def _tx_write(*args, **kwargs):
        translated_args = tuple(
            tx(a, lang) if isinstance(a, str) else a for a in args
        )
        return _orig_write(*translated_args, **kwargs)
    st.write = _tx_write

    # ── Alert functions ───────────────────────────────────────────────────
    for fn_name in ['info', 'success', 'warning', 'error']:
        _orig = getattr(st, f'_original_{fn_name}', getattr(st, fn_name))

        def _make_alert_wrapper(orig_fn):
            def _tx_alert(body, *args, icon=None, **kwargs):
                if isinstance(body, str):
                    body = tx(body, lang)
                return orig_fn(body, *args, icon=icon, **kwargs)
            return _tx_alert

        setattr(st, fn_name, _make_alert_wrapper(_orig))

    # ── st.caption ────────────────────────────────────────────────────────
    _orig_caption = st._original_caption if hasattr(st, '_original_caption') else st.caption

    def _tx_caption(body, *args, **kwargs):
        if isinstance(body, str):
            body = tx(body, lang)
        return _orig_caption(body, *args, **kwargs)
    st.caption = _tx_caption

    # ── st.subheader / st.header / st.title ──────────────────────────────
    for fn_name in ['subheader', 'header', 'title']:
        _orig = getattr(st, f'_original_{fn_name}', getattr(st, fn_name))

        def _make_heading_wrapper(orig_fn):
            def _tx_heading(body, *args, **kwargs):
                if isinstance(body, str):
                    body = tx(body, lang)
                return orig_fn(body, *args, **kwargs)
            return _tx_heading

        setattr(st, fn_name, _make_heading_wrapper(_orig))

    # ── st.metric ─────────────────────────────────────────────────────────
    _orig_metric = st._original_metric if hasattr(st, '_original_metric') else st.metric

    def _tx_metric(label, value, delta=None, *args, **kwargs):
        if isinstance(label, str):
            label = tx(label, lang)
        if isinstance(delta, str):
            delta = tx(delta, lang)
        return _orig_metric(label, value, delta, *args, **kwargs)
    st.metric = _tx_metric

    # ── st.toast ──────────────────────────────────────────────────────────
    if hasattr(st, 'toast'):
        _orig_toast = st._original_toast if hasattr(st, '_original_toast') else st.toast

        def _tx_toast(body, *args, **kwargs):
            if isinstance(body, str):
                body = tx(body, lang)
            return _orig_toast(body, *args, **kwargs)
        st.toast = _tx_toast


def _save_originals() -> None:
    """Save original Streamlit functions before patching."""
    for fn_name in ['markdown', 'write', 'info', 'success', 'warning', 'error',
                    'caption', 'subheader', 'header', 'title', 'metric', 'toast']:
        orig_key = f'_original_{fn_name}'
        if not hasattr(st, orig_key):
            fn = getattr(st, fn_name, None)
            if fn:
                setattr(st, orig_key, fn)


def _restore_originals() -> None:
    """Restore original Streamlit functions (called when switching to English)."""
    for fn_name in ['markdown', 'write', 'info', 'success', 'warning', 'error',
                    'caption', 'subheader', 'header', 'title', 'metric', 'toast']:
        orig_key = f'_original_{fn_name}'
        if hasattr(st, orig_key):
            setattr(st, fn_name, getattr(st, orig_key))




# ── Pre-warm translation cache ─────────────────────────────────────────────────

# Common strings that appear on every page — translate them all at once
# on first language switch to avoid per-string API calls during rendering.
_COMMON_STRINGS = [
    "Run Simulation", "Reset", "Loading…", "Complete ✓",
    "Fairness Score", "Accuracy", "Bias Intensity", "Scenario",
    "Algorithm", "Language", "Runs", "Records",
    "Precision", "Recall", "Export", "Compliance",
    "Explainable AI", "Longitudinal", "Federated",
    "Overall Outcome Rate", "Gender Gap", "Equity Score",
    "Warning", "Error", "Success", "Info",
    "Download", "Configuration", "Settings",
]


def prewarm_translations(lang: str | None = None) -> None:
    """
    Translate all common strings upfront in a single batch API call.
    Call this once after a language switch. Much faster than translating
    each string individually as it appears on screen.
    """
    lang = lang or active_lang()
    if lang == "en":
        return

    # Find which strings are not yet cached
    uncached = []
    for text in _COMMON_STRINGS:
        key = __import__("hashlib").md5(f"{lang}:{text}".encode()).hexdigest()
        if _cache_get(key) is None:
            uncached.append(text)

    if uncached:
        tx_batch(uncached, lang)   # single API call for all


# ── Language selector widget (replaces i18n.language_switcher) ────────────────

def language_switcher(location: str = "sidebar") -> None:
    """
    Render the language selector.
    Identical API to i18n.language_switcher() — drop-in replacement.
    """
    lang_key = LANG_KEY
    current  = st.session_state.get(lang_key, "en")

    options = ["en", "ha", "yo", "ig"]
    labels  = {
        "en": "🇬🇧 English",
        "ha": "🇳🇬 Hausa",
        "yo": "🇳🇬 Yorùbá",
        "ig": "🇳🇬 Igbo",
    }

    if location == "sidebar":
        st.sidebar.markdown(
            "<p style='font-size:.78rem;font-weight:600;margin-bottom:2px;"
            "color:#6d28d9;'>🌐 Language / Harshe / Èdè / Asụsụ</p>",
            unsafe_allow_html=True,
        )

        def _on_change():
            new_lang = st.session_state["_tx_lang_select"]
            st.session_state[lang_key] = new_lang
            # Clear session cache when language changes (file cache persists)
            st.session_state[CACHE_KEY] = {}
            st.session_state.pop(RATE_LIMIT_KEY, None)
            st.session_state.pop("_tx_patched_lang", None)
            # Pre-warm common strings for the new language
            prewarm_translations(new_lang)

        st.sidebar.selectbox(
            "Language",
            options,
            index=options.index(current),
            format_func=lambda c: labels[c],
            key="_tx_lang_select",
            label_visibility="collapsed",
            on_change=_on_change,
        )

        widget_val = st.session_state.get("_tx_lang_select", current)
        if widget_val != current:
            st.session_state[lang_key] = widget_val
            st.session_state[CACHE_KEY] = {}
            st.rerun()
    else:
        cols = st.columns(len(options))
        for col, code in zip(cols, options):
            btn_type = "primary" if code == current else "secondary"
            if col.button(labels[code], key=f"_tx_btn_{code}",
                          type=btn_type, use_container_width=True):
                st.session_state[lang_key] = code
                st.session_state[CACHE_KEY] = {}
                st.rerun()


# ── Convenience re-exports for backward compatibility with i18n.py ─────────────

def get_lang() -> str:
    """Alias for active_lang(). Compatible with i18n.get_lang()."""
    return active_lang()


def t(key: str, lang: Optional[str] = None) -> str:
    """
    Backward-compatible with i18n.t().
    First checks static i18n dictionary, then uses Google Translate.
    """
    lang = lang or active_lang()

    # Try static dictionary first (fast, no network)
    try:
        from components.i18n import TRANSLATIONS
        entry = TRANSLATIONS.get(key, {})
        val = entry.get(lang) or entry.get("en")
        if val:
            return val
    except ImportError:
        pass

    # If key is actually an English string, translate it directly
    return tx(key, lang)