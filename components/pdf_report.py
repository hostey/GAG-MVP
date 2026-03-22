# components/pdf_report.py
"""
GAGS Compliance Report PDF Generator — v1.0
=============================================
Generates a professional PDF compliance report from GAGS simulation results.
Uses reportlab only — no external dependencies beyond what is already installed.

Usage
-----
from components.pdf_report import generate_pdf_compliance_report

pdf_bytes = generate_pdf_compliance_report(
    compliance_report=cr,          # from generate_compliance_report()
    model_card=mc,                 # from ExplainableModel.model_card()
    simulation_metadata=meta,      # dict with accuracy, fairness_score, etc.
    domain="agrotech",
)
st.download_button("Download PDF", pdf_bytes, "compliance_report.pdf", "application/pdf")
"""

import io
from datetime import datetime
from typing import Any, Dict, Optional

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT

# ── Colour palette ────────────────────────────────────────────────────────────
_NAVY    = colors.HexColor("#1a2e4a")
_BLUE    = colors.HexColor("#2980b9")
_GREEN   = colors.HexColor("#27ae60")
_RED     = colors.HexColor("#e74c3c")
_ORANGE  = colors.HexColor("#f39c12")
_LGRAY   = colors.HexColor("#f5f5f5")
_MGRAY   = colors.HexColor("#95a5a6")
_DGRAY   = colors.HexColor("#2c3e50")
_WHITE   = colors.white
_BLACK   = colors.black

_DOMAIN_COLORS = {
    "healthcare":        colors.HexColor("#2980b9"),
    "national_security": colors.HexColor("#c0392b"),
    "security":          colors.HexColor("#c0392b"),
    "agrotech":          colors.HexColor("#27ae60"),
    "generic":           colors.HexColor("#8e44ad"),
}


def _build_styles():
    """Build all paragraph styles."""
    base = getSampleStyleSheet()
    styles = {}

    styles["title"] = ParagraphStyle(
        "gags_title",
        parent=base["Title"],
        fontSize=22, textColor=_WHITE,
        spaceAfter=4, spaceBefore=0,
        fontName="Helvetica-Bold",
    )
    styles["subtitle"] = ParagraphStyle(
        "gags_subtitle",
        parent=base["Normal"],
        fontSize=11, textColor=_WHITE,
        spaceAfter=0,
    )
    styles["h1"] = ParagraphStyle(
        "gags_h1",
        parent=base["Heading1"],
        fontSize=14, textColor=_NAVY,
        spaceBefore=16, spaceAfter=6,
        fontName="Helvetica-Bold",
        borderPad=4,
    )
    styles["h2"] = ParagraphStyle(
        "gags_h2",
        parent=base["Heading2"],
        fontSize=11, textColor=_DGRAY,
        spaceBefore=10, spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    styles["body"] = ParagraphStyle(
        "gags_body",
        parent=base["Normal"],
        fontSize=9, textColor=_DGRAY,
        leading=13, spaceAfter=4,
    )
    styles["small"] = ParagraphStyle(
        "gags_small",
        parent=base["Normal"],
        fontSize=8, textColor=_MGRAY,
        leading=11,
    )
    styles["code"] = ParagraphStyle(
        "gags_code",
        parent=base["Code"],
        fontSize=8, textColor=_NAVY,
        fontName="Courier",
        backColor=_LGRAY,
        leftIndent=6,
    )
    styles["pass"] = ParagraphStyle(
        "gags_pass",
        parent=base["Normal"],
        fontSize=9, textColor=_GREEN,
        fontName="Helvetica-Bold",
    )
    styles["fail"] = ParagraphStyle(
        "gags_fail",
        parent=base["Normal"],
        fontSize=9, textColor=_RED,
        fontName="Helvetica-Bold",
    )
    styles["bullet"] = ParagraphStyle(
        "gags_bullet",
        parent=base["Normal"],
        fontSize=9, textColor=_DGRAY,
        leading=13, leftIndent=12,
        bulletIndent=4,
    )
    return styles


def _header_table(domain: str, model_name: str, risk_level: str, generated_at: str, styles):
    """Full-width navy header banner."""
    dom_color = _DOMAIN_COLORS.get(domain.lower(), _BLUE)
    domain_icons = {
        "healthcare": "Healthcare", "national_security": "National Security",
        "security": "National Security", "agrotech": "Agrotech", "generic": "General",
    }
    domain_label = domain_icons.get(domain.lower(), domain.title())

    data = [[
        Paragraph(f"GAGS Compliance Report", styles["title"]),
        Paragraph(f"{domain_label} Domain  |  {risk_level}<br/>"
                  f"Model: {model_name}  |  {generated_at}", styles["subtitle"]),
    ]]
    t = Table(data, colWidths=[8*cm, 11*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), _NAVY),
        ("ROWPADDING",  (0,0), (-1,-1), 10),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
        ("LINEBELOW",   (0,0), (-1,-1), 3, dom_color),
    ]))
    return t


def _kpi_row(metrics: Dict[str, float], styles):
    """3-column KPI summary row."""
    fs   = metrics.get("fairness_score", 0)
    dp   = metrics.get("demographic_parity_gap", 0)
    acc  = metrics.get("accuracy", 0)
    eo   = metrics.get("equalized_odds_gap", 0)

    def _cell(label, value, fmt, good_threshold, lower_better=False):
        v = value if isinstance(value, (int, float)) else 0
        col = _GREEN if ((v >= good_threshold) != lower_better) else _RED
        return [
            Paragraph(label,             ParagraphStyle("kl", fontSize=8, textColor=_MGRAY, fontName="Helvetica")),
            Paragraph(fmt.format(v),     ParagraphStyle("kv", fontSize=18, textColor=col, fontName="Helvetica-Bold")),
        ]

    data = [
        [_cell("Fairness Score", fs, "{:.2f}", 0.70),
         _cell("Accuracy",       acc, "{:.1%}", 0.65),
         _cell("Parity Gap",     dp,  "{:.1%}", 0.10, lower_better=True),
         _cell("Eq. Odds Gap",   eo,  "{:.1%}", 0.10, lower_better=True)],
    ]
    flat = [[item for cell in row for item in cell] for row in data]
    t = Table(flat[0:1], colWidths=[4.75*cm]*4)
    t.setStyle(TableStyle([
        ("BOX",         (0,0), (-1,-1), 0.5, colors.HexColor("#ddd")),
        ("INNERGRID",   (0,0), (-1,-1), 0.5, colors.HexColor("#eee")),
        ("BACKGROUND",  (0,0), (-1,-1), _LGRAY),
        ("ROWPADDING",  (0,0), (-1,-1), 8),
        ("VALIGN",      (0,0), (-1,-1), "MIDDLE"),
    ]))
    return t


def _framework_table(fw_name: str, fw_data: Dict, styles):
    """Render a framework compliance table with pass/fail rows."""
    checks = fw_data.get("checks", {})
    if not checks:
        return None

    header = [
        Paragraph("Check", ParagraphStyle("th", fontSize=9, textColor=_WHITE, fontName="Helvetica-Bold")),
        Paragraph("Status", ParagraphStyle("th", fontSize=9, textColor=_WHITE, fontName="Helvetica-Bold", alignment=TA_CENTER)),
    ]
    rows = [header]
    for check, status in checks.items():
        passed = status == "PASS"
        status_style = styles["pass"] if passed else styles["fail"]
        rows.append([
            Paragraph(check, styles["body"]),
            Paragraph("PASS" if passed else "FAIL", status_style),
        ])

    # Add summary row if present
    for extra_key in ["article_9_status", "certification_ready", "nigeria_ready", "clinical_deployment_ready"]:
        if extra_key in fw_data:
            val = fw_data[extra_key]
            rows.append([
                Paragraph(f"Overall: {extra_key.replace('_', ' ').title()}", styles["h2"]),
                Paragraph(val, styles["pass"] if val == "PASS" else styles["fail"]),
            ])
            break

    t = Table(rows, colWidths=[15*cm, 4*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0), _NAVY),
        ("ROWPADDING",  (0, 0), (-1, -1), 5),
        ("INNERGRID",   (0, 0), (-1, -1), 0.3, colors.HexColor("#ddd")),
        ("BOX",         (0, 0), (-1, -1), 0.5, colors.HexColor("#ccc")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1), [_WHITE, _LGRAY]),
        ("ALIGN",       (1, 0), (1, -1), "CENTER"),
        ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
    ]))
    return t


def generate_pdf_compliance_report(
    compliance_report: Dict[str, Any],
    model_card: Optional[Dict[str, Any]] = None,
    simulation_metadata: Optional[Dict[str, Any]] = None,
    domain: str = "generic",
    output_path: Optional[str] = None,
) -> bytes:
    """
    Generate a multi-page PDF compliance report.

    Parameters
    ----------
    compliance_report   : output of generate_compliance_report()
    model_card          : output of ExplainableModel.model_card() (optional)
    simulation_metadata : dict with accuracy, fairness_score, n_samples, etc.
    domain              : "healthcare" | "national_security" | "agrotech" | "generic"
    output_path         : if provided, also write to disk

    Returns
    -------
    bytes : PDF file content for st.download_button or file write
    """
    buf    = io.BytesIO()
    styles = _build_styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
        title=f"GAGS Compliance Report — {domain.title()}",
        author="GAGS Resilience Framework v3.0",
    )

    meta     = simulation_metadata or {}
    cr       = compliance_report or {}
    summ     = cr.get("summary", {})
    mc       = model_card or {}
    story    = []
    dom_color = _DOMAIN_COLORS.get(domain.lower(), _BLUE)

    # ── Page 1: Header + KPIs + overall verdict ───────────────────────────
    story.append(_header_table(
        domain,
        cr.get("model_name", mc.get("model_name", "GAGS Model")),
        cr.get("overall_risk_level", "Unknown"),
        cr.get("generated_at", datetime.now().isoformat())[:19],
        styles,
    ))
    story.append(Spacer(1, 0.4*cm))

    # Overall verdict banner
    compliant = summ.get("overall_compliant", False)
    verdict_color = _GREEN if compliant else _RED
    verdict_text  = "COMPLIANT" if compliant else "NON-COMPLIANT"
    verdict_data  = [[
        Paragraph(
            f"Overall Compliance Status: <b>{verdict_text}</b>  "
            f"|  Report ID: {cr.get('report_id', 'N/A')}  "
            f"|  Bias Findings: {summ.get('bias_findings_count', 0)}",
            ParagraphStyle("verdict", fontSize=10, textColor=_WHITE, fontName="Helvetica-Bold"),
        )
    ]]
    vt = Table(verdict_data, colWidths=[19*cm])
    vt.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), verdict_color),
        ("ROWPADDING", (0,0), (-1,-1), 8),
    ]))
    story.append(vt)
    story.append(Spacer(1, 0.4*cm))

    # KPI metrics
    kpi_metrics = {
        "fairness_score":          summ.get("fairness_score", meta.get("fairness_score", 0)),
        "demographic_parity_gap":  summ.get("demographic_parity_gap", 0),
        "accuracy":                summ.get("accuracy", meta.get("accuracy", 0)),
        "equalized_odds_gap":      summ.get("equalized_odds_gap", 0),
    }
    story.append(_kpi_row(kpi_metrics, styles))
    story.append(Spacer(1, 0.5*cm))

    # Executive summary
    story.append(Paragraph("Executive Summary", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=dom_color))
    story.append(Spacer(1, 0.2*cm))

    exec_text = (
        f"This report documents the AI governance compliance assessment for model "
        f"<b>{cr.get('model_name', 'GAGS Model')}</b> in the "
        f"<b>{domain.replace('_',' ').title()}</b> domain. "
        f"The assessment maps simulation results to {len(cr.get('frameworks', {}))} "
        f"regulatory frameworks and identifies {summ.get('bias_findings_count', 0)} "
        f"bias finding(s). "
    )
    if compliant:
        exec_text += (
            "The model meets the minimum compliance threshold across all evaluated frameworks "
            "and may proceed to pilot deployment subject to ongoing monitoring."
        )
    else:
        exec_text += (
            "The model does NOT meet minimum compliance requirements. "
            "Bias mitigation and/or architectural changes are required before deployment."
        )
    story.append(Paragraph(exec_text, styles["body"]))
    story.append(Spacer(1, 0.4*cm))

    # ── Page 1 continued: Model card summary ─────────────────────────────
    if mc:
        story.append(Paragraph("Model Information", styles["h1"]))
        story.append(HRFlowable(width="100%", thickness=2, color=dom_color))
        story.append(Spacer(1, 0.2*cm))

        mc_rows = [
            ["Model Name",   mc.get("model_name", "—")],
            ["Domain",       mc.get("domain", "—").title()],
            ["Version",      mc.get("version", "—")],
            ["Training Data", mc.get("training_data_desc", "—")[:80]],
            ["Evaluation Data", mc.get("evaluation_data_desc", "—")[:80]],
        ]
        for label, value in mc_rows:
            story.append(Paragraph(
                f"<b>{label}:</b> {value}", styles["body"]
            ))

        if mc.get("intended_uses"):
            story.append(Paragraph("Intended Uses:", styles["h2"]))
            for u in mc["intended_uses"]:
                story.append(Paragraph(f"• {u}", styles["bullet"]))

        if mc.get("bias_findings"):
            story.append(Paragraph("Bias Findings:", styles["h2"]))
            for f_ in mc["bias_findings"]:
                finding_style = ParagraphStyle(
                    "finding", fontSize=9, textColor=_RED if "significant" in f_.lower() or "detected" in f_.lower() else _DGRAY,
                    leftIndent=12, leading=13,
                )
                story.append(Paragraph(f"• {f_}", finding_style))

    story.append(PageBreak())

    # ── Page 2+: Per-framework compliance tables ──────────────────────────
    frameworks = cr.get("frameworks", {})
    for fw_name, fw_data in frameworks.items():
        story.append(Paragraph(f"{fw_name} Compliance", styles["h1"]))
        story.append(HRFlowable(width="100%", thickness=2, color=dom_color))
        story.append(Spacer(1, 0.2*cm))

        # Status summary line
        for status_key in ["article_9_compliance", "status", "nigeria_ready",
                            "clinical_deployment_ready", "certification_ready"]:
            if status_key in fw_data:
                story.append(Paragraph(
                    f"<b>Status:</b> {fw_data[status_key]}", styles["body"]
                ))
                break

        if "risk_classification" in fw_data:
            rc = fw_data["risk_classification"]
            rc_color = _RED if "HIGH" in rc else (_ORANGE if "LIMITED" in rc else _GREEN)
            story.append(Paragraph(
                f"<b>Risk Classification:</b> {rc}", styles["body"]
            ))

        # Checks table
        tbl = _framework_table(fw_name, fw_data, styles)
        if tbl:
            story.append(Spacer(1, 0.2*cm))
            story.append(tbl)

        # Recommendations
        recs = fw_data.get("recommendations", [])
        if recs:
            story.append(Paragraph("Recommendations:", styles["h2"]))
            for r in recs:
                story.append(Paragraph(f"• {r}", styles["bullet"]))

        # NIST special layout
        if fw_name == "NIST AI RMF" and isinstance(fw_data, dict):
            for fn_name, fn_data in fw_data.items():
                if isinstance(fn_data, dict) and "evidence" in fn_data:
                    fn_status = fn_data.get("status", "")
                    sc = _GREEN if fn_status == "IMPLEMENTED" else (_ORANGE if fn_status == "PARTIAL" else _RED)
                    story.append(Paragraph(
                        f"<b>{fn_name}</b> — <font color='{sc.hexval() if hasattr(sc,'hexval') else '#888'}'>{fn_status}</font>",
                        styles["body"]
                    ))
                    for ev in fn_data.get("evidence", []):
                        story.append(Paragraph(f"  • {ev}", styles["small"]))
                    for gap in fn_data.get("gaps", []):
                        story.append(Paragraph(f"  ⚠ {gap}", ParagraphStyle(
                            "gap", fontSize=8, textColor=_ORANGE, leftIndent=12
                        )))

        story.append(Spacer(1, 0.5*cm))

    # ── Final page: Limitations + sign-off ───────────────────────────────
    story.append(PageBreak())
    story.append(Paragraph("Limitations & Ethical Considerations", styles["h1"]))
    story.append(HRFlowable(width="100%", thickness=2, color=dom_color))
    story.append(Spacer(1, 0.2*cm))

    if mc.get("limitations"):
        story.append(Paragraph("Model Limitations:", styles["h2"]))
        for lim in mc["limitations"]:
            story.append(Paragraph(f"• {lim}", styles["bullet"]))

    if mc.get("ethical_considerations"):
        story.append(Paragraph("Ethical Considerations:", styles["h2"]))
        for eth in mc["ethical_considerations"]:
            story.append(Paragraph(f"• {eth}", styles["bullet"]))

    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("Disclaimer", styles["h2"]))
    story.append(Paragraph(
        "This report was auto-generated by the GAGS Resilience Framework v3.0 "
        "from simulation results. It is intended for educational and research purposes. "
        "Real-world AI deployment requires comprehensive legal, regulatory, and clinical review "
        "by qualified professionals. This document does not constitute formal regulatory certification.",
        styles["small"]
    ))

    # Sign-off box
    signoff_data = [[
        Paragraph(
            f"Generated by GAGS Resilience Framework v3.0  |  "
            f"{cr.get('generated_at', datetime.now().isoformat())[:19]}  |  "
            f"Report ID: {cr.get('report_id', 'N/A')}",
            ParagraphStyle("so", fontSize=8, textColor=_WHITE)
        )
    ]]
    so_t = Table(signoff_data, colWidths=[19*cm])
    so_t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), _NAVY),
        ("ROWPADDING", (0,0), (-1,-1), 6),
    ]))
    story.append(Spacer(1, 0.5*cm))
    story.append(so_t)

    doc.build(story)
    pdf_bytes = buf.getvalue()

    if output_path:
        with open(output_path, "wb") as f:
            f.write(pdf_bytes)

    return pdf_bytes