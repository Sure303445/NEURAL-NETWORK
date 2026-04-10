"""
CareAI -- Full Medical Report Generator
=======================================
Generates a professional PDF radiology report with:
  1. Patient Information
  2. Examination Details
  3. Findings / Key Observations
  4. Impression / Conclusion
  5. Radiologist Name and Signature

Usage (standalone):
    python generate_report.py

Usage (from app.py / Streamlit):
    from generate_report import generate_careai_report, streamlit_download_button
"""

import io
import os
import random
import math
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether,
)
from reportlab.graphics.shapes import (
    Drawing, Rect, String, Line, Circle, Polygon,
)

# ── Colour palette ─────────────────────────────────────────
C_NAVY    = colors.HexColor('#0d4f7a')
C_BLUE    = colors.HexColor('#1c6fa8')
C_SKY     = colors.HexColor('#3a9fd6')
C_PALE    = colors.HexColor('#e8f4fe')
C_BORDER  = colors.HexColor('#d0e5f5')
C_WHITE   = colors.white
C_DARK    = colors.HexColor('#1a2a3a')
C_MID     = colors.HexColor('#4a6b80')
C_LIGHT   = colors.HexColor('#f5f9fe')
C_STRIP   = colors.HexColor('#0a3d60')
C_ACCENT  = colors.HexColor('#a8d4f0')

C_GREEN   = colors.HexColor('#1a7a3c')
C_GREENBG = colors.HexColor('#e2f5ea')
C_ORANGE  = colors.HexColor('#b35a00')
C_ORGBG   = colors.HexColor('#fff3e0')
C_RED     = colors.HexColor('#b52b2b')
C_REDBG   = colors.HexColor('#ffe8e8')

# ── Paragraph styles ────────────────────────────────────────
def _S():
    return {
        'sec':  ParagraphStyle('sec',  fontName='Helvetica-Bold', fontSize=8,
                    textColor=C_MID, spaceBefore=2, spaceAfter=4, letterSpacing=1),
        'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10,
                    textColor=C_DARK, leading=15),
        'bold': ParagraphStyle('bold', fontName='Helvetica-Bold', fontSize=10,
                    textColor=C_DARK, leading=15),
        'sm':   ParagraphStyle('sm',   fontName='Helvetica', fontSize=8.5,
                    textColor=C_MID, leading=13),
        'item': ParagraphStyle('item', fontName='Helvetica', fontSize=9.5,
                    textColor=C_DARK, leading=14, leftIndent=8, spaceAfter=2),
        'diag': ParagraphStyle('diag', fontName='Helvetica-Bold', fontSize=22,
                    leading=26),
        'disc': ParagraphStyle('disc', fontName='Helvetica-Oblique', fontSize=8,
                    textColor=C_MID, leading=12, alignment=TA_CENTER),
        'foot': ParagraphStyle('foot', fontName='Helvetica', fontSize=8,
                    textColor=C_MID, alignment=TA_CENTER),
        'just': ParagraphStyle('just', fontName='Helvetica', fontSize=10,
                    textColor=C_DARK, leading=15, alignment=TA_JUSTIFY),
        'adv':  ParagraphStyle('adv',  fontName='Helvetica', fontSize=9,
                    textColor=C_NAVY, leading=14, alignment=TA_JUSTIFY),
        'rpt':  ParagraphStyle('rpt',  fontName='Helvetica', fontSize=9,
                    textColor=C_MID, alignment=TA_RIGHT),
    }

# ── Header drawing (logo + title) ──────────────────────────
def _header(w):
    d = Drawing(w, 64)
    d.add(Rect(0, 0, w, 64, fillColor=C_NAVY, strokeColor=None))
    d.add(Rect(0, 0, 6, 64, fillColor=C_SKY,  strokeColor=None))
    # Hexagon icon
    cx, cy = 38, 32
    pts = []
    for i in range(6):
        a = math.radians(60*i - 30)
        pts += [cx + 18*math.cos(a), cy + 18*math.sin(a)]
    d.add(Polygon(pts, fillColor=C_BLUE, strokeColor=C_SKY, strokeWidth=1.5))
    # Cross
    d.add(Rect(cx-9, cy-2, 18, 4, fillColor=C_WHITE, strokeColor=None, rx=2))
    d.add(Rect(cx-2, cy-9,  4, 18, fillColor=C_WHITE, strokeColor=None, rx=2))
    d.add(Circle(cx, cy, 3, fillColor=C_NAVY, strokeColor=None))
    for dx, dy in [(19,4),(-18,-3),(3,20)]:
        d.add(Circle(cx+dx, cy+dy, 2, fillColor=C_SKY, strokeColor=None))
    # Text
    d.add(String(68, 40, 'CareAI',
                 fontName='Helvetica-Bold', fontSize=24, fillColor=C_WHITE))
    d.add(String(69, 24, 'Chest X-Ray Diagnostic System',
                 fontName='Helvetica', fontSize=9,
                 fillColor=colors.HexColor('#a8d4f0')))
    d.add(String(w-10, 40, 'Radiology Report',
                 fontName='Helvetica-Bold', fontSize=10,
                 fillColor=colors.HexColor('#a8d4f0'), textAnchor='end'))
    d.add(String(w-10, 24, 'AI-Assisted Differential Diagnosis',
                 fontName='Helvetica', fontSize=8,
                 fillColor=colors.HexColor('#7fb8d8'), textAnchor='end'))
    return d

# ── Probability bar drawing ─────────────────────────────────
def _prob_bar(label, pct, bar_color, width=420):
    d = Drawing(width, 22)
    bw     = width - 130
    filled = bw * (pct / 100)
    d.add(String(0, 6, label, fontName='Helvetica-Bold', fontSize=9,
                 fillColor=C_DARK))
    d.add(Rect(96, 4, bw, 12,
               fillColor=colors.HexColor('#e8f2fc'), strokeColor=None, rx=5))
    if filled > 0:
        d.add(Rect(96, 4, filled, 12,
                   fillColor=bar_color, strokeColor=None, rx=5))
    d.add(String(width-2, 6, f'{pct:.1f}%',
                 fontName='Helvetica-Bold', fontSize=9,
                 fillColor=bar_color, textAnchor='end'))
    return d

# ── Risk badge drawing ──────────────────────────────────────
def _badge(diagnosis):
    cfg = {
        'NORMAL':    ('LOW RISK',    C_GREEN,  C_GREENBG),
        'PNEUMONIA': ('MEDIUM RISK', C_ORANGE, C_ORGBG),
        'TB':        ('HIGH RISK',   C_RED,    C_REDBG),
    }
    label, fg, bg = cfg.get(diagnosis, ('UNKNOWN', C_MID, C_LIGHT))
    d = Drawing(120, 24)
    d.add(Rect(0, 0, 120, 24, fillColor=bg, strokeColor=fg,
               strokeWidth=1, rx=12))
    d.add(String(60, 7, label, fontName='Helvetica-Bold', fontSize=9,
                 fillColor=fg, textAnchor='middle'))
    return d

# ── Per-diagnosis clinical text ─────────────────────────────
_CLINICAL = {
    'NORMAL': {
        'findings': [
            'Lungs are clear and well-aerated bilaterally with no focal consolidation or opacity.',
            'No pleural effusion, pneumothorax, or hilar adenopathy identified.',
            'Cardiac silhouette is within normal limits; cardiothoracic ratio < 0.5.',
            'Costophrenic angles are sharp and well-defined bilaterally.',
            'Trachea is midline; mediastinal contours appear unremarkable.',
            'No bony abnormality of the visualised thoracic cage detected.',
            'No cavitary lesion, nodule, or mass identified in either lung field.',
        ],
        'impression': (
            'No acute cardiopulmonary pathology detected on this chest X-ray. '
            'Lung fields appear radiologically clear. '
            'Findings do not suggest active tuberculosis or pneumonia at this time.'
        ),
        'next_steps': [
            'No immediate radiological intervention required.',
            'Continue routine annual health screening as clinically appropriate.',
            'Patient advised to return if respiratory symptoms develop or worsen.',
            'Correlate with clinical history and physical examination findings.',
        ],
    },
    'PNEUMONIA': {
        'findings': [
            'Patchy airspace consolidation noted in the right lower lobe with ill-defined margins.',
            'Air bronchograms visible within the consolidation — consistent with alveolar filling.',
            'Increased interstitial markings observed in the right mid-zone.',
            'Mild blunting of the right costophrenic angle suggesting a small pleural effusion.',
            'Left lung field is relatively clear without significant abnormality.',
            'Cardiac silhouette partially obscured by adjacent right lower zone opacity.',
            'No cavitation or calcified granuloma identified.',
        ],
        'impression': (
            'Radiological findings are consistent with right lower lobe pneumonia. '
            'Clinical and microbiological correlation is strongly advised. '
            'Tuberculosis cannot be entirely excluded on imaging alone; further clinical evaluation recommended.'
        ),
        'next_steps': [
            'Urgent consultation with a pulmonologist or infectious disease specialist.',
            'Sputum culture and sensitivity, complete blood count, and CRP strongly advised.',
            'Consider empirical antibiotic therapy as guided by clinical assessment.',
            'Repeat chest X-ray in 4-6 weeks post-treatment to confirm resolution.',
            'Monitor oxygen saturation and respiratory rate closely.',
        ],
    },
    'TB': {
        'findings': [
            'Bilateral upper lobe infiltrates with cavitary changes identified — findings consistent with active pulmonary tuberculosis.',
            'Nodular opacities and fibrotic streaking noted in the apical segments bilaterally.',
            'Bilateral hilar lymphadenopathy suspected on PA projection.',
            'Calcified granulomata present, suggesting prior healed tuberculous infection.',
            'No significant pleural effusion identified; trachea remains midline.',
            'Right upper lobe cavity with air-fluid level detected — suggestive of active cavitary TB.',
            'Miliary pattern not identified; however, findings remain highly suspicious for active disease.',
        ],
        'impression': (
            'Findings are strongly consistent with active pulmonary tuberculosis. '
            'Cavitary lesions and bilateral upper lobe infiltrates raise significant concern. '
            'Immediate clinical evaluation, isolation precautions, and microbiological confirmation are mandatory. '
            'Suspicious for tuberculosis -- recommend further clinical evaluation without delay.'
        ),
        'next_steps': [
            'Immediate patient isolation and notification to public health authorities as per protocol.',
            'Sputum smear microscopy (AFB) and mycobacterial culture required urgently.',
            'Nucleic acid amplification test (NAAT / GeneXpert) recommended for rapid confirmation.',
            'Refer immediately to an infectious disease or respiratory specialist.',
            'Initiate DOTS (Directly Observed Treatment, Short-course) evaluation.',
            'Contact tracing for close contacts should be commenced without delay.',
            'Follow-up chest imaging recommended 2 months after treatment initiation.',
        ],
    },
}

# ════════════════════════════════════════════════════════════
# PUBLIC FUNCTION
# ════════════════════════════════════════════════════════════
def generate_careai_report(
    # 1. Patient Information
    patient_name:     str,
    patient_age:      int,
    patient_gender:   str,
    patient_id:       str,
    xray_date:        str,        # e.g. "14 March 2026"

    # 2. Examination Details
    exam_type:        str,        # e.g. "PA and Lateral View"
    exam_datetime:    str,        # e.g. "14 March 2026 at 10:30 AM"
    exam_reason:      str,        # e.g. "Persistent cough for 3 weeks"

    # 3. AI Diagnosis
    diagnosis:        str,        # "NORMAL" | "PNEUMONIA" | "TB"
    confidence:       float,      # 0-100
    prob_normal:      float,
    prob_pneumonia:   float,
    prob_tb:          float,

    # 5. Radiologist
    radiologist_name: str  = 'Dr. AI Diagnostic System',
    radiologist_id:   str  = 'RAD-AI-001',

    # Output
    output_path:      str  = None,
    return_bytes:     bool = False,
):
    """
    Generate a complete CareAI radiology PDF report.

    Returns:
        str   -- saved file path     (return_bytes=False, default)
        bytes -- raw PDF bytes       (return_bytes=True, for Streamlit)
    """
    diagnosis   = diagnosis.upper().strip()
    report_id   = f'RPT-{random.randint(100000, 999999)}'
    generated   = datetime.now().strftime('%d %B %Y  %H:%M:%S')
    signed_at   = datetime.now().strftime('%d %B %Y  at  %H:%M:%S')
    clinical    = _CLINICAL.get(diagnosis, _CLINICAL['NORMAL'])
    S           = _S()
    W, H        = A4
    marg        = 18*mm
    cw          = W - 2*marg

    diag_cfg = {
        'NORMAL':    (C_GREEN,  C_GREENBG, 'Lungs appear radiologically clear.'),
        'PNEUMONIA': (C_ORANGE, C_ORGBG,   'Signs of pneumonia detected.'),
        'TB':        (C_RED,    C_REDBG,   'Findings consistent with active pulmonary tuberculosis.'),
    }
    d_fg, d_bg, d_summary = diag_cfg.get(diagnosis, (C_MID, C_LIGHT, ''))

    # Buffer or file
    if return_bytes:
        buf  = io.BytesIO()
        dest = buf
    else:
        if output_path is None:
            fname = (f"CareAI_{patient_name.replace(' ','_')}"
                     f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            output_path = os.path.join(os.getcwd(), fname)
        dest = output_path

    doc = SimpleDocTemplate(
        dest, pagesize=A4,
        leftMargin=marg, rightMargin=marg,
        topMargin=10*mm, bottomMargin=18*mm,
        title=f'CareAI Report — {patient_name}',
        author='CareAI Diagnostic System',
        subject='Chest X-Ray Radiology Report',
    )

    story = []

    # ══ HEADER ═══════════════════════════════════════════════
    story.append(_header(cw))
    story.append(Spacer(1, 4*mm))

    # Report meta bar
    meta = [[
        Paragraph(f'<b>Report ID:</b> {report_id}', S['body']),
        Paragraph(f'<b>Generated:</b> {generated}', S['body']),
        Paragraph(f'<b>Signed:</b> {signed_at}', S['body']),
        Paragraph(f'<b>Radiologist:</b> {radiologist_name}', S['body']),
    ]]
    mt = Table(meta, colWidths=[cw/4]*4)
    mt.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), C_PALE),
        ('GRID',       (0,0),(-1,-1), 0.5, C_BORDER),
        ('FONTSIZE',   (0,0),(-1,-1), 9),
        ('PADDING',    (0,0),(-1,-1), 6),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
    ]))
    story += [mt, Spacer(1, 5*mm)]

    # ══ SECTION 1 — PATIENT INFORMATION ══════════════════════
    story.append(Paragraph('1.  PATIENT INFORMATION', S['sec']))
    story.append(HRFlowable(width=cw, thickness=1.5, color=C_BLUE, spaceAfter=4))
    pat = [
        [Paragraph('<b>Full Name</b>', S['sm']),
         Paragraph('<b>Age</b>', S['sm']),
         Paragraph('<b>Gender</b>', S['sm']),
         Paragraph('<b>Patient ID / MRN</b>', S['sm']),
         Paragraph('<b>Date of X-Ray</b>', S['sm'])],
        [Paragraph(patient_name, S['bold']),
         Paragraph(f'{patient_age} years', S['body']),
         Paragraph(patient_gender, S['body']),
         Paragraph(patient_id, S['body']),
         Paragraph(xray_date, S['body'])],
    ]
    pt = Table(pat, colWidths=[cw*0.26, cw*0.10, cw*0.12, cw*0.24, cw*0.28])
    pt.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), C_PALE),
        ('BACKGROUND', (0,1),(-1,1), C_WHITE),
        ('GRID',       (0,0),(-1,-1), 0.5, C_BORDER),
        ('PADDING',    (0,0),(-1,-1), 8),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
    ]))
    story += [pt, Spacer(1, 5*mm)]

    # ══ SECTION 2 — EXAMINATION DETAILS ══════════════════════
    story.append(Paragraph('2.  EXAMINATION DETAILS', S['sec']))
    story.append(HRFlowable(width=cw, thickness=1.5, color=C_BLUE, spaceAfter=4))
    exam = [
        [Paragraph('<b>Type of Examination</b>', S['sm']),
         Paragraph('<b>Date and Time of Exam</b>', S['sm']),
         Paragraph('<b>Reason for Examination</b>', S['sm'])],
        [Paragraph(f'Chest X-Ray -- {exam_type}', S['body']),
         Paragraph(exam_datetime, S['body']),
         Paragraph(exam_reason, S['body'])],
    ]
    et = Table(exam, colWidths=[cw*0.28, cw*0.28, cw*0.44])
    et.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), C_PALE),
        ('BACKGROUND', (0,1),(-1,1), C_WHITE),
        ('GRID',       (0,0),(-1,-1), 0.5, C_BORDER),
        ('PADDING',    (0,0),(-1,-1), 8),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
    ]))
    story += [et, Spacer(1, 5*mm)]

    # ══ SECTION 3 — FINDINGS / KEY OBSERVATIONS ══════════════
    story.append(KeepTogether([
        Paragraph('3.  FINDINGS / KEY OBSERVATIONS', S['sec']),
        HRFlowable(width=cw, thickness=1.5, color=C_BLUE, spaceAfter=4),
    ]))
    find_rows = []
    for i, obs in enumerate(clinical['findings']):
        find_rows.append([
            Paragraph(str(i+1), S['sm']),
            Paragraph(obs, S['item']),
        ])
    ft = Table(find_rows, colWidths=[8*mm, cw - 8*mm])
    ft.setStyle(TableStyle([
        ('ROWBACKGROUNDS', (0,0),(-1,-1), [C_WHITE, C_PALE]),
        ('GRID',           (0,0),(-1,-1), 0.5, C_BORDER),
        ('PADDING',        (0,0),(-1,-1), 7),
        ('ALIGN',          (0,0),(0,-1),  'CENTER'),
        ('VALIGN',         (0,0),(-1,-1), 'MIDDLE'),
        ('FONTNAME',       (0,0),(0,-1),  'Helvetica-Bold'),
        ('FONTSIZE',       (0,0),(0,-1),  9),
        ('TEXTCOLOR',      (0,0),(0,-1),  C_BLUE),
    ]))
    story += [ft, Spacer(1, 5*mm)]

    # AI probability breakdown (sub-section under Findings)
    story.append(Paragraph('AI PROBABILITY BREAKDOWN', S['sec']))
    story.append(HRFlowable(width=cw, thickness=1, color=C_BORDER, spaceAfter=6))
    bw = cw - 20
    pb_rows = [
        [_prob_bar('Normal',              prob_normal,    C_GREEN,  bw)],
        [_prob_bar('Pneumonia',           prob_pneumonia, C_ORANGE, bw)],
        [_prob_bar('Tuberculosis (TB)',   prob_tb,        C_RED,    bw)],
    ]
    pbt = Table(pb_rows, colWidths=[cw])
    pbt.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), C_LIGHT),
        ('GRID',       (0,0),(-1,-1), 0.5, C_BORDER),
        ('PADDING',    (0,0),(-1,-1), 8),
        ('LEFTPADDING',(0,0),(-1,-1), 12),
    ]))
    story += [pbt, Spacer(1, 5*mm)]

    # ══ SECTION 4 — IMPRESSION / CONCLUSION ══════════════════
    story.append(Paragraph('4.  IMPRESSION / CONCLUSION', S['sec']))
    story.append(HRFlowable(width=cw, thickness=1.5, color=C_BLUE, spaceAfter=4))

    # Primary diagnosis banner
    diag_inner = Table([[
        Paragraph(
            f'<font color="{d_fg.hexval()}"><b>{diagnosis}</b></font>',
            S['diag']
        ),
        _badge(diagnosis),
    ]], colWidths=[cw*0.65, cw*0.35])
    diag_inner.setStyle(TableStyle([
        ('VALIGN',  (0,0),(-1,-1), 'MIDDLE'),
        ('ALIGN',   (1,0),(1,0),   'RIGHT'),
        ('PADDING', (0,0),(-1,-1), 0),
    ]))

    conf_color = d_fg.hexval()
    diag_box = Table([
        [diag_inner],
        [Paragraph(
            f'Confidence Score: <b><font color="{conf_color}">{confidence:.1f}%</font></b>',
            S['body'])],
        [Paragraph(d_summary, S['bold'])],
        [Paragraph(clinical['impression'], S['just'])],
    ], colWidths=[cw])
    diag_box.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),(-1,-1), d_bg),
        ('GRID',          (0,0),(-1,-1), 0.5, d_fg),
        ('PADDING',       (0,0),(-1,-1), 12),
        ('LEFTPADDING',   (0,0),(-1,-1), 16),
        ('TOPPADDING',    (0,0),(0,0),   14),
        ('BOTTOMPADDING', (0,-1),(-1,-1),14),
    ]))
    story += [diag_box, Spacer(1, 4*mm)]

    # Recommendations
    rec_rows = []
    for rec in clinical['next_steps']:
        rec_rows.append([
            Paragraph('>', S['bold']),
            Paragraph(rec, S['body']),
        ])
    rt = Table(rec_rows, colWidths=[6*mm, cw - 6*mm])
    rt.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), d_bg),
        ('GRID',       (0,0),(-1,-1), 0.5, d_fg),
        ('PADDING',    (0,0),(-1,-1), 7),
        ('LEFTPADDING',(0,0),(0,-1),  10),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
        ('TEXTCOLOR',  (0,0),(0,-1),  d_fg),
        ('FONTNAME',   (0,0),(0,-1),  'Helvetica-Bold'),
    ]))
    story += [rt, Spacer(1, 5*mm)]

    # AI model metadata
    story.append(Paragraph('AI MODEL INFORMATION', S['sec']))
    story.append(HRFlowable(width=cw, thickness=1, color=C_BORDER, spaceAfter=4))
    model_rows = [
        [Paragraph('<b>Model</b>', S['sm']),
         Paragraph('<b>Architecture</b>', S['sm']),
         Paragraph('<b>Training Acc.</b>', S['sm']),
         Paragraph('<b>Test Acc.</b>', S['sm']),
         Paragraph('<b>Classes</b>', S['sm'])],
        [Paragraph('CareAI v2.1', S['body']),
         Paragraph('MobileNetV2 + Transfer Learning', S['body']),
         Paragraph('98.27%', S['body']),
         Paragraph('99.00%', S['body']),
         Paragraph('Normal | Pneumonia | TB', S['body'])],
    ]
    mdt = Table(model_rows, colWidths=[cw*0.14, cw*0.30, cw*0.16, cw*0.14, cw*0.26])
    mdt.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,0), C_PALE),
        ('BACKGROUND', (0,1),(-1,1), C_WHITE),
        ('GRID',       (0,0),(-1,-1), 0.5, C_BORDER),
        ('PADDING',    (0,0),(-1,-1), 7),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
    ]))
    story += [mdt, Spacer(1, 5*mm)]

    # ══ SECTION 5 — RADIOLOGIST SIGNATURE ════════════════════
    story.append(Paragraph('5.  RADIOLOGIST NAME AND SIGNATURE', S['sec']))
    story.append(HRFlowable(width=cw, thickness=1.5, color=C_BLUE, spaceAfter=4))
    sig = [[
        Paragraph(
            f'Reporting Radiologist: <b>{radiologist_name}</b><br/>'
            f'Radiologist ID: <b>{radiologist_id}</b><br/>'
            f'Report Generated: <b>{generated}</b><br/>'
            f'Report Signed: <b>{signed_at}</b><br/>'
            f'Report ID: <b>{report_id}</b>',
            S['body']
        ),
        Paragraph(
            f'<b>Digital Signature</b><br/><br/>'
            f'____________________________<br/>'
            f'<font size="10">{radiologist_name}</font><br/>'
            f'<font size="8" color="{C_MID.hexval()}">{radiologist_id}</font><br/>'
            f'<font size="8" color="{C_MID.hexval()}">{signed_at}</font>',
            ParagraphStyle('sig', fontName='Helvetica', fontSize=10,
                           alignment=TA_RIGHT, textColor=C_DARK, leading=15)
        ),
    ]]
    st = Table(sig, colWidths=[cw*0.58, cw*0.42])
    st.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), C_PALE),
        ('GRID',       (0,0),(-1,-1), 0.5, C_BORDER),
        ('PADDING',    (0,0),(-1,-1), 12),
        ('VALIGN',     (0,0),(-1,-1), 'MIDDLE'),
    ]))
    story += [st, Spacer(1, 5*mm)]

    # ══ PROFESSIONAL ADVISORY ═════════════════════════════════
    adv = Table([[Paragraph(
        '<b>PROFESSIONAL ADVISORY:</b>  We strongly advise you to consult a qualified '
        'medical professional for clinical confirmation of this AI-generated report. '
        'This report is produced by an artificial intelligence system and must NOT be '
        'used as the sole basis for any medical decision, treatment, or diagnosis. '
        'All findings require review by a licensed radiologist or clinician before '
        'any clinical action is taken.',
        S['adv']
    )]], colWidths=[cw])
    adv.setStyle(TableStyle([
        ('BACKGROUND', (0,0),(-1,-1), C_PALE),
        ('GRID',       (0,0),(-1,-1), 1.0, C_BLUE),
        ('PADDING',    (0,0),(-1,-1), 10),
        ('LEFTPADDING',(0,0),(-1,-1), 14),
    ]))
    story += [adv, Spacer(1, 4*mm)]

    # ══ FOOTER ════════════════════════════════════════════════
    story.append(HRFlowable(width=cw, thickness=0.5, color=C_BORDER, spaceAfter=4))
    story.append(Paragraph(
        'DISCLAIMER: This report is generated by an AI diagnostic tool for EDUCATIONAL '
        'PURPOSES ONLY. It is not a substitute for professional medical advice, diagnosis, '
        'or treatment. Always consult a qualified healthcare provider with questions regarding '
        'a medical condition.',
        S['disc']
    ))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        f'CareAI Diagnostic System  |  Neural Network-Driven Differential Diagnosis  '
        f'|  Report ID: {report_id}  |  Generated: {generated}  |  CONFIDENTIAL',
        S['foot']
    ))

    doc.build(story)

    if return_bytes:
        return buf.getvalue()
    print(f'[CareAI] Report saved -> {output_path}')
    return output_path


# ════════════════════════════════════════════════════════════
# STREAMLIT INTEGRATION HELPER
# ════════════════════════════════════════════════════════════
def streamlit_download_button(st, patient: dict, exam: dict, ai: dict,
                               radiologist: str = 'Dr. AI Diagnostic System',
                               radiologist_id: str = 'RAD-AI-001'):
    """
    Drop-in helper for app.py.  Call after model prediction to add a
    "Download PDF Report" button to your Streamlit page.

    Parameters
    ----------
    st           : the streamlit module
    patient      : {name, age, gender, id, xray_date}
    exam         : {type, datetime, reason}
    ai           : {diagnosis, confidence, prob_normal, prob_pneumonia, prob_tb}
    radiologist  : str
    radiologist_id: str

    Example in app.py
    -----------------
    from generate_report import streamlit_download_button
    streamlit_download_button(st,
        patient={'name':'Sureka R.','age':24,'gender':'Female',
                 'id':'PAT-2026-001','xray_date':'14 March 2026'},
        exam={'type':'PA View','datetime':'14 March 2026 at 10:30 AM',
              'reason':'Persistent cough for 3 weeks'},
        ai={'diagnosis':'TB','confidence':99.1,
            'prob_normal':0.5,'prob_pneumonia':0.4,'prob_tb':99.1}
    )
    """
    pdf_bytes = generate_careai_report(
        patient_name     = patient['name'],
        patient_age      = patient['age'],
        patient_gender   = patient['gender'],
        patient_id       = patient.get('id', 'N/A'),
        xray_date        = patient.get('xray_date', datetime.now().strftime('%d %B %Y')),
        exam_type        = exam.get('type', 'PA View'),
        exam_datetime    = exam.get('datetime', datetime.now().strftime('%d %B %Y %H:%M')),
        exam_reason      = exam.get('reason', 'Not specified'),
        diagnosis        = ai['diagnosis'],
        confidence       = ai['confidence'],
        prob_normal      = ai['prob_normal'],
        prob_pneumonia   = ai['prob_pneumonia'],
        prob_tb          = ai['prob_tb'],
        radiologist_name = radiologist,
        radiologist_id   = radiologist_id,
        return_bytes     = True,
    )
    fname = f"CareAI_{patient['name'].replace(' ','_')}_{ai['diagnosis']}.pdf"
    st.download_button(
        label     = 'Download PDF Report',
        data      = pdf_bytes,
        file_name = fname,
        mime      = 'application/pdf',
    )


# ════════════════════════════════════════════════════════════
# DEMO — generate 3 sample PDFs
# ════════════════════════════════════════════════════════════
if __name__ == '__main__':
    samples = [
        dict(
            patient_name='Sureka R.',      patient_age=24,  patient_gender='Female',
            patient_id='PAT-2026-001',     xray_date='14 March 2026',
            exam_type='PA View',
            exam_datetime='14 March 2026 at 10:30 AM',
            exam_reason='Persistent cough and night sweats for 4 weeks',
            diagnosis='TB',               confidence=99.1,
            prob_normal=0.5, prob_pneumonia=0.4, prob_tb=99.1,
            radiologist_name='Dr. AI Diagnostic System',
            output_path='/home/claude/CareAI_Report_TB.pdf',
        ),
        dict(
            patient_name='Arjun S.',       patient_age=28,  patient_gender='Male',
            patient_id='PAT-2026-002',     xray_date='14 March 2026',
            exam_type='PA and Lateral View',
            exam_datetime='14 March 2026 at 11:15 AM',
            exam_reason='Annual health screening',
            diagnosis='NORMAL',           confidence=98.2,
            prob_normal=98.2, prob_pneumonia=1.2, prob_tb=0.6,
            radiologist_name='Dr. AI Diagnostic System',
            output_path='/home/claude/CareAI_Report_Normal.pdf',
        ),
        dict(
            patient_name='Meena R.',       patient_age=45,  patient_gender='Female',
            patient_id='PAT-2026-003',     xray_date='14 March 2026',
            exam_type='PA View',
            exam_datetime='14 March 2026 at 02:00 PM',
            exam_reason='Fever, productive cough, and breathlessness for 1 week',
            diagnosis='PNEUMONIA',        confidence=91.4,
            prob_normal=3.1, prob_pneumonia=91.4, prob_tb=5.5,
            radiologist_name='Dr. AI Diagnostic System',
            output_path='/home/claude/CareAI_Report_Pneumonia.pdf',
        ),
    ]
    for s in samples:
        generate_careai_report(**s)
    print('\nAll 3 sample reports generated.')
