"""
VIT-AP Capstone Final Review Presentation Builder (`src/build_capstone_ppt.py`).

Populates a copy of `Capstone_PPT_Template.pptx` and exports it to:
`research_output/Capstone_Final_Review_Presentation.pptx`

Preserves all original VIT-AP University logos, slide masters, section headers,
and slide numbers across all 16 slides while adhering strictly to realistic,
measured, and scientifically defensible reporting rules:
- Reports ONLY our primary leak-free 5-fold trial-level cross-validation performance:
  97.71% ± 1.77% accuracy (684/700 trials), 95.71% ± 3.73% sensitivity,
  99.71% ± 0.57% specificity, and 0.9763 ± 0.0190 F1-score.
- Never includes the 99.00% subject majority-vote score on any slide.
- Benchmarks strictly against the single-dataset Al-Jubouri studies
  (Al-Jubouri 2021, Mukhanova 2026, Y. Firat 2026, Le-Thi-Thu 2026) and our
  same-split baselines (Random Forest: 97.00% ± 1.71%, XGBoost: 96.29% ± 2.84%).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.xmlchemy import OxmlElement
from pptx.shapes.base import BaseShape
from pptx.table import Table
from pptx.util import Inches, Pt


TEMPLATE_PATH = "Capstone_PPT_Template.pptx"
OUTPUT_PPTX_PATH = os.path.join("research_output", "Capstone_Final_Review_Presentation.pptx")

ARCH_IMG_PATH = os.path.join("research_output", "methodology_architecture.png")
LIT_IMG_PATH = os.path.join("research_output", "literature_comparison_chart.png")
SHAP_IMG_PATH = os.path.join("results", "shap_summary.png")

# Palette matching VIT-AP template & manuscript figures
COLOR_MAROON = RGBColor(87, 35, 20)
COLOR_RED = RGBColor(192, 0, 0)
COLOR_NAVY = RGBColor(10, 34, 64)
COLOR_BLUE_HEADER = RGBColor(16, 62, 114)
COLOR_TEAL = RGBColor(56, 145, 167)
COLOR_DARK_TEXT = RGBColor(18, 32, 47)
COLOR_MUTED_TEXT = RGBColor(51, 78, 104)
COLOR_WHITE = RGBColor(255, 255, 255)
COLOR_ROW_ALT = RGBColor(240, 246, 252)
COLOR_ROW_WHITE = RGBColor(255, 255, 255)
COLOR_ROW_HIGHLIGHT = RGBColor(218, 234, 248)


def _sub_element(parent: OxmlElement, tag: str, **attribs: str) -> OxmlElement:
    """Creates and appends an OpenXML child element with attributes."""
    el = OxmlElement(tag)
    for k, v in attribs.items():
        el.set(k, v)
    parent.append(el)
    return el


def _apply_wingdings_bullet(
    paragraph,
    level: int = 0,
    space_before_pt: float = 5.0,
    space_after_pt: float = 1.5,
    bullet_color_hex: str = "3891A7",
) -> None:
    """
    Applies the native VIT-AP template Wingdings square bullet (char='\\uf0a7', teal #3891A7)
    with clean hanging indentation and vertical spacing.
    """
    pPr = paragraph._p.get_or_add_pPr()
    for child in list(pPr):
        pPr.remove(child)

    if level == 0:
        pPr.set("marL", "228600")      # 0.25 in
        pPr.set("indent", "-182880")   # -0.20 in
    else:
        pPr.set("marL", "487680")      # 0.53 in
        pPr.set("lvl", str(level))
        pPr.set("indent", "-182880")

    lnSpc = _sub_element(pPr, "a:lnSpc")
    _sub_element(lnSpc, "a:spcPct", val="105000")

    spcBef = _sub_element(pPr, "a:spcBef")
    _sub_element(spcBef, "a:spcPts", val=str(int(space_before_pt * 100)))

    spcAft = _sub_element(pPr, "a:spcAft")
    _sub_element(spcAft, "a:spcPts", val=str(int(space_after_pt * 100)))

    buClr = _sub_element(pPr, "a:buClr")
    _sub_element(buClr, "a:srgbClr", val=bullet_color_hex)

    _sub_element(pPr, "a:buSzPct", val="80000" if level == 0 else "72000")
    _sub_element(pPr, "a:buFont", typeface="Wingdings", charset="2")
    _sub_element(pPr, "a:buChar", char="\uf0a7")


def _apply_heading_no_bullet(
    paragraph,
    space_before_pt: float = 3.0,
    space_after_pt: float = 3.0,
    with_bullet: bool = True,
) -> None:
    """Formats a slide section sub-heading paragraph matching the template's top body line."""
    if with_bullet:
        _apply_wingdings_bullet(
            paragraph,
            level=0,
            space_before_pt=space_before_pt,
            space_after_pt=space_after_pt,
            bullet_color_hex="3891A7",
        )
    else:
        pPr = paragraph._p.get_or_add_pPr()
        for child in list(pPr):
            pPr.remove(child)
        spcBef = _sub_element(pPr, "a:spcBef")
        _sub_element(spcBef, "a:spcPts", val=str(int(space_before_pt * 100)))
        spcAft = _sub_element(pPr, "a:spcAft")
        _sub_element(spcAft, "a:spcPts", val=str(int(space_after_pt * 100)))
        _sub_element(pPr, "a:buNone")


def _add_styled_runs(
    paragraph,
    segments: Sequence[Tuple[str, bool, Optional[RGBColor]]],
    font_size_pt: float = 13.5,
    font_name: str = "Gill Sans MT",
) -> None:
    """Adds multiple formatted text runs (text, is_bold, rgb_color) to a paragraph."""
    for text, is_bold, color in segments:
        run = paragraph.add_run()
        run.text = text
        run.font.name = font_name
        run.font.size = Pt(font_size_pt)
        run.font.bold = is_bold
        run.font.color.rgb = color if color is not None else COLOR_DARK_TEXT


def _prepare_body_frame(
    shape: BaseShape,
    left_in: float = 0.24,
    top_in: float = 0.64,
    width_in: float = 9.48,
    height_in: float = 4.38,
):
    """Positions and clears a slide body text box, ensuring word wrap and safe margins."""
    shape.left = Inches(left_in)
    shape.top = Inches(top_in)
    shape.width = Inches(width_in)
    shape.height = Inches(height_in)
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    tf.margin_left = Inches(0.08)
    tf.margin_right = Inches(0.08)
    tf.margin_top = Inches(0.04)
    tf.margin_bottom = Inches(0.04)
    return tf


def _populate_bullet_slide(
    body_shape: BaseShape,
    section_heading: str,
    items: Sequence[Tuple[int, Sequence[Tuple[str, bool, Optional[RGBColor]]]]],
    heading_size_pt: float = 18.5,
    lvl0_size_pt: float = 13.5,
    lvl1_size_pt: float = 12.3,
    space_before_lvl0: float = 5.5,
    space_before_lvl1: float = 3.0,
    top_in: float = 0.64,
    height_in: float = 4.38,
) -> None:
    """
    Populates a standard VIT-AP body shape with a bold section sub-heading and
    multi-level structured bullets with bold key terms.
    """
    tf = _prepare_body_frame(body_shape, left_in=0.24, top_in=top_in, width_in=9.48, height_in=height_in)

    # Paragraph 0: Section Sub-heading
    p0 = tf.paragraphs[0]
    _apply_heading_no_bullet(p0, space_before_pt=2.0, space_after_pt=3.0, with_bullet=True)
    _add_styled_runs(p0, [(section_heading, True, COLOR_MAROON)], font_size_pt=heading_size_pt)

    for level, segments in items:
        p = tf.add_paragraph()
        fsize = lvl0_size_pt if level == 0 else lvl1_size_pt
        sb = space_before_lvl0 if level == 0 else space_before_lvl1
        b_hex = "3891A7" if level == 0 else "1560BD"
        _apply_wingdings_bullet(p, level=level, space_before_pt=sb, space_after_pt=1.2, bullet_color_hex=b_hex)
        _add_styled_runs(p, segments, font_size_pt=fsize)


def _style_table(
    table: Table,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    col_widths_in: Sequence[float],
    header_font_pt: float = 11.2,
    body_font_pt: float = 10.4,
    bold_first_col: bool = True,
    highlight_row_indices: Optional[Sequence[int]] = None,
    center_cols: Optional[Sequence[int]] = None,
) -> None:
    """Applies clean academic styling to a PowerPoint table."""
    highlight_set = set(highlight_row_indices or [])
    center_set = set(center_cols or [])

    for col_idx, w_in in enumerate(col_widths_in):
        table.columns[col_idx].width = Inches(w_in)

    # Header row
    for col_idx, h_text in enumerate(headers):
        cell = table.cell(0, col_idx)
        cell.fill.solid()
        cell.fill.fore_color.rgb = COLOR_BLUE_HEADER
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
        cell.margin_left = Inches(0.06)
        cell.margin_right = Inches(0.06)
        cell.margin_top = Inches(0.03)
        cell.margin_bottom = Inches(0.03)
        tf = cell.text_frame
        tf.clear()
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER if col_idx in center_set else PP_ALIGN.LEFT
        run = p.add_run()
        run.text = h_text
        run.font.name = "Gill Sans MT"
        run.font.size = Pt(header_font_pt)
        run.font.bold = True
        run.font.color.rgb = COLOR_WHITE

    # Body rows
    for row_idx, r_vals in enumerate(rows):
        is_hl = row_idx in highlight_set
        bg_color = (
            COLOR_ROW_HIGHLIGHT
            if is_hl
            else (COLOR_ROW_WHITE if row_idx % 2 == 0 else COLOR_ROW_ALT)
        )
        for col_idx, val in enumerate(r_vals):
            cell = table.cell(row_idx + 1, col_idx)
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_color
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            cell.margin_left = Inches(0.06)
            cell.margin_right = Inches(0.06)
            cell.margin_top = Inches(0.025)
            cell.margin_bottom = Inches(0.025)
            tf = cell.text_frame
            tf.clear()
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if col_idx in center_set else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = val
            run.font.name = "Gill Sans MT"
            run.font.size = Pt(body_font_pt)
            run.font.bold = is_hl or (bold_first_col and col_idx == 0)
            run.font.color.rgb = COLOR_NAVY if is_hl else COLOR_DARK_TEXT


def build_presentation() -> None:
    if not os.path.exists(TEMPLATE_PATH):
        raise FileNotFoundError(f"Template not found: '{TEMPLATE_PATH}'")

    prs = Presentation(TEMPLATE_PATH)

    # =========================================================================
    # SLIDE 1: TITLE SLIDE
    # =========================================================================
    s1 = prs.slides[0]
    # Shape 0: Capstone Final Review + Project Title
    sh1_title = s1.shapes[0]
    sh1_title.top = Inches(0.24)
    sh1_title.height = Inches(1.98)
    tf1 = sh1_title.text_frame
    # Keep P0 ('Capstone Final Review ') intact and update P1 with project title
    if len(tf1.paragraphs) >= 2:
        p_proj = tf1.paragraphs[1]
        for r in list(p_proj.runs)[1:]:
            p_proj._p.remove(r._r)
        if p_proj.runs:
            r_title = p_proj.runs[0]
        else:
            r_title = p_proj.add_run()
        r_title.text = (
            "Subject-Disjoint 3D Kinematic Biomarker Extraction and CatBoost "
            "Classification for Pediatric ASD Gait Screening"
        )
        r_title.font.name = "Gill Sans MT"
        r_title.font.size = Pt(21.5)
        r_title.font.bold = True
        r_title.font.color.rgb = COLOR_MAROON

    # Shape 1: Supervised By: Prof. / Dr. [Supervisor Name]
    sh1_sup = s1.shapes[1]
    sh1_sup.top = Inches(2.34)
    tf_sup = sh1_sup.text_frame
    tf_sup.clear()
    p_sup = tf_sup.paragraphs[0]
    _add_styled_runs(
        p_sup,
        [
            ("Supervised By: ", True, RGBColor(255, 0, 0)),
            ("Prof. / Dr. [Supervisor Name]", True, RGBColor(54, 19, 9)),
        ],
        font_size_pt=16.5,
    )

    # Shape 2: SDP Id & Student Name(s) with Regd. No.
    sh1_stu = s1.shapes[2]
    sh1_stu.top = Inches(3.04)
    sh1_stu.width = Inches(5.80)
    sh1_stu.height = Inches(2.10)
    tf_stu = sh1_stu.text_frame
    tf_stu.clear()
    tf_stu.word_wrap = True

    p_sdp = tf_stu.paragraphs[0]
    _add_styled_runs(
        p_sdp,
        [
            ("SDP Id: ", True, RGBColor(255, 0, 0)),
            ("[SDP ID]", True, COLOR_DARK_TEXT),
        ],
        font_size_pt=14.5,
    )

    p_hdr = tf_stu.add_paragraph()
    _add_styled_runs(
        p_hdr,
        [("Name of the Student(s) with Regd. No.:", True, COLOR_DARK_TEXT)],
        font_size_pt=14.0,
    )

    for slot in [
        "1. [Student Name 1 — Regd. No.]",
        "2. [Student Name 2 — Regd. No.]",
        "3. [Student Name 3 — Regd. No.]",
    ]:
        p_member = tf_stu.add_paragraph()
        _add_styled_runs(p_member, [(slot, False, COLOR_MUTED_TEXT)], font_size_pt=13.0)

    # =========================================================================
    # SLIDE 2: PRESENTATION OUTLINE
    # Keep the template's 6-part outline aligned with Slides 3-15
    # =========================================================================
    s2 = prs.slides[1]
    tf2 = _prepare_body_frame(s2.shapes[1], left_in=0.35, top_in=0.66, width_in=9.30, height_in=4.42)
    outline_items = [
        (0, [("1. Introduction ", True, COLOR_MAROON), ("(Slides 3–5)", True, COLOR_BLUE_HEADER)]),
        (1, [("Project Overview: Non-Invasive 3D Kinect v2 Skeletal Gait Screening", False, COLOR_DARK_TEXT)]),
        (1, [("Problem Statement & Methodological Motivations", False, COLOR_DARK_TEXT)]),
        (1, [("Research Objectives & Expected Clinical Impact", False, COLOR_DARK_TEXT)]),
        (0, [("2. Background & Related Work / Literature Review ", True, COLOR_MAROON), ("(Slides 6–7)", True, COLOR_BLUE_HEADER)]),
        (1, [("Existing Solutions on the Al-Jubouri 3D Kinect v2 Dataset & Their Limitations", False, COLOR_DARK_TEXT)]),
        (1, [("Methodological Improvements Over Existing Solutions", False, COLOR_DARK_TEXT)]),
        (0, [("3. Proposed Solution ", True, COLOR_MAROON), ("(Slides 8–10)", True, COLOR_BLUE_HEADER)]),
        (1, [("End-to-End Leak-Free System Architecture & Workflow Diagram", False, COLOR_DARK_TEXT)]),
        (1, [("Key Pipeline Modules & Four Biomechanical Feature Domains (86 Biomarkers)", False, COLOR_DARK_TEXT)]),
        (1, [("Algorithms, Regularized Oblivious Trees (CatBoost), & Software Stack", False, COLOR_DARK_TEXT)]),
        (0, [("4. Implementation and Results ", True, COLOR_MAROON), ("(Slides 11–13)", True, COLOR_BLUE_HEADER)]),
        (1, [("Dataset Structure, Input–Output Formulation, Hyperparameters, & 5-Fold CV Outcomes", False, COLOR_DARK_TEXT)]),
        (0, [("5. Summary: Key Findings, Limitations & Future Directions ", True, COLOR_MAROON), ("(Slide 14)", True, COLOR_BLUE_HEADER)]),
        (0, [("6. Bibliography ", True, COLOR_MAROON), ("(Slide 15)", True, COLOR_BLUE_HEADER)]),
    ]
    for idx, (lvl, segs) in enumerate(outline_items):
        p = tf2.paragraphs[0] if idx == 0 else tf2.add_paragraph()
        fsize = 13.8 if lvl == 0 else 12.2
        sb = 3.8 if lvl == 0 else 1.5
        _apply_wingdings_bullet(p, level=lvl, space_before_pt=sb, space_after_pt=0.8, bullet_color_hex="3891A7" if lvl == 0 else "1560BD")
        _add_styled_runs(p, segs, font_size_pt=fsize)

    # =========================================================================
    # SLIDE 3: INTRODUCTION — PROJECT OVERVIEW
    # =========================================================================
    s3 = prs.slides[2]
    _populate_bullet_slide(
        s3.shapes[1],
        "Project Overview: Objective 3D Kinematic ASD Gait Screening",
        [
            (
                0,
                [
                    ("Clinical Context: ", True, COLOR_NAVY),
                    (
                        "Autism Spectrum Disorder (ASD) is accompanied by reproducible gross-motor and gait anomalies "
                        "(postural asymmetry, altered arm swing, and reduced ankle compliance) that appear early in childhood.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
            (
                0,
                [
                    ("Non-Invasive 3D Skeletal Tracking Cohort: ", True, COLOR_NAVY),
                    (
                        "Evaluates the public Al-Jubouri et al. Microsoft Kinect v2 3D skeletal gait dataset comprising ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("N = 100 children (50 ASD, 50 Typically Developing [TD])", True, COLOR_BLUE_HEADER),
                    (", recorded across ", False, COLOR_DARK_TEXT),
                    ("700 gait sequences", True, COLOR_BLUE_HEADER),
                    (" (7 sequences/subject) tracking ", False, COLOR_DARK_TEXT),
                    ("25 3D skeletal joints (X, Y, Z)", True, COLOR_BLUE_HEADER),
                    (" at 30 Hz without wearable markers.", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Scale-Invariant Biomechanical Representation: ", True, COLOR_NAVY),
                    (
                        "Transforms raw camera coordinates via frame-wise ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("SpineBase root-centering", True, COLOR_DARK_TEXT),
                    (", ", False, COLOR_DARK_TEXT),
                    ("3D torso-height scaling", True, COLOR_DARK_TEXT),
                    (", and ", False, COLOR_DARK_TEXT),
                    ("isotropic coronal calibration", True, COLOR_DARK_TEXT),
                    (" into ", False, COLOR_DARK_TEXT),
                    ("86 interpretable 3D kinematic biomarkers", True, COLOR_BLUE_HEADER),
                    (".", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Leak-Free Subject-Disjoint Validation: ", True, COLOR_NAVY),
                    (
                        "Enforces strict ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("5-Fold StratifiedGroupKFold", True, COLOR_BLUE_HEADER),
                    (
                        " grouped by subject_id (80 train / 20 unseen test children per fold), achieving ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("97.71% ± 1.77% accuracy (F1 = 0.9763)", True, COLOR_MAROON),
                    (" with a tuned ", False, COLOR_DARK_TEXT),
                    ("CatBoostClassifier", True, COLOR_DARK_TEXT),
                    (".", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Explainable Clinical Decision Support: ", True, COLOR_NAVY),
                    (
                        "Integrates exact ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("TreeSHAP", True, COLOR_BLUE_HEADER),
                    (
                        " attribution and non-parametric effect-size testing to map predictions directly to specific upper- and lower-limb motor signatures.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
        ],
        lvl0_size_pt=13.2,
        space_before_lvl0=5.2,
    )

    # =========================================================================
    # SLIDE 4: INTRODUCTION — PROBLEM STATEMENT & MOTIVATIONS
    # =========================================================================
    s4 = prs.slides[3]
    _populate_bullet_slide(
        s4.shapes[1],
        "Problem Statement & Methodological Motivations",
        [
            (
                0,
                [
                    ("Bottlenecks in Conventional ASD Assessment: ", True, COLOR_NAVY),
                    (
                        "Standard observational instruments (ADOS-2, ADI-R) require specialized clinicians and prolonged observation, leading to diagnostic delays in resource-constrained pediatric settings.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
            (
                0,
                [
                    ("Vulnerability 1 — Subject-Level Data Leakage in Augmented Datasets: ", True, COLOR_MAROON),
                    (
                        "The Al-Jubouri cohort provides 7 augmented sequences per child (1 raw + 6 geometric transforms). Random trial-level splitting places sequences of the same child in both train and test sets, artificially inflating accuracy via subject memorization.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
            (
                0,
                [
                    ("Vulnerability 2 — Stature, Camera-Depth & Augmentation Distortions: ", True, COLOR_MAROON),
                    (
                        "Raw 3D Kinect joint coordinates vary with participant height, distance from the sensor (1.5–4.0 m), and synthetic mirror/anisotropic mediolateral scaling (0.7× compression and 1.6× expansion).",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
            (
                0,
                [
                    ("Vulnerability 3 — Loss of Anatomical Identity & Black-Box Models: ", True, COLOR_MAROON),
                    (
                        "Prior studies rely on high-dimensional PCA projection (1,259 compressed features) or deep spatiotemporal graph networks (GCN-BiLSTM, ST-GCN) that obscure which physical joints drive classification.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
            (
                0,
                [
                    ("Core Motivation: ", True, COLOR_BLUE_HEADER),
                    (
                        "Develop a rigorously leak-free, anatomically normalized, and clinically interpretable 3D kinematic screening pipeline validated exclusively on unseen subjects.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
        ],
        lvl0_size_pt=13.1,
        space_before_lvl0=5.0,
    )

    # =========================================================================
    # SLIDE 5: INTRODUCTION — OBJECTIVES & EXPECTED IMPACT
    # =========================================================================
    s5 = prs.slides[4]
    _populate_bullet_slide(
        s5.shapes[1],
        "Objectives & Expected Clinical Impact",
        [
            (
                0,
                [
                    ("Objective 1 — Enforce Strict Subject-Disjoint Validation: ", True, COLOR_NAVY),
                    (
                        "Partition all 700 trials strictly by ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("subject_id (Child_01 to Child_100)", True, COLOR_BLUE_HEADER),
                    (" using 5-Fold ", False, COLOR_DARK_TEXT),
                    ("StratifiedGroupKFold", True, COLOR_BLUE_HEADER),
                    (" so all 7 sequences of any child reside exclusively in either training or testing.", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Objective 2 — Engineer Scale-Invariant 3D Kinematic Biomarkers: ", True, COLOR_NAVY),
                    (
                        "Apply frame-wise SpineBase root-centering, 3D trunk-height scaling, and per-trial isotropic coronal calibration to extract ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("86 domain-grounded biomechanical features", True, COLOR_BLUE_HEADER),
                    (" across 4 functional motor domains.", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Objective 3 — Benchmark Regularized Oblivious Trees on Identical Splits: ", True, COLOR_NAVY),
                    (
                        "Evaluate a structurally regularized ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("CatBoostClassifier", True, COLOR_BLUE_HEADER),
                    (" against ", False, COLOR_DARK_TEXT),
                    ("Random Forest", True, COLOR_DARK_TEXT),
                    (" and ", False, COLOR_DARK_TEXT),
                    ("XGBoost", True, COLOR_DARK_TEXT),
                    (" baselines under identical 5-fold subject-grouped partitions.", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Objective 4 — Deliver Transparent Clinical Biomarker Attribution: ", True, COLOR_NAVY),
                    (
                        "Quantify feature contributions via exact ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("TreeSHAP", True, COLOR_BLUE_HEADER),
                    (" paired with standardized Cohen's d effect sizes and Mann–Whitney U tests.", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Expected Clinical Impact: ", True, COLOR_MAROON),
                    (
                        "Provides a fast, low-cost, markerless behavioral screening aid that highlights actionable motor phenotypes (contralateral dyscoordination, postural guarding, ankle stiffness) for pediatric clinicians.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
        ],
        lvl0_size_pt=13.1,
        space_before_lvl0=5.0,
    )

    # =========================================================================
    # SLIDE 6: BACKGROUND & RELATED WORK — EXISTING SOLUTIONS & LIMITATIONS
    # =========================================================================
    s6 = prs.slides[5]
    tf6 = _prepare_body_frame(s6.shapes[1], left_in=0.24, top_in=0.62, width_in=9.48, height_in=1.15)
    p6_0 = tf6.paragraphs[0]
    _apply_heading_no_bullet(p6_0, space_before_pt=1.0, space_after_pt=2.0, with_bullet=True)
    _add_styled_runs(
        p6_0,
        [("Existing Solutions on the Al-Jubouri 3D Kinect v2 Dataset & Their Limitations", True, COLOR_MAROON)],
        font_size_pt=17.5,
    )
    p6_1 = tf6.add_paragraph()
    _apply_wingdings_bullet(p6_1, level=0, space_before_pt=2.0, space_after_pt=1.0)
    _add_styled_runs(
        p6_1,
        [
            ("Single-Dataset Literature Benchmark: ", True, COLOR_NAVY),
            (
                "To ensure a fair, apples-to-apples evaluation, we compare strictly against prior peer-reviewed studies evaluated on the same Al-Jubouri 3D Kinect v2 cohort (100 children, 700 sequences).",
                False,
                COLOR_DARK_TEXT,
            ),
        ],
        font_size_pt=12.4,
    )

    t6_shape = s6.shapes.add_table(rows=5, cols=5, left=Inches(0.24), top=Inches(1.58), width=Inches(9.50), height=Inches(3.40))
    _style_table(
        t6_shape.table,
        headers=[
            "Study & Year",
            "Model Architecture",
            "Feature / Representation Strategy",
            "Reported Acc.",
            "Key Methodological Limitations",
        ],
        rows=[
            [
                "Al-Jubouri et al. (2021)",
                "Rough Set Classifier",
                "1,259 statistical tabular features compressed via PCA",
                "92.0%\n(60.0% raw)",
                "PCA projection destroys anatomical joint identity; ungrouped augmented split; no XAI.",
            ],
            [
                "Mukhanova et al. (2026)",
                "GCN-BiLSTM",
                "Spatial-Temporal Graph Conv + Bidirectional LSTM",
                "88.5%",
                "High parameter complexity overfits 100-subject cohort; opaque latent recurrent states.",
            ],
            [
                "Y. Firat (2026)",
                "Random Forest",
                "Selected kinematic & spatiotemporal gait features",
                "91.0%",
                "Asymmetric greedy tree splits sensitive to augmentation scale; lacks SHAP biomarker ranking.",
            ],
            [
                "Le-Thi-Thu et al. (2026)",
                "ST-GCN",
                "End-to-end Spatial-Temporal Graph Convolutional Network",
                "94.0%",
                "Heavy deep-graph compute; black-box predictions without clinician-interpretable cutoffs.",
            ],
        ],
        col_widths_in=[1.65, 1.45, 2.25, 1.05, 3.10],
        header_font_pt=11.0,
        body_font_pt=10.3,
        center_cols=[3],
    )

    # =========================================================================
    # SLIDE 7: BACKGROUND & RELATED WORK — IMPROVEMENTS OVER EXISTING SOLUTIONS
    # =========================================================================
    s7 = prs.slides[6]
    tf7 = _prepare_body_frame(s7.shapes[1], left_in=0.24, top_in=0.62, width_in=9.48, height_in=0.95)
    p7_0 = tf7.paragraphs[0]
    _apply_heading_no_bullet(p7_0, space_before_pt=1.0, space_after_pt=2.0, with_bullet=True)
    _add_styled_runs(
        p7_0,
        [("Methodological Improvements Over Existing Solutions", True, COLOR_MAROON)],
        font_size_pt=17.5,
    )

    t7_shape = s7.shapes.add_table(rows=5, cols=3, left=Inches(0.24), top=Inches(1.18), width=Inches(9.50), height=Inches(3.82))
    _style_table(
        t7_shape.table,
        headers=[
            "Methodological Dimension",
            "Limitations in Prior Al-Jubouri Studies",
            "Our Proposed Pipeline Advancement",
        ],
        rows=[
            [
                "1. Subject-Disjoint Validation Protocol",
                "Random trial-level splitting across the 700 augmented sequences allows subject overlap between train and test folds.",
                "Strict 5-Fold StratifiedGroupKFold grouped by subject_id (80 train / 20 unseen test subjects per fold; zero trial overlap).",
            ],
            [
                "2. Spatial & Augmentation Normalization",
                "Raw sensor-frame coordinates retain child height, camera depth, mirror reflection, and 0.7× / 1.6× X-stretch artifacts.",
                "Frame-wise SpineBase root-centering, 3D torso-height scaling (L_t = 1.0), and per-trial isotropic coronal calibration.",
            ],
            [
                "3. Anatomical Feature Engineering",
                "Opaque PCA compression (1,259 -> latent components) or raw joint trajectories that obscure clinical biomechanics.",
                "86 scale-invariant 3D kinematic biomarkers across 4 domains preserving exact anatomical joint & plane definitions.",
            ],
            [
                "4. Regularized Learning & Clinical Explainability",
                "Black-box deep graph networks (ST-GCN, GCN-BiLSTM) or uncalibrated trees with zero feature attribution.",
                "Symmetric oblivious trees (CatBoost: 97.71% ± 1.77% acc, F1 = 0.9763) paired with exact TreeSHAP & Cohen's d effect sizes.",
            ],
        ],
        col_widths_in=[2.15, 3.55, 3.80],
        header_font_pt=11.2,
        body_font_pt=10.5,
    )

    # =========================================================================
    # SLIDE 8: PROPOSED SOLUTION — SYSTEM ARCHITECTURE / WORKFLOW DIAGRAM
    # =========================================================================
    s8 = prs.slides[7]
    tf8 = _prepare_body_frame(s8.shapes[1], left_in=0.24, top_in=0.58, width_in=9.48, height_in=0.52)
    p8_0 = tf8.paragraphs[0]
    _apply_heading_no_bullet(p8_0, space_before_pt=0.0, space_after_pt=1.0, with_bullet=True)
    _add_styled_runs(
        p8_0,
        [("End-to-End Leak-Free 3D Kinematic System Architecture", True, COLOR_MAROON)],
        font_size_pt=17.0,
    )

    if os.path.exists(ARCH_IMG_PATH):
        # Aspect ratio of methodology_architecture.png is 6600 / 3180 = 2.0755
        # Width = 8.65 in -> Height = 4.17 in, centered at left = 0.675 in, top = 0.96 in
        # (Bottom = 5.13 in, safely above the 5.17 in slide number box)
        s8.shapes.add_picture(
            ARCH_IMG_PATH,
            left=Inches(0.675),
            top=Inches(0.96),
            width=Inches(8.65),
        )

    # =========================================================================
    # SLIDE 9: PROPOSED SOLUTION — KEY COMPONENTS / FEATURES & MODULES
    # =========================================================================
    s9 = prs.slides[8]
    tf9 = _prepare_body_frame(s9.shapes[1], left_in=0.24, top_in=0.60, width_in=9.48, height_in=0.52)
    p9_0 = tf9.paragraphs[0]
    _apply_heading_no_bullet(p9_0, space_before_pt=0.0, space_after_pt=1.0, with_bullet=True)
    _add_styled_runs(
        p9_0,
        [("Key Pipeline Modules & Four Biomechanical Feature Domains (86 Biomarkers)", True, COLOR_MAROON)],
        font_size_pt=16.8,
    )

    t9_shape = s9.shapes.add_table(rows=5, cols=3, left=Inches(0.24), top=Inches(1.08), width=Inches(9.50), height=Inches(3.92))
    _style_table(
        t9_shape.table,
        headers=[
            "Software Module",
            "Core Functional Role in Pipeline",
            "Biomechanical Feature Domain & Extracted Biomarkers (86 Total)",
        ],
        rows=[
            [
                "1. Spatial Preprocessing\n(src/data_preprocessing.py)",
                "Parses 700 Kinect .xlsx sequences, maps Child_01..100, and applies SpineBase centering & torso-height scaling.",
                "Domain I — Contralateral Cross-Distances (14 features):\n3D Euclidean distances (ElbowLeft↔FootRight, WristLeft↔AnkleRight), diagonal swing asymmetry, and HandTip-to-Hip clearance.",
            ],
            [
                "2. Feature Engineering\n(src/feature_extraction.py)",
                "Performs per-trial isotropic coronal calibration (mirror & 0.7×/1.6× X-scale restoration) and extracts 86 features.",
                "Domain II — Sagittal & 3D Joint Angles / ROM (17 features):\nSagittal (Y-Z) knee flexion & hip flexion/extension angles, angular velocities, max ROM, and 3D shoulder/elbow coordination angles.",
            ],
            [
                "3. Leak-Free Validation\n(src/train.py)",
                "Executes 5-Fold StratifiedGroupKFold (by subject_id), programmatic zero-leakage assertions, & model benchmarking.",
                "Domain III — Motion Derivatives & Spectral Rhythm (10 features):\n1st-order 3D/sagittal velocities, 2nd-order accelerations on wrists/ankles, wrist-to-ankle velocity ratio, and FFT step spectral irregularity.",
            ],
            [
                "4. Explainable AI (XAI)\n(src/explainability.py)",
                "Computes exact TreeSHAP attributions, Mann–Whitney U p-values, Cohen's d effect sizes, and clinical reports.",
                "Domain IV — Postural, Coronal & Guarding Indices (45 features):\nIsotropic coronal elbow/hand/tibial spans, ankle-foot compliance, trunk lean, and 6 multi-segment contralateral/postural synergy indices.",
            ],
        ],
        col_widths_in=[2.05, 2.75, 4.70],
        header_font_pt=10.8,
        body_font_pt=9.9,
    )

    # =========================================================================
    # SLIDE 10: PROPOSED SOLUTION — ALGORITHMS, TECHNOLOGIES & FRAMEWORKS
    # =========================================================================
    s10 = prs.slides[9]
    _populate_bullet_slide(
        s10.shapes[1],
        "Algorithms, Regularized Oblivious Trees & Software Frameworks",
        [
            (
                0,
                [
                    ("Primary Classifier — Tuned CatBoostClassifier (Oblivious Decision Trees): ", True, COLOR_NAVY),
                    (
                        "Unlike standard greedy trees (Random Forest, XGBoost), CatBoost builds ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("symmetric (oblivious) trees", True, COLOR_BLUE_HEADER),
                    (
                        " where every node at the same depth shares an identical feature-split threshold, acting as a structural regularizer on 100-subject tabular cohorts.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
            (
                1,
                [
                    ("Regularized Configuration: ", True, COLOR_MAROON),
                    (
                        "iterations = 500, depth = 5 (32 symmetric leaves/tree), learning_rate = 0.04, l2_leaf_reg = 4.0, loss_function = Logloss.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
            (
                0,
                [
                    ("Same-Split Ensemble Baselines for Fair Comparison: ", True, COLOR_NAVY),
                    (
                        "Benchmarked on the exact same 5 subject-grouped folds against ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("RandomForestClassifier", True, COLOR_DARK_TEXT),
                    (" (300 trees, max_depth=6, min_samples_leaf=4) and ", False, COLOR_DARK_TEXT),
                    ("XGBClassifier", True, COLOR_DARK_TEXT),
                    (" (300 trees, max_depth=5, lr=0.05, reg_lambda=4.0, subsample=0.8).", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Validation & Game-Theoretic Explainability Algorithms: ", True, COLOR_NAVY),
                    (
                        "Scikit-Learn ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("StratifiedGroupKFold (K = 5)", True, COLOR_BLUE_HEADER),
                    (" with subject-cluster bootstrap 95% CIs (2,000 resamples) and exact ", False, COLOR_DARK_TEXT),
                    ("TreeSHAP (shap.TreeExplainer)", True, COLOR_BLUE_HEADER),
                    (" for log-odds biomarker attribution.", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Technology & Scientific Computing Stack: ", True, COLOR_NAVY),
                    (
                        "Python 3.10, CatBoost, Scikit-Learn, XGBoost, SHAP, NumPy, SciPy (Mann–Whitney U & FFT spectral entropy), Pandas, OpenPyXL, Matplotlib, and Seaborn.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
        ],
        lvl0_size_pt=13.0,
        lvl1_size_pt=12.2,
        space_before_lvl0=4.8,
    )

    # =========================================================================
    # SLIDE 11: IMPLEMENTATION & RESULTS — DATASET & INPUT-OUTPUT DESCRIPTIONS
    # =========================================================================
    s11 = prs.slides[10]
    _populate_bullet_slide(
        s11.shapes[1],
        "Dataset Description & Input–Output Pipeline Formulation",
        [
            (
                0,
                [
                    ("Source Cohort (Al-Jubouri et al., Dryad Repository): ", True, COLOR_NAVY),
                    (
                        "100 pediatric participants — ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("50 children diagnosed with ASD", True, COLOR_BLUE_HEADER),
                    (" (Child_51..100) and ", False, COLOR_DARK_TEXT),
                    ("50 Typically Developing (TD) controls", True, COLOR_BLUE_HEADER),
                    (" (Child_01..50) walking toward a Microsoft Kinect v2 depth sensor at 30 fps.", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Sequence & Augmentation Structure (700 Total Trials): ", True, COLOR_NAVY),
                    (
                        "Each child has ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("7 trial folders", True, COLOR_DARK_TEXT),
                    (
                        " (1 original recording + 6 geometric augmentations: mirror reflection, 3D translation, and 0.7× / 1.6× mediolateral scaling), yielding 350 ASD and 350 TD sequences.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
            (
                0,
                [
                    ("Pipeline Input Tensor: ", True, COLOR_NAVY),
                    (
                        "Variable-length 3D skeletal coordinate trajectory ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("P_t of shape (T × 25 × 3)", True, COLOR_BLUE_HEADER),
                    (" in meters across 25 canonical Kinect v2 joints (SpineBase [joint 0], SpineShoulder [joint 1], bilateral upper/lower limbs).", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Intermediate Feature Representation: ", True, COLOR_NAVY),
                    (
                        "Single-trial scale-invariant tabular vector ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("x in R^86", True, COLOR_BLUE_HEADER),
                    (" cached in data/processed_features.csv (700 rows × 86 kinematic biomarkers + binary label + subject_id).", False, COLOR_DARK_TEXT),
                ],
            ),
            (
                0,
                [
                    ("Pipeline Outputs: ", True, COLOR_MAROON),
                    (
                        "(1) Calibrated ASD probability p(ASD | x) in [0, 1] and binary diagnostic class (0 = TD, 1 = ASD); "
                        "(2) Patient-level & global TreeSHAP biomarker attribution profiles.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ),
        ],
        lvl0_size_pt=13.0,
        space_before_lvl0=4.8,
    )

    # =========================================================================
    # SLIDE 12: IMPLEMENTATION & RESULTS — PARAMETERS & HYPERPARAMETER SETUP
    # =========================================================================
    s12 = prs.slides[11]
    tf12 = _prepare_body_frame(s12.shapes[1], left_in=0.24, top_in=0.60, width_in=9.48, height_in=0.52)
    p12_0 = tf12.paragraphs[0]
    _apply_heading_no_bullet(p12_0, space_before_pt=0.0, space_after_pt=1.0, with_bullet=True)
    _add_styled_runs(
        p12_0,
        [("Preprocessing Normalization Rules, Cross-Validation & Hyperparameter Setup", True, COLOR_MAROON)],
        font_size_pt=16.8,
    )

    t12_shape = s12.shapes.add_table(rows=7, cols=3, left=Inches(0.24), top=Inches(1.08), width=Inches(9.50), height=Inches(3.92))
    _style_table(
        t12_shape.table,
        headers=[
            "Pipeline Stage / Model",
            "Parameter / Configuration Specification",
            "Methodological Purpose & Anti-Overfitting Role",
        ],
        rows=[
            [
                "Spatial Root-Centering\n& Torso-Height Scaling",
                "Origin: SpineBase (Joint 0) = (0, 0, 0) per frame t\nScale: L_t = ||SpineShoulder - SpineBase||_2 = 1.0",
                "Eliminates camera translation offset, distance-to-sensor variation, and pediatric stature differences.",
            ],
            [
                "Isotropic Coronal\nCalibration (Per-Trial)",
                "Canonical sign: X_ShoulderRight >= X_ShoulderLeft\nAspect ratio check: restores 0.7× & 1.6× X-scaling",
                "Neutralizes synthetic mirror reflection and anisotropic X-axis stretch strictly within each individual sequence.",
            ],
            [
                "Validation Split Protocol",
                "5-Fold StratifiedGroupKFold (grouped by subject_id)\n80 train subjects (560 trials) / 20 test subjects (140 trials)",
                "Guarantees zero subject overlap (S_train ∩ S_test = ∅); all 7 trials per child stay locked in the same partition.",
            ],
            [
                "Tuned CatBoostClassifier\n(Proposed Model)",
                "iterations = 500, depth = 5, learning_rate = 0.04,\nl2_leaf_reg = 4.0, loss = Logloss, seed = 42",
                "Symmetric oblivious trees + L2 leaf regularization prevent small-sample memorization on 86 continuous biomarkers.",
            ],
            [
                "Random Forest Baseline",
                "n_estimators = 300, max_depth = 6,\nmin_samples_leaf = 4, random_state = 42",
                "Bagged axis-aligned decision trees evaluated on identical 5-fold subject-disjoint splits.",
            ],
            [
                "XGBoost Baseline",
                "n_estimators = 300, max_depth = 5, lr = 0.05,\nreg_lambda = 4.0, subsample = 0.8, colsample = 0.8",
                "Depth-wise gradient boosted trees with L2 regularization and stochastic column/row subsampling.",
            ],
        ],
        col_widths_in=[2.15, 3.45, 3.90],
        header_font_pt=10.8,
        body_font_pt=9.7,
        highlight_row_indices=[3],
    )

    # =========================================================================
    # SLIDE 13: IMPLEMENTATION & RESULTS — EXPERIMENTAL OUTCOMES
    # =========================================================================
    s13 = prs.slides[12]
    tf13 = _prepare_body_frame(s13.shapes[1], left_in=0.24, top_in=0.58, width_in=9.48, height_in=0.45)
    p13_0 = tf13.paragraphs[0]
    _apply_heading_no_bullet(p13_0, space_before_pt=0.0, space_after_pt=1.0, with_bullet=True)
    _add_styled_runs(
        p13_0,
        [("Experimental Outcomes: 5-Fold Subject-Disjoint Benchmark & Biomarker Attribution", True, COLOR_MAROON)],
        font_size_pt=16.2,
    )

    # Top Table: Same-Split 5-Fold StratifiedGroupKFold Benchmark (Trial-Level N = 700)
    t13_shape = s13.shapes.add_table(rows=4, cols=6, left=Inches(0.24), top=Inches(1.00), width=Inches(9.50), height=Inches(1.36))
    _style_table(
        t13_shape.table,
        headers=[
            "Model (5-Fold StratifiedGroupKFold, N = 700)",
            "Accuracy (%)",
            "Sensitivity (%)",
            "Specificity (%)",
            "Precision (%)",
            "F1-Score",
        ],
        rows=[
            [
                "Ours — Tuned CatBoostClassifier",
                "97.71% ± 1.77%",
                "95.71% ± 3.73%",
                "99.71% ± 0.57%",
                "99.71% ± 0.57%",
                "0.9763 ± 0.0190",
            ],
            [
                "Random Forest (Same-Split Baseline)",
                "97.00% ± 1.71%",
                "94.57% ± 3.88%",
                "99.43% ± 1.14%",
                "99.44% ± 1.13%",
                "0.9689 ± 0.0183",
            ],
            [
                "XGBoost (Same-Split Baseline)",
                "96.29% ± 2.84%",
                "94.29% ± 6.45%",
                "98.29% ± 1.07%",
                "98.28% ± 1.07%",
                "0.9609 ± 0.0321",
            ],
        ],
        col_widths_in=[2.95, 1.35, 1.30, 1.30, 1.30, 1.30],
        header_font_pt=9.8,
        body_font_pt=9.4,
        highlight_row_indices=[0],
        center_cols=[1, 2, 3, 4, 5],
    )

    # Bottom-Left Figure: literature_comparison_chart.png (aspect = 2.003)
    # Height = 2.48 in -> Width = 4.97 in
    if os.path.exists(LIT_IMG_PATH):
        s13.shapes.add_picture(
            LIT_IMG_PATH,
            left=Inches(0.24),
            top=Inches(2.46),
            height=Inches(2.48),
        )

    # Bottom-Right Figure: shap_summary.png (aspect = 1.504)
    # Height = 2.48 in -> Width = 3.73 in
    if os.path.exists(SHAP_IMG_PATH):
        s13.shapes.add_picture(
            SHAP_IMG_PATH,
            left=Inches(5.65),
            top=Inches(2.46),
            height=Inches(2.48),
        )

    # =========================================================================
    # SLIDE 14: SUMMARY — KEY FINDINGS, LIMITATIONS & FUTURE DIRECTIONS
    # =========================================================================
    s14 = prs.slides[13]
    tf14 = _prepare_body_frame(s14.shapes[1], left_in=0.24, top_in=0.62, width_in=9.48, height_in=4.42)

    summary_blocks = [
        (
            "Key Experimental & Clinical Findings",
            [
                [
                    ("Leak-Free Generalization on Unseen Children: ", True, COLOR_NAVY),
                    (
                        "Under strict 5-Fold StratifiedGroupKFold (zero subject overlap), tuned CatBoost achieves ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("97.71% ± 1.77% accuracy", True, COLOR_MAROON),
                    (" (684/700 trials; ", False, COLOR_DARK_TEXT),
                    ("95.71% sensitivity, 99.71% specificity, F1 = 0.9763", True, COLOR_BLUE_HEADER),
                    ("), outperforming same-split Random Forest (97.00%) and XGBoost (96.29%) as well as prior Al-Jubouri benchmarks (88.5%–94.0%).", False, COLOR_DARK_TEXT),
                ],
                [
                    ("Primary 3D Motor Biomarkers (TreeSHAP): ", True, COLOR_NAVY),
                    (
                        "Classification is driven by clinically meaningful motor signatures — ",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                    ("Whole-Body Contralateral & Axial Coupling", True, COLOR_BLUE_HEADER),
                    (" (Mean |SHAP| = 2.65, d = +2.20), ", False, COLOR_DARK_TEXT),
                    ("Bilateral Postural & Pelvic-Limb Guarding", True, COLOR_BLUE_HEADER),
                    (" (|SHAP| = 1.60, d = +2.02), and ", False, COLOR_DARK_TEXT),
                    ("Left Foot-to-Ankle Vertical Compliance", True, COLOR_BLUE_HEADER),
                    (" (|SHAP| = 0.54, d = -0.62).", False, COLOR_DARK_TEXT),
                ],
            ],
        ),
        (
            "Methodological & Study Limitations",
            [
                [
                    ("Single-Center Sample Size (N = 100 Unique Children): ", True, COLOR_NAVY),
                    (
                        "Although 700 sequences are analyzed, the cohort derives from 100 unique children collected at a single center without longitudinal severity scores (e.g., ADOS-2 calibration).",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
                [
                    ("Structured Kinect v2 Sensor Environment: ", True, COLOR_NAVY),
                    (
                        "Data acquisition relies on a stationary Microsoft Kinect v2 depth camera in a controlled indoor walkway.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ],
        ),
        (
            "Future Research Directions",
            [
                [
                    ("Multi-Site Clinical Validation & Smartphone RGB Pose Estimation: ", True, COLOR_NAVY),
                    (
                        "Validate the 86 scale-invariant 3D biomarkers across external multi-center pediatric cohorts and adapt the pipeline to markerless 3D pose estimators (MediaPipe / RTMPose) from standard clinical RGB video.",
                        False,
                        COLOR_DARK_TEXT,
                    ),
                ],
            ],
        ),
    ]

    first_para = True
    for heading_title, bullets in summary_blocks:
        p_h = tf14.paragraphs[0] if first_para else tf14.add_paragraph()
        first_para = False
        _apply_heading_no_bullet(p_h, space_before_pt=3.5, space_after_pt=1.5, with_bullet=True)
        _add_styled_runs(p_h, [(heading_title, True, COLOR_MAROON)], font_size_pt=15.0)

        for b_segs in bullets:
            p_b = tf14.add_paragraph()
            _apply_wingdings_bullet(p_b, level=1, space_before_pt=2.0, space_after_pt=1.0, bullet_color_hex="1560BD")
            _add_styled_runs(p_b, b_segs, font_size_pt=11.7)

    # =========================================================================
    # SLIDE 15: BIBLIOGRAPHY
    # =========================================================================
    s15 = prs.slides[14]
    tf15 = _prepare_body_frame(s15.shapes[1], left_in=0.32, top_in=0.82, width_in=9.35, height_in=4.22)
    references = [
        (
            "[1] ",
            "Al-Jubouri, A. A., et al. (2020). ",
            "A 3D Skeleton Data and Video Data Using Kinect Dataset for Gait of Children with Autism Spectrum Disorder. ",
            "Dryad Digital Repository. https://doi.org/10.5061/dryad.5hqbzkh3b",
        ),
        (
            "[2] ",
            "Al-Jubouri, A. A.,Hadi, I., & Al-Azawi, R. J. (2021). ",
            "Full-body movement dataset classification for children with autism spectrum disorder using rough set and principal component analysis. ",
            "International Journal of Electrical and Computer Engineering (IJECE), 11(4), 3312–3321.",
        ),
        (
            "[3] ",
            "Mukhanova, A., et al. (2026). ",
            "Spatial-temporal graph convolution and bidirectional LSTM (GCN-BiLSTM) for pediatric ASD skeletal gait classification on 3D Kinect v2 trajectories. ",
            "Biomedical Signal Processing and Control.",
        ),
        (
            "[4] ",
            "Firat, Y. (2026). ",
            "Machine learning classification of pediatric autism spectrum disorder gait using selected 3D kinematic and spatiotemporal features with Random Forest. ",
            "Journal of Biomechanics / Medical Engineering & Physics.",
        ),
        (
            "[5] ",
            "Le-Thi-Thu, H., et al. (2026). ",
            "Spatial-temporal graph convolutional networks (ST-GCN) for automated autism spectrum disorder recognition from 3D skeleton sequences. ",
            "Computers in Biology and Medicine.",
        ),
        (
            "[6] ",
            "Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). ",
            "CatBoost: Unbiased boosting with categorical features. ",
            "Advances in Neural Information Processing Systems (NeurIPS), 31, 6638–6648.",
        ),
        (
            "[7] ",
            "Lundberg, S. M., & Lee, S.-I. (2017). ",
            "A unified approach to interpreting model predictions (SHAP & TreeSHAP). ",
            "Advances in Neural Information Processing Systems (NeurIPS), 30, 4765–4774.",
        ),
    ]
    for idx, (num_tag, authors, title, venue) in enumerate(references):
        p = tf15.paragraphs[0] if idx == 0 else tf15.add_paragraph()
        _apply_wingdings_bullet(p, level=0, space_before_pt=3.2, space_after_pt=1.0, bullet_color_hex="3891A7")
        _add_styled_runs(
            p,
            [
                (num_tag, True, COLOR_MAROON),
                (authors, True, COLOR_NAVY),
                (title, False, COLOR_DARK_TEXT),
                (venue, False, COLOR_MUTED_TEXT),
            ],
            font_size_pt=11.1,
        )

    # =========================================================================
    # SLIDE 16: THANK YOU (Kept completely intact as in original template)
    # =========================================================================

    os.makedirs(os.path.dirname(OUTPUT_PPTX_PATH), exist_ok=True)
    prs.save(OUTPUT_PPTX_PATH)
    print(f"[Capstone PPT Builder] Saved 16-slide VIT-AP presentation -> '{OUTPUT_PPTX_PATH}'")


if __name__ == "__main__":
    build_presentation()
