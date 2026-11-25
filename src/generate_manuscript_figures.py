"""
Publication-Grade Manuscript Figures Generator for the Leak-Free ASD 3D Kinematic Pipeline.

Generates two 300 DPI publication figures saved inside `research_output/`:
1. `research_output/methodology_architecture.png`:
   Clean block-and-arrow system architecture diagram modeled on the visual layout,
   color harmony, and structured block style of `refrence for methodology digram.png`.
2. `research_output/literature_comparison_chart.png`:
   Minimal, uncluttered horizontal accuracy comparison chart comparing published studies
   on the Al-Jubouri 3D Kinect v2 Dryad dataset (`paper studies table.png`) against our
   leak-free Tuned CatBoost model (97.71% ± 1.77% accuracy, 0.9990 AUROC).
"""

from __future__ import annotations

import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
import seaborn as sns


OUTPUT_DIR = os.path.join("research_output")


def configure_typography() -> None:
    """Sets clean publication-grade matplotlib and seaborn defaults."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "axes.edgecolor": "#334E68",
            "axes.linewidth": 0.9,
            "figure.facecolor": "#FFFFFF",
            "savefig.facecolor": "#FFFFFF",
            "text.color": "#102A43",
        }
    )


def _draw_rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    facecolor: str,
    edgecolor: str = "#243B53",
    lw: float = 1.25,
    linestyle: str = "-",
    pad: float = 0.08,
    rounding_size: float = 0.18,
    zorder: int = 2,
    alpha: float = 1.0,
) -> FancyBboxPatch:
    """Helper to add a clean rounded rectangle to an Axes in data coordinates."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad={pad},rounding_size={rounding_size}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(box)
    return box


def _draw_arrow(
    ax: plt.Axes,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str = "#102A43",
    lw: float = 1.7,
    mutation_scale: float = 15.0,
    zorder: int = 5,
    connectionstyle: str = "arc3,rad=0.0",
) -> None:
    """Helper to draw a crisp directed arrow between two coordinates."""
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle=connectionstyle,
        zorder=zorder,
    )
    ax.add_patch(arrow)


def _draw_skeleton_glyph(
    ax: plt.Axes,
    cx: float,
    cy: float,
    scale: float,
    color: str = "#1D4E89",
    root_highlight: bool = False,
    zorder: int = 4,
) -> None:
    """Draws a schematic 25-joint 3D Kinect v2 human skeleton stick figure."""
    joints = {
        "head": (cx, cy + 0.85 * scale),
        "neck": (cx, cy + 0.62 * scale),
        "spine_sh": (cx, cy + 0.48 * scale),
        "spine_mid": (cx, cy + 0.24 * scale),
        "spine_base": (cx, cy + 0.00 * scale),
        "sh_l": (cx - 0.32 * scale, cy + 0.46 * scale),
        "el_l": (cx - 0.48 * scale, cy + 0.21 * scale),
        "wr_l": (cx - 0.56 * scale, cy - 0.03 * scale),
        "sh_r": (cx + 0.32 * scale, cy + 0.46 * scale),
        "el_r": (cx + 0.48 * scale, cy + 0.21 * scale),
        "wr_r": (cx + 0.56 * scale, cy - 0.03 * scale),
        "hip_l": (cx - 0.20 * scale, cy - 0.04 * scale),
        "kn_l": (cx - 0.24 * scale, cy - 0.42 * scale),
        "an_l": (cx - 0.26 * scale, cy - 0.78 * scale),
        "hip_r": (cx + 0.20 * scale, cy - 0.04 * scale),
        "kn_r": (cx + 0.24 * scale, cy - 0.42 * scale),
        "an_r": (cx + 0.26 * scale, cy - 0.78 * scale),
    }
    bones = [
        ("head", "neck"),
        ("neck", "spine_sh"),
        ("spine_sh", "spine_mid"),
        ("spine_mid", "spine_base"),
        ("spine_sh", "sh_l"),
        ("sh_l", "el_l"),
        ("el_l", "wr_l"),
        ("spine_sh", "sh_r"),
        ("sh_r", "el_r"),
        ("el_r", "wr_r"),
        ("spine_base", "hip_l"),
        ("hip_l", "kn_l"),
        ("kn_l", "an_l"),
        ("spine_base", "hip_r"),
        ("hip_r", "kn_r"),
        ("kn_r", "an_r"),
    ]
    for j1, j2 in bones:
        p1, p2 = joints[j1], joints[j2]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=1.45, zorder=zorder)

    xs = [p[0] for p in joints.values()]
    ys = [p[1] for p in joints.values()]
    ax.scatter(xs, ys, s=16, color=color, edgecolors="#FFFFFF", linewidths=0.45, zorder=zorder + 1)

    if root_highlight:
        sb = joints["spine_base"]
        ss = joints["spine_sh"]
        ax.plot([sb[0], ss[0]], [sb[1], ss[1]], color="#D9534F", lw=2.3, zorder=zorder + 2)
        ax.scatter(
            [sb[0]],
            [sb[1]],
            s=40,
            color="#D9534F",
            edgecolors="#FFFFFF",
            linewidths=0.9,
            zorder=zorder + 3,
        )


def generate_methodology_architecture(
    output_path: str = os.path.join(OUTPUT_DIR, "methodology_architecture.png"),
) -> None:
    """
    Generates Figure 1: Complete Methodology Architecture Diagram (`methodology_architecture.png`).
    Uses a compact, high-legibility 2-Row Serpentine Pipeline Layout (`figsize=(13.0, 8.2)`, 300 DPI)
    designed so every single label (>= 10.5 pt minimum font size floor) is razor-sharp and 100%
    legible when embedded at standard A4/IEEE manuscript width (~6.5 inches) without zooming.
    """
    configure_typography()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig = plt.figure(figsize=(13.0, 8.2), dpi=300, facecolor="#FFFFFF")
    ax = fig.add_axes([0.008, 0.008, 0.984, 0.984])
    ax.set_xlim(0.0, 13.0)
    ax.set_ylim(0.0, 8.2)
    ax.axis("off")

    # Outer clean border frame
    outer_frame = Rectangle(
        (0.10, 0.10),
        12.80,
        8.00,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.5,
        zorder=0,
    )
    ax.add_patch(outer_frame)

    # =========================================================================
    # TOP ROW: STAGES 1 -> 2 -> 3 (Left-to-Right, y = 4.35 .. 7.45)
    # =========================================================================

    # -------------------------------------------------------------------------
    # STAGE 1: INPUT COHORT (KINECT V2) (x = 0.25 .. 3.45)
    # -------------------------------------------------------------------------
    ax.text(
        1.85,
        7.74,
        "1. Input Cohort (Kinect v2)",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
    )
    _draw_rounded_box(
        ax,
        x=0.25,
        y=4.35,
        w=3.20,
        h=3.10,
        facecolor="#F8FAFC",
        edgecolor="#475569",
        lw=1.6,
        pad=0.04,
        rounding_size=0.14,
    )

    # Sub-box 1A: 100 Subjects & 700 Trials (with minimalist 25-joint skeleton icon)
    _draw_rounded_box(
        ax,
        x=0.40,
        y=5.96,
        w=2.90,
        h=1.32,
        facecolor="#EFF6FF",
        edgecolor="#2563EB",
        lw=1.5,
        pad=0.04,
        rounding_size=0.12,
        zorder=3,
    )
    _draw_skeleton_glyph(
        ax,
        cx=0.72,
        cy=6.58,
        scale=0.46,
        color="#1D4ED8",
        root_highlight=True,
        zorder=4,
    )
    ax.text(
        2.15,
        6.92,
        "100 Subjects",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        2.15,
        6.34,
        "50 ASD  |  50 TD\n700 Trials (7 per child)",
        ha="center",
        va="center",
        fontsize=10.6,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # Sub-box 1B: 25 3D Skeletal Joints
    _draw_rounded_box(
        ax,
        x=0.40,
        y=4.52,
        w=2.90,
        h=1.24,
        facecolor="#EFF6FF",
        edgecolor="#2563EB",
        lw=1.5,
        pad=0.04,
        rounding_size=0.12,
        zorder=3,
    )
    ax.text(
        1.85,
        5.42,
        "25 3D Skeletal Joints",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        1.85,
        4.86,
        "30 Hz Depth Stream\n(X, Y, Z) in meters",
        ha="center",
        va="center",
        fontsize=10.6,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # -------------------------------------------------------------------------
    # STAGE 2: SPATIAL NORMALIZATION (x = 4.00 .. 7.35)
    # -------------------------------------------------------------------------
    ax.text(
        5.675,
        7.74,
        "2. Spatial Normalization",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
    )
    _draw_rounded_box(
        ax,
        x=4.00,
        y=4.35,
        w=3.35,
        h=3.10,
        facecolor="#F8FAFC",
        edgecolor="#475569",
        lw=1.6,
        pad=0.04,
        rounding_size=0.14,
    )

    # Step 2A: SpineBase Root-Centering
    _draw_rounded_box(
        ax,
        x=4.15,
        y=6.48,
        w=3.05,
        h=0.82,
        facecolor="#FEFCE8",
        edgecolor="#CA8A04",
        lw=1.5,
        pad=0.03,
        rounding_size=0.10,
        zorder=3,
    )
    ax.text(
        5.675,
        7.06,
        "SpineBase Root-Centering",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        5.675,
        6.70,
        "Removes camera offset",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # Step 2B: Trunk-Length Scaling
    _draw_rounded_box(
        ax,
        x=4.15,
        y=5.49,
        w=3.05,
        h=0.82,
        facecolor="#FEFCE8",
        edgecolor="#CA8A04",
        lw=1.5,
        pad=0.03,
        rounding_size=0.10,
        zorder=3,
    )
    ax.text(
        5.675,
        6.07,
        "Trunk-Length Scaling",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        5.675,
        5.71,
        "Removes height & depth bias",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # Step 2C: Isotropic Coronal Calibration
    _draw_rounded_box(
        ax,
        x=4.15,
        y=4.50,
        w=3.05,
        h=0.82,
        facecolor="#FAF5FF",
        edgecolor="#9333EA",
        lw=1.5,
        pad=0.03,
        rounding_size=0.10,
        zorder=3,
    )
    ax.text(
        5.675,
        5.08,
        "Isotropic Coronal Calibration",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        5.675,
        4.72,
        "Aligns mirror/mediolateral scale",
        ha="center",
        va="center",
        fontsize=10.6,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # Arrow Stage 1 -> Stage 2
    _draw_arrow(
        ax,
        3.50,
        5.90,
        3.95,
        5.90,
        color="#334155",
        lw=2.2,
        mutation_scale=18.0,
    )

    # -------------------------------------------------------------------------
    # STAGE 3: 86 KINEMATIC BIOMARKERS (x = 7.90 .. 12.75)
    # Clean 2x2 Grid of 4 Concise Domain Cards
    # -------------------------------------------------------------------------
    ax.text(
        10.325,
        7.74,
        "3. 86 Kinematic Biomarkers",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
    )
    _draw_rounded_box(
        ax,
        x=7.90,
        y=4.35,
        w=4.85,
        h=3.10,
        facecolor="#F8FAFC",
        edgecolor="#475569",
        lw=1.6,
        linestyle="--",
        pad=0.04,
        rounding_size=0.14,
    )

    # Domain I (Top-Left)
    _draw_rounded_box(
        ax,
        x=8.06,
        y=5.96,
        w=2.18,
        h=1.32,
        facecolor="#FFF7ED",
        edgecolor="#EA580C",
        lw=1.5,
        pad=0.03,
        rounding_size=0.10,
        zorder=3,
    )
    ax.text(
        9.15,
        6.92,
        "Domain I (14)",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        9.15,
        6.34,
        "Contralateral\nCross-Distances",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # Domain II (Top-Right)
    _draw_rounded_box(
        ax,
        x=10.41,
        y=5.96,
        w=2.18,
        h=1.32,
        facecolor="#FEFCE8",
        edgecolor="#CA8A04",
        lw=1.5,
        pad=0.03,
        rounding_size=0.10,
        zorder=3,
    )
    ax.text(
        11.50,
        6.92,
        "Domain II (17)",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        11.50,
        6.34,
        "Sagittal & Coronal\nJoint Angles",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # Domain III (Bottom-Left)
    _draw_rounded_box(
        ax,
        x=8.06,
        y=4.50,
        w=2.18,
        h=1.32,
        facecolor="#F0FDF4",
        edgecolor="#16A34A",
        lw=1.5,
        pad=0.03,
        rounding_size=0.10,
        zorder=3,
    )
    ax.text(
        9.15,
        5.46,
        "Domain III (10)",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        9.15,
        4.88,
        "Velocity, Accel. &\nSpectral Entropy",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # Domain IV (Bottom-Right)
    _draw_rounded_box(
        ax,
        x=10.41,
        y=4.50,
        w=2.18,
        h=1.32,
        facecolor="#EFF6FF",
        edgecolor="#2563EB",
        lw=1.5,
        pad=0.03,
        rounding_size=0.10,
        zorder=3,
    )
    ax.text(
        11.50,
        5.46,
        "Domain IV (45)",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        11.50,
        4.88,
        "Postural & Pelvic-Limb\nGuarding",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # Arrow Stage 2 -> Stage 3
    _draw_arrow(
        ax,
        7.40,
        5.90,
        7.85,
        5.90,
        color="#334155",
        lw=2.2,
        mutation_scale=18.0,
    )

    # =========================================================================
    # SERPENTINE CONNECTOR: STAGE 3 (Top-Right) -> STAGE 4 (Bottom-Right)
    # =========================================================================
    _draw_arrow(
        ax,
        10.325,
        4.30,
        10.325,
        3.82,
        color="#334155",
        lw=2.4,
        mutation_scale=19.0,
    )
    ax.text(
        10.12,
        4.06,
        "86-D Feature Matrix (700 × 86)",
        ha="right",
        va="center",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
    )

    # =========================================================================
    # BOTTOM ROW: STAGE 4 (Right) -> STAGE 5 (Left) (Serpentine Flow, y = 0.30 .. 3.40)
    # =========================================================================

    # -------------------------------------------------------------------------
    # STAGE 4: SUBJECT-DISJOINT VALIDATION (x = 7.05 .. 12.75)
    # -------------------------------------------------------------------------
    ax.text(
        9.90,
        3.66,
        "4. Subject-Disjoint Validation",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
    )
    _draw_rounded_box(
        ax,
        x=7.05,
        y=0.30,
        w=5.70,
        h=3.10,
        facecolor="#F8FAFC",
        edgecolor="#475569",
        lw=1.6,
        pad=0.04,
        rounding_size=0.14,
    )

    # Box 4A: 5-Fold StratifiedGroupKFold
    _draw_rounded_box(
        ax,
        x=7.22,
        y=1.92,
        w=5.36,
        h=1.30,
        facecolor="#FFF7ED",
        edgecolor="#EA580C",
        lw=1.5,
        pad=0.04,
        rounding_size=0.12,
        zorder=3,
    )
    ax.text(
        9.90,
        2.86,
        "5-Fold StratifiedGroupKFold (by subject_id)",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        9.90,
        2.30,
        "80 Train / 20 Unseen Test Subjects per Fold\nAll 7 trials per child locked together (Zero Leakage)",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # Box 4B: Tuned CatBoost Classifier
    _draw_rounded_box(
        ax,
        x=7.22,
        y=0.46,
        w=5.36,
        h=1.30,
        facecolor="#FEFCE8",
        edgecolor="#CA8A04",
        lw=1.5,
        pad=0.04,
        rounding_size=0.12,
        zorder=3,
    )
    ax.text(
        9.90,
        1.40,
        "Tuned CatBoost Classifier (Oblivious Trees)",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        9.90,
        0.84,
        "500 trees  |  depth = 5  |  lr = 0.04  |  L2 = 4\nBenchmarked vs. Random Forest & XGBoost on same folds",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # -------------------------------------------------------------------------
    # STAGE 5: CLINICAL OUTPUT & XAI (x = 0.25 .. 6.25)
    # -------------------------------------------------------------------------
    ax.text(
        3.25,
        3.66,
        "5. Clinical Output & XAI",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color="#0F172A",
    )
    _draw_rounded_box(
        ax,
        x=0.25,
        y=0.30,
        w=6.00,
        h=3.10,
        facecolor="#F8FAFC",
        edgecolor="#475569",
        lw=1.6,
        pad=0.04,
        rounding_size=0.14,
    )

    # Outcome Badge 5A (Left): Class 0: TD
    _draw_rounded_box(
        ax,
        x=0.42,
        y=2.24,
        w=2.72,
        h=0.98,
        facecolor="#F0FDF4",
        edgecolor="#16A34A",
        lw=1.6,
        pad=0.03,
        rounding_size=0.10,
        zorder=3,
    )
    ax.text(
        1.78,
        2.92,
        "Class 0: TD",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        1.78,
        2.50,
        "Neurotypical Gait",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#111827",
        zorder=4,
    )

    # Outcome Badge 5B (Right): Class 1: ASD
    _draw_rounded_box(
        ax,
        x=3.36,
        y=2.24,
        w=2.72,
        h=0.98,
        facecolor="#FFF1F2",
        edgecolor="#E11D48",
        lw=1.6,
        pad=0.03,
        rounding_size=0.10,
        zorder=3,
    )
    ax.text(
        4.72,
        2.92,
        "Class 1: ASD",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        4.72,
        2.50,
        "Atypical Motor Phenotype",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#111827",
        zorder=4,
    )

    # Performance & TreeSHAP XAI Card (Bottom of Stage 5)
    _draw_rounded_box(
        ax,
        x=0.42,
        y=0.46,
        w=5.66,
        h=1.60,
        facecolor="#EFF6FF",
        edgecolor="#2563EB",
        lw=1.6,
        pad=0.04,
        rounding_size=0.12,
        zorder=3,
    )
    ax.text(
        3.25,
        1.72,
        "97.71% ± 1.77% Accuracy  (AUROC 0.9990 | F1 0.9763)",
        ha="center",
        va="center",
        fontsize=12.0,
        fontweight="bold",
        color="#111827",
        zorder=4,
    )
    ax.text(
        3.25,
        1.02,
        "TreeSHAP Biomarkers:\nContralateral-Axial & Postural Guarding",
        ha="center",
        va="center",
        fontsize=10.8,
        fontweight="semibold",
        color="#1E293B",
        zorder=4,
    )

    # Serpentine Arrow Stage 4 (Right) -> Stage 5 (Left)
    _draw_arrow(
        ax,
        7.00,
        1.85,
        6.30,
        1.85,
        color="#334155",
        lw=2.4,
        mutation_scale=19.0,
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    shutil.copy2(output_path, os.path.join("results", "figures", "methodology_architecture.png"))
    print(f"[Manuscript Figures] Saved Methodology Architecture Diagram -> '{output_path}' & 'results/figures/methodology_architecture.png'")


def generate_literature_comparison_chart(
    output_path: str = os.path.join(OUTPUT_DIR, "literature_comparison_chart.png"),
) -> None:
    """
    Generates Figure 2: Compact, high-readability Literature Accuracy Comparison Graph
    (`literature_comparison_chart.png`, figsize=(8.8, 4.3), 300 DPI) comparing the 5 verified
    single-dataset studies on the Al-Jubouri 3D Kinect v2 dataset with symmetrical in-bar
    white percentage labels and zero line-text overlap.
    """
    configure_typography()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 5 verified single-dataset studies ordered bottom-to-top so Proposed sits at top
    studies = [
        {
            "label": "Mukhanova et al. (2026) — GCN-BiLSTM",
            "acc": 88.50,
            "val_text": "88.50%",
            "is_ours": False,
            "color": "#64748B",
        },
        {
            "label": "Y. Fırat (2026) — Random Forest",
            "acc": 91.00,
            "val_text": "91.00%",
            "is_ours": False,
            "color": "#64748B",
        },
        {
            "label": "Al-Jubouri et al. (2021) — Rough Set + PCA",
            "acc": 92.00,
            "val_text": "92.00%",
            "is_ours": False,
            "color": "#64748B",
        },
        {
            "label": "Le-Thi-Thu et al. (2026) — ST-GCN",
            "acc": 94.00,
            "val_text": "94.00%",
            "is_ours": False,
            "color": "#64748B",
        },
        {
            "label": "Proposed (Tuned CatBoost, 5-Fold GroupCV)",
            "acc": 97.71,
            "val_text": "97.71% ± 1.77%",
            "is_ours": True,
            "color": "#0F3D68",
        },
    ]

    fig, ax = plt.subplots(figsize=(8.8, 4.3), dpi=300, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")
    ax.set_axisbelow(True)

    y_pos = np.arange(len(studies))
    accs = [s["acc"] for s in studies]
    colors = [s["color"] for s in studies]
    edge_colors = ["#082542" if s["is_ours"] else "#475569" for s in studies]
    line_widths = [1.4 if s["is_ours"] else 1.0 for s in studies]

    # Subtle vertical gridlines strictly behind bars (zorder=1)
    ax.grid(axis="x", linestyle=":", linewidth=0.75, color="#E2E8F0", alpha=0.95, zorder=1)
    ax.grid(axis="y", visible=False)

    # Vertical dashed reference line at 94.0% strictly behind bars (zorder=1)
    ax.axvline(
        94.0,
        color="#64748B",
        linestyle="--",
        linewidth=1.3,
        alpha=0.90,
        zorder=1,
    )

    bars = ax.barh(
        y_pos,
        accs,
        height=0.56,
        color=colors,
        edgecolor=edge_colors,
        linewidth=line_widths,
        zorder=3,
    )

    # Symmetrical in-bar percentage labels in crisp bold 10.5 pt white text (zorder=5)
    for bar, s in zip(bars, studies):
        x_val = bar.get_width()
        ax.text(
            x_val - 0.45,
            bar.get_y() + bar.get_height() / 2.0,
            s["val_text"],
            va="center",
            ha="right",
            fontsize=10.5,
            fontweight="bold",
            color="#FFFFFF",
            zorder=5,
        )

    # Opaque callout badge for Prior SOTA (94.00%) so no dashed line crosses behind text
    ax.text(
        94.0,
        0.02,
        "Prior SOTA (94.00%)",
        ha="center",
        va="center",
        fontsize=9.5,
        fontweight="bold",
        color="#475569",
        bbox=dict(
            boxstyle="round,pad=0.26",
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            linewidth=0.9,
        ),
        zorder=5,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [s["label"] for s in studies],
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
    )

    ax.set_xlim(80.0, 101.5)
    ax.set_ylim(-0.55, 4.55)
    ax.set_xticks([80, 84, 88, 92, 96, 100])
    ax.set_xticklabels(
        ["80%", "84%", "88%", "92%", "96%", "100%"],
        fontsize=10.5,
        fontweight="bold",
        color="#0F172A",
    )
    ax.set_xlabel(
        "Reported Classification Accuracy (%)",
        fontsize=10.8,
        fontweight="bold",
        color="#0F172A",
        labelpad=8,
    )

    # Clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334155")
    ax.spines["left"].set_linewidth(1.1)
    ax.spines["bottom"].set_color("#334155")
    ax.spines["bottom"].set_linewidth(1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.14)
    plt.close(fig)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    shutil.copy2(output_path, os.path.join("results", "figures", "literature_comparison_chart.png"))
    print(f"[Manuscript Figures] Saved Literature Comparison Chart -> '{output_path}' & 'results/figures/literature_comparison_chart.png'")


def main() -> None:
    generate_methodology_architecture()
    generate_literature_comparison_chart()


if __name__ == "__main__":
    main()
