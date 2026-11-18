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
    Matches the clean block-and-arrow visual layout, color harmony, vertical feature-vector strip,
    dashed domain grouping containers, and structured left-to-right flow of the reference diagram.
    """
    configure_typography()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Use explicit axes position without bbox_inches='tight' distortion
    fig = plt.figure(figsize=(22.0, 10.6), dpi=300, facecolor="#FFFFFF")
    ax = fig.add_axes([0.008, 0.008, 0.984, 0.984])
    ax.set_xlim(0.0, 22.0)
    ax.set_ylim(0.60, 11.15)
    ax.axis("off")

    # Outer border frame (matching the clean figure frame of the reference image)
    outer_frame = Rectangle(
        (0.15, 0.72),
        21.70,
        10.30,
        facecolor="#FFFFFF",
        edgecolor="#9FB3C8",
        linewidth=1.2,
        zorder=0,
    )
    ax.add_patch(outer_frame)

    # Top Title Banner
    ax.text(
        11.0,
        10.64,
        "Leak-Free 3D Kinematic Biomarker & Subject-Disjoint CatBoost Architecture for Pediatric ASD Gait Screening",
        ha="center",
        va="center",
        fontsize=14.0,
        fontweight="bold",
        color="#102A43",
    )

    # =========================================================================
    # STAGE 1: INPUT COHORT (x = 0.45 .. 3.25)
    # =========================================================================
    ax.text(
        1.85,
        9.96,
        "1. Input Cohort\n(3D Kinect v2 Dataset)",
        ha="center",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color="#102A43",
    )

    _draw_rounded_box(
        ax,
        x=0.50,
        y=2.85,
        w=2.70,
        h=6.60,
        facecolor="#FDF4F2",
        edgecolor="#486581",
        lw=1.3,
        pad=0.06,
    )

    # Top sub-card: Cohort Demographics
    _draw_rounded_box(
        ax,
        x=0.68,
        y=6.78,
        w=2.34,
        h=2.45,
        facecolor="#FADBD8",
        edgecolor="#B03A2E",
        lw=1.1,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        1.85,
        8.96,
        "Al-Jubouri Dryad Cohort",
        ha="center",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color="#641E16",
        zorder=4,
    )
    _draw_skeleton_glyph(ax, cx=1.08, cy=7.78, scale=0.58, color="#78281F", root_highlight=False)
    ax.text(
        2.10,
        7.82,
        "N = 100 Subjects\n• 50 ASD\n• 50 TD\n\n700 Sequences\n(7 trials/child)",
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="#102A43",
        zorder=4,
    )

    # Bottom sub-card: Raw 3D Skeletal Stream
    _draw_rounded_box(
        ax,
        x=0.68,
        y=3.12,
        w=2.34,
        h=3.34,
        facecolor="#EBF5FB",
        edgecolor="#2874A6",
        lw=1.1,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        1.85,
        6.16,
        "25 3D Skeletal Joints",
        ha="center",
        va="center",
        fontsize=9.4,
        fontweight="bold",
        color="#1B4F72",
        zorder=4,
    )
    ax.text(
        1.85,
        4.60,
        "30 Hz Frame Trajectories\n\n3D Joint Coordinates:\n"
        r"$\mathbf{P}_t \in \mathbb{R}^{T \times 25 \times 3}$"
        "\n"
        r"$(X, Y, Z)$ in meters"
        "\n\nStrict Subject Mapping:\nChild_01 to Child_100",
        ha="center",
        va="center",
        fontsize=8.3,
        color="#102A43",
        zorder=4,
    )

    # =========================================================================
    # STAGE 2: SPATIAL & AUGMENTATION NORMALIZATION (x = 3.75 .. 7.15)
    # =========================================================================
    ax.text(
        5.45,
        9.96,
        "2. Spatial & Augmentation\nNormalization",
        ha="center",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color="#102A43",
    )

    _draw_rounded_box(
        ax,
        x=3.82,
        y=2.85,
        w=3.26,
        h=6.60,
        facecolor="#F8FAFC",
        edgecolor="#334E68",
        lw=1.3,
        pad=0.06,
    )

    # Sub-box 2A: SpineBase Root-Centering
    _draw_rounded_box(
        ax,
        x=4.02,
        y=7.46,
        w=2.86,
        h=1.80,
        facecolor="#D6EAF8",
        edgecolor="#21618C",
        lw=1.1,
        pad=0.05,
        zorder=3,
    )
    _draw_skeleton_glyph(ax, cx=4.34, cy=8.30, scale=0.40, color="#1B4F72", root_highlight=True)
    ax.text(
        5.82,
        8.36,
        "SpineBase Root-Centering\n"
        r"$\tilde{\mathbf{p}}_i(t) = \mathbf{p}_i(t) - \mathbf{p}_{\mathrm{SpineBase}}(t)$"
        "\nSets Pelvis Origin at (0, 0, 0)\n(Removes camera translation)",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#102A43",
        zorder=4,
    )

    # Sub-box 2B: Frame-Wise 3D Torso-Height Scaling
    _draw_rounded_box(
        ax,
        x=4.02,
        y=5.28,
        w=2.86,
        h=1.82,
        facecolor="#FCF3CF",
        edgecolor="#B7950B",
        lw=1.1,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        5.45,
        6.19,
        "Torso-Height Scaling\n"
        r"$L_t = \|\mathrm{SpineShoulder} - \mathrm{SpineBase}\|_2$"
        "\n"
        r"$\hat{\mathbf{p}}_i(t) = \tilde{\mathbf{p}}_i(t) \,/\, L_t$"
        "\nRemoves child stature & depth bias",
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color="#102A43",
        zorder=4,
    )

    # Sub-box 2C: Isotropic Coronal Calibration
    _draw_rounded_box(
        ax,
        x=4.02,
        y=3.10,
        w=2.86,
        h=1.82,
        facecolor="#E8DAEF",
        edgecolor="#6C3483",
        lw=1.1,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        5.45,
        4.01,
        "Isotropic Coronal Calibration\n"
        "Per-trial mirror canonicalization\n"
        r"($X_{\mathrm{ShoulderRight}} \geq X_{\mathrm{ShoulderLeft}}$)"
        "\nCorrects "
        r"$0.7\times / 1.6\times$"
        " mediolateral scaling",
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color="#102A43",
        zorder=4,
    )

    # Internal vertical flow arrows inside Stage 2
    _draw_arrow(ax, 5.45, 7.41, 5.45, 7.15, color="#243B53", lw=1.3, mutation_scale=11)
    _draw_arrow(ax, 5.45, 5.23, 5.45, 4.97, color="#243B53", lw=1.3, mutation_scale=11)

    # Arrow from Stage 1 to Stage 2
    _draw_arrow(ax, 3.26, 6.15, 3.76, 6.15, color="#102A43", lw=1.8, mutation_scale=15)

    # =========================================================================
    # STAGE 3: BIOMECHANICAL FEATURE ENGINEERING (x = 7.65 .. 13.55)
    # Two-branch dashed domain blocks (like DSRSENet / Handcraft boxes in reference)
    # =========================================================================
    ax.text(
        10.60,
        9.96,
        "3. Biomechanical Feature Engineering:\n86 Scale-Invariant 3D Biomarkers",
        ha="center",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color="#102A43",
    )

    # Upper Dashed Container: Inter-Limb & Joint Angular Kinematics
    _draw_rounded_box(
        ax,
        x=7.72,
        y=6.32,
        w=5.76,
        h=3.13,
        facecolor="#FDFEFE",
        edgecolor="#243B53",
        lw=1.25,
        linestyle="--",
        pad=0.06,
        zorder=2,
    )
    ax.text(
        10.60,
        9.25,
        "Spatial Inter-Limb & Sagittal/3D Joint Angular Kinematics",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#1B4F72",
        zorder=4,
    )

    # Domain Box 3A
    _draw_rounded_box(
        ax,
        x=7.90,
        y=6.52,
        w=2.60,
        h=2.46,
        facecolor="#FADBD8",
        edgecolor="#A93226",
        lw=1.1,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        9.20,
        8.68,
        "Domain I: Contralateral\nCross-Distances (14)",
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color="#641E16",
        zorder=4,
    )
    ax.text(
        9.20,
        7.42,
        "• ElbowLeft ↔ FootRight (mean/std/min)\n"
        "• ElbowRight ↔ FootLeft (mean/std/min)\n"
        "• WristLeft ↔ AnkleRight (mean/min)\n"
        "• WristRight ↔ AnkleLeft (mean/std/min)\n"
        "• Diagonal Wrist-Ankle Asymmetry\n"
        "• HandTipRight-Hip & Radial Reach",
        ha="center",
        va="center",
        fontsize=7.65,
        color="#102A43",
        zorder=4,
    )

    # Domain Box 3B
    _draw_rounded_box(
        ax,
        x=10.72,
        y=6.52,
        w=2.58,
        h=2.46,
        facecolor="#FCF3CF",
        edgecolor="#B7950B",
        lw=1.1,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        12.01,
        8.68,
        "Domain II: Sagittal/Coronal\nJoint Angles & ROM (17)",
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color="#7D6608",
        zorder=4,
    )
    ax.text(
        12.01,
        7.42,
        "• Sagittal (Y-Z) Knee Flexion\n  (mean, std, angular velocity)\n"
        "• Sagittal Hip Flexion/Extension\n  (mean, std, max ROM, ang. vel.)\n"
        "• 3D Knee Flexion Angle\n"
        "• 3D Shoulder & Bilateral Elbow\n  Flexion/Abduction Angles",
        ha="center",
        va="center",
        fontsize=7.65,
        color="#102A43",
        zorder=4,
    )

    # Lower Dashed Container: Dynamic Derivatives & Postural/Guarding Coupling
    _draw_rounded_box(
        ax,
        x=7.72,
        y=2.85,
        w=5.76,
        h=3.13,
        facecolor="#FDFEFE",
        edgecolor="#243B53",
        lw=1.25,
        linestyle="--",
        pad=0.06,
        zorder=2,
    )
    ax.text(
        10.60,
        5.78,
        "Dynamic Motion Derivatives & Postural/Pelvic-Limb Guarding Indices",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#1B4F72",
        zorder=4,
    )

    # Domain Box 3C
    _draw_rounded_box(
        ax,
        x=7.90,
        y=3.05,
        w=2.60,
        h=2.46,
        facecolor="#D5F5E3",
        edgecolor="#1E8449",
        lw=1.1,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        9.20,
        5.21,
        "Domain III: Motion Derivatives\n& Spectral Irregularity (10)",
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color="#145A32",
        zorder=4,
    )
    ax.text(
        9.20,
        3.95,
        "• 1st-Order 3D & Sagittal Velocity\n  (AnkleLeft/Right, WristRight)\n"
        "• 2nd-Order 3D Acceleration\n  (AnkleRight, WristLeft mean/std)\n"
        "• Wrist-to-Ankle Velocity Ratio\n"
        "• FFT Step Spectral Irregularity\n  & Dominant Rhythm Ratio",
        ha="center",
        va="center",
        fontsize=7.65,
        color="#102A43",
        zorder=4,
    )

    # Domain Box 3D
    _draw_rounded_box(
        ax,
        x=10.72,
        y=3.05,
        w=2.58,
        h=2.46,
        facecolor="#D6EAF8",
        edgecolor="#21618C",
        lw=1.1,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        12.01,
        5.21,
        "Domain IV: Postural/Pelvic-Limb\nGuarding Indices (45)",
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color="#1B4F72",
        zorder=4,
    )
    ax.text(
        12.01,
        3.95,
        "• Isotropic Coronal Span (Elbows,\n  Wrists, Hands, Thumbs, Tibia)\n"
        "• Distal Foot-Ankle Compliance\n"
        "• Arm Swing Asymmetry & Trunk Lean\n"
        "• 6 Multi-Segment Synergy Indices:\n  Whole-Body Contralateral-Axial,\n  Bilateral Postural & Arm Guarding",
        ha="center",
        va="center",
        fontsize=7.55,
        color="#102A43",
        zorder=4,
    )

    # Branching arrows from Stage 2 to Upper and Lower Stage 3 blocks
    _draw_arrow(ax, 7.14, 7.25, 7.66, 7.85, color="#102A43", lw=1.6, mutation_scale=14)
    _draw_arrow(ax, 7.14, 5.05, 7.66, 4.42, color="#102A43", lw=1.6, mutation_scale=14)

    # =========================================================================
    # FEATURE FUSION VECTOR STRIP (x = 13.98 .. 14.44)
    # Mirrors the vertical segmented feature strip in the reference image
    # =========================================================================
    ax.text(
        14.21,
        9.42,
        "86-D\nBiomarker\nVector",
        ha="center",
        va="center",
        fontsize=8.7,
        fontweight="bold",
        color="#102A43",
    )

    strip_x = 13.99
    strip_w = 0.44
    n_cells = 14
    cell_h = 0.43
    strip_y0 = 2.95
    palette_strip = [
        "#F5B7B1",
        "#F5B7B1",
        "#F5B7B1",
        "#F9E79F",
        "#F9E79F",
        "#F9E79F",
        "#A9DFBF",
        "#A9DFBF",
        "#A9DFBF",
        "#AED6F1",
        "#AED6F1",
        "#AED6F1",
        "#D2B4DE",
        "#D2B4DE",
    ]
    for idx in range(n_cells):
        cy_cell = strip_y0 + idx * cell_h
        rect = Rectangle(
            (strip_x, cy_cell),
            strip_w,
            cell_h - 0.03,
            facecolor=palette_strip[n_cells - 1 - idx],
            edgecolor="#334E68",
            linewidth=0.85,
            zorder=4,
        )
        ax.add_patch(rect)

    # Arrows from Upper & Lower Feature Engineering containers into the 86-D Feature Vector Strip
    _draw_arrow(ax, 13.54, 7.85, 13.96, 7.55, color="#102A43", lw=1.55, mutation_scale=13)
    _draw_arrow(ax, 13.54, 4.42, 13.96, 4.65, color="#102A43", lw=1.55, mutation_scale=13)

    # =========================================================================
    # STAGE 4: LEAK-FREE SUBJECT-DISJOINT VALIDATION (x = 14.95 .. 18.35)
    # Modeled on the grey shaded classifier/optimization container in reference
    # =========================================================================
    ax.text(
        16.65,
        9.96,
        "4. Leak-Free Subject-Disjoint\nValidation",
        ha="center",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color="#102A43",
    )

    _draw_rounded_box(
        ax,
        x=14.95,
        y=2.85,
        w=3.40,
        h=6.60,
        facecolor="#EAEEF3",
        edgecolor="#334E68",
        lw=1.3,
        pad=0.06,
    )

    # Sub-box 4A: 5-Fold StratifiedGroupKFold
    _draw_rounded_box(
        ax,
        x=15.13,
        y=7.02,
        w=3.04,
        h=2.22,
        facecolor="#FDEBD0",
        edgecolor="#CA6F1E",
        lw=1.15,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        16.65,
        8.86,
        "5-Fold StratifiedGroupKFold\n(Grouped by subject_id)",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#784212",
        zorder=4,
    )
    ax.text(
        16.65,
        7.72,
        "• 80 Train Subjects (560 Trials) / Fold\n"
        "• 20 Unseen Test Subjects (140 Trials)\n"
        "• All 7 trials/child locked together\n"
        "• Zero Trial / Subject Overlap\n"
        r"  ($S_{\mathrm{train}} \cap S_{\mathrm{test}} = \emptyset$)",
        ha="center",
        va="center",
        fontsize=7.95,
        fontweight="bold",
        color="#102A43",
        zorder=4,
    )

    # Sub-box 4B: Tuned CatBoost Classifier
    _draw_rounded_box(
        ax,
        x=15.13,
        y=4.80,
        w=3.04,
        h=1.90,
        facecolor="#FADBD8",
        edgecolor="#922B21",
        lw=1.2,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        16.65,
        6.36,
        "Tuned CatBoostClassifier",
        ha="center",
        va="center",
        fontsize=9.2,
        fontweight="bold",
        color="#641E16",
        zorder=4,
    )
    ax.text(
        16.65,
        5.48,
        "iterations = 500 | depth = 5\n"
        "learning_rate = 0.04 | l2_leaf_reg = 4\n"
        "Loss = Logloss (Oblivious Decision Trees)",
        ha="center",
        va="center",
        fontsize=7.95,
        color="#102A43",
        zorder=4,
    )

    # Sub-box 4C: Same-Split Baseline Comparison
    _draw_rounded_box(
        ax,
        x=15.13,
        y=3.10,
        w=3.04,
        h=1.38,
        facecolor="#D6EAF8",
        edgecolor="#21618C",
        lw=1.05,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        16.65,
        3.79,
        "Same-Split Baseline Comparison\n"
        "• Random Forest: 97.00% ± 1.71%\n"
        "• XGBoost: 96.29% ± 2.84%",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#1B4F72",
        zorder=4,
    )

    # Internal arrows inside Stage 4
    _draw_arrow(ax, 16.65, 6.97, 16.65, 6.75, color="#243B53", lw=1.3, mutation_scale=11)
    _draw_arrow(ax, 16.65, 4.75, 16.65, 4.53, color="#243B53", lw=1.3, mutation_scale=11)

    # Arrow from 86-D Feature Vector Strip into Stage 4
    _draw_arrow(ax, 14.47, 5.95, 14.90, 5.95, color="#102A43", lw=1.8, mutation_scale=15)

    # =========================================================================
    # STAGE 5: EXPLAINABLE CLINICAL OUTPUT (x = 18.88 .. 21.62)
    # Includes Benign/TD (Green) & ASD (Crimson) output nodes + TreeSHAP & Metrics
    # =========================================================================
    ax.text(
        20.25,
        9.96,
        "5. Explainable Clinical\nOutput",
        ha="center",
        va="center",
        fontsize=10.6,
        fontweight="bold",
        color="#102A43",
    )

    # Output Class Nodes (matching the green/red output boxes on the right of reference)
    _draw_rounded_box(
        ax,
        x=18.96,
        y=8.32,
        w=2.58,
        h=1.08,
        facecolor="#58A631",
        edgecolor="#1E8449",
        lw=1.3,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        20.25,
        8.86,
        "Typically Developing (TD)\nClass 0: Neurotypical Gait Kinematics",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#FFFFFF",
        zorder=4,
    )

    _draw_rounded_box(
        ax,
        x=18.96,
        y=6.92,
        w=2.58,
        h=1.08,
        facecolor="#900C3F",
        edgecolor="#581845",
        lw=1.3,
        pad=0.05,
        zorder=3,
    )
    ax.text(
        20.25,
        7.46,
        "Autism Spectrum (ASD)\nClass 1: Atypical Motor Phenotype",
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        color="#FFFFFF",
        zorder=4,
    )

    # Exact TreeSHAP & Performance Card
    _draw_rounded_box(
        ax,
        x=18.92,
        y=2.85,
        w=2.66,
        h=3.74,
        facecolor="#F4F9FD",
        edgecolor="#1560BD",
        lw=1.35,
        pad=0.06,
        zorder=3,
    )
    ax.text(
        20.25,
        6.26,
        "Verified Performance &\nExact TreeSHAP Attribution",
        ha="center",
        va="center",
        fontsize=8.9,
        fontweight="bold",
        color="#083D77",
        zorder=4,
    )
    ax.text(
        20.25,
        4.54,
        "• Accuracy (5-Fold CV):\n"
        "  97.71% ± 1.77% (684/700)\n"
        "• AUROC: 0.9990 ± 0.0008\n"
        "• Precision (PPV): 99.71%\n"
        "• F1-Score: 0.9763 ± 0.0190\n\n"
        "Top TreeSHAP Biomarkers:\n"
        "1. Contralateral-Axial Index\n"
        "   (|SHAP|=2.65, d=+2.20)\n"
        "2. Postural Integration Index\n"
        "   (|SHAP|=1.60, d=+2.02)",
        ha="center",
        va="center",
        fontsize=7.85,
        fontweight="bold",
        color="#102A43",
        zorder=4,
    )

    # Forked output arrows from Stage 4 to TD, ASD, and TreeSHAP Output cards
    _draw_arrow(ax, 18.40, 8.86, 18.91, 8.86, color="#102A43", lw=1.6, mutation_scale=13)
    _draw_arrow(ax, 18.40, 7.46, 18.91, 7.46, color="#102A43", lw=1.6, mutation_scale=13)
    _draw_arrow(ax, 18.40, 4.72, 18.87, 4.72, color="#102A43", lw=1.6, mutation_scale=13)

    # =========================================================================
    # BOTTOM SUMMARY STRIP
    # =========================================================================
    _draw_rounded_box(
        ax,
        x=0.50,
        y=1.08,
        w=21.08,
        h=1.35,
        facecolor="#F0F4F8",
        edgecolor="#829AB1",
        lw=1.0,
        pad=0.05,
        zorder=2,
    )
    ax.text(
        11.04,
        1.75,
        "End-to-End Methodological Guarantees:  "
        "(1) Frame-wise SpineBase root-centering & 3D trunk-height scaling remove camera translation and child stature biases   |   "
        "(2) Per-trial isotropic coronal calibration corrects mirror reflection & 0.7×/1.6× mediolateral scaling\n"
        "(3) 5-Fold StratifiedGroupKFold locks all 7 trials per child into the same fold (80 train / 20 unseen test subjects per fold; zero trial overlap)   |   "
        "(4) Exact TreeSHAP attribution links model outputs to clinical 3D motor biomarkers.",
        ha="center",
        va="center",
        fontsize=8.65,
        fontweight="bold",
        color="#243B53",
        zorder=4,
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
    Generates Figure 2: Simple Literature Accuracy Comparison Graph (`literature_comparison_chart.png`).
    Reads the exact study names, methods, validation protocols, and reported accuracies from
    `paper studies table.png` (TABLE 2. Comparison with published results on the Al-Jubouri 3D Kinect v2 dataset)
    and compares them against our Tuned CatBoost model (97.71% ± 1.77%).
    """
    configure_typography()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Exact studies from `paper studies table.png` (Al-Jubouri 3D Kinect v2 dataset)
    studies = [
        {
            "label": "Movement Signs Kinect (2024) — AdaBoost / NB\n[Positions & angles + medical rules | LOSO]",
            "acc": 87.00,
            "val_text": "87.0% (AdaBoost)",
            "is_ours": False,
            "color": "#829AB1",
        },
        {
            "label": "Mukhanova et al. — GCN-BiLSTM\n[Spatial-Temporal Graph + BiLSTM | Cross-Validation]",
            "acc": 88.50,
            "val_text": "88.5%",
            "is_ours": False,
            "color": "#627D98",
        },
        {
            "label": "Firat et al. — Random Forest\n[Selected kinematic & spatiotemporal features | Cross-Validation]",
            "acc": 91.00,
            "val_text": "91.0%",
            "is_ours": False,
            "color": "#486581",
        },
        {
            "label": "Al-Jubouri et al. (2021) — Rough Set + PCA\n[1,259 tabular features using PCA | Augmented Split]",
            "acc": 92.00,
            "val_text": "92.0%",
            "is_ours": False,
            "color": "#334E68",
        },
        {
            "label": "Le-Thi-Thu et al. — ST-GCN\n[Spatial-Temporal Graph ConvNet | Cross-Validation]",
            "acc": 94.00,
            "val_text": "94.0%",
            "is_ours": False,
            "color": "#2B7BBA",
        },
        {
            "label": "Ours (Tuned CatBoost)\n[86 Root-Centered, Torso-Scaled 3D Kinematic Biomarkers | 5-Fold StratifiedGroupKFold]",
            "acc": 97.71,
            "val_text": "97.71% ± 1.77%",
            "is_ours": True,
            "color": "#083D77",
        },
    ]

    fig, ax = plt.subplots(figsize=(13.6, 7.2), dpi=300)
    ax.set_facecolor("#F8FBFF")

    y_pos = np.arange(len(studies))
    accs = [s["acc"] for s in studies]
    colors = [s["color"] for s in studies]
    edge_colors = ["#041E42" if s["is_ours"] else "#243B53" for s in studies]
    line_widths = [1.5 if s["is_ours"] else 0.95 for s in studies]

    bars = ax.barh(
        y_pos,
        accs,
        height=0.56,
        color=colors,
        edgecolor=edge_colors,
        linewidth=line_widths,
        zorder=3,
    )

    # Subtle highlight band for our leak-free row
    ax.axhspan(4.52, 5.52, color="#E6F0FA", alpha=0.70, zorder=1)

    # Subtle dashed vertical threshold line at 96.0% showing our model surpassing prior SOTA
    ax.axvline(
        96.0,
        color="#B03A2E",
        linestyle="--",
        linewidth=1.55,
        alpha=0.88,
        zorder=4,
    )

    # Place the 96.0% threshold callout cleanly in open whitespace at bottom right of the line
    ax.annotate(
        "96.0% Benchmark Threshold\n(Surpasses prior 94.0% SOTA)",
        xy=(96.0, 1.35),
        xytext=(97.15, 1.35),
        ha="left",
        va="center",
        fontsize=8.9,
        fontweight="bold",
        color="#922B21",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#FDEDEC", edgecolor="#E6B0AA", lw=0.85),
        arrowprops=dict(arrowstyle="-|>", color="#B03A2E", lw=1.25),
        zorder=5,
    )

    # Error bar / range indicator on our 5-Fold StratifiedGroupKFold result (97.71% +/- 1.77%)
    trial_idx = 5
    ax.errorbar(
        [97.71],
        [trial_idx],
        xerr=[[1.77], [1.77]],
        fmt="none",
        ecolor="#041E42",
        elinewidth=1.8,
        capsize=4.5,
        capthick=1.6,
        zorder=6,
    )

    # Bold numeric percentage labels directly beside each bar
    for idx, (bar, s) in enumerate(zip(bars, studies)):
        x_val = bar.get_width()
        offset = 2.05 if idx == trial_idx else 0.35
        text_color = "#041E42" if s["is_ours"] else "#102A43"
        font_size = 10.4 if s["is_ours"] else 9.7
        ax.text(
            x_val + offset,
            bar.get_y() + bar.get_height() / 2.0,
            s["val_text"],
            va="center",
            ha="left",
            fontsize=font_size,
            fontweight="bold",
            color=text_color,
            zorder=6,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(
        [s["label"] for s in studies],
        fontsize=9.7,
        fontweight="bold",
        color="#102A43",
    )

    ax.set_xlim(78.0, 105.5)
    ax.set_ylim(-1.05, 5.75)
    ax.set_xticks([80, 84, 88, 92, 96, 100])
    ax.set_xticklabels(["80%", "84%", "88%", "92%", "96%", "100%"], fontsize=10.0, fontweight="bold")
    ax.set_xlabel("Reported Classification Accuracy (%)", fontsize=11.0, fontweight="bold", labelpad=10)
    ax.set_title(
        "Comparison with Published Studies on the Al-Jubouri 3D Kinect v2 Gait Dataset",
        fontsize=13.0,
        fontweight="bold",
        color="#041E42",
        pad=14,
    )

    # Clean spines & gridlines
    ax.grid(axis="x", linestyle="--", linewidth=0.7, color="#D5E3F0", alpha=0.9, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#334E68")
    ax.spines["bottom"].set_color("#334E68")

    # Legend in open bottom-right space with generous vertical margin below bar 0
    legend_handles = [
        mpatches.Patch(
            facecolor="#486581",
            edgecolor="#243B53",
            label="Published Studies on Al-Jubouri Dataset (87.0% – 94.0%)",
        ),
        mpatches.Patch(
            facecolor="#083D77",
            edgecolor="#041E42",
            label="Ours (Tuned CatBoost) — Leak-Free 5-Fold StratifiedGroupKFold (97.71% ± 1.77%, AUROC: 0.9990)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        fontsize=8.8,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#BCCCDC",
        framealpha=0.98,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    os.makedirs(os.path.join("results", "figures"), exist_ok=True)
    shutil.copy2(output_path, os.path.join("results", "figures", "literature_comparison_chart.png"))
    print(f"[Manuscript Figures] Saved Literature Comparison Chart -> '{output_path}' & 'results/figures/literature_comparison_chart.png'")


def main() -> None:
    generate_methodology_architecture()
    generate_literature_comparison_chart()


if __name__ == "__main__":
    main()
