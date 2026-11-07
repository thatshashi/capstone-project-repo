"""
Explainability & Publication Artifact Module for the Leak-Free ASD Kinematic Pipeline.

Computes TreeSHAP (`shap.TreeExplainer`) on the CatBoost model and generates:
1. `results/shap_summary.png` — SHAP Beeswarm plot of top 15 motor biomarkers.
2. `results/feature_importance.csv` — Ranked table of top 15 motor biomarkers with
   mean |SHAP| attribution and clinical biomechanical interpretation.
3. `results/roc_curve.png` — Cross-validated ROC curve across the 5 subject-grouped
   StratifiedGroupKFold splits (with per-fold curves, mean +/- 1 SD band, and baseline comparison).
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import auc, roc_curve

from src.feature_extraction import DEFAULT_PROCESSED_CSV
from src.train import build_catboost_model, load_processed_dataset, run_benchmark


CLINICAL_BIOMARKER_GLOSSARY: Dict[str, str] = {
    "cross_dist_left_elbow_right_foot_mean": "Mean Contralateral Cross-Distance (Left Elbow to Right Foot) - Diagonal inter-limb coordination",
    "cross_dist_left_elbow_right_foot_std": "Std of Contralateral Cross-Distance (Left Elbow to Right Foot) - Variability in diagonal gait coupling",
    "cross_dist_left_elbow_right_foot_min": "Min Contralateral Cross-Distance (Left Elbow to Right Foot) - Peak contralateral limb convergence",
    "cross_dist_right_elbow_left_foot_mean": "Mean Contralateral Cross-Distance (Right Elbow to Left Foot) - Diagonal inter-limb coordination",
    "cross_dist_right_elbow_left_foot_std": "Std of Contralateral Cross-Distance (Right Elbow to Left Foot) - Variability in diagonal gait coupling",
    "cross_dist_right_elbow_left_foot_min": "Min Contralateral Cross-Distance (Right Elbow to Left Foot) - Peak contralateral limb convergence",
    "cross_dist_left_wrist_right_ankle_mean": "Mean Contralateral Cross-Distance (Left Wrist to Right Ankle) - Distal diagonal inter-limb coordination",
    "cross_dist_left_wrist_right_ankle_std": "Std of Contralateral Cross-Distance (Left Wrist to Right Ankle) - Distal diagonal swing variability",
    "cross_dist_left_wrist_right_ankle_min": "Min Contralateral Cross-Distance (Left Wrist to Right Ankle) - Minimum distal diagonal limb separation",
    "cross_dist_right_wrist_left_ankle_mean": "Mean Contralateral Cross-Distance (Right Wrist to Left Ankle) - Distal diagonal inter-limb coordination",
    "cross_dist_right_wrist_left_ankle_std": "Std of Contralateral Cross-Distance (Right Wrist to Left Ankle) - Distal diagonal swing variability",
    "cross_dist_right_wrist_left_ankle_min": "Min Contralateral Cross-Distance (Right Wrist to Left Ankle) - Minimum distal diagonal limb separation",
    "cross_dist_elbow_foot_asymmetry": "Contralateral Elbow-to-Foot Cross-Distance Asymmetry Index - Bilateral diagonal gait dyscoordination",
    "cross_dist_wrist_ankle_asymmetry": "Contralateral Wrist-to-Ankle Cross-Distance Asymmetry Index - Distal inter-limb coupling asymmetry",
    "arm_swing_asymmetry_index": "3D Arm Swing Asymmetry Index (ASI) - Bilateral upper-limb swing disparity during locomotion",
    "arm_swing_sagittal_asymmetry": "Sagittal Arm Swing Asymmetry Index - Anterior-posterior reciprocal arm swing imbalance",
    "arm_swing_vertical_asymmetry": "Vertical Arm Swing Asymmetry Index - Bilateral wrist vertical oscillation asymmetry",
    "step_cadence_variability": "Step Cadence Variability (CV) - Temporal irregularity of rhythmic foot alternation",
    "step_spectral_irregularity": "Step Spectral Entropy - Frequency-domain irregularity of lower-limb gait periodicity",
    "step_dominant_rhythm_ratio": "Dominant Step Rhythm Ratio - Harmonic concentration of the fundamental stride frequency",
    "wrist_left_rel_height_mean": "Mean Left Wrist Vertical Elevation relative to SpineBase - Upper-limb guarding posture",
    "wrist_right_rel_height_mean": "Mean Right Wrist Vertical Elevation relative to SpineBase - Upper-limb guarding posture",
    "elbow_left_rel_height_mean": "Mean Left Elbow Vertical Elevation relative to SpineBase - Forearm flexion/guarding elevation",
    "elbow_right_rel_height_mean": "Mean Right Elbow Vertical Elevation relative to SpineBase - Forearm flexion/guarding elevation",
    "shoulder_angle_left_mean": "Mean Left Shoulder Abduction/Flexion Angle (deg) - Upper-arm postural abduction",
    "shoulder_angle_right_mean": "Mean Right Shoulder Abduction/Flexion Angle (deg) - Upper-arm postural abduction",
    "elbow_angle_left_mean": "Mean Left Elbow Flexion Angle (deg) - Forearm flexion posture during walking",
    "elbow_angle_right_mean": "Mean Right Elbow Flexion Angle (deg) - Forearm flexion posture during walking",
    "trunk_tilt_angle_mean": "Mean 3D Trunk Tilt Angle relative to Vertical Y-Axis (deg) - Postural inclination during walking",
    "trunk_tilt_angle_std": "Std of 3D Trunk Tilt Angle relative to Vertical Y-Axis (deg) - Dynamic upper-body postural instability",
    "trunk_tilt_angle_max": "Max 3D Trunk Tilt Angle relative to Vertical Y-Axis (deg) - Peak trunk postural excursion",
    "trunk_tilt_angle_rom": "Trunk Tilt Range of Motion relative to Vertical Y-Axis (deg) - Total trunk angular sway",
    "trunk_tilt_ang_vel_mean": "Mean Trunk Tilt Angular Velocity (deg/s) - Rate of postural trunk sway",
    "trunk_sagittal_lean_mean": "Mean Sagittal Trunk Lean Angle (deg) - Forward/backward torso flexion angle",
    "trunk_coronal_sway_std": "Coronal Lateral Trunk Sway Std (deg) - Medio-lateral trunk balance oscillation",
    "head_tilt_angle_mean": "Mean Head-Neck Tilt Angle relative to Vertical Y-Axis (deg) - Cervical postural inclination",
    "wrist_to_ankle_velocity_ratio": "Wrist-to-Ankle Velocity Ratio - Upper-to-lower extremity kinetic coupling ratio",
}


def translate_feature_to_clinical(feature_name: str) -> str:
    """
    Translates engineered 3D kinematic feature names into clinical biomechanical descriptions.
    """
    feat = feature_name.strip()
    if feat in CLINICAL_BIOMARKER_GLOSSARY:
        return CLINICAL_BIOMARKER_GLOSSARY[feat]

    if feat.startswith("sagittal_knee_flexion_"):
        side = "Left" if "_left_" in feat else "Right"
        metric = feat.split(f"_{side.lower()}_")[-1].upper().replace("_", " ")
        return f"Sagittal {side} Knee Flexion Angle ({metric}) - Lower-limb sagittal flexion dynamics"

    if feat.startswith("sagittal_hip_angle_"):
        side = "Left" if "_left_" in feat else "Right"
        metric = feat.split(f"_{side.lower()}_")[-1].upper().replace("_", " ")
        return f"Sagittal {side} Hip Angle ({metric}) - Pelvic-femoral sagittal flexion/extension"

    if feat.startswith(("ankle_left_", "ankle_right_", "wrist_left_", "wrist_right_")) and (
        "_vel_" in feat or "_acc_" in feat
    ):
        parts = feat.split("_")
        joint = f"{parts[1].capitalize()} {parts[0].capitalize()}"
        deriv = "1st-Order Velocity" if "vel" in parts else "2nd-Order Acceleration"
        stat = parts[-1].upper()
        return f"{joint} {deriv} ({stat}) - Distal extremity kinematic derivative"

    if feat.startswith("norm_"):
        parts = feat.split("_")
        axis = parts[1].upper()
        joint = parts[2]
        stat = parts[3].capitalize()
        return f"{stat} Normalized {joint} Position along {axis}-Axis (SpineBase-centered, torso-scaled)"

    return f"3D Kinematic Gait Biomarker: {feat}"


def generate_shap_artifacts(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    shap_plot_path: str = os.path.join("results", "shap_summary.png"),
    importance_csv_path: str = os.path.join("results", "feature_importance.csv"),
    top_k: int = 15,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computes out-of-fold / full-cohort TreeSHAP explanations on the tuned CatBoostClassifier
    and saves:
    - `results/shap_summary.png` (Beeswarm plot of top 15 motor biomarkers)
    - `results/feature_importance.csv` (Top 15 ranked motor biomarkers)
    """
    model = build_catboost_model(random_state=random_state)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    shap_values = np.asarray(shap_values, dtype=np.float64)

    os.makedirs(os.path.dirname(shap_plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(importance_csv_path), exist_ok=True)

    # 1. Generate SHAP Beeswarm Plot (results/shap_summary.png)
    plt.figure(figsize=(11, 7.5))
    shap.summary_plot(
        shap_values,
        X,
        max_display=top_k,
        plot_type="dot",
        show=False,
    )
    plt.title(
        "SHAP Motor Biomarker Attribution (CatBoost — Leak-Free 3D Kinect v2 Pipeline)",
        fontsize=13,
        weight="bold",
        pad=14,
    )
    plt.tight_layout()
    plt.savefig(shap_plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Generate Top-15 Feature Importance CSV (results/feature_importance.csv)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_k]

    rows: List[Dict[str, Any]] = []
    for rank, col_idx in enumerate(top_indices, start=1):
        feat_name = str(X.columns[col_idx])
        score = float(mean_abs_shap[col_idx])
        rows.append(
            {
                "rank": rank,
                "feature_name": feat_name,
                "mean_abs_shap": round(score, 6),
                "clinical_interpretation": translate_feature_to_clinical(feat_name),
            }
        )

    importance_df = pd.DataFrame(rows)
    importance_df.to_csv(importance_csv_path, index=False)

    return importance_df


def generate_cross_validated_roc_plot(
    benchmark_results: Dict[str, Dict[str, Any]],
    roc_plot_path: str = os.path.join("results", "roc_curve.png"),
) -> None:
    """
    Plots and saves the 5-fold StratifiedGroupKFold cross-validated ROC curve
    (`results/roc_curve.png`) featuring per-fold CatBoost traces, mean CatBoost ROC
    with +/- 1 SD shaded confidence band, and comparative curves for Random Forest and XGBoost.
    """
    os.makedirs(os.path.dirname(roc_plot_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.0, 6.5))
    mean_fpr = np.linspace(0.0, 1.0, 200)

    cb_res = benchmark_results["CatBoost (Tuned)"]
    tprs: List[np.ndarray] = []

    for fold_idx, (y_fold, probs_fold) in enumerate(cb_res["fold_roc_data"], start=1):
        fpr, tpr, _ = roc_curve(y_fold, probs_fold)
        fold_auc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        ax.plot(
            fpr,
            tpr,
            lw=1.1,
            alpha=0.35,
            linestyle="-",
            label=f"CatBoost Fold {fold_idx} (AUC = {fold_auc:.3f})",
        )

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)

    cb_auc_mean = cb_res["auroc_mean"]
    cb_auc_std = cb_res["auroc_std"]

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="#1b4f72",
        lw=2.8,
        label=f"CatBoost Mean ROC (AUC = {cb_auc_mean:.3f} ± {cb_auc_std:.3f})",
    )
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0.0),
        np.minimum(mean_tpr + std_tpr, 1.0),
        color="#2e86c1",
        alpha=0.18,
        label="CatBoost ±1 SD Band (5-Fold GroupCV)",
    )

    # Overlay Baseline Models (Random Forest & XGBoost) mean ROC curves
    baseline_styles = [
        ("Random Forest (Baseline)", "#239b56", "--"),
        ("XGBoost (Baseline)", "#b9770e", "-."),
    ]
    for model_name, color, ls in baseline_styles:
        if model_name in benchmark_results:
            res = benchmark_results[model_name]
            base_tprs = []
            for y_fold, probs_fold in res["fold_roc_data"]:
                fpr, tpr, _ = roc_curve(y_fold, probs_fold)
                itpr = np.interp(mean_fpr, fpr, tpr)
                itpr[0] = 0.0
                base_tprs.append(itpr)
            m_tpr = np.mean(base_tprs, axis=0)
            m_tpr[-1] = 1.0
            ax.plot(
                mean_fpr,
                m_tpr,
                color=color,
                linestyle=ls,
                lw=2.0,
                label=f"{model_name} (AUC = {res['auroc_mean']:.3f} ± {res['auroc_std']:.3f})",
            )

    ax.plot([0, 1], [0, 1], color="grey", lw=1.2, linestyle=":", label="Chance Level (AUC = 0.500)")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, weight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / ASD Recall)", fontsize=11, weight="bold")
    ax.set_title(
        "Cross-Validated ROC Curve (5-Fold StratifiedGroupKFold by Subject ID)",
        fontsize=12.5,
        weight="bold",
        pad=12,
    )
    ax.legend(loc="lower right", fontsize=8.8, frameon=True)
    ax.grid(alpha=0.3, linestyle=":")
    plt.tight_layout()
    plt.savefig(roc_plot_path, dpi=300)
    plt.close()


def run_explainability_pipeline(
    csv_path: str = DEFAULT_PROCESSED_CSV,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Runs the leak-free cross-validation benchmark and generates all required SHAP and ROC artifacts:
    - `results/shap_summary.png`
    - `results/feature_importance.csv`
    - `results/roc_curve.png`
    """
    benchmark_results, X, y, groups = run_benchmark(
        csv_path=csv_path,
        n_splits=5,
        random_state=random_state,
        save_artifacts=True,
    )

    importance_df = generate_shap_artifacts(
        X=X,
        y=y,
        groups=groups,
        shap_plot_path=os.path.join("results", "shap_summary.png"),
        importance_csv_path=os.path.join("results", "feature_importance.csv"),
        top_k=15,
        random_state=random_state,
    )

    generate_cross_validated_roc_plot(
        benchmark_results=benchmark_results,
        roc_plot_path=os.path.join("results", "roc_curve.png"),
    )

    print("=" * 88)
    print("  TOP 15 MOTOR BIOMARKERS (TreeSHAP Attribution on Leak-Free CatBoost)")
    print("=" * 88)
    for _, row in importance_df.iterrows():
        print(
            f"  {int(row['rank']):2d}. {row['feature_name']:<42} | "
            f"Mean |SHAP| = {row['mean_abs_shap']:.4f}"
        )
        print(f"      -> {row['clinical_interpretation']}")
    print("=" * 88)
    print("  Saved Explainability Artifacts:")
    print("    * results/shap_summary.png")
    print("    * results/feature_importance.csv")
    print("    * results/roc_curve.png")
    print("=" * 88 + "\n")

    return importance_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate TreeSHAP motor biomarker attributions and GroupKFold ROC curves."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_PROCESSED_CSV,
        help="Path to processed features CSV (default: data/processed_features.csv).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    args = parser.parse_args()
    run_explainability_pipeline(csv_path=args.data, random_state=args.seed)
