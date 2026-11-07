"""
Kinematic Feature Extraction Module for the Al-Jubouri 3D Kinect v2 ASD Gait Dataset.

Converts continuous root-centered, torso-scaled 3D joint sequences (T, 25, 3) into a
biomechanically interpretable tabular feature matrix (86 kinematic features) and caches
the result to `data/processed_features.csv` with columns:
[<kinematic_features>, label, subject_id].

Extracted Feature Families:
1. Contralateral Cross-Distances:
   - Left Elbow to Right Foot & Right Elbow to Left Foot (mean, std, min)
   - Left Wrist to Right Ankle & Right Wrist to Left Ankle (mean, std, min)
   - Contralateral cross-distance asymmetry ratios
2. Sagittal & 3D Joint Angles (Max ROM, Mean, Std, and Angular Velocity):
   - Sagittal Knee flexion angle (Hip-Knee-Ankle in sagittal Y-Z plane)
   - Sagittal Hip flexion/extension angle (SpineShoulder-Hip-Knee in sagittal Y-Z plane)
   - Upper-limb Shoulder and Elbow 3D coordination angles
3. Motion Derivatives (1st-Order Velocity & 2nd-Order Acceleration):
   - 3D velocity and acceleration dynamics on Left/Right Ankles and Wrists
   - Upper-to-lower extremity velocity coupling ratio
4. Stride & Postural Coordination Metrics:
   - Arm swing asymmetry index (3D, sagittal Z, vertical Y) & upper-limb guarding posture
   - Step cadence variability, spectral rhythm regularity, dynamic step width, and inter-foot clearance
5. Trunk Tilt Angle Relative to Vertical Y-Axis:
   - 3D trunk inclination from vertical Y-axis (mean, std, max, ROM, angular velocity),
     sagittal forward lean, coronal lateral sway, and head-neck tilt angle
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List
import numpy as np
import pandas as pd

from src.data_preprocessing import (
    DEFAULT_RAW_DATASET_DIR,
    JOINT_INDEX,
    TrialSequence,
    load_all_normalized_trials,
)

DEFAULT_PROCESSED_CSV = os.path.join("data", "processed_features.csv")


def _joint(coords: np.ndarray, name: str) -> np.ndarray:
    """Returns trajectory of shape (T, 3) for the named joint."""
    return coords[:, JOINT_INDEX[name], :]


def _euclidean_dist(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Frame-wise 3D Euclidean distance between two joint trajectories of shape (T, 3)."""
    return np.linalg.norm(p1 - p2, axis=1)


def _angle_3d_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Computes frame-wise 3D joint angle ABC (in degrees) with vertex at B.
    Vectors are u = A - B and v = C - B.
    """
    u = a - b
    v = c - b
    u_norm = np.linalg.norm(u, axis=1) + eps
    v_norm = np.linalg.norm(v, axis=1) + eps
    cos_theta = np.sum(u * v, axis=1) / (u_norm * v_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _angle_sagittal_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Computes frame-wise sagittal-plane (Y-Z plane, where Y is vertical and Z is depth/walking axis)
    joint angle ABC (in degrees) with vertex at B.
    """
    u_yz = a[:, 1:3] - b[:, 1:3]
    v_yz = c[:, 1:3] - b[:, 1:3]
    u_norm = np.linalg.norm(u_yz, axis=1) + eps
    v_norm = np.linalg.norm(v_yz, axis=1) + eps
    cos_theta = np.sum(u_yz * v_yz, axis=1) / (u_norm * v_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def _step_cadence_variability(
    inter_ankle_dist: np.ndarray,
    ankle_left_y: np.ndarray,
    ankle_right_y: np.ndarray,
    dt: float,
) -> Dict[str, float]:
    """
    Computes step cadence variability and rhythmic regularity metrics from lower-limb
    oscillation signals across the gait cycle.
    """
    sig = inter_ankle_dist - np.mean(inter_ankle_dist)
    alt_y = ankle_left_y - ankle_right_y
    alt_y_centered = alt_y - np.mean(alt_y)

    zc_indices = np.where(np.diff(np.signbit(alt_y_centered)))[0]
    if len(zc_indices) >= 3:
        half_step_intervals = np.diff(zc_indices) * dt
        cadence_var = float(np.std(half_step_intervals) / (np.mean(half_step_intervals) + 1e-6))
    else:
        step_rate = np.abs(np.diff(inter_ankle_dist)) / dt
        cadence_var = float(np.std(step_rate) / (np.mean(step_rate) + 1e-6))

    fft_mag = np.abs(np.fft.rfft(sig))
    if len(fft_mag) > 1 and np.sum(fft_mag[1:]) > 1e-8:
        power = fft_mag[1:] ** 2
        power_prob = power / np.sum(power)
        spectral_irregularity = float(-np.sum(power_prob * np.log(power_prob + 1e-12)))
        dominant_freq_ratio = float(np.max(power) / np.sum(power))
    else:
        spectral_irregularity = 0.0
        dominant_freq_ratio = 1.0

    return {
        "step_cadence_variability": cadence_var,
        "step_spectral_irregularity": spectral_irregularity,
        "step_dominant_rhythm_ratio": dominant_freq_ratio,
    }


def extract_features_from_sequence(seq: TrialSequence) -> Dict[str, float]:
    """
    Extracts the 86 biomechanical and kinematic features from a single normalized TrialSequence.
    """
    coords = seq.norm_coords  # Shape (T, 25, 3)
    ts = seq.timestamps_sec
    if len(ts) > 1:
        dt = float(np.median(np.diff(ts)))
        if dt <= 1e-5 or not np.isfinite(dt):
            dt = 1.0 / 30.0
    else:
        dt = 1.0 / 30.0

    feats: Dict[str, float] = {}

    # Retrieve 3D joint trajectories (already root-centered at SpineBase and scaled by torso height)
    spine_base = _joint(coords, "SpineBase")
    spine_shoulder = _joint(coords, "SpineShoulder")
    head = _joint(coords, "Head")

    sh_l = _joint(coords, "ShoulderLeft")
    sh_r = _joint(coords, "ShoulderRight")
    el_l = _joint(coords, "ElbowLeft")
    el_r = _joint(coords, "ElbowRight")
    wr_l = _joint(coords, "WristLeft")
    wr_r = _joint(coords, "WristRight")

    hip_l = _joint(coords, "HipLeft")
    hip_r = _joint(coords, "HipRight")
    knee_l = _joint(coords, "KneeLeft")
    knee_r = _joint(coords, "KneeRight")
    ank_l = _joint(coords, "AnkleLeft")
    ank_r = _joint(coords, "AnkleRight")
    foot_l = _joint(coords, "FootLeft")
    foot_r = _joint(coords, "FootRight")

    # =========================================================================
    # 1. CONTRALATERAL CROSS-DISTANCES
    #    - Left Elbow to Right Foot & Right Elbow to Left Foot (mean, std, min)
    #    - Left Wrist to Right Ankle & Right Wrist to Left Ankle (mean, std, min)
    # =========================================================================
    dist_elL_foR = _euclidean_dist(el_l, foot_r)
    dist_elR_foL = _euclidean_dist(el_r, foot_l)

    feats["cross_dist_left_elbow_right_foot_mean"] = float(np.mean(dist_elL_foR))
    feats["cross_dist_left_elbow_right_foot_std"] = float(np.std(dist_elL_foR))
    feats["cross_dist_left_elbow_right_foot_min"] = float(np.min(dist_elL_foR))

    feats["cross_dist_right_elbow_left_foot_mean"] = float(np.mean(dist_elR_foL))
    feats["cross_dist_right_elbow_left_foot_std"] = float(np.std(dist_elR_foL))
    feats["cross_dist_right_elbow_left_foot_min"] = float(np.min(dist_elR_foL))

    dist_wrL_anR = _euclidean_dist(wr_l, ank_r)
    dist_wrR_anL = _euclidean_dist(wr_r, ank_l)
    feats["cross_dist_left_wrist_right_ankle_mean"] = float(np.mean(dist_wrL_anR))
    feats["cross_dist_left_wrist_right_ankle_std"] = float(np.std(dist_wrL_anR))
    feats["cross_dist_left_wrist_right_ankle_min"] = float(np.min(dist_wrL_anR))
    feats["cross_dist_right_wrist_left_ankle_mean"] = float(np.mean(dist_wrR_anL))
    feats["cross_dist_right_wrist_left_ankle_std"] = float(np.std(dist_wrR_anL))
    feats["cross_dist_right_wrist_left_ankle_min"] = float(np.min(dist_wrR_anL))

    feats["cross_dist_elbow_foot_asymmetry"] = float(
        abs(np.mean(dist_elL_foR) - np.mean(dist_elR_foL))
        / (0.5 * (np.mean(dist_elL_foR) + np.mean(dist_elR_foL)) + 1e-6)
    )
    feats["cross_dist_wrist_ankle_asymmetry"] = float(
        abs(np.mean(dist_wrL_anR) - np.mean(dist_wrR_anL))
        / (0.5 * (np.mean(dist_wrL_anR) + np.mean(dist_wrR_anL)) + 1e-6)
    )

    # =========================================================================
    # 2. SAGITTAL JOINT ANGLES (Knee Flexion & Hip Angle: Max ROM & Angular Velocity)
    # =========================================================================
    knee_sag_l = _angle_sagittal_deg(hip_l, knee_l, ank_l)
    knee_sag_r = _angle_sagittal_deg(hip_r, knee_r, ank_r)
    for side, ang_sag in [("left", knee_sag_l), ("right", knee_sag_r)]:
        ang_vel_sag = np.abs(np.diff(ang_sag)) / dt
        feats[f"sagittal_knee_flexion_{side}_max_rom"] = float(np.ptp(ang_sag))
        feats[f"sagittal_knee_flexion_{side}_mean"] = float(np.mean(ang_sag))
        feats[f"sagittal_knee_flexion_{side}_std"] = float(np.std(ang_sag))
        feats[f"sagittal_knee_flexion_{side}_ang_vel_mean"] = float(np.mean(ang_vel_sag))
        feats[f"sagittal_knee_flexion_{side}_ang_vel_max"] = float(np.max(ang_vel_sag))

    hip_sag_l = _angle_sagittal_deg(spine_shoulder, hip_l, knee_l)
    hip_sag_r = _angle_sagittal_deg(spine_shoulder, hip_r, knee_r)
    for side, ang_sag in [("left", hip_sag_l), ("right", hip_sag_r)]:
        ang_vel_sag = np.abs(np.diff(ang_sag)) / dt
        feats[f"sagittal_hip_angle_{side}_max_rom"] = float(np.ptp(ang_sag))
        feats[f"sagittal_hip_angle_{side}_mean"] = float(np.mean(ang_sag))
        feats[f"sagittal_hip_angle_{side}_std"] = float(np.std(ang_sag))
        feats[f"sagittal_hip_angle_{side}_ang_vel_mean"] = float(np.mean(ang_vel_sag))

    shoulder_ang_l = _angle_3d_deg(spine_shoulder, sh_l, el_l)
    shoulder_ang_r = _angle_3d_deg(spine_shoulder, sh_r, el_r)
    elbow_ang_l = _angle_3d_deg(sh_l, el_l, wr_l)
    elbow_ang_r = _angle_3d_deg(sh_r, el_r, wr_r)
    feats["shoulder_angle_left_mean"] = float(np.mean(shoulder_ang_l))
    feats["shoulder_angle_right_mean"] = float(np.mean(shoulder_ang_r))
    feats["elbow_angle_left_mean"] = float(np.mean(elbow_ang_l))
    feats["elbow_angle_right_mean"] = float(np.mean(elbow_ang_r))

    # =========================================================================
    # 3. MOTION DERIVATIVES (1st-Order Velocity & 2nd-Order Acceleration on Ankles & Wrists)
    # =========================================================================
    for joint_label, traj in [
        ("ankle_left", ank_l),
        ("ankle_right", ank_r),
        ("wrist_left", wr_l),
        ("wrist_right", wr_r),
    ]:
        vel_vec = np.diff(traj, axis=0) / dt
        vel_mag = np.linalg.norm(vel_vec, axis=1)
        acc_vec = np.diff(vel_vec, axis=0) / dt
        acc_mag = np.linalg.norm(acc_vec, axis=1)

        feats[f"{joint_label}_vel_mean"] = float(np.mean(vel_mag))
        feats[f"{joint_label}_vel_std"] = float(np.std(vel_mag))
        if joint_label.startswith("ankle_"):
            feats[f"{joint_label}_vel_z_std"] = float(np.std(vel_vec[:, 2]))
        feats[f"{joint_label}_acc_mean"] = float(np.mean(acc_mag))
        feats[f"{joint_label}_acc_std"] = float(np.std(acc_mag))

    mean_wrist_vel = 0.5 * (feats["wrist_left_vel_mean"] + feats["wrist_right_vel_mean"])
    mean_ankle_vel = 0.5 * (feats["ankle_left_vel_mean"] + feats["ankle_right_vel_mean"])
    feats["wrist_to_ankle_velocity_ratio"] = float(mean_wrist_vel / (mean_ankle_vel + 1e-6))

    # =========================================================================
    # 4. STRIDE & POSTURAL COORDINATION METRICS
    #    - Arm swing asymmetry index & step cadence variability
    # =========================================================================
    wr_l_disp = np.linalg.norm(wr_l - np.mean(wr_l, axis=0, keepdims=True), axis=1)
    wr_r_disp = np.linalg.norm(wr_r - np.mean(wr_r, axis=0, keepdims=True), axis=1)
    rom_wr_l = float(np.ptp(wr_l_disp))
    rom_wr_r = float(np.ptp(wr_r_disp))
    feats["arm_swing_asymmetry_index"] = float(
        abs(rom_wr_l - rom_wr_r) / (0.5 * (rom_wr_l + rom_wr_r) + 1e-6)
    )

    rom_wr_l_z = float(np.ptp(wr_l[:, 2]))
    rom_wr_r_z = float(np.ptp(wr_r[:, 2]))
    feats["arm_swing_sagittal_asymmetry"] = float(
        abs(rom_wr_l_z - rom_wr_r_z) / (0.5 * (rom_wr_l_z + rom_wr_r_z) + 1e-6)
    )
    rom_wr_l_y = float(np.ptp(wr_l[:, 1]))
    rom_wr_r_y = float(np.ptp(wr_r[:, 1]))
    feats["arm_swing_vertical_asymmetry"] = float(
        abs(rom_wr_l_y - rom_wr_r_y) / (0.5 * (rom_wr_l_y + rom_wr_r_y) + 1e-6)
    )

    inter_ankle_dist = _euclidean_dist(ank_l, ank_r)
    inter_foot_dist = _euclidean_dist(foot_l, foot_r)
    step_width_x = np.abs(ank_l[:, 0] - ank_r[:, 0])

    cadence_metrics = _step_cadence_variability(
        inter_ankle_dist=inter_ankle_dist,
        ankle_left_y=ank_l[:, 1],
        ankle_right_y=ank_r[:, 1],
        dt=dt,
    )
    feats.update(cadence_metrics)

    feats["step_width_lateral_mean"] = float(np.mean(step_width_x))
    feats["step_width_lateral_std"] = float(np.std(step_width_x))
    feats["inter_foot_distance_mean"] = float(np.mean(inter_foot_dist))
    feats["inter_foot_distance_max"] = float(np.max(inter_foot_dist))
    feats["inter_foot_distance_min"] = float(np.min(inter_foot_dist))

    # Normalized upper/lower limb elevation and guarding relative to SpineBase
    feats["wrist_left_rel_height_mean"] = float(np.mean(wr_l[:, 1]))
    feats["wrist_right_rel_height_mean"] = float(np.mean(wr_r[:, 1]))
    feats["elbow_left_rel_height_mean"] = float(np.mean(el_l[:, 1]))
    feats["elbow_right_rel_height_mean"] = float(np.mean(el_r[:, 1]))
    feats["norm_y_HipLeft_mean"] = float(np.mean(hip_l[:, 1]))
    feats["norm_y_HipRight_mean"] = float(np.mean(hip_r[:, 1]))
    feats["norm_y_FootLeft_mean"] = float(np.mean(foot_l[:, 1]))
    feats["norm_y_FootRight_mean"] = float(np.mean(foot_r[:, 1]))
    feats["norm_z_ElbowLeft_mean"] = float(np.mean(el_l[:, 2]))
    feats["norm_z_ElbowRight_mean"] = float(np.mean(el_r[:, 2]))
    feats["norm_y_WristRight_std"] = float(np.std(wr_r[:, 1]))
    feats["norm_y_WristLeft_std"] = float(np.std(wr_l[:, 1]))

    # =========================================================================
    # 5. TRUNK TILT ANGLE RELATIVE TO VERTICAL Y-AXIS
    # =========================================================================
    trunk_vec = spine_shoulder - spine_base
    trunk_norm = np.linalg.norm(trunk_vec, axis=1) + 1e-8
    cos_vertical = np.clip(trunk_vec[:, 1] / trunk_norm, -1.0, 1.0)
    trunk_tilt_deg = np.degrees(np.arccos(cos_vertical))
    trunk_tilt_vel = np.abs(np.diff(trunk_tilt_deg)) / dt

    trunk_sagittal_lean_deg = np.degrees(np.arctan2(trunk_vec[:, 2], np.maximum(trunk_vec[:, 1], 1e-6)))
    trunk_coronal_sway_deg = np.degrees(np.arctan2(trunk_vec[:, 0], np.maximum(trunk_vec[:, 1], 1e-6)))

    head_neck_vec = head - spine_shoulder
    hn_norm = np.linalg.norm(head_neck_vec, axis=1) + 1e-8
    head_tilt_deg = np.degrees(np.arccos(np.clip(head_neck_vec[:, 1] / hn_norm, -1.0, 1.0)))

    feats["trunk_tilt_angle_mean"] = float(np.mean(trunk_tilt_deg))
    feats["trunk_tilt_angle_std"] = float(np.std(trunk_tilt_deg))
    feats["trunk_tilt_angle_max"] = float(np.max(trunk_tilt_deg))
    feats["trunk_tilt_angle_rom"] = float(np.ptp(trunk_tilt_deg))
    feats["trunk_tilt_ang_vel_mean"] = float(np.mean(trunk_tilt_vel))
    feats["trunk_sagittal_lean_mean"] = float(np.mean(trunk_sagittal_lean_deg))
    feats["trunk_coronal_sway_std"] = float(np.std(trunk_coronal_sway_deg))
    feats["head_tilt_angle_mean"] = float(np.mean(head_tilt_deg))

    return feats


def build_and_cache_feature_matrix(
    raw_dataset_dir: str = DEFAULT_RAW_DATASET_DIR,
    output_csv_path: str = DEFAULT_PROCESSED_CSV,
    force_recompute: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extracts normalized kinematic features across all 700 trial folders and caches the
    resulting DataFrame to `data/processed_features.csv`.
    """
    if not force_recompute and os.path.exists(output_csv_path):
        df_cached = pd.read_csv(output_csv_path)
        if len(df_cached) == 700 and "subject_id" in df_cached.columns and "label" in df_cached.columns:
            if verbose:
                print(
                    f"[Feature Extraction] Loaded cached feature matrix from '{output_csv_path}' "
                    f"(shape={df_cached.shape}, subjects={df_cached['subject_id'].nunique()})."
                )
            return df_cached

    sequences: List[TrialSequence] = load_all_normalized_trials(
        raw_dataset_dir=raw_dataset_dir,
        verbose=verbose,
    )

    records: List[Dict[str, object]] = []
    for seq in sequences:
        row_dict: Dict[str, object] = extract_features_from_sequence(seq)
        row_dict["label"] = int(seq.label)
        row_dict["subject_id"] = seq.subject_id
        records.append(row_dict)

    df = pd.DataFrame(records)

    feature_cols = [c for c in df.columns if c not in ("label", "subject_id")]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    if df[feature_cols].isnull().sum().sum() > 0:
        df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)

    if verbose:
        print(
            f"[Feature Extraction] Extracted {len(feature_cols)} kinematic features across "
            f"{len(df)} trials ({df['subject_id'].nunique()} unique subjects)."
        )
        print(f"[Feature Extraction] Cached tabular dataset to '{output_csv_path}'.")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 3D kinematic gait features and cache to CSV.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=DEFAULT_RAW_DATASET_DIR,
        help="Path to raw Al-Jubouri Dataset directory.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_PROCESSED_CSV,
        help="Output CSV path (default: data/processed_features.csv).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-extraction even if cached CSV exists.",
    )
    args = parser.parse_args()
    build_and_cache_feature_matrix(
        raw_dataset_dir=args.raw_dir,
        output_csv_path=args.output,
        force_recompute=args.force,
        verbose=True,
    )
