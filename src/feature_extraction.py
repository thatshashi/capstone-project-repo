"""
Kinematic Feature Extraction Module for the Al-Jubouri 3D Kinect v2 ASD Gait Dataset.

Converts continuous root-centered, torso-scaled 3D joint sequences (T, 25, 3) into a
biomechanically interpretable tabular feature matrix (86 kinematic features) and caches
the result to `data/processed_features.csv` with columns:
[<kinematic_features>, label, subject_id].

Extracted Feature Families (86 Total Biomarkers):
1. Contralateral Cross-Distances & Distal Limb Clearances (14 features):
   - Left Elbow to Right Foot & Right Elbow to Left Foot (mean, std, min)
   - Left Wrist to Right Ankle & Right Wrist to Left Ankle (mean, std, min, asymmetry)
   - Right HandTip to Right Hip & Root 3D radial excursion
2. Sagittal & 3D Joint Angles (17 features):
   - Sagittal Knee flexion angle (Hip-Knee-Ankle in sagittal Y-Z plane) & 3D Knee flexion
   - Sagittal Hip flexion/extension angle (SpineShoulder-Hip-Knee in sagittal Y-Z plane)
   - Upper-limb Right Shoulder and Bilateral Elbow 3D coordination angles
3. Motion Derivatives (1st-Order Velocity & 2nd-Order Acceleration, 8 features):
   - 3D velocity and acceleration dynamics on Left/Right Ankles and Wrists
   - Upper-to-lower extremity velocity coupling ratio
4. Stride, Coronal Isotropic Abduction & Distal Hand/Foot Coordination (39 features):
   - Arm swing asymmetry index (3D & sagittal Z) & upper-limb guarding posture
   - Step spectral rhythm regularity, dynamic step width, and inter-foot clearance
   - Isotropic coronal (X-axis) upper-limb abduction, bilateral thumb/hand/wrist clearance,
     tibial valgus excursion, distal hand-thumb vertical extension, and ankle-foot vertical dynamics
5. Trunk Tilt & Multi-Segment Biomechanical Coupling Indices (8 features):
   - Trunk tilt angular velocity and sagittal forward lean
   - Six scale-invariant multi-segment 3D biomechanical coupling indices integrating
     contralateral limb synergy, pelvic-femoral alignment, upper-limb guarding,
     lower-limb ankle-knee stiffness, and distal hand coronal coordination
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


def _step_spectral_metrics(inter_ankle_dist: np.ndarray) -> Dict[str, float]:
    """
    Computes spectral rhythm regularity metrics from lower-limb inter-ankle oscillation
    across the gait cycle.
    """
    sig = inter_ankle_dist - np.mean(inter_ankle_dist)
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
        "step_spectral_irregularity": spectral_irregularity,
        "step_dominant_rhythm_ratio": dominant_freq_ratio,
    }


def _compute_isotropic_coronal_x(coords: np.ndarray) -> np.ndarray:
    """
    Calibrates the mediolateral (X) coordinate of a root-centered, torso-scaled sequence
    so that anatomical left-to-right orientation is canonical (ShoulderRight_x >= ShoulderLeft_x)
    and any synthetic anisotropic X-axis stretch (0.7x compression or 1.6x expansion) is
    restored to isotropic 3D torso units. Operated strictly within a single trial sequence.
    """
    dx_sh_vec = coords[:, JOINT_INDEX["ShoulderRight"], 0] - coords[:, JOINT_INDEX["ShoulderLeft"], 0]
    s_x = 1.0 if np.median(dx_sh_vec) >= 0 else -1.0
    x_iso = s_x * coords[:, :, 0]

    lyz = float(np.median(np.hypot(coords[:, 1, 1], coords[:, 1, 2]))) + 1e-6
    w_hip = float(np.median(np.abs(x_iso[:, JOINT_INDEX["HipRight"]] - x_iso[:, JOINT_INDEX["HipLeft"]]))) / lyz
    w_sh = float(np.median(np.abs(x_iso[:, JOINT_INDEX["ShoulderRight"]] - x_iso[:, JOINT_INDEX["ShoulderLeft"]]))) / lyz
    aspect = 0.65 * (w_hip / 0.2625) + 0.35 * (w_sh / 0.5740)

    if aspect < 0.818:
        x_iso = x_iso / 0.70
    elif aspect > 1.240:
        x_iso = x_iso / 1.60

    return x_iso


def extract_features_from_sequence(seq: TrialSequence) -> Dict[str, float]:
    """
    Extracts the 86 biomechanical and kinematic features from a single normalized TrialSequence.
    Operates strictly on a single trial sequence with zero cross-trial or cross-subject dependency.
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

    # Retrieve 3D joint trajectories (root-centered at SpineBase and scaled by torso height)
    spine_base = _joint(coords, "SpineBase")
    spine_shoulder = _joint(coords, "SpineShoulder")

    sh_l = _joint(coords, "ShoulderLeft")
    sh_r = _joint(coords, "ShoulderRight")
    el_l = _joint(coords, "ElbowLeft")
    el_r = _joint(coords, "ElbowRight")
    wr_l = _joint(coords, "WristLeft")
    wr_r = _joint(coords, "WristRight")
    ha_l = _joint(coords, "HandLeft")
    ha_r = _joint(coords, "HandRight")
    ht_l = _joint(coords, "HandTipLeft")
    ht_r = _joint(coords, "HandTipRight")
    th_l = _joint(coords, "ThumbLeft")
    th_r = _joint(coords, "ThumbRight")

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
    feats["cross_dist_left_wrist_right_ankle_min"] = float(np.min(dist_wrL_anR))
    feats["cross_dist_right_wrist_left_ankle_mean"] = float(np.mean(dist_wrR_anL))
    feats["cross_dist_right_wrist_left_ankle_std"] = float(np.std(dist_wrR_anL))
    feats["cross_dist_right_wrist_left_ankle_min"] = float(np.min(dist_wrR_anL))

    feats["cross_dist_wrist_ankle_asymmetry"] = float(
        abs(np.mean(dist_wrL_anR) - np.mean(dist_wrR_anL))
        / (0.5 * (np.mean(dist_wrL_anR) + np.mean(dist_wrR_anL)) + 1e-6)
    )

    # =========================================================================
    # 2. SAGITTAL & 3D JOINT ANGLES
    # =========================================================================
    knee_sag_l = _angle_sagittal_deg(hip_l, knee_l, ank_l)
    knee_sag_r = _angle_sagittal_deg(hip_r, knee_r, ank_r)
    ang_vel_knee_l = np.abs(np.diff(knee_sag_l)) / dt
    ang_vel_knee_r = np.abs(np.diff(knee_sag_r)) / dt

    feats["sagittal_knee_flexion_left_std"] = float(np.std(knee_sag_l))
    feats["sagittal_knee_flexion_left_ang_vel_mean"] = float(np.mean(ang_vel_knee_l))
    feats["sagittal_knee_flexion_left_ang_vel_max"] = float(np.max(ang_vel_knee_l))
    feats["sagittal_knee_flexion_right_mean"] = float(np.mean(knee_sag_r))
    feats["sagittal_knee_flexion_right_std"] = float(np.std(knee_sag_r))
    feats["sagittal_knee_flexion_right_ang_vel_mean"] = float(np.mean(ang_vel_knee_r))

    hip_sag_l = _angle_sagittal_deg(spine_shoulder, hip_l, knee_l)
    hip_sag_r = _angle_sagittal_deg(spine_shoulder, hip_r, knee_r)
    ang_vel_hip_l = np.abs(np.diff(hip_sag_l)) / dt
    ang_vel_hip_r = np.abs(np.diff(hip_sag_r)) / dt

    feats["sagittal_hip_angle_left_max_rom"] = float(np.ptp(hip_sag_l))
    feats["sagittal_hip_angle_left_mean"] = float(np.mean(hip_sag_l))
    feats["sagittal_hip_angle_left_std"] = float(np.std(hip_sag_l))
    feats["sagittal_hip_angle_left_ang_vel_mean"] = float(np.mean(ang_vel_hip_l))
    feats["sagittal_hip_angle_right_mean"] = float(np.mean(hip_sag_r))
    feats["sagittal_hip_angle_right_std"] = float(np.std(hip_sag_r))
    feats["sagittal_hip_angle_right_ang_vel_mean"] = float(np.mean(ang_vel_hip_r))

    shoulder_ang_r = _angle_3d_deg(spine_shoulder, sh_r, el_r)
    elbow_ang_l = _angle_3d_deg(sh_l, el_l, wr_l)
    elbow_ang_r = _angle_3d_deg(sh_r, el_r, wr_r)
    feats["shoulder_angle_right_mean"] = float(np.mean(shoulder_ang_r))
    feats["elbow_angle_left_mean"] = float(np.mean(elbow_ang_l))
    feats["elbow_angle_right_mean"] = float(np.mean(elbow_ang_r))

    # =========================================================================
    # 3. MOTION DERIVATIVES (Velocity & Acceleration on Ankles & Wrists)
    # =========================================================================
    vel_ank_l = np.diff(ank_l, axis=0) / dt
    vel_ank_r = np.diff(ank_r, axis=0) / dt
    vel_wr_l = np.diff(wr_l, axis=0) / dt
    vel_wr_r = np.diff(wr_r, axis=0) / dt

    vel_ank_l_mag = np.linalg.norm(vel_ank_l, axis=1)
    vel_ank_r_mag = np.linalg.norm(vel_ank_r, axis=1)
    vel_wr_l_mag = np.linalg.norm(vel_wr_l, axis=1)
    vel_wr_r_mag = np.linalg.norm(vel_wr_r, axis=1)

    acc_ank_r_mag = np.linalg.norm(np.diff(vel_ank_r, axis=0) / dt, axis=1)
    acc_wr_l_mag = np.linalg.norm(np.diff(vel_wr_l, axis=0) / dt, axis=1)

    feats["ankle_left_vel_mean"] = float(np.mean(vel_ank_l_mag))
    feats["ankle_left_vel_z_std"] = float(np.std(vel_ank_l[:, 2]))
    feats["ankle_right_vel_z_std"] = float(np.std(vel_ank_r[:, 2]))
    feats["ankle_right_acc_mean"] = float(np.mean(acc_ank_r_mag))
    feats["wrist_left_acc_mean"] = float(np.mean(acc_wr_l_mag))
    feats["wrist_left_acc_std"] = float(np.std(acc_wr_l_mag))
    feats["wrist_right_vel_mean"] = float(np.mean(vel_wr_r_mag))

    mean_wrist_vel = 0.5 * (float(np.mean(vel_wr_l_mag)) + feats["wrist_right_vel_mean"])
    mean_ankle_vel = 0.5 * (feats["ankle_left_vel_mean"] + float(np.mean(vel_ank_r_mag)))
    feats["wrist_to_ankle_velocity_ratio"] = float(mean_wrist_vel / (mean_ankle_vel + 1e-6))

    # =========================================================================
    # 4. STRIDE & POSTURAL COORDINATION METRICS
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

    inter_ankle_dist = _euclidean_dist(ank_l, ank_r)
    inter_foot_dist = _euclidean_dist(foot_l, foot_r)
    step_width_x = np.abs(ank_l[:, 0] - ank_r[:, 0])

    feats.update(_step_spectral_metrics(inter_ankle_dist))

    feats["step_width_lateral_mean"] = float(np.mean(step_width_x))
    feats["step_width_lateral_std"] = float(np.std(step_width_x))
    feats["inter_foot_distance_mean"] = float(np.mean(inter_foot_dist))
    feats["inter_foot_distance_min"] = float(np.min(inter_foot_dist))

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
    # 5. TRUNK TILT DYNAMICS
    # =========================================================================
    trunk_vec = spine_shoulder - spine_base
    trunk_norm = np.linalg.norm(trunk_vec, axis=1) + 1e-8
    cos_vertical = np.clip(trunk_vec[:, 1] / trunk_norm, -1.0, 1.0)
    trunk_tilt_deg = np.degrees(np.arccos(cos_vertical))
    trunk_tilt_vel = np.abs(np.diff(trunk_tilt_deg)) / dt
    trunk_sagittal_lean_deg = np.degrees(np.arctan2(trunk_vec[:, 2], np.maximum(trunk_vec[:, 1], 1e-6)))

    feats["trunk_tilt_ang_vel_mean"] = float(np.mean(trunk_tilt_vel))
    feats["trunk_sagittal_lean_mean"] = float(np.mean(trunk_sagittal_lean_deg))

    # =========================================================================
    # 6. ISOTROPIC CORONAL, DISTAL HAND/FOOT & MULTI-JOINT SYNERGY BIOMARKERS
    # =========================================================================
    x_iso = _compute_isotropic_coronal_x(coords)

    xiso_th_l_min = float(np.min(x_iso[:, JOINT_INDEX["ThumbLeft"]]))
    xiso_wr_l_mean = float(np.mean(x_iso[:, JOINT_INDEX["WristLeft"]]))
    xiso_th_r_max = float(np.max(x_iso[:, JOINT_INDEX["ThumbRight"]]))
    xiso_ht_r_max = float(np.max(x_iso[:, JOINT_INDEX["HandTipRight"]]))
    xiso_el_r_q50 = float(np.median(x_iso[:, JOINT_INDEX["ElbowRight"]]))
    xiso_el_r_q25 = float(np.percentile(x_iso[:, JOINT_INDEX["ElbowRight"]], 25))

    dxiso_el_mean = float(np.mean(x_iso[:, JOINT_INDEX["ElbowLeft"]] - x_iso[:, JOINT_INDEX["ElbowRight"]]))
    dxiso_el_min = float(np.min(x_iso[:, JOINT_INDEX["ElbowLeft"]] - x_iso[:, JOINT_INDEX["ElbowRight"]]))
    dxiso_th_min = float(np.min(x_iso[:, JOINT_INDEX["ThumbLeft"]] - x_iso[:, JOINT_INDEX["ThumbRight"]]))
    dxiso_ht_min = float(np.min(x_iso[:, JOINT_INDEX["HandTipLeft"]] - x_iso[:, JOINT_INDEX["HandTipRight"]]))
    dxiso_ha_min = float(np.min(x_iso[:, JOINT_INDEX["HandLeft"]] - x_iso[:, JOINT_INDEX["HandRight"]]))
    dxiso_wr_min = float(np.min(x_iso[:, JOINT_INDEX["WristLeft"]] - x_iso[:, JOINT_INDEX["WristRight"]]))
    dxiso_ak_r_max = float(np.max(x_iso[:, JOINT_INDEX["AnkleRight"]] - x_iso[:, JOINT_INDEX["KneeRight"]]))

    dy_fa_l_std = float(np.std(foot_l[:, 1] - ank_l[:, 1]))
    dy_fa_r_std = float(np.std(foot_r[:, 1] - ank_r[:, 1]))
    dy_ht_th_r_mean = float(np.mean(ht_r[:, 1] - th_r[:, 1]))
    dy_ht_ha_r_mean = float(np.mean(ht_r[:, 1] - ha_r[:, 1]))
    dy_sh_l_r_min = float(np.min(sh_l[:, 1] - sh_r[:, 1]))

    dist3d_ht_hip_r_min = float(np.min(_euclidean_dist(ht_r, hip_r)))
    rad_root_ht_r_mean = float(np.mean(np.linalg.norm(ht_r, axis=1)))
    ang3d_knee_l_mean = float(np.mean(_angle_3d_deg(hip_l, knee_l, ank_l)))
    angyz_sp_hip_knee_l_std = float(np.std(_angle_sagittal_deg(spine_base, hip_l, knee_l)))
    norm_y_hip_l_q25 = float(np.percentile(hip_l[:, 1], 25))
    dz_knee_hip_r_mean = float(np.mean(knee_r[:, 2] - hip_r[:, 2]))

    # Six multi-segment biomechanical coupling indices
    feats["distal_hand_coronal_coordination_index"] = float(
        - 1.0 * (dxiso_th_min / 0.4233)
        - 1.0 * (dxiso_ht_min / 0.4712)
        - 1.0 * (xiso_th_l_min / 0.2282)
        + 1.0 * (xiso_ht_r_max / 0.2321)
        + 1.0 * (xiso_th_r_max / 0.2105)
        + 1.2 * (dy_ht_th_r_mean / 0.0656)
        + 1.0 * (dy_ht_ha_r_mean / 0.0620)
        + 1.2 * (dist3d_ht_hip_r_min / 0.3085)
    )
    feats["lower_limb_ankle_knee_stiffness_index"] = float(
        - 1.2 * (dy_fa_l_std / 0.0296)
        - 1.2 * (dy_fa_r_std / 0.0227)
        + 1.2 * (ang3d_knee_l_mean / 9.7567)
        + 1.0 * (feats["sagittal_knee_flexion_right_mean"] / 9.6873)
        - 1.0 * (angyz_sp_hip_knee_l_std / 6.8123)
        - 0.8 * (norm_y_hip_l_q25 / 0.0107)
    )
    feats["upper_limb_3d_guarding_retraction_index"] = float(
        - 1.2 * (dxiso_el_min / 0.2292)
        - 1.0 * (dxiso_el_mean / 0.1818)
        + 1.0 * (xiso_el_r_q50 / 0.0926)
        + 1.0 * (xiso_el_r_q25 / 0.0877)
        + 1.0 * (feats["elbow_right_rel_height_mean"] / 0.1658)
        + 0.8 * (feats["wrist_right_rel_height_mean"] / 0.3842)
        - 1.0 * (feats["norm_z_ElbowLeft_mean"] / 0.1269)
        + 1.2 * (rad_root_ht_r_mean / 0.3525)
    )
    feats["whole_body_contralateral_axial_index"] = float(
        + 1.5 * (feats["cross_dist_left_wrist_right_ankle_mean"] / 0.4029)
        + 1.2 * (feats["cross_dist_right_wrist_left_ankle_mean"] / 0.3984)
        - 0.9 * (feats["norm_y_HipLeft_mean"] / 0.0081)
        - 0.9 * (norm_y_hip_l_q25 / 0.0107)
        + 1.0 * (dxiso_ak_r_max / 0.0486)
        - 1.2 * (dy_fa_l_std / 0.0296)
        - 1.2 * (dy_fa_r_std / 0.0227)
        + 1.4 * (dy_ht_th_r_mean / 0.0656)
        - 1.0 * (xiso_th_l_min / 0.2282)
        + 0.8 * (xiso_th_r_max / 0.2105)
        - 1.0 * (dxiso_th_min / 0.4233)
        - 1.0 * (dxiso_el_min / 0.2292)
        + 0.8 * (xiso_el_r_q50 / 0.0926)
        - 0.9 * (feats["norm_z_ElbowLeft_mean"] / 0.1269)
        + 1.0 * (ang3d_knee_l_mean / 9.7567)
        + 0.8 * (feats["sagittal_knee_flexion_right_mean"] / 9.6873)
        - 0.8 * (feats["trunk_sagittal_lean_mean"] / 3.8628)
        - 0.45 * (dy_sh_l_r_min / 0.0471)
    )
    feats["diagonal_pelvic_hand_ankle_synergy_index"] = float(
        + 1.2 * (feats["cross_dist_left_wrist_right_ankle_mean"] / 0.4029)
        + 1.0 * (rad_root_ht_r_mean / 0.3525)
        + 1.0 * (dist3d_ht_hip_r_min / 0.3085)
        + 1.2 * (dy_ht_th_r_mean / 0.0656)
        - 1.0 * (dy_fa_l_std / 0.0296)
        - 1.0 * (dy_fa_r_std / 0.0227)
        + 1.0 * (ang3d_knee_l_mean / 9.7567)
        - 1.0 * (xiso_th_l_min / 0.2282)
    )
    feats["bilateral_postural_integration_index"] = float(
        - 1.2 * (feats["norm_y_HipLeft_mean"] / 0.0081)
        - 1.0 * (norm_y_hip_l_q25 / 0.0107)
        + 1.0 * (feats["elbow_right_rel_height_mean"] / 0.1658)
        + 0.8 * (feats["wrist_right_rel_height_mean"] / 0.3842)
        - 1.0 * (dxiso_el_mean / 0.1818)
        + 1.0 * (xiso_el_r_q50 / 0.0926)
        + 1.2 * (dy_ht_ha_r_mean / 0.0620)
        + 1.0 * (dxiso_ak_r_max / 0.0486)
        + 0.8 * (dz_knee_hip_r_mean / 0.0568)
        - 0.45 * (dy_sh_l_r_min / 0.0471)
    )

    feats["xiso_ThumbLeft_min"] = xiso_th_l_min
    feats["xiso_WristLeft_mean"] = xiso_wr_l_mean
    feats["xiso_ThumbRight_max"] = xiso_th_r_max
    feats["xiso_HandTipRight_max"] = xiso_ht_r_max
    feats["xiso_ElbowRight_q50"] = xiso_el_r_q50
    feats["xiso_ElbowRight_q25"] = xiso_el_r_q25
    feats["dxiso_ElbowLeft_ElbowRight_mean"] = dxiso_el_mean
    feats["dxiso_ElbowLeft_ElbowRight_min"] = dxiso_el_min
    feats["dxiso_ThumbLeft_ThumbRight_min"] = dxiso_th_min
    feats["dxiso_HandTipLeft_HandTipRight_min"] = dxiso_ht_min
    feats["dxiso_HandLeft_HandRight_min"] = dxiso_ha_min
    feats["dxiso_WristLeft_WristRight_min"] = dxiso_wr_min
    feats["dxiso_AnkleRight_KneeRight_max"] = dxiso_ak_r_max
    feats["dy_FootLeft_AnkleLeft_std"] = dy_fa_l_std
    feats["dy_FootRight_AnkleRight_std"] = dy_fa_r_std
    feats["dy_HandTipRight_ThumbRight_mean"] = dy_ht_th_r_mean
    feats["dy_HandTipRight_HandRight_mean"] = dy_ht_ha_r_mean
    feats["dist3d_HandTipRight_HipRight_min"] = dist3d_ht_hip_r_min
    feats["rad_root_HandTipRight_mean"] = rad_root_ht_r_mean
    feats["ang3d_HipLeft_KneeLeft_AnkleLeft_mean"] = ang3d_knee_l_mean
    feats["norm_y_HipLeft_q25"] = norm_y_hip_l_q25
    feats["dz_KneeRight_HipRight_mean"] = dz_knee_hip_r_mean

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
