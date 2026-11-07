"""
Data Preprocessing Module for the Al-Jubouri 3D Kinect v2 ASD Gait Dataset.

Traverses the 700 trial folders (50 Typically Developing [TD] children and 50 Autism
Spectrum Disorder [ASD] children, 7 trials/augmentations per child), parses the 25 3D
skeletal joint trajectories (X, Y, Z), maps each folder strictly to its true underlying
`subject_id` (Child_01 to Child_100), and normalizes coordinates via:
1. Root-centering: Translates all joints per frame so SpineBase (joint 0) is at (0, 0, 0).
2. Torso-scaling: Scales all joint coordinates by the frame-wise 3D trunk height
   (Euclidean distance between SpineShoulder and SpineBase) to eliminate participant
   height and camera-distance biases.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import openpyxl


# Default location of the extracted 700-folder Al-Jubouri Kinect v2 dataset
DEFAULT_RAW_DATASET_DIR = os.environ.get(
    "ASD_RAW_DATA_DIR",
    r"C:\Users\thats\Downloads\Dataset-2\Dataset",
)

# Canonical 25 Kinect v2 joints with SpineBase as Joint 0 and SpineShoulder as Joint 1
CANONICAL_JOINTS: Tuple[str, ...] = (
    "SpineBase",       # 0: Root joint (origin after root-centering)
    "SpineShoulder",   # 1: Upper trunk reference for torso-height scaling
    "Midspain",        # 2: Mid-thoracic spine (SpineMid)
    "Neck",            # 3: Cervical spine / Neck
    "Head",            # 4: Cranium / Head
    "ShoulderLeft",    # 5: Left Glenohumeral Joint
    "ElbowLeft",       # 6: Left Humeroulnar Joint
    "WristLeft",       # 7: Left Radiocarpal Joint
    "HandLeft",        # 8: Left Metacarpal Center
    "HandTipLeft",     # 9: Left Distal Phalanx
    "ThumbLeft",       # 10: Left Pollex
    "ShoulderRight",   # 11: Right Glenohumeral Joint
    "ElbowRight",      # 12: Right Humeroulnar Joint
    "WristRight",      # 13: Right Radiocarpal Joint
    "HandRight",       # 14: Right Metacarpal Center
    "HandTipRight",    # 15: Right Distal Phalanx
    "ThumbRight",      # 16: Right Pollex
    "HipLeft",         # 17: Left Acetabulofemoral Joint
    "KneeLeft",        # 18: Left Tibiofemoral Joint
    "AnkleLeft",       # 19: Left Talocrural Joint
    "FootLeft",        # 20: Left Distal Foot
    "HipRight",        # 21: Right Acetabulofemoral Joint
    "KneeRight",       # 22: Right Tibiofemoral Joint
    "AnkleRight",      # 23: Right Talocrural Joint
    "FootRight",       # 24: Right Distal Foot
)

JOINT_INDEX: Dict[str, int] = {name: idx for idx, name in enumerate(CANONICAL_JOINTS)}

# Standard column offsets (1-based column indices in the raw Al-Jubouri .xlsx files)
# Row 0 header places these 25 joints at columns 1, 4, 7, ..., 73:
RAW_FILE_JOINT_COLUMNS: Dict[str, int] = {
    "Midspain": 1,
    "AnkleLeft": 4,
    "AnkleRight": 7,
    "ElbowLeft": 10,
    "ElbowRight": 13,
    "FootLeft": 16,
    "FootRight": 19,
    "HandLeft": 22,
    "HandRight": 25,
    "HandTipLeft": 28,
    "HandTipRight": 31,
    "Head": 34,
    "HipLeft": 37,
    "HipRight": 40,
    "KneeLeft": 43,
    "KneeRight": 46,
    "Neck": 49,
    "ShoulderLeft": 52,
    "ShoulderRight": 55,
    "SpineBase": 58,
    "SpineShoulder": 61,
    "ThumbLeft": 64,
    "ThumbRight": 67,
    "WristLeft": 70,
    "WristRight": 73,
}


@dataclass
class TrialSequence:
    """Container for a single normalized 3D skeletal gait trial."""

    subject_id: str          # e.g., "Child_01" .. "Child_100"
    label: int               # 0 = Typically Developing (TD), 1 = ASD
    trial_id: int            # 1 .. 7
    filepath: str            # Source coordinate .xlsx path
    timestamps_sec: np.ndarray  # Shape (T,) in seconds
    raw_coords: np.ndarray      # Shape (T, 25, 3) raw Kinect v2 coordinates (meters)
    norm_coords: np.ndarray     # Shape (T, 25, 3) SpineBase-centered & torso-scaled
    torso_lengths: np.ndarray   # Shape (T,) Euclidean distance ||SpineShoulder - SpineBase||


def _parse_timestamp_to_seconds(ts_val: object, fallback_idx: int, fps: float = 30.0) -> float:
    """Parses Kinect timestamp string 'H:M:S:MS' into elapsed seconds."""
    if ts_val is None:
        return fallback_idx / fps
    s = str(ts_val).strip()
    parts = s.replace(".", ":").split(":")
    if len(parts) == 4:
        try:
            hh = float(parts[0])
            mm = float(parts[1])
            ss = float(parts[2])
            ms = float(parts[3])
            return hh * 3600.0 + mm * 60.0 + ss + (ms / 1000.0)
        except ValueError:
            pass
    return fallback_idx / fps


def normalize_skeleton_sequence(
    raw_coords: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalizes a 3D skeletal sequence of shape (T, 25, 3).

    Steps:
    1. Root-centering: Translates all joints so SpineBase (joint 0) is at (0, 0, 0)
       at every frame t.
    2. Torso-scaling: Divides all joint coordinates at frame t by the 3D trunk height
       L_t = ||SpineShoulder (joint 1) - SpineBase (joint 0)||_2 to remove subject
       stature and camera-distance biases.

    Parameters
    ----------
    raw_coords : np.ndarray
        Array of shape (T, 25, 3) containing raw (X, Y, Z) coordinates.
    eps : float, default=1e-6
        Numerical stability floor for torso length division.

    Returns
    -------
    norm_coords : np.ndarray
        Normalized coordinates of shape (T, 25, 3) with SpineBase == (0, 0, 0)
        and ||SpineShoulder - SpineBase||_2 == 1.0.
    torso_lengths : np.ndarray
        Frame-wise 3D trunk heights of shape (T,).
    """
    if raw_coords.ndim != 3 or raw_coords.shape[1:] != (25, 3):
        raise ValueError(f"Expected raw_coords of shape (T, 25, 3), got {raw_coords.shape}")

    # 1. Root-centering around SpineBase (joint 0)
    spine_base = raw_coords[:, 0:1, :]  # (T, 1, 3)
    centered = raw_coords - spine_base  # (T, 25, 3), joint 0 is now (0, 0, 0)

    # 2. Torso-scaling by trunk height ||SpineShoulder (joint 1) - SpineBase (joint 0)||_2
    spine_shoulder = centered[:, 1, :]  # (T, 3)
    torso_lengths = np.linalg.norm(spine_shoulder, axis=1)  # (T,)

    # Guard against degenerate frames by substituting median valid trunk height
    valid_mask = torso_lengths > eps
    if np.any(valid_mask):
        median_torso = float(np.median(torso_lengths[valid_mask]))
    else:
        median_torso = 1.0
    safe_torso = np.where(valid_mask, torso_lengths, max(median_torso, eps))

    norm_coords = centered / safe_torso[:, np.newaxis, np.newaxis]
    return norm_coords, safe_torso


def parse_coordinate_excel(
    filepath: str,
    subject_id: str,
    label: int,
    trial_id: int,
) -> TrialSequence:
    """
    Reads a single trial coordinate .xlsx file (columns 0..75), extracts the 25 3D
    joint trajectories, and applies root-centering and torso-scaling.
    """
    wb = openpyxl.load_workbook(filepath, read_only=True, data_only=True)
    try:
        ws = wb.active
        rows = list(ws.iter_rows(min_col=1, max_col=76, values_only=True))
    finally:
        wb.close()

    if len(rows) < 2:
        raise ValueError(f"Coordinate file '{filepath}' has fewer than 2 rows.")

    header_row = rows[0]
    # Build mapping from joint name -> 0-based column index in `rows`
    col_map: Dict[str, int] = {}
    for col_idx, cell_val in enumerate(header_row):
        if cell_val is not None:
            clean_name = str(cell_val).strip()
            if clean_name in JOINT_INDEX:
                col_map[clean_name] = col_idx

    # Fallback to standard column layout if any header string has a typo
    for j_name, default_col in RAW_FILE_JOINT_COLUMNS.items():
        if j_name not in col_map:
            col_map[j_name] = default_col

    timestamps: List[float] = []
    frames_xyz: List[np.ndarray] = []

    for row_idx, row in enumerate(rows[1:]):
        if row is None or len(row) < 76:
            continue
        # Check if core joints have numeric coordinates on this row
        sb_col = col_map["SpineBase"]
        if row[sb_col] is None or row[sb_col + 1] is None or row[sb_col + 2] is None:
            continue

        frame_joints = np.zeros((25, 3), dtype=np.float64)
        valid_frame = True
        for j_name, canonical_idx in JOINT_INDEX.items():
            c = col_map[j_name]
            try:
                x = float(row[c])
                y = float(row[c + 1])
                z = float(row[c + 2])
            except (TypeError, ValueError):
                valid_frame = False
                break
            if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(z)):
                valid_frame = False
                break
            frame_joints[canonical_idx, 0] = x
            frame_joints[canonical_idx, 1] = y
            frame_joints[canonical_idx, 2] = z

        if valid_frame:
            ts = _parse_timestamp_to_seconds(row[0], len(frames_xyz))
            timestamps.append(ts)
            frames_xyz.append(frame_joints)

    if len(frames_xyz) < 3:
        raise ValueError(f"Insufficient valid 3D frames ({len(frames_xyz)}) in '{filepath}'.")

    raw_coords = np.stack(frames_xyz, axis=0)  # (T, 25, 3)
    ts_arr = np.asarray(timestamps, dtype=np.float64)
    # Convert timestamps to relative elapsed seconds from frame 0; handle midnight wrap or non-monotonic noise
    ts_arr = ts_arr - ts_arr[0]
    diffs = np.diff(ts_arr)
    if np.any(diffs <= 0) or np.any(diffs > 1.0):
        # Standard Kinect v2 frame interval is 1/30 s (~0.0333 s)
        ts_arr = np.arange(len(raw_coords), dtype=np.float64) / 30.0

    norm_coords, torso_lengths = normalize_skeleton_sequence(raw_coords)

    return TrialSequence(
        subject_id=subject_id,
        label=label,
        trial_id=trial_id,
        filepath=filepath,
        timestamps_sec=ts_arr,
        raw_coords=raw_coords,
        norm_coords=norm_coords,
        torso_lengths=torso_lengths,
    )


def discover_700_trial_folders(
    raw_dataset_dir: str = DEFAULT_RAW_DATASET_DIR,
) -> List[Tuple[str, int, int, str]]:
    """
    Discovers all 700 trial coordinate files across the 100 unique children:
    - Child_01 to Child_50: Typically Developing (TD, label=0), folders Typical/1..50/augmentation/1..7
    - Child_51 to Child_100: ASD (label=1), folders Autism/children with ASD/1..50/augmentation/1..7

    Returns
    -------
    manifest : List[Tuple[str, int, int, str]]
        List of (subject_id, label, trial_id, excel_filepath) sorted deterministically.
    """
    if not os.path.exists(raw_dataset_dir):
        raise FileNotFoundError(
            f"Raw dataset directory not found at '{raw_dataset_dir}'. "
            "Pass --raw-dir or set ASD_RAW_DATA_DIR."
        )

    manifest: List[Tuple[str, int, int, str]] = []

    # 1. Typically Developing (TD): Child_01 .. Child_50 (label = 0)
    typical_root = os.path.join(raw_dataset_dir, "Typical")
    for child_num in range(1, 51):
        subject_id = f"Child_{child_num:02d}"
        for trial_id in range(1, 8):
            trial_dir = os.path.join(typical_root, str(child_num), "augmentation", str(trial_id))
            xlsx_path = os.path.join(trial_dir, f"{trial_id}.xlsx")
            if not os.path.exists(xlsx_path):
                raise FileNotFoundError(f"Expected TD coordinate file missing: '{xlsx_path}'")
            manifest.append((subject_id, 0, trial_id, xlsx_path))

    # 2. Autism Spectrum Disorder (ASD): Child_51 .. Child_100 (label = 1)
    asd_root = os.path.join(raw_dataset_dir, "Autism", "children with ASD")
    for child_num in range(1, 51):
        subject_id = f"Child_{child_num + 50:02d}"
        for trial_id in range(1, 8):
            trial_dir = os.path.join(asd_root, str(child_num), "augmentation", str(trial_id))
            xlsx_path = os.path.join(trial_dir, f"{trial_id}.xlsx")
            if not os.path.exists(xlsx_path):
                raise FileNotFoundError(f"Expected ASD coordinate file missing: '{xlsx_path}'")
            manifest.append((subject_id, 1, trial_id, xlsx_path))

    return manifest


def load_all_normalized_trials(
    raw_dataset_dir: str = DEFAULT_RAW_DATASET_DIR,
    max_workers: int = 8,
    verbose: bool = True,
) -> List[TrialSequence]:
    """
    Loads and normalizes all 700 trial sequences across the 100 children in parallel.
    """
    manifest = discover_700_trial_folders(raw_dataset_dir)
    if verbose:
        print(f"[Data Preprocessing] Discovered {len(manifest)} coordinate folders across 100 subjects.")

    def _worker(item: Tuple[str, int, int, str]) -> TrialSequence:
        subj_id, lbl, tr_id, fpath = item
        return parse_coordinate_excel(fpath, subj_id, lbl, tr_id)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        sequences = list(pool.map(_worker, manifest))

    if verbose:
        n_subjects = len({s.subject_id for s in sequences})
        n_td = sum(1 for s in sequences if s.label == 0)
        n_asd = sum(1 for s in sequences if s.label == 1)
        # Verify SpineBase root-centering and SpineShoulder torso-scaling invariants
        max_root_err = max(float(np.abs(s.norm_coords[:, 0, :]).max()) for s in sequences)
        max_scale_err = max(
            float(np.abs(np.linalg.norm(s.norm_coords[:, 1, :], axis=1) - 1.0).max())
            for s in sequences
        )
        print(
            f"[Data Preprocessing] Parsed {len(sequences)} trials | "
            f"{n_subjects} unique subjects ({n_td} TD trials, {n_asd} ASD trials)."
        )
        print(
            f"[Data Preprocessing] Normalization check -> "
            f"Max SpineBase origin deviation: {max_root_err:.2e} | "
            f"Max torso-unit scale deviation: {max_scale_err:.2e}"
        )

    return sequences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess 700 3D Kinect v2 gait trial folders.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=DEFAULT_RAW_DATASET_DIR,
        help="Path to extracted Al-Jubouri Dataset root containing 'Typical' and 'Autism' folders.",
    )
    args = parser.parse_args()
    trials = load_all_normalized_trials(raw_dataset_dir=args.raw_dir, verbose=True)
