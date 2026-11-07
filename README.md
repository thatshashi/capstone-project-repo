# Leak-Free 3D Kinematic ASD Classification Pipeline (Al-Jubouri Kinect v2 Dataset)

A peer-review-grade machine learning and biomechanical explainability pipeline for classifying **Autism Spectrum Disorder (ASD)** vs. **Typically Developing (TD)** children from 3D Microsoft Kinect v2 skeletal gait recordings on the **Al-Jubouri et al. (2020)** dataset.

---

## 1. Methodological Rigor: Resolving the 700-Folder Subject-Level Data Leakage Problem

### Why Naive Random Splits Produce Invalid ~99% Accuracy
The Al-Jubouri 3D Kinect v2 dataset consists of **700 augmented coordinate folders** (`350 ASD + 350 TD`) derived from **100 unique children** (`50 ASD, 50 TD`; **7 augmented trials per child** generated via 3D geometric and temporal transformations of each child's gait cycle).

In prior implementations and several published studies on this dataset, evaluation was performed using a **naive random 90/10 train/test split** or standard **`StratifiedKFold`** across the 700 (or 800) trial rows without grouping by participant identity:
1. **Subject-Identity Memorization**: When 5–6 augmented trials of `Child_01` are placed in the training set and 1–2 augmented trials of `Child_01` are placed in the test set, decision trees memorize participant-specific anthropometric ratios and baseline trajectories, yielding an **artificially inflated ~99.25% accuracy**.
2. **Camera-Distance & Stature Confounds**: Raw Kinect v2 $(X, Y, Z)$ coordinates are recorded in sensor-frame meters ($Z \in [1.5\text{ m}, 4.0\text{ m}]$). Without spatial normalization, classifiers exploit systematic differences in participant height or camera starting distance rather than genuine motor kinematics.

### Our Three-Pillar Leak-Free Solution
1. **Strict Subject-ID Mapping (`Child_01` to `Child_100`)**:
   - Every folder in `Typical/1..50/augmentation/1..7` (`Child_01`–`Child_50`, `label=0`) and `Autism/children with ASD/1..50/augmentation/1..7` (`Child_51`–`Child_100`, `label=1`) is deterministically mapped to its ground-truth `subject_id`.
2. **Per-Frame Skeletal Normalization (`src/data_preprocessing.py`)**:
   - **Root-Centering**: At every frame $t$, all 25 3D joints are translated so `SpineBase` (joint 0) is anchored at $(0, 0, 0)$:
     $$\mathbf{P}'_{t, j} = \mathbf{P}_{t, j} - \mathbf{P}_{t, \text{SpineBase}}$$
   - **Torso-Height Scaling**: All joint coordinates are scaled by the frame-wise 3D trunk height $L_{\text{torso}, t} = \|\mathbf{P}_{t, \text{SpineShoulder}} - \mathbf{P}_{t, \text{SpineBase}}\|_2$:
     $$\tilde{\mathbf{P}}_{t, j} = \frac{\mathbf{P}'_{t, j}}{\|\mathbf{P}_{t, \text{SpineShoulder}} - \mathbf{P}_{t, \text{SpineBase}}\|_2}$$
     eliminating participant stature and camera-depth biases ($0.00\times 10^0$ origin deviation, $2.22\times 10^{-16}$ unit-scale deviation across all 700 folders).
3. **Subject-Grouped Cross-Validation (`StratifiedGroupKFold(n_splits=5)`)**:
   - All 7 trials of any child are held strictly in **either** the training split **or** the test split on every fold (`assert set(train_subjects).isdisjoint(set(test_subjects))`). Every test prediction is made on **100% unseen children**.

---

## 2. Leak-Free Model Benchmark & Literature Comparison

### 2.1 Same-Split 5-Fold `StratifiedGroupKFold` Benchmark (`src/train.py`)
All three classifiers are evaluated on the exact same 5 subject-grouped folds (`700 trials, 100 unique children, 86 normalized 3D kinematic features`):

| Model | Accuracy (%) | AUROC | Sensitivity / ASD Recall (%) | Specificity / TD Recall (%) | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CatBoost (`iterations=500, depth=5, lr=0.04, l2=4`)** | **92.57 ± 2.70** | **0.9802 ± 0.0251** | **90.29 ± 6.54** | **94.86 ± 5.31** | **0.9233 ± 0.0296** |
| **Random Forest (Baseline)** | 89.57 ± 5.32 | 0.9635 ± 0.0429 | 87.71 ± 7.65 | 91.43 ± 6.94 | 0.8933 ± 0.0545 |
| **XGBoost (Baseline)** | 89.14 ± 6.61 | 0.9694 ± 0.0330 | 84.57 ± 7.19 | 93.71 ± 7.80 | 0.8863 ± 0.0684 |

#### CatBoost Per-Fold Leak-Free Breakdown (20 Unseen Children / 140 Trials per Fold)
- **Fold 1**: Accuracy = `89.29%` | AUROC = `0.9955` | Sensitivity = `78.57%` | Specificity = `100.00%` | F1 = `0.8800`
- **Fold 2**: Accuracy = `95.00%` | AUROC = `0.9973` | Sensitivity = `90.00%` | Specificity = `100.00%` | F1 = `0.9474`
- **Fold 3**: Accuracy = `89.29%` | AUROC = `0.9314` | Sensitivity = `91.43%` | Specificity = `87.14%` | F1 = `0.8951`
- **Fold 4**: Accuracy = `94.29%` | AUROC = `0.9812` | Sensitivity = `98.57%` | Specificity = `90.00%` | F1 = `0.9452`
- **Fold 5**: Accuracy = `95.00%` | AUROC = `0.9957` | Sensitivity = `92.86%` | Specificity = `97.14%` | F1 = `0.9489`

---

### 2.2 Benchmark Comparison Against Published Literature on the Al-Jubouri 3D Kinect v2 Dataset

| Study / Architecture | Feature / Representation Strategy | Validation Protocol | Reported Accuracy | Clinical Explainability (XAI) | Methodological Notes |
| :--- | :--- | :--- | :---: | :---: | :--- |
| **Al-Jubouri et al. (2021)** — *Rough Set + PCA* | 1,259 tabular features compressed via PCA | Augmented Split | **92.0%** *(60.0% raw)* | No | PCA destroys anatomical joint identity; unaugmented accuracy was 60.0%. |
| **Mukhanova et al.** — *GCN-BiLSTM* | Spatial-Temporal Graph + Bidirectional LSTM | Cross-Validation | **88.5%** | No | Deep sequence model prone to overfitting on 100-subject cohort; black-box latent state. |
| **Firat et al.** — *Random Forest* | Selected kinematic & spatiotemporal features | Cross-Validation | **91.0% – 93.0%** | No | Asymmetric greedy decision trees without game-theoretic biomarker attribution. |
| **Le-Thi-Thu et al.** — *ST-GCN* | Spatial-Temporal Graph Convolutional Network | Cross-Validation | **94.0% – 96.0%** | No | High computational overhead; lacks direct clinical tabular cutoffs for clinicians. |
| **Ours (Tuned CatBoost)** | **86 Root-Centered, Torso-Scaled 3D Kinematic Biomarkers** | **5-Fold `StratifiedGroupKFold` (by `subject_id`)** | **92.57% ± 2.70%** *(AUROC: 0.9802)* | **Yes (TreeSHAP Top-15 Biomarkers)** | **100% zero subject leakage**, scale-invariant 3D biomechanics, outperforms RF (89.57%) & XGBoost (89.14%) on identical splits. |

---

## 3. Extracted 3D Kinematic Feature Families (`src/feature_extraction.py`)

Each normalized `(T, 25, 3)` skeletal sequence is transformed into **86 scale-invariant biomechanical features** across 5 clinical domains and cached to [`data/processed_features.csv`](file:///C:/Users/thats/Downloads/capstone-main%20(1)/capstone-main/data/processed_features.csv):

1. **Contralateral Cross-Distances**:
   - 3D Euclidean distances between **Left Elbow and Right Foot** and **Right Elbow and Left Foot** (`mean`, `std`, `min`), **Left Wrist and Right Ankle** / **Right Wrist and Left Ankle** (`mean`, `std`, `min`), and contralateral cross-limb asymmetry indices.
2. **Sagittal Joint Angles**:
   - **Knee Flexion Angle** ($\angle(\text{Hip}, \text{Knee}, \text{Ankle})$ in the $Y\text{-}Z$ sagittal plane) and **Hip Flexion/Extension Angle** ($\angle(\text{SpineShoulder}, \text{Hip}, \text{Knee})$ in the sagittal plane): maximum range of motion (`max_rom`), `mean`, `std`, and 1st-order **angular velocity** (`ang_vel_mean`, `ang_vel_max`), plus 3D Shoulder and Elbow coordination angles.
3. **Distal Extremity Motion Derivatives**:
   - **1st-order velocity** ($\|\mathbf{v}_t\|_2$ and sagittal progression variability $v_z$) and **2nd-order acceleration** ($\|\mathbf{a}_t\|_2$) on **Left/Right Ankles** and **Left/Right Wrists**, plus upper-to-lower extremity velocity coupling ratio.
4. **Stride & Upper-Limb Coordination Metrics**:
   - **Arm Swing Asymmetry Index (ASI)** (3D, sagittal $Z$, and vertical $Y$), bilateral wrist/elbow elevation guarding relative to `SpineBase`, **step cadence variability**, spectral rhythm entropy, dynamic lateral step width, and inter-foot clearance.
5. **Trunk Tilt Angle Relative to Vertical Y-Axis**:
   - 3D trunk inclination angle relative to the vertical $+Y$ axis (`mean`, `std`, `max`, `rom`, `ang_vel_mean`), sagittal forward torso lean, coronal medio-lateral sway, and cervical head-neck tilt angle.

---

## 4. Clinical Explainability: Top 15 Motor Biomarkers (`src/explainability.py`)

Computed via exact `shap.TreeExplainer` on the CatBoost model (saved to [`results/feature_importance.csv`](file:///C:/Users/thats/Downloads/capstone-main%20(1)/capstone-main/results/feature_importance.csv) and [`results/shap_summary.png`](file:///C:/Users/thats/Downloads/capstone-main%20(1)/capstone-main/results/shap_summary.png)):

| Rank | Feature Name | Mean \|SHAP\| | Clinical & Biomechanical Interpretation |
| :---: | :--- | :---: | :--- |
| **1** | `norm_y_HipLeft_mean` | `1.9875` | Mean Normalized HipLeft Position along Y-Axis (SpineBase-centered, torso-scaled pelvic geometry) |
| **2** | `sagittal_knee_flexion_left_ang_vel_mean` | `0.6932` | Sagittal Left Knee Flexion Angular Velocity (Mean) — Lower-limb sagittal flexion dynamics |
| **3** | `cross_dist_left_wrist_right_ankle_mean` | `0.6719` | Mean Contralateral Cross-Distance (Left Wrist to Right Ankle) — Distal diagonal inter-limb coordination |
| **4** | `elbow_right_rel_height_mean` | `0.6686` | Mean Right Elbow Vertical Elevation relative to SpineBase — Forearm flexion/guarding posture |
| **5** | `wrist_right_rel_height_mean` | `0.5784` | Mean Right Wrist Vertical Elevation relative to SpineBase — Upper-limb guarding posture |
| **6** | `norm_y_WristRight_std` | `0.5291` | Std Normalized WristRight Position along Y-Axis — Vertical wrist oscillation variability |
| **7** | `norm_z_ElbowLeft_mean` | `0.5219` | Mean Normalized ElbowLeft Position along Z-Axis — Sagittal upper-limb retraction posture |
| **8** | `step_dominant_rhythm_ratio` | `0.3560` | Dominant Step Rhythm Ratio — Harmonic concentration of the fundamental stride frequency |
| **9** | `ankle_right_vel_z_std` | `0.3386` | Right Ankle 1st-Order Sagittal Velocity (Std) — Distal swing progression variability |
| **10** | `sagittal_knee_flexion_right_mean` | `0.3164` | Sagittal Right Knee Flexion Angle (Mean) — Lower-limb sagittal flexion dynamics |
| **11** | `cross_dist_right_wrist_left_ankle_mean` | `0.3162` | Mean Contralateral Cross-Distance (Right Wrist to Left Ankle) — Distal diagonal inter-limb coordination |
| **12** | `cross_dist_left_elbow_right_foot_mean` | `0.2618` | Mean Contralateral Cross-Distance (Left Elbow to Right Foot) — Diagonal inter-limb coordination |
| **13** | `arm_swing_sagittal_asymmetry` | `0.2464` | Sagittal Arm Swing Asymmetry Index — Anterior-posterior reciprocal arm swing imbalance |
| **14** | `shoulder_angle_right_mean` | `0.1858` | Mean Right Shoulder Abduction/Flexion Angle (deg) — Upper-arm postural abduction |
| **15** | `step_spectral_irregularity` | `0.1857` | Step Spectral Entropy — Frequency-domain irregularity of lower-limb gait periodicity |

---

## 5. Repository Structure

```text
capstone-main/
├── data/
│   └── processed_features.csv       # Cached 700-trial × 86-feature leak-free matrix + label + subject_id
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py        # Traverses 700 raw Kinect folders, maps Child_01..Child_100, root-centers & torso-scales
│   ├── feature_extraction.py        # Extracts 86 3D kinematic biomarkers & caches data/processed_features.csv
│   ├── train.py                     # 5-fold StratifiedGroupKFold benchmark (CatBoost vs RF vs XGBoost)
│   └── explainability.py            # TreeSHAP Beeswarm plot, top-15 biomarker CSV, and cross-validated ROC curve
├── results/
│   ├── shap_summary.png             # SHAP Beeswarm plot of top 15 motor biomarkers
│   ├── feature_importance.csv       # Top 15 motor biomarkers ranked by Mean |SHAP|
│   ├── roc_curve.png                # 5-fold StratifiedGroupKFold ROC curve (CatBoost vs RF vs XGBoost)
│   ├── benchmark_metrics.csv        # Mean +/- Std cross-validation metrics across all benchmarked models
│   └── models/
│       └── catboost_asd_model.cbm   # Serialized tuned CatBoost classifier
├── test_model.py                    # Standalone CLI inference & SHAP verification script
├── requirements.txt                 # Pinned reproducible Python dependencies
└── README.md                        # Methodological documentation, literature benchmark & CLI guide
```

---

## 6. Step-by-Step CLI Execution Instructions

### Step 1: Install Pinned Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2: Run 3D Coordinate Preprocessing & Normalization (700 Folders $\to$ Normalized 3D Skeletons)
By default, `src/data_preprocessing.py` reads the 700 coordinate folders from `ASD_RAW_DATA_DIR` (or `--raw-dir`):
```bash
python -m src.data_preprocessing --raw-dir "C:\Users\thats\Downloads\Dataset-2\Dataset"
```

### Step 3: Extract 3D Kinematic Features & Cache to `data/processed_features.csv`
```bash
python -m src.feature_extraction --raw-dir "C:\Users\thats\Downloads\Dataset-2\Dataset" --output data/processed_features.csv --force
```

### Step 4: Execute Leak-Free 5-Fold `StratifiedGroupKFold` Training & Benchmarking
```bash
python -m src.train --data data/processed_features.csv --splits 5
```

### Step 5: Generate SHAP Explainability & Cross-Validated ROC Artifacts
Generates `results/shap_summary.png`, `results/feature_importance.csv`, and `results/roc_curve.png`:
```bash
python -m src.explainability --data data/processed_features.csv
```

### Step 6: Run Standalone Model Verification & Biomarker Inspection
```bash
python test_model.py --input data/processed_features.csv --explain
```
