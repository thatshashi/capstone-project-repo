# Comprehensive Methodology Documentation

## 1. Raw Dataset Architecture and Forensic Provenance Audit

### 1.1 Primary Data Source
The experimental cohort originates from the publicly archived three-dimensional (3D) pediatric gait and full-body movement dataset collected by Al-Jubouri (AbdulRahman), Hadi (Ali), and Rajihy (2020) [2, 3] using a markerless Microsoft Kinect v2 depth and infrared time-of-flight camera (`30 fps`, 25 skeletal joints in 3D Cartesian space $[X, Y, Z]$ measured in meters) and hosted on the Dryad Digital Repository (`DOI: 10.5061/dryad.s7h44j150`).

The raw directory tree (`Dataset-2/Dataset/`) comprises two primary diagnostic cohorts:
- **Typically Developing (TD) Controls**: 50 nominal subject folders (`Typical/1` through `Typical/50`), assigned canonical IDs `Child_01` through `Child_50` ($y_i = 0$).
- **Autism Spectrum Disorder (ASD) Cohort**: 50 nominal walking subject folders (`Autism/children with ASD/1` through `50`), assigned canonical IDs `Child_51` through `Child_100` ($y_i = 1$). An additional 9 severe ASD cases (`Severe autism/1..9`) contain raw video files without 3D skeletal coordinate `.xlsx` spreadsheets and are therefore excluded from quantitative 3D skeletal modeling.

Within each of the 100 nominal subject folders:
- `video/8.xlsx` stores the **original unaugmented 3D Kinect v2 recording** (corresponding to row $8k + 0$ in the pre-aggregated `Final dataset.xlsx` spreadsheet).
- `augmentation/1/1.xlsx` through `augmentation/7/7.xlsx` store **7 synthetic coordinate-augmented trials** (corresponding to rows $8k + 1$ through $8k + 7$ in `Final dataset.xlsx`).

### 1.2 Demographic Confound Analysis
Inspection of the official cohort metadata (`Dataset/readme.docx`) revealed a statistically and biomechanically critical demographic confound between the two groups:
- **TD Cohort ($N=50$ nominal)**: 26 males (52.0%), 24 females (48.0%); mean age $6.66 \pm 3.08\text{ years}$ (range $2\text{--}14\text{ years}$); mean recorded height $3.75 \pm 0.67\text{ ft}$ ($1.158 \pm 0.195\text{ m}$).
- **ASD Cohort ($N=50$ nominal)**: 42 males (84.0%), 8 females (16.0%); mean age $8.62 \pm 2.19\text{ years}$ (range $5\text{--}15\text{ years}$); mean recorded height $1.324 \pm 0.154\text{ m}$ (range $0.976\text{--}1.707\text{ m}$).

Because the ASD participants are on average **$1.96\text{ years}$ older ($t = 3.67, p = 0.0004$) and $16.6\text{ cm}$ taller ($t = 4.72, p < 0.0001$)** than the TD controls, any classifier trained directly on unnormalized meter-scale coordinates (`Final dataset.xlsx`, 1,259 raw features) can trivially separate ASD from TD by measuring absolute joint height ($Y$) or limb bone length rather than motor pathology.

---

## 2. Frame-by-Frame 3D Anthropometric Skeleton Normalization

To mathematically eliminate both camera position offset and the $16.6\text{ cm}$ inter-group stature confound, every raw 3D skeletal trajectory $\mathbf{P}_i \in \mathbb{R}^{T_i \times 25 \times 3}$ is normalized frame-by-frame prior to feature extraction (`src/data_preprocessing.py`):

### Step 1: Pelvis Root-Centering (`SpineBase`, Joint $j = 0$)
At each timestamp frame $t \in \{1, \dots, T_i\}$, the 3D position of `SpineBase` ($\mathbf{P}_i(t, 0) \in \mathbb{R}^3$) is subtracted from all 25 joints:
$$\mathbf{P}_i^{\text{centered}}(t, j) = \mathbf{P}_i(t, j) - \mathbf{P}_i(t, 0), \quad \forall j \in \{0, \dots, 24\}$$
This places the anatomical pelvis root at the exact origin $(0, 0, 0)$ at every frame, making the representation strictly invariant to global 3D translation $(+\Delta X, +\Delta Y, +\Delta Z)$ relative to the Kinect v2 sensor.

### Step 2: 3D Trunk-Height Scaling (`SpineShoulder` to `SpineBase`)
Next, we compute the instantaneous 3D Euclidean trunk length $L_i(t)$ between `SpineShoulder` ($j = 1$) and `SpineBase` ($j = 0$):
$$L_i(t) = \left\| \mathbf{P}_i^{\text{centered}}(t, 1) - \mathbf{P}_i^{\text{centered}}(t, 0) \right\|_2 = \left\| \mathbf{P}_i(t, 1) - \mathbf{P}_i(t, 0) \right\|_2$$
To guard against transient depth-sensor drop-outs, any frame with $L_i(t) \le \epsilon$ ($\epsilon = 10^{-6}\text{ m}$) substitutes the trial's median valid trunk length $\tilde{L}_i$. All 25 centered joint coordinates are then divided by $L_i(t)$:
$$\tilde{\mathbf{P}}_i(t, j) = \frac{\mathbf{P}_i(t, j) - \mathbf{P}_i(t, 0)}{L_i(t)}$$
In this normalized coordinate space (measured in dimensionless **Torso Units [TU]**), $\|\tilde{\mathbf{P}}_i(t, 1)\|_2 = 1.000 \pm 0.000$ across both ASD and TD cohorts, completely removing the stature confound.

---

## 3. Forensic Discovery of Two-Level Data Leakage

### 3.1 Level-1 Leakage: Translation- and Reflection-Invariance of Augmentations 1, 2, 3, and 5
By comparing the raw coordinate matrices of `augmentation/1..7` against the original camera recording `video/8.xlsx` across all 100 folders, we established the exact mathematical formulas used by Al-Jubouri et al. [2, 3] to generate the 7 augmentations:
1. **Augmentation 1 (`1.xlsx`)**: Pure $+0.05\text{ m}$ shift along the mediolateral $X$-axis ($X' = X + 0.05, Y' = Y, Z' = Z$).
2. **Augmentation 2 (`2.xlsx`)**: Pure $+0.35\text{ m}$ shift along the $X$-axis ($X' = X + 0.35$).
3. **Augmentation 3 (`3.xlsx`)**: Pure $-0.25\text{ m}$ shift along the $X$-axis ($X' = X - 0.25$).
4. **Augmentation 4 (`4.xlsx`)**: 3D camera pitch/perspective transformation with additive coordinate perturbation.
5. **Augmentation 5 (`5.xlsx`)**: Pure horizontal mirror reflection across the sagittal plane ($X' = -X, Y' = Y, Z' = Z$).
6. **Augmentation 6 (`6.xlsx`)**: Anisotropic mediolateral compression ($X' = 0.7 X, Y' = Y, Z' = Z$).
7. **Augmentation 7 (`7.xlsx`)**: Anisotropic mediolateral expansion ($X' = 1.6 X, Y' = Y, Z' = Z$).

**Mathematical Consequence**: Under `SpineBase` root-centering, any global translation $\mathbf{P}(t, j) + \mathbf{c}$ cancels identically:
$$(\mathbf{P}(t, j) + \mathbf{c}) - (\mathbf{P}(t, 0) + \mathbf{c}) = \mathbf{P}(t, j) - \mathbf{P}(t, 0)$$
Consequently, **Augmentations 1, 2, and 3 produce feature vectors that are identical to the original trial (`video/8.xlsx`) up to IEEE-754 floating-point precision ($\max |\Delta| \le 1.92 \times 10^{-13}$)**, and **Augmentation 5 ($-X$ reflection)** produces identical values across all bilateral/magnitude features! Therefore, ungrouped trial-level cross-validation (`Regime 1`) places up to 4 bit-for-bit identical copies of the same child's trial into both training and test folds.

### 3.2 Level-2 Leakage: Discovery of 12 Cross-Folder Duplicate Recordings (88 Unique Clusters / 87 Metadata Clusters)
Pairwise correlation auditing across all 100 nominal folders (`Child_01` through `Child_100`) revealed that **12 subject folders are exact coordinate clones ($r = 1.0000$) of 7 other folders within the same diagnostic class**, reducing the 100 nominal folders to **88 strictly unique coordinate clusters** ($47\text{ unique ASD} + 41\text{ unique TD}$):
- **TD Clone Clusters (9 duplicate folders across 4 primary subjects -> 41 unique TD subjects)**:
  - `Cluster_00` (5 folders): `Child_01` (`Typical/1`), `Child_07` (`Typical/7`), `Child_16` (`Typical/16`), `Child_20` (`Typical/20`), `Child_29` (`Typical/29`).
  - `Cluster_01` (3 folders): `Child_02` (`Typical/2`), `Child_26` (`Typical/26`), `Child_40` (`Typical/40`).
  - `Cluster_03` (3 folders): `Child_04` (`Typical/4`), `Child_23` (`Typical/23`), `Child_28` (`Typical/28`).
  - `Cluster_12` (2 folders): `Child_13` (`Typical/13`), `Child_14` (`Typical/14`).
- **ASD Clone Clusters (3 duplicate folders across 3 primary subjects -> 47 unique ASD subjects)**:
  - `Cluster_50` (2 folders): `Child_51` (`ASD Case 1`), `Child_84` (`ASD Case 34`).
  - `Cluster_56` (2 folders): `Child_57` (`ASD Case 7`), `Child_68` (`ASD Case 18`).
  - `Cluster_79` (2 folders): `Child_80` (`ASD Case 30`), `Child_90` (`ASD Case 40`).
- **87 Metadata Clusters in `Final dataset.xlsx`**: In the pre-extracted `Final dataset.xlsx`, two genuinely distinct ASD recordings (`Child_75` [Block 24] and `Child_93` [Block 42]) share the same 5-decimal rounded `mean-x-Midspain = -0.09342`, collapsing the 88 coordinate clusters into **87 metadata clusters** ($46\text{ ASD} + 41\text{ TD}$).

---

## 4. Biomechanical Feature Engineering (86 Normalized 3D Features)

From each normalized sequence $\tilde{\mathbf{P}}_i \in \mathbb{R}^{T_i \times 25 \times 3}$, `src/feature_extraction.py` extracts $d = 86$ clinically interpretable kinematic features across five biomechanical domains:
1. **3D Joint Angles (20 features — Temporal Mean & SD across 10 anatomical angles)**:
   $$\theta_{abc}(t) = \arccos\!\left(\text{clip}\!\left(\frac{(\tilde{\mathbf{p}}_a(t) - \tilde{\mathbf{p}}_b(t)) \cdot (\tilde{\mathbf{p}}_c(t) - \tilde{\mathbf{p}}_b(t))}{\|\tilde{\mathbf{p}}_a(t) - \tilde{\mathbf{p}}_b(t)\|_2 \, \|\tilde{\mathbf{p}}_c(t) - \tilde{\mathbf{p}}_b(t)\|_2}, -1, 1\right)\right) \cdot \frac{180^\circ}{\pi}$$
   Computed for Left/Right Knee (`Hip–Knee–Ankle`), Left/Right Hip (`SpineBase–Hip–Knee`), Left/Right Elbow (`Shoulder–Elbow–Wrist`), Left/Right Shoulder (`SpineShoulder–Shoulder–Elbow`), Neck Flexion (`SpineShoulder–Neck–Head`), and Trunk Alignment (`SpineBase–Midspain–SpineShoulder`).
2. **Joint Velocity & Acceleration Dynamics (26 features)**:
   Instantaneous 3D speed $v_j(t) = \|\tilde{\mathbf{P}}(t, j) - \tilde{\mathbf{P}}(t-1, j)\|_2 / \Delta t$ and acceleration magnitude $a_j(t) = |v_j(t) - v_j(t-1)| / \Delta t$ summarized via temporal mean, SD, and 95th/max peak across 13 key joints.
3. **Postural Sway, Balance & Step Width (12 features)**:
   Includes 3D and mediolateral (`X`)/anteroposterior (`Z`) `Head`–`SpineBase` sway (`posture_head_spine_sway_mean/std`), dynamic 3D ankle separation (`step_width_3d_mean/std`), and sagittal/coronal trunk inclination angles.
4. **Bilateral Asymmetry & Upper-Limb Stereotypy Distances (14 features)**:
   Absolute left–right differences in knee, hip, elbow, and shoulder angles, plus normalized 3D hand-to-head and hand-to-chest distances (`dist_hand_head_mean/min`, `dist_hand_spine_shoulder_mean`) quantifying upper-extremity motor stereotypies and arm-guarding postures.
5. **Spatial Excursion & Temporal Trajectory Variability (14 features)**:
   Interquartile and peak-to-peak 3D spatial excursions of distal extremities (`Wrist`, `HandTip`, `Ankle`, `Foot`) in normalized torso units.

---

## 5. Ordered CatBoost Architecture and Four-Regime Validation Hierarchy

### 5.1 Ordered Boosting with Symmetrical Oblivious Trees
Unlike depthwise gradient boosted decision trees (`XGBoost` [23], `LightGBM` [24]) that split nodes greedily and suffer from **prediction shift** (target leakage caused by estimating gradients on the same observations used to fit previous trees), **CatBoost** [19] introduces two structural regularizers ideal for small-sample biomedical datasets ($N \le 100$ subjects):
1. **Oblivious (Symmetric) Decision Trees**: Every node at depth $l \in \{1, \dots, D\}$ applies the exact same feature-threshold split $(f_l, \tau_l)$ across the entire tree, compiling into a $2^D$-entry decision table with only $D$ split parameters per tree rather than $2^D - 1$.
2. **Ordered Boosting**: Uses random permutations $\sigma$ of the training instances so residual gradients for sample $i$ are computed strictly from models trained on preceding samples $\{j : \sigma(j) < \sigma(i)\}$, preventing within-fold overfitting.

We evaluate both:
- **`CatBoost (Tuned)`**: `iterations = 500`, `depth = 5`, `learning_rate = 0.04`, `l2_leaf_reg = 4.0`, `random_seed = 42`.
- **`CatBoost (Hist-300)`**: Historical baseline configuration (`VarianceThreshold(1e-5)` inside each fold + `iterations = 300`, `depth = 4`, `learning_rate = 0.05`, `l2_leaf_reg = 3.0`, `random_seed = 42`).

### 5.2 Four-Regime Validation Hierarchy
To provide complete transparency, all 10 classifiers are evaluated under four progressively stricter 5-fold cross-validation regimes (`seed = 42`):
- **Regime 1 (`R1_Naive_Ungrouped_700`)**: 5-fold `StratifiedKFold` across $N = 700$ trials without subject grouping (demonstrates Level-1 + Level-2 leakage).
- **Regime 2 (`R2_Nominal_100Subj_SGKF_700`)**: 5-fold `StratifiedGroupKFold` grouped by the 100 nominal subject folders (`Child_01..Child_100`), plus subject-level majority voting ($N = 100$).
- **Regime 3A (`R3A_87Cluster_SGKF_700`) & Regime 3B (`R3B_Deduplicated_88Subj_SGKF_616`)**:
  - `R3A` groups all 700 trials by the 87 metadata clusters (and 88 coordinate clusters).
  - `R3B` permanently removes the 12 duplicate clone folders prior to partitioning, evaluating 5-fold `StratifiedGroupKFold` across the **88 strictly unique children** ($88 \times 7 = 616$ trials; $47\text{ ASD}, 41\text{ TD}$).
- **Regime 4A (`R4A_Original_100_GroupedBy87`) & Regime 4B (`R4B_Deduplicated_Original_88`)**:
  - Evaluates **only the original unaugmented camera recordings (`video/8.xlsx`)** with zero synthetic augmentations ($N = 100$ nominal original recordings grouped by 87 clusters in `R4A`, and $N = 88$ strictly unique children with 1 original recording per child in `R4B`).
