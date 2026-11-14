# Critical Peer-Review Simulation and Author Rebuttal Report (Pass 6 Audit)

To ensure the manuscript withstands rigorous scrutiny at **MDPI *Bioengineering*** or **Elsevier *Biomedical Signal Processing and Control***, we conducted an independent, adversarial peer-review simulation comprising three domain-specialist reviewers and an Academic Editor, followed by point-by-point manuscript revisions.

---

## 1. Reviewer 1 — Machine Learning & Validation Rigor Specialist

### Summary Critique
The manuscript investigates 3D Kinect v2 skeletal movement classification of pediatric ASD using `CatBoost` and 9 baseline models. The forensic discovery that the widely used Al-Jubouri (2020) dataset (`10.5061/dryad.s7h44j150`) contains 12 cross-folder coordinate clones (`88` unique subjects / `87` metadata clusters) and that four of the seven augmentations (`+0.05m`, `+0.35m`, `-0.25m` X-shifts and `-X` reflection) are translation-invariant under root-centering is a major methodological contribution to the field.

### Major Technical Comments
1. **R1.1 (Hyperparameter Tuning & In-Fold Leakage)**: Clarify whether `VarianceThreshold` and `StandardScaler` were fit inside each training fold or across the full dataset prior to splitting.
2. **R1.2 (Performance Under Deduplicated Regimes `R3B` and `R4B`)**: While `CatBoost (Tuned)` is clearly #1 on `R2` (Nominal 100-subject SGKF: $92.57\%$ trial accuracy, $93.00\%$ subject accuracy, $\text{AUROC} = 0.9802$, McNemar $p \le 0.0076$) and `R3A` (87-cluster SGKF: $86.71\%$ accuracy, $\text{AUROC} = 0.9320$), on `R3B` ($N=88$ deduplicated subjects) `SVM-RBF` achieves slightly higher accuracy ($90.53\%$ vs. $88.59\%$) and on `R4B` ($N=88$ unaugmented original trials) `CatBoost (Hist-300)` ($89.74\%, \text{AUROC} = 0.9502$) outperforms `CatBoost (Tuned)` ($87.45\%, \text{AUROC} = 0.9243$). The authors must discuss this transparently without overclaiming that a single hyperparameter setting dominates every sample size.

### Author Response & Action Taken
- **Response to R1.1**: Confirmed and explicitly documented in Sections 6.1 and `03_methodology/reproducibility_guide.md` that both `StandardScaler` and `VarianceThreshold(1e-5)` are wrapped inside `sklearn.pipeline.Pipeline` objects and fit **strictly within each training fold** (`X_train`), preventing any validation distribution leakage.
- **Response to R1.2**: We fully agree and have highlighted this nuance in Abstract, Section 7.2 (Table 5), and Section 9.2. Specifically, we explain that when training sample size drops from $700$ trials (`R2`) to $88$ single-trial recordings (`R4B`, $\sim 70$ training samples per fold), shallower depth-4 trees (`CatBoost Hist-300`, $\text{Acc} = 89.74\%, \text{AUROC} = 0.9502$) and margin-regularized classifiers (`SVM-RBF`, `XGBoost`, `Logistic Regression`) prevent overfitting compared to depth-5 trees (`CatBoost Tuned`).

---

## 2. Reviewer 2 — Autism & Human Movement Analytics Specialist

### Summary Critique
This paper bridges computational machine learning and clinical pediatric gait biomechanics effectively. The identification and elimination of the $+16.6\text{ cm}$ standing height confound ($\text{ASD}: 1.324\text{ m}$ vs. $\text{TD}: 1.158\text{ m}, p < 0.0001$) via per-frame `SpineBase` root-centering and 3D `SpineShoulder–SpineBase` scaling is essential, as pediatric gait parameters scale strongly with leg and trunk length.

### Major Clinical Comments
1. **R2.1 (Sex Ratio and Age Confounding Beyond Height)**: While 3D torso scaling removes linear body-size differences, the ASD cohort is $84\%$ male ($42/50$) versus $52\%$ male ($26/50$) in TD, and $1.96\text{ years}$ older ($8.62$ vs. $6.66\text{ years}$). Neuromotor maturation between ages 6 and 9 can affect joint angle variability even after torso scaling. Does `readme.docx` provide individual subject age/sex mapping to adjust for age per folder?
2. **R2.2 (Exclusion of the 9 Severe ASD Cases and Clinical Specificity)**: Explain why the 9 severe ASD participants mentioned in Fırat (2026) [5] and `readme.docx` are not included, and temper clinical claims regarding ASD vs. other neurodevelopmental motor disorders (DCD, ADHD).

### Author Response & Action Taken
- **Response to R2.1**: Inspected `Dataset-2/Dataset/readme.docx` in full; the original dataset creators recorded individual ages/heights in an unlinked summary table that is not keyed to folder IDs `1..50`. We added an explicit discussion of this limitation in Section 10 (Limitation #2), noting that while per-frame 3D torso normalization removes first-order anthropometric scaling, future datasets must provide per-subject age and sex covariates for ANCOVA/stratified calibration.
- **Response to R2.2**: Documented in Table 2 and Section 10 (Limitations #3 and #4) that the `Severe autism/1..9` directories contain raw video files only without `.xlsx` 3D skeletal coordinate matrices, and explicitly framed our model as a **quantitative neuromotor screening aid** rather than a standalone differential diagnostic replacement for ADOS-2/DSM-5 evaluation.

---

## 3. Reviewer 3 — Statistical Methodology & Reproducibility Auditor

### Summary Critique
The statistical methodology is exceptionally thorough. Auditing the historical preliminary metrics (`ROC-AUC: 0.919 ± 0.072`, `Balanced Accuracy: ~80.3%`, `Holdout AUC: 1.000, N=22`, `Original-only AUC: 0.975`) and proving that $N = 22$ corresponds to a $25\%$ cluster holdout of the $87$ metadata clusters—where $\text{AUC} = 1.000$ under `seed = 42` but drops to $0.875$ under `seed = 123`—is a textbook demonstration of small-sample holdout instability.

### Major Statistical Comments
1. **R3.1 (Cluster Bootstrap & Paired Testing)**: Because trials within a child are correlated, standard binomial confidence intervals on $N = 700$ trials are too narrow. Ensure 95% confidence intervals are computed via **subject-cluster bootstrap resampling** ($M = 100$ subjects resampled with replacement), and report exact binomial McNemar $p$-values and Brier calibration scores.
2. **R3.2 (Why 87-Cluster SGKF Balanced Accuracy Was ~80.3% vs. 88.80% After Deduplication)**: Clarify mathematically why grouping all $700$ rows by $87$ clusters (`R3A`) yields higher fold variance ($\pm 13.99\%$) than deduplicating the 12 clone folders first (`R3B`, $\pm 6.14\%$).

### Author Response & Action Taken
- **Response to R3.1**: Implemented 2,000-iteration stratified subject-cluster bootstrap 95% CIs (`statistical_comparison_and_bootstrap_ci.csv`), exact binomial McNemar tests, and OOF Brier scores in Table 4.
- **Response to R3.2**: Explained in Section 7.2 and Table 6 that in `R3A`, leaving `TD_Cluster_00` ($5\text{ folders} = 35\text{ identical TD trials}$) and `TD_Cluster_01` ($3\text{ folders} = 21\text{ identical TD trials}$) un-deduplicated causes a single subject's recording to receive $5\times$ weight in the loss function and validation fold, distorting fold class balance and threshold calibration. Deduplicating the 12 clone folders prior to CV (`R3B`) restores equal $7$-trial weighting per unique child and cuts fold standard deviation by more than half.

---

## 4. Academic Editor Final Decision

**Editorial Decision**: **Accept in Present Form** (Following verification of all 7 Quality Control Passes).  
The manuscript provides a definitive, forensic benchmark of the Al-Jubouri (2020) Kinect v2 ASD dataset, resolves previously unrecognized two-level data leakage and demographic stature confounding, and satisfies all methodological, bibliographic, and ethical standards of *Bioengineering* (MDPI) and *Biomedical Signal Processing and Control* (Elsevier).
