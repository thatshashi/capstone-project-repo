# Seven-Pass Quality Control and Manuscript Revision Log

This document records the systematic audit findings, corrective actions, and verification checks executed across all **Seven Quality Control Passes** mandated by Part XI of the Master Prompt.

---

## Pass 1 — Research Foundation Audit
- **Scope**: Literature coverage, dataset provenance (`Category A/B/C/D`), and novelty claims.
- **Findings**:
  1. Verified focal scoping review `10.3390/bioengineering12111136` (Rakotomanana & Rouhafzay, 2025, *Bioengineering*, 12(11), 1136) [1].
  2. Identified primary dataset provenance as Al-Jubouri (AbdulRahman), Hadi, & Rajihy (2020/2021) (`DOI: 10.5061/dryad.s7h44j150` and `DOI: 10.1088/1742-6596/1818/1/012201`) [2, 3].
  3. Identified recent 2024–2026 Category A studies on the identical dataset: Yazdi et al. (2024) [4], Fırat (2026) [5], Le-Thi-Thu et al. (2026) [6], and Amirbay et al. (2025) [7].
- **Revisions Applied**:
  - Built a 23-row comparative literature matrix (`02_literature_review/literature_matrix.xlsx`) and refined our novelty claims to emphasize what no prior study on this dataset had done: uncovering the 12 cross-folder clones ($88$ unique subjects / $87$ metadata clusters), proving `Aug 1/2/3/5` translation-invariance, eliminating the $+16.6\text{ cm}$ stature confound via per-frame 3D torso normalization, and benchmarking Ordered `CatBoost` across a 4-regime leakage hierarchy.

---

## Pass 2 — Methodological Audit
- **Scope**: Preprocessing, normalization, feature extraction, grouping, and leakage prevention.
- **Findings**:
  1. Inspected `Dataset-2/Dataset/readme.docx` and discovered that the ASD cohort is $1.96\text{ years}$ older ($p = 0.0004$) and $16.6\text{ cm}$ taller ($1.324\text{ m}$ vs. $1.158\text{ m}, p < 0.0001$) than the TD cohort.
  2. Verified that `src/data_preprocessing.py` performs frame-by-frame `SpineBase` root-centering and 3D `SpineShoulder–SpineBase` torso-height scaling, reducing trunk length to $1.000 \pm 0.000\text{ TU}$ across both groups.
  3. Verified that `StandardScaler` and `VarianceThreshold(1e-5)` are encapsulated inside `sklearn.pipeline.Pipeline` objects fit strictly inside each training fold.
- **Revisions Applied**:
  - Added formal mathematical derivations of root-centering, torso scaling, and `Aug 1/2/3/5` translation-invariance to Section 4, Section 5, and `03_methodology/methodology_documentation.md`.

---

## Pass 3 — Experimental Audit
- **Scope**: Multi-regime benchmark execution, baseline comparisons, historical metric reconciliation, and statistical significance testing.
- **Findings**:
  1. Extracted 86 normalized features for all $100$ unaugmented original `video/8.xlsx` trials (`original_unaugmented_100_trials.csv`).
  2. Executed 10 classifiers across 6 validation regimes (`master_multi_regime_benchmark.csv`).
  3. Reconciled all four historical preliminary metrics (`5-fold SGKF ROC-AUC: 0.919 ± 0.072`, `Balanced accuracy: ~80.3%`, `Holdout AUC: 1.000, N=22`, `Original-only AUC: 0.975`) via `historical_87cluster_holdout22_audit.csv` and `normalized86_87cluster_holdout22_audit.csv`, proving that $N = 22$ is a $25\%$ cluster holdout of the $87$ metadata clusters whose $\text{AUC}$ spans $0.875\text{--}1.000$ across random seeds.
- **Revisions Applied**:
  - Added Table 4 (10-model `Regime 2` benchmark with 2,000-iteration cluster bootstrap 95% CIs and exact McNemar $p$-values), Table 5 (6-regime comparison matrix), and Table 6 (forensic reconciliation of historical metrics).

---

## Pass 4 — Manuscript Development & Visual Audit
- **Scope**: Scientific prose, logical structure, equations, tables, and 300-DPI figure inspection.
- **Findings**:
  1. Visual inspection of `fig02_kinect_skeleton_normalization.png` revealed minor Z-axis tick crowding in Panel (a) and annotation proximity to the error bar in Panel (c).
  2. Verified confusion matrix cell counts in `fig06_confusion_matrices.png` against `master_multi_regime_benchmark.csv` (`R3B`: $TN=37, FP=4, FN=6, TP=41$; `R4B`: $TN=35, FP=6, FN=3, TP=44$).
- **Revisions Applied**:
  - Regenerated `fig02_kinect_skeleton_normalization.png` and `fig06_confusion_matrices.png` at 300 DPI and verified visual clarity via `read_file`.

---

## Pass 5 — Reference and Originality Audit
- **Scope**: Verification of all 26 citations, DOI resolution, and zero-plagiarism similarity check.
- **Findings**:
  - Confirmed 100% bidirectional match between in-text citations `[1]`–`[26]` and the Reference list / `verified_references.bib`.
- **Revisions Applied**:
  - Documented itemized verification in `07_quality_audit/reference_verification_report.md` and originality/AI transparency log in `07_quality_audit/similarity_risk_audit.md`.

---

## Pass 6 — Simulated Critical Peer Review
- **Scope**: Independent review by 3 domain reviewers + Academic Editor.
- **Findings & Revisions Applied**:
  - Addressed Reviewer 1's comment on comparing `CatBoost (Tuned)` (`depth=5`) vs. `CatBoost (Hist-300)` (`depth=4`) and `SVM-RBF` across `R2`, `R3B`, and `R4B`.
  - Addressed Reviewer 2's comment on sex imbalance ($84\%$ male in ASD vs. $52\%$ in TD), age distribution, exclusion of the 9 video-only severe ASD cases, and clinical specificity vs. DCD/ADHD.
  - Addressed Reviewer 3's comment on cluster bootstrap confidence intervals and the mathematical mechanism of cluster-size imbalance in `R3A` vs. `R3B`.

---

## Pass 7 — Final Publication Audit & Deliverable Compilation
- **Scope**: Compilation of `final_manuscript.docx` and `final_manuscript.pdf`, directory completeness check, and final publication checklist.
- **Status**: **Completed and Verified** (`07_quality_audit/final_publication_checklist.md`).
