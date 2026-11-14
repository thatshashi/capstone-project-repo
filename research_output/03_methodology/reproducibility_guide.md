# Reproducibility Guide and Execution Instructions

## 1. Environment and Dependency Specifications

All data preprocessing, kinematic feature extraction, multi-regime cross-validation benchmarks, statistical significance tests, and TreeSHAP explainability analyses were executed deterministically under **Windows 10/11 (x86_64)** using **Python 3.12+** inside a project-isolated virtual environment (`.venv`).

### 1.1 Exact Pinned Package Versions (`requirements.txt`)
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `numpy` | `2.4.2` | Vectorized 3D skeletal tensor algebra, root-centering, torso scaling, and bootstrap resampling |
| `pandas` | `2.3.3` | Tabular feature serialization, cross-validation metric aggregation, and Excel/CSV I/O |
| `scipy` | `1.17.0` | Mann–Whitney $U$ non-parametric tests, exact binomial McNemar tests, and statistical distributions |
| `scikit-learn` | `1.8.0` | `StratifiedGroupKFold`, `StratifiedKFold`, baseline classifiers, `StandardScaler`, and evaluation metrics |
| `catboost` | `1.2.10` | Ordered `CatBoostClassifier` training and native exact `Pool` / `TreeSHAP` computation |
| `xgboost` | `3.2.0` | `XGBClassifier` gradient boosted decision tree baseline |
| `shap` | `0.51.0` | Exact TreeSHAP local and global feature attribution |
| `matplotlib` | `3.10.8` | 300-DPI publication figure rendering |
| `seaborn` | `0.13.2` | Statistical heatmaps, confusion matrices, and violin/distribution plots |
| `openpyxl` | `3.1.5` | Parsing raw Kinect v2 `.xlsx` coordinate spreadsheets and generating formatted Excel workbooks |
| `joblib` | `1.5.3` | Model serialization |

---

## 2. Deterministic Seeds and Partitioning Guarantees

Every stochastic component in the repository is pinned to `RANDOM_SEED = 42`:
- **Cross-Validation Splitters**: `StratifiedGroupKFold(n_splits=5)` and `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`.
- **Classifier Initialization**: `random_seed=42` (`CatBoostClassifier`) and `random_state=42` (`XGBClassifier`, `HistGradientBoostingClassifier`, `RandomForestClassifier`, `AdaBoostClassifier`, `SVC`, `LogisticRegression`, `DecisionTreeClassifier`).
- **Cluster Bootstrap Confidence Intervals**: `np.random.RandomState(42)` with $B = 2,000$ stratified subject-cluster resamples.
- **Zero In-Fold Preprocessing Leakage**: All feature scaling (`StandardScaler` for SVM and Logistic Regression) and variance filtering (`VarianceThreshold(1e-5)` for `CatBoost Hist-300`) are encapsulated inside `sklearn.pipeline.Pipeline` objects so mean/variance parameters are fit **exclusively on each training fold** (`X_train`) and never exposed to validation folds (`X_val`).

---

## 3. Step-by-Step Reproduction Commands

From the repository root (`C:\Users\thats\Downloads\capstone-main (1)\capstone-main`), activate the virtual environment or invoke `.\.venv\Scripts\python.exe` directly:

### Step 1: Preprocess Raw Kinect v2 Folders and Extract 86 Normalized 3D Kinematic Features
```powershell
.\.venv\Scripts\python.exe -m src.feature_extraction --raw-dir "C:\Users\thats\Downloads\Dataset-2\Dataset" --output "data/processed_features.csv"
```
- **Input**: 700 raw `.xlsx` coordinate files across `Typical/1..50` and `Autism/children with ASD/1..50`.
- **Output**: `data/processed_features.csv` (`700 rows × 88 columns`: `subject_id`, `label`, and 86 normalized 3D kinematic features).

### Step 2: Run the Primary Leak-Free 5-Fold `StratifiedGroupKFold` Benchmark & Generate Figures
```powershell
.\.venv\Scripts\python.exe -m src.train
```
- **Outputs**:
  - `results/benchmark_metrics.csv` and `results/metrics/evaluation_metrics.csv`
  - `results/metrics/unseen_test_predictions.csv` (Out-of-fold trial and subject predictions)
  - `results/models/catboost_asd_model.cbm` and `results/models/model_metadata.json`

### Step 3: Run Exact TreeSHAP Explainability & Clinical Mann–Whitney $U$ / Cohen's $d$ Analysis
```powershell
.\.venv\Scripts\python.exe -m src.explainability
```
- **Outputs**:
  - `results/feature_importance.csv` (Full 86-feature ranked table with mean $|\text{SHAP}|$, class means/SDs, Cohen's $d$, and two-sided Mann–Whitney $U$ $p$-values)
  - `results/metrics/top_features_clinical_report.csv` (Top 15 motor biomarkers mapped to clinical ASD biomechanics)
  - `results/figures/fig6_shap_beeswarm.png` and `results/figures/fig7_shap_bar_cohens_d.png`

### Step 4: Run the Automated Anti-Leakage & Reproducibility Verification Suite
```powershell
.\.venv\Scripts\python.exe test_model.py
```
- Verifies that `SpineBase` root-centering equals `(0, 0, 0)`, torso length equals `1.0`, zero `subject_id` overlap occurs across any train/test fold, and all saved metrics and figures match deterministic expectations.

---

## 4. Directory Map of Preserved Originals vs. Newly Generated Deliverables

- **Preserved Original Repository Files (Untouched)**:
  - `Final dataset.xlsx`, `Final dataset (1).xlsx`, `data/unseen_test_samples.xlsx`
  - `data/processed_features.csv`
  - `src/data_preprocessing.py`, `src/feature_extraction.py`, `src/train.py`, `src/explainability.py`
  - `results/` (copied into `research_output/04_experiments/verified_existing_results/`)
- **Newly Generated Research Deliverables (`research_output/`)**:
  - `research_output/01_manuscript/`: `editable_source.md`, `final_manuscript.docx`, `final_manuscript.pdf`
  - `research_output/02_literature_review/`: `literature_matrix.xlsx`, `citation_network_analysis.xlsx`, `verified_references.bib`
  - `research_output/03_methodology/`: `methodology_documentation.md`, `reproducibility_guide.md`, `experimental_protocol.md`
  - `research_output/04_experiments/new_experiments/`:
    - `original_unaugmented_100_trials.csv` (86 normalized features extracted from the 100 original `video/8.xlsx` recordings)
    - `master_multi_regime_benchmark.csv` (10 models $\times$ 6 validation regimes = 60 full evaluation rows)
    - `statistical_comparison_and_bootstrap_ci.csv` (2,000-iteration cluster bootstrap 95% CIs and exact McNemar $p$-values)
    - `historical_87cluster_holdout22_audit.csv` and `normalized86_87cluster_holdout22_audit.csv` (8-seed audit of the $N=22$ cluster holdout)
    - `oof_predictions_regime2_100subjects.csv`
  - `research_output/05_figures/`: `fig01` through `fig10` (all 300-DPI publication figures)
  - `research_output/06_journal_selection/`: `journal_comparison.xlsx`, `target_journal_guidelines.md`
  - `research_output/07_quality_audit/`: `reference_verification_report.md`, `similarity_risk_audit.md`, `peer_review_report.md`, `revision_log.md`, `final_publication_checklist.md`
