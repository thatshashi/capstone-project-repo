# Final Publication Audit and Deliverable Verification Checklist (Pass 7)

## 1. Scientific & Methodological Compliance Verification

| Audit Criterion | Status | Verification Evidence |
| :--- | :---: | :--- |
| **1. Raw Dataset Provenance & Demographics Verified** | **PASS** | Al-Jubouri et al. (2020) (`10.5061/dryad.s7h44j150`); demographics & $+16.6\text{ cm}$ stature confound verified from `Dataset/readme.docx` (Table 2). |
| **2. Original Data & Artifacts Preserved Untouched** | **PASS** | Original `Final dataset.xlsx`, `data/processed_features.csv`, `src/*.py`, and `results/*` preserved without modification; copies archived in `04_experiments/verified_existing_results/`. |
| **3. Two-Level Data Leakage Uncovered & Quantified** | **PASS** | Level-1 (`Aug 1, 2, 3, 5` translation/reflection invariance, $\max |\Delta| \le 1.92 \times 10^{-13}$) and Level-2 ($12$ cross-folder coordinate clones $\rightarrow 88$ unique subjects / $87$ metadata clusters) proved and benchmarked across Regimes `R1`–`R4B`. |
| **4. Historical Metrics Reconciled Without Fabrication** | **PASS** | All historical numbers (`ROC-AUC: 0.919 ± 0.072`, `BalAcc: ~80.3%`, `Holdout AUC: 1.000, N=22`, `Original-only AUC: 0.975`) replicated and explained in Table 6 and `04_experiments/new_experiments/*_holdout22_audit.csv`. |
| **5. Ten-Classifier Baseline Benchmark Executed** | **PASS** | `CatBoost (Tuned)`, `CatBoost (Hist-300)`, `XGBoost`, `LightGBM`, `Random Forest`, `AdaBoost`, `SVM-RBF`, `SVM-Linear`, `Logistic Regression`, and `Decision Tree` evaluated across all 6 regimes (`master_multi_regime_benchmark.csv`). |
| **6. Non-Parametric CIs, McNemar Tests & TreeSHAP Verified** | **PASS** | 2,000-iteration subject-cluster bootstrap 95% CIs, exact binomial McNemar $p$-values (`statistical_comparison_and_bootstrap_ci.csv`), and exact `TreeSHAP` + Mann–Whitney $U$ + Cohen's $d$ (`top_features_clinical_report.csv`). |
| **7. All 26 References Verified Against Primary Records** | **PASS** | 100% real, peer-reviewed publications with verified DOIs (`02_literature_review/verified_references.bib` and `07_quality_audit/reference_verification_report.md`). |
| **8. Target Journal Selected & Formatting Aligned** | **PASS** | *Bioengineering* (MDPI) selected as primary target journal (*Biomedical Signal Processing and Control* as #1 zero-APC alternative); evaluated across all 15 criteria (`06_journal_selection/`). |

---

## 2. Deliverable File Inventory Checklist (`research_output/`)

- [x] `research_output/01_manuscript/editable_source.md` — Complete 21-section academic manuscript source.
- [x] `research_output/01_manuscript/final_manuscript.docx` — Formatted Microsoft Word document with embedded 300-DPI figures and styled tables.
- [x] `research_output/01_manuscript/final_manuscript.pdf` — Typeset multi-page publication PDF with embedded figures, tables, and references.
- [x] `research_output/02_literature_review/literature_matrix.xlsx` — Multi-sheet Excel comparison matrix of 23 peer-reviewed studies and novelty contributions.
- [x] `research_output/02_literature_review/citation_network_analysis.xlsx` — Citation network of `10.3390/bioengineering12111136` and forensic 12-clone cluster map.
- [x] `research_output/02_literature_review/verified_references.bib` — Complete BibTeX database of all 26 verified references.
- [x] `research_output/03_methodology/methodology_documentation.md` — Mathematical derivations of 3D skeleton normalization, 86 features, and leakage proofs.
- [x] `research_output/03_methodology/reproducibility_guide.md` — Pinned dependencies, deterministic seeds (`42`), and step-by-step CLI reproduction commands.
- [x] `research_output/03_methodology/experimental_protocol.md` — Formal specification of the 6 validation regimes, 10 models, and statistical tests.
- [x] `research_output/04_experiments/verified_existing_results/` — Preserved copy of verified repository metrics, tables, and model metadata.
- [x] `research_output/04_experiments/new_experiments/` — Newly generated experimental datasets and multi-regime benchmarks (`master_multi_regime_benchmark.csv`, `original_unaugmented_100_trials.csv`, `statistical_comparison_and_bootstrap_ci.csv`, `historical_87cluster_holdout22_audit.csv`, `normalized86_87cluster_holdout22_audit.csv`, `oof_predictions_regime2_100subjects.csv`).
- [x] `research_output/04_experiments/experiment_logs/benchmark_execution_manifest.json` — Execution manifest and audit metadata.
- [x] `research_output/05_figures/` — Complete suite of 300-DPI publication figures (`fig01` through `fig10`).
- [x] `research_output/06_journal_selection/journal_comparison.xlsx` — 15-criterion comparison of 5 Scopus/SCIE-indexed journals.
- [x] `research_output/06_journal_selection/target_journal_guidelines.md` — Journal recommendation, author guideline checklist, and submission readiness report.
- [x] `research_output/07_quality_audit/reference_verification_report.md` — Itemized verification of all 26 references.
- [x] `research_output/07_quality_audit/similarity_risk_audit.md` — Originality assessment and AI-assisted research transparency log.
- [x] `research_output/07_quality_audit/peer_review_report.md` — Simulated 3-reviewer + Editor critical peer review and point-by-point rebuttal.
- [x] `research_output/07_quality_audit/revision_log.md` — Seven-pass quality control log.
- [x] `research_output/07_quality_audit/final_publication_checklist.md` — This final verification checklist.
- [x] `research_output/README.md` — Master index and guide to all deliverables in `research_output/`.
