# Autonomous Academic Research & Journal Publication Package (`research_output/`)

## Research Paper Title
**Movement-Based Classification of Autism Spectrum Disorder in Children Using CatBoost: A Methodologically Rigorous Benchmark**

- **Primary Target Journal**: *Bioengineering* (MDPI, ISSN: 2306-5354; SCIE Impact Factor: 3.8, Scopus CiteScore: 4.8)
- **Zero-APC Hybrid Alternative**: *Biomedical Signal Processing and Control* (Elsevier, ISSN: 1746-8094; SCIE Impact Factor: 4.9, Scopus CiteScore: 8.2)
- **Primary Dataset Audited**: Al-Jubouri (AbdulRahman), Hadi, & Rajihy (2020) Microsoft Kinect v2 3D Pediatric ASD Gait Dataset (`DOI: 10.5061/dryad.s7h44j150`)
- **Focal Scoping Review Investigated**: Rakotomanana & Rouhafzay (2025), *Bioengineering*, 12(11), 1136 (`DOI: 10.3390/bioengineering12111136`)

---

## Complete Directory & Deliverable Inventory

```text
research_output/
├── 01_manuscript/
│   ├── editable_source.md          # Complete 21-section academic manuscript in Markdown/LaTeX
│   ├── final_manuscript.docx       # Formatted Word manuscript with 8 embedded 300-DPI figures & 7 tables
│   └── final_manuscript.pdf        # Typeset multi-page publication PDF ready for journal submission
│
├── 02_literature_review/
│   ├── literature_matrix.xlsx      # 23-study comparison matrix (Categories A/B/C/D) & Novelty Matrix
│   ├── citation_network_analysis.xlsx # Citation network of 10.3390/bioengineering12111136 & 12-Clone Map
│   └── verified_references.bib     # Complete BibTeX database of all 26 verified peer-reviewed references
│
├── 03_methodology/
│   ├── methodology_documentation.md # Mathematical derivations of 3D normalization, 86 features & leakage proofs
│   ├── reproducibility_guide.md    # Pinned dependencies, deterministic seeds (42), and CLI reproduction commands
│   └── experimental_protocol.md    # Formal specification of the 6 validation regimes and 10 classifiers
│
├── 04_experiments/
│   ├── verified_existing_results/  # Untouched copy of original repository metrics, tables, and figures
│   ├── new_experiments/
│   │   ├── master_multi_regime_benchmark.csv             # 10 classifiers × 6 validation regimes (60 full rows)
│   │   ├── original_unaugmented_100_trials.csv           # 86 normalized features for all 100 original video/8.xlsx files
│   │   ├── statistical_comparison_and_bootstrap_ci.csv   # 2,000-iteration cluster bootstrap 95% CIs & exact McNemar p-values
│   │   ├── historical_87cluster_holdout22_audit.csv      # 8-seed audit of the N=22 cluster holdout on raw Final dataset.xlsx
│   │   ├── normalized86_87cluster_holdout22_audit.csv    # 8-seed audit of the N=22 cluster holdout on 86 normalized features
│   │   └── oof_predictions_regime2_100subjects.csv       # Out-of-fold trial probabilities & predictions across all 10 models
│   └── experiment_logs/
│       └── benchmark_execution_manifest.json             # Execution provenance, dataset dimensions, and random seed log
│
├── 05_figures/
│   ├── fig01_methodology_workflow.png          # End-to-end methodology & 4-regime validation architecture
│   ├── fig02_kinect_skeleton_normalization.png # Real 3D Kinect v2 skeleton before/after torso scaling & stature audit
│   ├── fig03_leakage_and_regime_comparison.png # Multi-regime accuracy & 10-model × 6-regime ROC-AUC heatmap
│   ├── fig04_roc_curves_multimodel.png         # 5-fold cross-validated ROC curves
│   ├── fig05_precision_recall_curves.png       # 5-fold cross-validated Precision-Recall curves
│   ├── fig06_confusion_matrices.png            # 4-panel OOF confusion matrices (R2 Trial, R2 Subj, R3B Dedup, R4B Orig)
│   ├── fig07_metrics_benchmark_bar.png         # Multi-metric bar chart across baseline classifiers
│   ├── fig08_probability_calibration.png       # Out-of-fold probability density and calibration profile
│   ├── fig09_shap_beeswarm.png                 # Exact TreeSHAP beeswarm plot of top 15 3D kinematic biomarkers
│   ├── fig10_shap_importance_cohens_d.png      # Ranked mean |SHAP| bar chart paired with Cohen's d & Mann-Whitney U
│   └── fig11_naive_vs_grouped_delta.png        # Naive vs. subject-grouped generalization delta chart
│
├── 06_journal_selection/
│   ├── journal_comparison.xlsx       # 15-criterion comparison of 5 Scopus & Web of Science (SCIE) journals
│   └── target_journal_guidelines.md  # Journal recommendation, author guideline checklist & readiness report
│
└── 07_quality_audit/
    ├── reference_verification_report.md # Itemized DOI & publisher verification of all 26 citations
    ├── similarity_risk_audit.md         # Originality audit (<4% similarity risk) & AI transparency log
    ├── peer_review_report.md            # Simulated 3-reviewer + Editor critical peer review & author rebuttal
    ├── revision_log.md                  # Documentation of all 7 Quality Control Passes
    └── final_publication_checklist.md   # Final pre-submission verification checklist
```

---

## Summary of Key Forensic and Experimental Findings

1. **Elimination of Demographic Stature Confound**:
   - The raw ASD walking cohort is $1.96\text{ years}$ older ($8.62 \pm 2.19$ vs. $6.66 \pm 3.08\text{ years}, p = 0.0004$) and $16.6\text{ cm}$ taller ($1.324 \pm 0.154\text{ m}$ vs. $1.158 \pm 0.195\text{ m}, p < 0.0001$) than the TD control cohort. Frame-by-frame `SpineBase` root-centering and 3D `SpineShoulder–SpineBase` torso-height scaling reduce trunk length to $1.000 \pm 0.000\text{ TU}$ across both cohorts, eliminating stature bias.
2. **Discovery of Two-Level Data Leakage**:
   - **Level-1 (Augmentation Invariance)**: Four of the seven coordinate augmentations per folder (`Aug 1: +0.05m X`, `Aug 2: +0.35m X`, `Aug 3: -0.25m X`, and `Aug 5: -X reflection`) yield identical normalized feature vectors ($\max |\Delta| \le 1.92 \times 10^{-13}$) to the original trial (`video/8.xlsx`), causing ungrouped `StratifiedKFold` (`R1`) to inflate `CatBoost` accuracy to $99.00\% \pm 0.73\%$ ($\text{AUROC} = 0.9997$).
   - **Level-2 (Cross-Folder Coordinate Clones)**: Pairwise coordinate auditing across the $100$ nominal subject folders uncovered **$12$ exact duplicate folders ($r = 1.0000$: $9\text{ TD}, 3\text{ ASD}$)** across $7$ multi-folder clusters, proving the dataset contains **$88$ strictly unique children** ($47\text{ ASD}, 41\text{ TD}$) and **$87$ metadata clusters** in `Final dataset.xlsx`.
3. **Multi-Regime Benchmark Performance**:
   - **`R2` (Nominal 100-Subject `StratifiedGroupKFold`, $N=700$ trials)**: `CatBoost (Tuned)` ranks **#1 across all 10 models** with **$92.57\% \pm 2.70\%$** trial accuracy, **$93.00\%$ ($93/100$, $95\%\text{ CI}: [88.0\%, 97.0\%]$)** subject majority-vote accuracy, **$\text{AUROC} = 0.9802 \pm 0.0251$**, and Brier score $= 0.0618$ (exact McNemar $p \le 0.0076$ vs. all non-CatBoost baselines).
   - **`R3A` (87-Cluster `StratifiedGroupKFold`, $N=700$ trials)**: `CatBoost (Tuned)` ranks **#1 across all 10 models** with **$86.71\%$** trial accuracy, **$88.00\%$** subject accuracy, and **$\text{AUROC} = 0.9320$** ($\text{Subject AUROC} = 0.9488$).
   - **`R3B` (Strictly Deduplicated 88-Subject `StratifiedGroupKFold`, $N=616$ trials)**: `CatBoost (Tuned)` achieves **$88.59\% \pm 6.14\%$** trial accuracy ($88.64\%$ [$78/88$] subject accuracy) and **$\text{AUROC} = 0.9717 \pm 0.0237$**.
   - **`R4B` (Strictly Deduplicated Unaugmented Original Recordings `video/8.xlsx`, $N=88$)**: `CatBoost (Hist-300)` achieves **$89.74\% \pm 8.56\%$** accuracy and **$\text{AUROC} = 0.9502 \pm 0.0453$** ($\text{OOF Original-Only 100 AUROC} = 0.9756$).
4. **Clinically Grounded `TreeSHAP` Motor Biomarkers**:
   - Exact `TreeSHAP` attribution coupled with two-sided Mann–Whitney $U$ testing isolates three primary stature-invariant motor signatures of ASD: reduced right knee sagittal flexion (`angle_Right_Knee_mean`, Cohen's $d = +1.18, p = 2.77 \times 10^{-45}$), elevated 3D head-to-pelvis postural sway (`posture_head_spine_sway_std`, $d = +0.90, p = 2.94 \times 10^{-38}$), and bilateral upper-limb elbow flexion guarding (`angle_Right_Elbow_mean`, $d = -0.77, p = 9.04 \times 10^{-25}$; `angle_Left_Elbow_mean`, $d = -0.61, p = 1.12 \times 10^{-19}$).
