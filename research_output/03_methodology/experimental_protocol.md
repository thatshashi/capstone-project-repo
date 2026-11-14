# Experimental Protocol and Statistical Evaluation Design

## 1. Experimental Objectives and Hypotheses

1. **Primary Hypothesis ($H_1$ — Classifier Architecture)**: Ordered Gradient Boosting on Oblivious Decision Trees (`CatBoostClassifier`) will achieve higher subject-level generalization accuracy and ROC-AUC, and lower Brier probability calibration error, than depthwise gradient boosting (`XGBoost`, `LightGBM`), bagged ensembles (`Random Forest`, `AdaBoost`), kernel/linear margin classifiers (`SVM-RBF`, `SVM-Linear`, `Logistic Regression`), and single `Decision Trees` on 3D pediatric gait kinematics.
2. **Methodological Hypothesis ($H_2$ — Two-Level Data Leakage Quantification)**:
   - *Level-1 Leakage*: Naive trial-level cross-validation (`StratifiedKFold` across 700 augmented trials) will artificially inflate classification accuracy by $>6\%$ across all classifiers due to within-subject memorization of translation-invariant augmentations (`Aug 1, 2, 3, 5`).
   - *Level-2 Leakage*: Nominal folder-grouped cross-validation (`StratifiedGroupKFold` across 100 folders) retains residual cross-folder clone leakage from 12 duplicate directories; strict coordinate-cluster deduplication ($N = 88$ unique subjects) and unaugmented original-only evaluation (`video/8.xlsx`, $N = 88$) will establish the true, leakage-free generalization bound.
3. **Biomechanical Hypothesis ($H_3$ — Stature-Invariant Motor Signatures)**: After per-frame `SpineBase` root-centering and 3D `SpineShoulder–SpineBase` torso-height scaling, exact TreeSHAP attributions will isolate genuine neuromotor features—specifically lower-limb sagittal knee flexion, axial head-to-spine postural sway, and upper-extremity stereotypy—that remain statistically significant under two-sided Mann–Whitney $U$ testing.

---

## 2. Complete Specification of the Six Validation Regimes

| Regime Code | Partitioning Strategy | Sample Size ($N$ Trials / $M$ Groups) | Leakage Level Addressed | Scientific Purpose |
| :--- | :--- | :--- | :--- | :--- |
| **`R1_Naive_Ungrouped_700`** | 5-Fold `StratifiedKFold` (`shuffle=True, seed=42`) | $N = 700$ trials (Ungrouped) | None (Suffers from Level-1 & Level-2 Leakage) | Quantifies the optimistic bias ($96.3\%\text{--}99.1\%$ accuracy) reported by prior ungrouped studies [3, 7]. |
| **`R2_Nominal_100Subj_SGKF_700`** | 5-Fold `StratifiedGroupKFold` grouped by `Child_01..Child_100` | $N = 700$ trials across $M = 100$ nominal folders ($50\text{ ASD}, 50\text{ TD}$) | Eliminates Level-1 within-folder augmentation leakage | Direct benchmark comparator against nominal subject-wise literature [4, 5]. |
| **`R3A_87Cluster_SGKF_700`** | 5-Fold `StratifiedGroupKFold` grouped by the 87 metadata clusters | $N = 700$ trials across $M = 87$ clusters ($46\text{ ASD}, 41\text{ TD}$) | Eliminates Level-1 & Level-2 leakage (retains uneven cluster weights, e.g., `Cluster_00` = 35 trials) | Reconciles historical 87-cluster metrics (`ROC-AUC ≈ 0.92–0.93`). |
| **`R3B_Deduplicated_88Subj_SGKF_616`** | 5-Fold `StratifiedGroupKFold` after removing 12 clone folders | $N = 616$ trials across $M = 88$ strictly unique children ($47\text{ ASD}, 41\text{ TD}$; exactly 7 trials/child) | **Gold-Standard Multi-Trial Benchmark**: Zero Level-1 or Level-2 leakage + equal weight per child | Evaluates true cross-subject generalization when utilizing valid geometric perturbations (`Aug 4, 6, 7`). |
| **`R4A_Original_100_GroupedBy87`** | 5-Fold `StratifiedGroupKFold` on original `video/8.xlsx` trials only | $N = 100$ original trials grouped by $M = 87$ clusters | Zero synthetic augmentation; cluster-disjoint | Evaluates unaugmented camera recordings across the 100 nominal folders. |
| **`R4B_Deduplicated_Original_88`** | 5-Fold `StratifiedKFold` on deduplicated original `video/8.xlsx` trials | $N = 88$ original trials across $M = 88$ unique children ($47\text{ ASD}, 41\text{ TD}$; 1 trial/child) | **Gold-Standard Unaugmented Benchmark**: Zero synthetic augmentation & zero clone folders | Establishes pure single-trial diagnostic generalization on raw camera captures. |

---

## 3. Model Configurations and Hyperparameter Protocol

All 10 classifiers were evaluated across identical fold assignments in every regime:

1. **`CatBoost (Tuned)`**: `CatBoostClassifier(iterations=500, depth=5, learning_rate=0.04, l2_leaf_reg=4.0, random_seed=42, verbose=0)`
2. **`CatBoost (Hist-300)`**: `Pipeline([("vt", VarianceThreshold(1e-5)), ("clf", CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42, verbose=0))])`
3. **`XGBoost`**: `XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=4.0, random_state=42, eval_metric="logloss")`
4. **`HistGradientBoosting / LightGBM`**: `HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.05, l2_regularization=4.0, random_state=42)`
5. **`Random Forest`**: `RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=-1)`
6. **`AdaBoost`**: `AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)`
7. **`SVM (RBF)`**: `Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))])`
8. **`SVM (Linear)`**: `Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=0.1, kernel="linear", probability=True, random_state=42))])`
9. **`Logistic Regression`**: `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=42))])`
10. **`Decision Tree`**: `DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=42)`

---

## 4. Evaluation Metrics and Statistical Testing Protocol

### 4.1 Observation-Level (Trial-Level) Metrics
For each of the 5 cross-validation folds $k \in \{1, \dots, 5\}$, we compute:
- **Accuracy**: $\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$
- **Balanced Accuracy**: $\text{BalAcc} = \frac{1}{2}\left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right)$
- **Sensitivity (Recall / True Positive Rate)**: $\text{Sens} = \frac{TP}{TP + FN}$
- **Specificity (True Negative Rate)**: $\text{Spec} = \frac{TN}{TN + FP}$
- **Precision (Positive Predictive Value)**: $\text{Prec} = \frac{TP}{TP + FP}$
- **F1-Score**: $F_1 = \frac{2 \cdot \text{Prec} \cdot \text{Sens}}{\text{Prec} + \text{Sens}}$
- **ROC-AUC** and **PR-AUC** (Average Precision)
- **Out-of-Fold (OOF) Calibration & Agreement**: Brier Score ($\frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2$), Cohen's $\kappa$, and Matthews Correlation Coefficient (MCC).

### 4.2 Child-Level (Subject-Level) Aggregation
For multi-trial regimes (`R2`, `R3A`, `R3B`), out-of-fold trial predictions for each child $s \in \{1, \dots, M\}$ are aggregated into a single child-level diagnostic decision:
- **Subject Mean Probability**: $\bar{p}_s = \frac{1}{|T_s|} \sum_{i \in T_s} \hat{p}_i^{\text{OOF}}$ (used to compute **Subject-Level ROC-AUC**).
- **Subject Majority Vote**: $\hat{y}_s = \mathbb{I}\!\left(\frac{1}{|T_s|} \sum_{i \in T_s} \hat{y}_i^{\text{OOF}} \ge 0.5\right)$ (used to compute **Subject-Level Accuracy, Sensitivity, Specificity, and Confusion Matrix**).

### 4.3 Non-Parametric Confidence Intervals and Hypothesis Tests
1. **Stratified Subject-Cluster Bootstrap ($B = 2,000$ iterations)**: Subjects are resampled with replacement within each diagnostic class to construct 95% percentile confidence intervals ($[P_{2.5}, P_{97.5}]$) for subject-level accuracy and ROC-AUC.
2. **Exact Binomial McNemar's Test**: Compares paired out-of-fold classification decisions between `CatBoost (Tuned)` and each baseline classifier using discordant counts $(b, c)$:
   $$p_{\text{McNemar}} = 2 \sum_{k=\max(b,c)}^{b+c} \binom{b+c}{k} 0.5^{b+c}$$
3. **Biomarker Effect Size & Significance**: Each feature's discriminative magnitude is quantified via standardized mean difference (**Cohen's $d$** using pooled standard deviation) and two-sided **Mann–Whitney $U$ test** $p$-values across the cohort.
