# Movement-Based Classification of Autism Spectrum Disorder in Children Using CatBoost: A Methodologically Audited Benchmark

---

## Abstract

Autism Spectrum Disorder (ASD) is a neurodevelopmental condition characterized by social communication difficulties and repetitive behaviors. Clinical research indicates that children with ASD also exhibit distinct movement and walking abnormalities at an early age. Markerless three-dimensional (3D) depth cameras, such as the Microsoft Kinect v2 sensor, provide a non-invasive method to capture skeletal movement data for automated ASD screening. However, recent reviews of artificial intelligence in autism detection show that many published studies report inflated classification accuracies between 95% and 99% on small datasets. These high accuracies often result from trial-level cross-validation data leakage and unaddressed physical height differences between subject groups. In this work, a leakage-audited machine learning framework is proposed for 3D skeletal movement-based ASD classification using Ordered Gradient Boosting on Oblivious Trees (CatBoost). The experimental evaluation uses the publicly available Al-Jubouri (2020) Microsoft Kinect v2 pediatric gait dataset, which contains 25 skeletal joints recorded across 100 nominal child folders (50 ASD and 50 Typically Developing [TD], with 1 original recording and 7 augmented trials per folder). To remove a significant group height difference between the ASD group ($1.324 \pm 0.154\text{ m}$, age $8.62 \pm 2.19\text{ years}$) and the TD group ($1.158 \pm 0.195\text{ m}$, age $6.66 \pm 3.08\text{ years}$; $p < 0.001$), the proposed system applies frame-by-frame pelvis root-centering at the `SpineBase` joint and 3D trunk-height scaling between the `SpineShoulder` and `SpineBase` joints. Subsequently, 86 biomechanical features are extracted across joint angles, joint velocities, postural sway, bilateral asymmetry, and spatial limb excursion. Ten machine learning classifiers are evaluated across a four-regime validation hierarchy (`seed = 42`), and model decisions are interpreted using exact TreeSHAP and two-sided Mann–Whitney $U$ statistical tests. A coordinate-level audit of the dataset reveals two previously undocumented sources of data leakage. First, four of the seven coordinate augmentations per folder (three mediolateral $X$-axis translations and one sagittal reflection) produce identical normalized feature vectors ($\max |\Delta| \le 1.92 \times 10^{-13}$) to the original recording after root-centering, which inflates naive ungrouped cross-validation (`R1`) accuracy to $99.00\% \pm 0.73\%$ ($\text{AUROC} = 0.9997$). Second, 12 of the 100 nominal subject folders are exact coordinate duplicates ($r = 1.0000$) of 7 other folders within the same class, reducing the cohort to 88 unique children (47 ASD and 41 TD) and 87 metadata clusters. Under nominal 100-subject grouped cross-validation (`R2`), tuned CatBoost achieves $92.57\% \pm 2.70\%$ trial-level accuracy, $93.00\%$ ($93/100$, $95\%\text{ CI}: [88.0\%, 97.0\%]$) subject majority-vote accuracy, and an area under the receiver operating characteristic curve ($\text{AUROC}$) of $0.9802 \pm 0.0251$ (Brier score $= 0.0618$). This performance is statistically superior ($p \le 0.0076$) to XGBoost ($89.14\%$), LightGBM ($89.86\%$), Random Forest ($89.57\%$), Support Vector Machine with a radial basis function kernel ($89.00\%$), and Logistic Regression ($88.29\%$). On the strictly deduplicated 88-child cohort (`R3B`, 616 trials), tuned CatBoost achieves $88.59\% \pm 6.14\%$ accuracy and $\text{AUROC} = 0.9717 \pm 0.0237$. On the 88 unaugmented original recordings (`R4B`), historical depth-4 CatBoost achieves $89.74\% \pm 8.56\%$ accuracy and $\text{AUROC} = 0.9502 \pm 0.0453$. Furthermore, multi-seed testing demonstrates that a previously reported single-split holdout $\text{AUROC}$ of $1.000$ ($N = 22$ clusters) is a small-sample artifact that varies between $0.875$ and $1.000$ across random seeds. TreeSHAP analysis identifies reduced right knee flexion ($\text{Cohen's } d = 1.18, p = 2.77 \times 10^{-45}$), increased 3D head-to-spine postural sway ($d = 0.90, p = 2.94 \times 10^{-38}$), and reduced bilateral elbow extension ($d = -0.77, p = 9.04 \times 10^{-25}$) as the primary height-invariant motor markers of ASD. The major limitations of this study include the small effective sample size of 88 unique children from a single clinical center, a male-predominant sex ratio in the ASD group, and the absence of non-ASD neurodevelopmental control groups.

## Keywords

Autism Spectrum Disorder; Kinect v2 skeletal tracking; gait kinematics; CatBoost; data leakage; explainable artificial intelligence

---

## 1. Introduction

Autism Spectrum Disorder (ASD) is a complex neurodevelopmental condition defined by persistent difficulties in social interaction, verbal and non-verbal communication, and repetitive patterns of behavior [1, 8]. Standard clinical diagnosis relies on behavioral observation schedules, such as the *Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition* (DSM-5) and the *Autism Diagnostic Observation Schedule, Second Edition* (ADOS-2). While these clinical instruments focus primarily on social and behavioral symptoms, clinical biomechanics studies have established that between 50% and 88% of children with ASD also experience significant motor coordination and gait impairments [8, 25, 26]. These motor differences include a wider walking base, reduced knee and ankle flexion during walking, increased head and trunk sway, asymmetric arm swing, and repetitive upper-limb movements [8, 10, 12]. Furthermore, motor abnormalities often appear during the first two years of life, before clear language and social delays can be formally assessed [8, 17, 25].

Despite the clinical value of early identification, traditional ASD diagnosis requires lengthy evaluations by specialized multidisciplinary teams. As a result, many children face long waiting periods that delay early behavioral and physical interventions [1, 26]. Objective movement analysis can assist clinicians by providing quantitative motor screening metrics. However, laboratory three-dimensional (3D) optical motion capture systems, such as multi-camera Vicon setups with skin-mounted reflective markers and force plates [9, 10], are expensive and restricted to specialized clinics. In addition, attaching physical markers to the skin often causes distress for children with ASD who have tactile sensory sensitivities [4, 8]. Therefore, markerless 3D depth cameras—particularly the Microsoft Kinect v2 time-of-flight sensor, which tracks 25 body joints at 30 frames per second (fps) without physical contact—have been widely adopted to record full-body movement in natural environments [2–6, 14, 15].

Several machine learning and deep learning methods have been developed to classify ASD and Typically Developing (TD) children from skeletal tracking data [3–7, 11, 15, 18]. Existing approaches extract spatiotemporal gait parameters, statistical coordinate summaries, or joint angle trajectories, and classify them using Rough Set classifiers [3], Support Vector Machines (SVM) [9, 11, 15], Random Forests [5, 14], adaptive boosting (AdaBoost) [4], or deep neural networks [6, 7]. A prominent benchmark in this domain is the publicly archived 3D Kinect v2 pediatric gait dataset introduced by Al-Jubouri et al. (2020) [2, 3], which contains 100 nominal subject folders (50 ASD and 50 TD) and 7 synthetic coordinate augmentations per folder.

However, existing computational studies on small skeletal movement datasets suffer from severe methodological limitations [1, 21, 22]. First, many studies apply trial-level cross-validation, where multiple augmented recordings from the same child are randomly divided between the training and test sets [3, 7]. This practice causes subject-level data leakage and produces overly optimistic accuracy estimates between 95% and 99% that fail to generalize to unseen children [1, 21, 22]. Second, prior studies on the Al-Jubouri (2020) dataset have not audited the raw coordinate files for cross-folder subject duplication or tested whether the provided synthetic coordinate shifts remain distinct after anatomical normalization [2–7]. Third, raw camera coordinates are measured in meters and depend directly on the physical height of the participant and the distance from the camera. If the ASD and TD groups differ in age and height, models trained on unnormalized coordinates may classify children based on body size rather than movement patterns. Fourth, standard gradient boosted decision trees, such as XGBoost [23] and LightGBM [24], build trees using greedy depthwise or leaf-wise splits and estimate gradients using the same samples used to construct prior trees. On small clinical datasets ($N \le 100$), this process causes prediction shift and model overfitting [18, 19].

Hence, a clear research gap exists regarding the true, leakage-free classification performance achievable on the Al-Jubouri (2020) 3D Kinect v2 dataset once height confounding, within-folder augmentation redundancy, and cross-folder subject duplication are removed. In addition, Ordered Gradient Boosting on Oblivious Trees (CatBoost) [19] has not previously been evaluated or compared against standard tree ensembles and kernel classifiers on this benchmark under a unified validation hierarchy.

In this work, a methodologically audited pipeline is proposed for 3D movement-based ASD classification using frame-wise anatomical normalization, biomechanical feature extraction, Ordered CatBoost classification, and exact TreeSHAP explainability [20]. The main contributions of this study are summarized as follows:

1. **Coordinate-Level Audit of Cross-Folder Duplication**: Pairwise trajectory comparisons across all 100 nominal subject folders (`Child_01` to `Child_100`) demonstrate that 12 folders are exact coordinate duplicates ($r = 1.0000$) of 7 other folders within the same diagnostic group. This reduces the 100 nominal folders to 88 strictly unique children (47 ASD and 41 TD) and 87 metadata clusters in the pre-compiled spreadsheet (`Final dataset.xlsx`).
2. **Mathematical Verification of Augmentation Translation-Invariance**: Algebraic and numerical analysis proves that four of the seven synthetic augmentations per folder (`Aug 1: +0.05 m X`, `Aug 2: +0.35 m X`, `Aug 3: -0.25 m X`, and `Aug 5: -X reflection`) yield identical normalized feature vectors ($\max |\Delta| \le 1.92 \times 10^{-13}$) to the original camera recording (`video/8.xlsx`) after pelvis root-centering. This explains the artificial 99% accuracy observed under ungrouped cross-validation.
3. **Removal of Demographic Height Confounding**: Analysis of the dataset documentation reveals that the ASD group is significantly older ($8.62 \pm 2.19\text{ years}$ vs. $6.66 \pm 3.08\text{ years}$, $p = 0.0004$) and taller ($1.324 \pm 0.154\text{ m}$ vs. $1.158 \pm 0.195\text{ m}$, $p < 0.0001$) than the TD group. Frame-by-frame pelvis root-centering (`SpineBase`) and 3D trunk-height normalization (`SpineShoulder–SpineBase` distance) are applied to eliminate this $16.6\text{ cm}$ stature bias before extracting 86 biomechanical features.
4. **Multi-Regime Benchmark of Ordered CatBoost**: Two CatBoost configurations (tuned depth-5 and historical depth-4) are evaluated against eight baseline models (XGBoost, LightGBM, Random Forest, AdaBoost, SVM-RBF, SVM-Linear, Logistic Regression, and Decision Tree) across six cross-validation regimes (`R1` through `R4B`). This hierarchy separates trial-level leakage, nominal folder grouping, cluster deduplication, and unaugmented original recordings.
5. **Reconciliation of Historical Holdout Metrics**: Multi-seed experiments reconcile previously reported preliminary metrics (`ROC-AUC: 0.919 ± 0.072`, `Holdout AUC: 1.000, N = 22`, and `Original-only AUC: 0.975`) and demonstrate that a single 22-cluster holdout split is highly sensitive to random seed selection ($\text{AUROC}$ ranging from $0.8750$ to $1.0000$).
6. **Biomechanical Interpretation via Exact TreeSHAP**: Exact TreeSHAP feature attribution is combined with non-parametric Mann–Whitney $U$ tests and Cohen's $d$ effect sizes to identify height-invariant motor markers of ASD across knee flexion, head-to-spine postural sway, and elbow flexion.

The remainder of this paper is organized as follows. Section 2 reviews related studies on motor abnormalities in ASD and machine learning classification, presents a comparative analysis table, and defines the research gap. Section 3 outlines the motivation and technical challenges of the study. Section 4 describes the proposed system model and architecture. Section 5 details the methodology, including parameter selection, dataset preparation, preprocessing, feature extraction, model formulation, and evaluation procedures. Section 6 describes the experimental setup and forensic dataset audit. Section 7 presents the quantitative results, statistical comparisons, and biomechanical discussion. Section 8 discusses the study limitations, Section 9 outlines future research directions, and Section 10 concludes the paper.

---

## 2. Related Works

### 2.1 Review of Existing Literature

Clinical biomechanics studies have long investigated motor coordination differences in children with ASD. Fournier et al. (2010) [25] conducted a meta-analysis of 51 clinical studies to quantify motor coordination deficits in individuals with ASD compared to neurotypical controls. Their methodology synthesized standardized effect sizes across postural stability, upper-limb coordination, and gait parameters. The authors found a large overall effect size ($d = 1.20$), indicating widespread motor impairment in ASD. This work provides the clinical foundation for using movement measurements as objective markers of autism. However, Fournier et al. [25] relied on aggregate clinical rating scales and laboratory observations rather than automated markerless 3D skeletal classification.

Li, Mache, and Todd (2021) [12] investigated age-related changes in gait kinematics among children with ASD between 6 and 14 years of age. The authors used wearable inertial measurement units (IMUs) to measure joint angles and spatiotemporal gait parameters during walking. Their findings showed that pelvic tilt, knee flexion, and stride regularity change significantly with chronological age and physical growth. This study is directly relevant to the current work because it demonstrates that gait kinematics are influenced by age and physical maturation. Unlike Li et al. [12], who used body-worn IMU sensors and statistical group comparisons, the present study uses markerless 3D Kinect v2 coordinates and applies per-frame trunk-height normalization to remove age-related stature differences prior to machine learning classification.

Olivas et al. (2022) [10] examined lower-extremity joint mechanics in children with ASD when walking at different speeds. Using a multi-camera 3D optical motion capture system, the authors measured sagittal ankle, knee, and hip angles across slow, self-selected, and fast walking speeds. The results indicated that children with ASD use altered ankle and knee stiffness strategies to regulate forward progression. This study supports the biomechanical relevance of lower-limb joint angles in ASD detection. However, Olivas et al. [10] required reflective skin markers in a laboratory setting and did not develop a supervised classification model.

Ioannidou et al. (2025) [8] performed a systematic review of pediatric ASD gait patterns across recent biomechanical investigations. By synthesizing kinematic and kinetic findings across lower-body and upper-body segments, the authors identified three consistent motor deviations in ASD: reduced sagittal knee and ankle range of motion, increased mediolateral and anteroposterior head and trunk sway, and abnormal upper-limb posture during walking. Similarly, Zampella et al. (2021) [26] reviewed clinical motor skill differences across the autism spectrum and highlighted repetitive upper-limb movements and coordination delays. These two clinical reviews guide the selection of the 86 biomechanical features in the present study, whereas neither review performed computational modeling or dataset auditing.

In the area of automated movement analysis using touchscreens and optical markers, Anzulewicz, Sobota, and Delafield-Butt (2016) [13] studied smart-tablet gesture kinematics in 82 children (37 ASD and 45 TD). They extracted 262 kinematic and touch-pressure features during tablet gameplay and trained a Regularized Greedy Forest classifier, achieving 93.0% accuracy ($\text{AUROC} = 0.93$). Similarly, Simeoli et al. (2021) [16] recorded 16 finger-dragging trajectory descriptors from 60 children (30 ASD and 30 TD) on a tablet interface and achieved 93.0% accuracy using an Artificial Neural Network (ANN). While both studies confirm that fine motor kinematics can distinguish ASD from TD children, they evaluate fine finger movements on a 2D screen rather than full-body 3D locomotion and postural balance.

Pradhan, Chester, and Padhiar (2022) [9] developed a machine learning pipeline to classify ASD gait using a 12-camera Vicon motion capture system. Their study evaluated 40 children (19 ASD and 21 TD) and extracted 168 multisegment foot and ankle kinematic features. Using nested subject-level cross-validation, an SVM classifier with a radial basis function (RBF) kernel achieved 96.3% accuracy on 9 selected foot angle features, which were interpreted using SHAP and SAGE. This study is methodologically strong due to its subject-disjoint validation and explainability analysis. However, it relied on a 12-camera Vicon laboratory with physical skin markers and focused exclusively on foot segments rather than markerless full-body 3D skeletal tracking.

To avoid physical markers, several researchers have used Kinect depth cameras on independent clinical cohorts. Ardalan et al. (2019) [14] recorded whole-body movement variability using a Kinect for Xbox 360 sensor while 62 participants (39 ASD and 23 TD) played a balance videogame. Using a Random Forest classifier on joint variability features, they reported 89.0% classification accuracy. Alcañiz Raya et al. (2020) [15] used a Kinect v2 camera inside a virtual reality (VR) room to track head, trunk, and limb displacements in 49 children (24 ASD and 25 TD). By combining recursive feature elimination (RFE) with an SVM classifier, they achieved accuracies between 82.98% and 89.36%. Aljabiri and Hamdan (2024) [11] analyzed lower-body Kinect trajectories from 30 children (15 ASD and 15 TD) during an augmented reality ball-kicking task using Linear Discriminant Analysis (LDA) and SVM. Although these studies demonstrate the feasibility of Kinect-based motor assessment, they used task-specific VR or gaming hardware on smaller cohorts ($N = 30\text{--}62$) and did not evaluate natural corridor walking on the public Al-Jubouri (2020) benchmark.

Markerless RGB video pose estimation has also been explored in recent literature. Doi et al. (2022) [17] analyzed spontaneous body movements in 4-month-old infants using 2D video recordings to predict autistic traits at 18 months of age. Ganai et al. (2025) [18] extracted 49 gait deviation and joint angle features using MediaPipe pose estimation from video recordings of 61 children (32 ASD and 29 TD). Under a subject-level split, regularized Logistic Regression achieved 82.0% accuracy ($\text{AUROC} = 0.97$) and outperformed XGBoost and Random Forest, which overfit the small sample size. The observation by Ganai et al. [18] that standard depthwise tree ensembles overfit small gait cohorts directly motivates the evaluation of Ordered CatBoost with symmetric oblivious trees [19] in the present work.

A central group of studies directly utilizes the Al-Jubouri et al. (2020) 3D Kinect v2 gait dataset (`DOI: 10.5061/dryad.s7h44j150`) [2]. AbdulRahman (Al-Jubouri), Hadi, and Rajihy (2020) [2] and Al-Jubouri, Ali, and Rajihy (2021) [3] collected the original dataset of 50 nominal ASD and 50 nominal TD children walking toward a Kinect v2 sensor. To increase the dataset size from 100 recordings to 800 rows (`Final dataset.xlsx`), the authors applied 7 geometric coordinate transformations per folder and extracted 1,259 raw meter-scale coordinate summary statistics. Using Principal Component Analysis (PCA) and a Rough Set classifier, they reported 60.0% accuracy on the 100 unaugmented trials and 92.0% accuracy on the 800 augmented trials [3]. However, their evaluation used an ungrouped trial-level split without trunk-height normalization, allowing augmented copies of the same child and raw height differences to leak across training and test sets.

Yazdi, Janghorbani, and Maleki (2024) [4] evaluated the same 100-folder Al-Jubouri dataset by extracting kinematic summary features and applying filter and wrapper feature selection methods, including minimum Redundancy Maximum Relevance (mRMR), ReliefF, and LASSO. Using 10-fold cross-validation and Leave-One-Subject-Out (LOSO) validation across the 100 nominal folders, their AdaBoost classifier achieved 87.0% accuracy, outperforming Naive Bayes (84.0%), K-Nearest Neighbors (KNN), and SVM. While Yazdi et al. [4] recognized the need for subject-level validation across the 100 folders, they did not audit the dataset for cross-folder duplicate recordings, did not evaluate CatBoost, and did not provide SHAP-based biomechanical attributions.

Fırat (2026) [5] proposed a SHAP-based explainable AI framework on the Al-Jubouri dataset, analyzing 463 pre-extracted statistical features across 109 nominal subjects (50 TD, 50 moderate ASD, and 9 severe ASD cases). After selecting the top 50 features via mRMR, Fırat [5] evaluated Random Forest, SVM, XGBoost, Multilayer Perceptron (MLP), KNN, and Logistic Regression under 5-fold cross-validation, reporting 86.36% binary classification accuracy and 84.62% three-class accuracy with Random Forest. Although Fırat [5] incorporated SHAP analysis, the study relied on pre-extracted statistical features without frame-wise 3D trunk-height scaling, did not test CatBoost, and did not detect that 12 of the 100 skeletal folders are exact coordinate clones or that 4 of the 7 augmentations are translation-invariant duplicates.

Le-Thi-Thu et al. (2026) [6] investigated skeleton gait normalization and dynamic motion features (velocity and acceleration) on the Al-Jubouri Kinect v2 dataset using deep learning architectures, including Bidirectional Long Short-Term Memory (BiLSTM) networks and Spatio-Temporal Graph Convolutional Networks (ST-GCN). Similarly, Amirbay et al. (2025) [7] applied hybrid convolutional and recurrent models to motor and facial data, reporting accuracies between 88.0% and 96.0% under trial-level partitioning. Both studies focused on black-box neural network architectures without auditing cross-folder coordinate clones or providing exact game-theoretic feature explanations.

Finally, the methodological design of the present work is informed by scoping reviews and statistical audits of data leakage in biomedical machine learning. Rakotomanana and Rouhafzay (2025) [1] reviewed 158 AI-based behavioral autism detection studies published between 2015 and 2025. They identified small sample sizes, male gender bias, and trial-level data leakage between training and test splits as the primary barriers to clinical reliability. This finding aligns with foundational studies by Saeb et al. (2017) [21] and Kapoor and Narayanan (2023) [22], who demonstrated that record-wise cross-validation on multi-trial subject data invalidates generalization claims across digital health applications.

### 2.2 Comparative Analysis

To provide a structured comparison of existing research, Table 1 summarizes the focus, methodology, reported results, and relationship with the proposed work across primary studies on the Al-Jubouri (2020) Kinect v2 dataset and related pediatric ASD gait benchmarks.

**Table 1. Comparative analysis of existing movement-based ASD classification studies and the proposed research.**

| Reference | Main Research Focus | Methodology or Contribution | Relationship with the Proposed Research |
| :--- | :--- | :--- | :--- |
| **Rakotomanana and Rouhafzay (2025)** [1] | Scoping review of 158 AI studies in behavioral autism detection | Identified small cohorts, gender skew, and train-test data leakage as widespread flaws across 8 behavioral modalities | Provides the methodological motivation to audit and eliminate two-tier data leakage and height confounding in 3D skeletal data |
| **Al-Jubouri et al. (2020, 2021)** [2, 3] | Creation and baseline classification of the 3D Kinect v2 pediatric ASD gait dataset | Collected 100 nominal folders (800 files with 7 augmentations); extracted 1,259 raw meter-scale features with PCA and Rough Set classification (60.0% raw, 92.0% augmented accuracy) | Primary dataset source; the proposed work replaces raw meter features and ungrouped splitting with 3D trunk-scaled features and grouped cross-validation |
| **Yazdi et al. (2024)** [4] | Feature selection and classical ML classification on the Al-Jubouri Kinect v2 dataset | Applied mRMR, ReliefF, and LASSO on kinematic summary features; achieved 87.0% accuracy with AdaBoost under nominal 100-folder validation | Benchmark comparator; the proposed work improves nominal 100-subject accuracy to 93.00% with CatBoost and uncovers 12 cross-folder clones overlooked in [4] |
| **Fırat (2026)** [5] | Explainable ASD classification using pre-extracted statistical features on the Al-Jubouri dataset | Selected top 50 mRMR features from 463 statistics; evaluated RF, SVM, XGBoost, MLP, KNN, and LR with SHAP (86.36% binary accuracy) | Benchmark comparator; the proposed work introduces frame-wise 3D torso normalization, evaluates Ordered CatBoost, and audits augmentation invariance and folder clones |
| **Le-Thi-Thu et al. (2026)** [6] | Deep sequence modeling of normalized Kinect v2 gait trajectories | Evaluated BiLSTM and ST-GCN models using skeleton normalization and velocity/acceleration dynamics | Confirms the value of motion derivatives; the proposed work shows that interpretable biomechanical features with CatBoost achieve high accuracy without black-box deep networks |
| **Amirbay et al. (2025)** [7] | Multimodal and motor-based autism classification | Applied hybrid CNN-BiLSTM and ML models under trial-level $k$-fold partitioning (88.0%–96.0% accuracy) | Illustrates trial-level leakage risk; the proposed work quantifies how ungrouped splits artificially inflate accuracy by 6.4% to 16.4% |
| **Pradhan et al. (2022)** [9] | Laboratory 3D foot kinematics classification in pediatric ASD ($N = 40$) | Extracted 168 Vicon foot features; achieved 96.3% nested CV accuracy using RBF-SVM with SHAP and SAGE | Methodological reference for subject-grouped validation and SHAP attribution, adapted here to markerless Kinect v2 full-body tracking |
| **Alcañiz Raya et al. (2020)** [15] | Kinect v2 body movement tracking inside a VR environment ($N = 49$) | Extracted head, trunk, and limb displacement features; achieved 82.98%–89.36% accuracy using SVM-RFE | Supports axial head and trunk sway features; differs by requiring an immersive VR room rather than a simple corridor walk |
| **Ganai et al. (2025)** [18] | Markerless RGB video gait analysis using MediaPipe ($N = 61$) | Extracted 49 gait features; found Logistic Regression (82.0% accuracy, $\text{AUROC} = 0.97$) outperformed overfitting XGBoost and RF | Demonstrates small-sample overfitting in standard depthwise tree ensembles, motivating the use of symmetric oblivious trees in CatBoost |
| **Proposed Work** | Leakage-audited 3D movement classification and biomechanical attribution on the Al-Jubouri dataset | Proves 12 clone folders ($M = 88$ unique children) and 4 translation-invariant augmentations; eliminates $16.6\text{ cm}$ stature bias via 3D trunk scaling; benchmarks Ordered CatBoost + TreeSHAP across 4 validation regimes | Achieves 92.57% trial accuracy (93.00% subject accuracy, $\text{AUROC} = 0.9802$) in `R2`, $\text{AUROC} = 0.9717$ in deduplicated `R3B`, and $89.74\%$ accuracy ($\text{AUROC} = 0.9502$) on unaugmented `R4B` |

### 2.3 Research Gap

Existing studies have demonstrated that 3D skeletal tracking and machine learning can identify movement differences between children with ASD and TD controls [2–9, 14, 15, 18]. Previous research on the Al-Jubouri (2020) Kinect v2 dataset has investigated Rough Set classification [3], filter and wrapper feature selection with AdaBoost [4], Random Forest classification with SHAP on pre-extracted statistical features [5], and deep neural networks [6, 7].

However, four specific aspects remain insufficiently investigated in the existing literature:

1. **Absence of Cross-Folder Duplicate Auditing**: All prior studies on the Al-Jubouri dataset [2–7] assumed that the 100 numbered directories (`Typical/1..50` and `Autism/children with ASD/1..50`) represent 100 distinct individuals. No prior work checked whether identical 3D coordinate recordings were copied across multiple subject folders, which causes cross-folder data leakage even under nominal subject-grouped cross-validation.
2. **Unexamined Redundancy of Synthetic Coordinate Augmentations**: Previous studies used the 7 augmented coordinate files per folder as independent training or testing samples without verifying how spatial translations (`+0.05 m`, `+0.35 m`, `-0.25 m`) and reflections (`-X`) behave after anatomical root-centering.
3. **Uncontrolled Physical Stature Confounding**: The raw coordinates and pre-compiled summary table (`Final dataset.xlsx`) are recorded in meters. Because the ASD cohort is on average $1.96\text{ years}$ older and $16.6\text{ cm}$ taller than the TD cohort ($p < 0.0001$), models trained without 3D trunk-height normalization can classify subjects based on physical stature rather than movement pathology.
4. **Lack of Ordered CatBoost Evaluation Across Leakage Regimes**: Standard depthwise tree ensembles (such as XGBoost, LightGBM, and Random Forest) are prone to prediction shift and leaf-level overfitting on small clinical cohorts [18, 19]. Ordered CatBoost [19], which uses symmetric oblivious decision trees as a structural regularizer, has not been benchmarked against classical and ensemble baselines across progressively stricter deduplicated and unaugmented validation regimes on this dataset.

The present work directly addresses these four gaps by combining a coordinate-level dataset audit, frame-by-frame 3D trunk normalization, an 86-feature biomechanical representation, a 10-model benchmark across six validation regimes, and exact TreeSHAP statistical attribution.

---

## 3. Motivation

### 3.1 Motivation

Early and objective screening for ASD is an important clinical priority because behavioral interventions are most effective when initiated during early childhood development [1, 8, 26]. While behavioral assessments require specialized clinicians and extended observation periods, a brief corridor walking test captured by a non-contact 3D depth camera requires less than one minute and places no physical sensors on the child [2, 4]. If movement signatures can be quantified reliably, markerless 3D gait analysis can serve as a rapid, low-cost screening aid to prioritize children for comprehensive clinical evaluation.

However, for a computational screening model to be clinically trustworthy, its reported accuracy must reflect true generalization to unseen children, and its predictions must rely on genuine motor abnormalities rather than dataset artifacts [1, 21, 22]. Establishing a rigorous benchmark that removes physical height bias, eliminates duplicate recordings, and links model decisions to verifiable biomechanical features (such as knee stiffness, postural sway, and upper-limb guarding) is therefore essential to advance movement-based ASD research from optimistic preliminary prototypes toward reliable clinical decision support.

### 3.2 Challenges

Implementing a methodologically sound classification pipeline on 3D pediatric skeletal data involves several practical and technical challenges:

1. **Inter-Group Demographic and Stature Bias**: In pediatric cohorts collected in clinical settings, diagnostic groups are rarely matched perfectly by age and standing height. In the Al-Jubouri (2020) cohort, the ASD group is $16.6\text{ cm}$ taller on average than the TD group. In raw camera coordinates, taller children exhibit larger vertical joint heights ($Y$), longer limb segments, and larger raw stride excursions, creating a strong confounding signal.
2. **Multi-Level Data Leakage in Augmented Skeletal Repositories**: Public movement datasets often include pre-generated data augmentations to compensate for small participant numbers. When global coordinate shifts are combined with translation-invariant preprocessing, multiple trials become mathematically identical. Furthermore, undocumented file duplication across subject folders introduces hidden data leakage that standard subject-ID grouping cannot prevent.
3. **High Feature-to-Subject Ratio and Small-Sample Overfitting**: Extracting comprehensive kinematics across 25 3D joints yields dozens of correlated features ($d = 86$), while the true number of unique children is small ($M = 88$). Under this high-dimensional, small-sample regime, flexible non-linear classifiers easily memorize subject-specific noise unless constrained by strong structural regularization.
4. **Variable Trial Lengths and Recording Frame Rates**: Natural walking trials vary in duration and frame count depending on the walking speed of each child, and depth sensors occasionally experience transient frame drop-outs or timestamp irregularities. Converting variable-length 3D trajectories into fixed-length, scale-invariant feature vectors without losing temporal dynamics requires careful signal processing.

---

## 4. Proposed System Model

### 4.1 Overall Architecture and System Components

The proposed system is designed as an end-to-end, leakage-audited processing pipeline that transforms raw 3D Kinect v2 skeletal coordinate sequences into calibrated ASD classification probabilities and biomechanical feature explanations. Figure 1 illustrates the overall architecture of the proposed system and the four-regime validation hierarchy used to evaluate model generalization.

![Figure 1: Methodological Architecture and Four-Regime Validation Hierarchy](../05_figures/fig01_methodology_workflow.png)
*Figure 1. Overall architecture of the proposed system, showing raw Kinect v2 coordinate auditing, frame-by-frame SpineBase root-centering and 3D SpineShoulder–SpineBase trunk-height normalization, 86-feature biomechanical extraction, Ordered CatBoost classification with TreeSHAP attribution, and the four-regime cross-validation hierarchy (`R1` to `R4`).*

The system consists of five core processing modules:

1. **Data Ingestion and Forensic Provenance Auditor**: Parses the raw `.xlsx` coordinate spreadsheets from the subject directories, standardizes the 25 anatomical joint channels, parses frame timestamps, and audits pairwise trajectory correlations to identify both translation-invariant augmentations (Level-1 leakage) and cross-folder subject clones (Level-2 leakage).
2. **Frame-by-Frame 3D Anthropometric Normalizer**: Translates all 25 joints at each time frame so that the anatomical pelvis center (`SpineBase`) is located at the origin $(0, 0, 0)$, and divides all joint coordinates by the instantaneous 3D Euclidean trunk length between `SpineShoulder` and `SpineBase`.
3. **Biomechanical Feature Engineering Engine**: Computes $d = 86$ clinically interpretable kinematic features from the normalized 3D trajectories across five movement domains: 3D and sagittal joint angles, first- and second-order motion derivatives (velocity and acceleration), axial postural sway and step width, bilateral limb asymmetry and upper-limb stereotypy distances, and distal spatial excursion.
4. **Multi-Regime Cross-Validation Partitioner**: Enforces strict separation between training and validation folds across six validation protocols (`R1` Naive ungrouped, `R2` Nominal 100-subject grouped, `R3A` 87-cluster grouped, `R3B` Deduplicated 88-subject grouped, `R4A` Nominal 100 original-only, and `R4B` Deduplicated 88 original-only).
5. **Ordered CatBoost Classifier and TreeSHAP Explainer**: Trains an ensemble of symmetric oblivious decision trees using ordered boosting to predict trial-level and subject-level ASD probabilities, and computes exact TreeSHAP values paired with non-parametric statistical tests for clinical interpretation.

### 4.2 Input Parameters, Processing Stages, and Decision-Making Mechanism

The input to the system for each walking trial $i$ is a variable-length multivariate time series $\mathbf{P}_i \in \mathbb{R}^{T_i \times 25 \times 3}$, where $T_i$ is the number of recorded video frames, $J = 25$ is the number of tracked skeletal joints, and each joint $j \in \{0, \dots, 24\}$ is represented by its 3D Cartesian coordinates $(X_{i,t,j}, Y_{i,t,j}, Z_{i,t,j})$ in meters relative to the Kinect v2 camera, accompanied by frame timestamps $t \in \{1, \dots, T_i\}$.

During the processing stages, each raw tensor $\mathbf{P}_i$ is first converted by the Anthropometric Normalizer into a dimensionless, root-centered and trunk-scaled tensor $\tilde{\mathbf{P}}_i \in \mathbb{R}^{T_i \times 25 \times 3}$ measured in Torso Units (TU). Next, the Feature Engineering Engine aggregates temporal dynamics across the $T_i$ frames to map $\tilde{\mathbf{P}}_i$ into a fixed-dimensional feature vector $\mathbf{x}_i \in \mathbb{R}^{86}$.

In the decision-making stage, the feature vector $\mathbf{x}_i$ is passed through an ensemble of $M_{\text{trees}}$ symmetric oblivious decision trees trained via Ordered CatBoost. Each tree $m$ evaluates $D$ global feature-threshold split conditions across all nodes at depth $l \in \{1, \dots, D\}$ to assign $\mathbf{x}_i$ to one of $2^D$ leaves with weight $w_{m,v}$. The summed log-odds score $F(\mathbf{x}_i) = \sum_{m=1}^{M_{\text{trees}}} h_m(\mathbf{x}_i)$ is converted via the logistic sigmoid function $\hat{p}_i = (1 + \exp(-F(\mathbf{x}_i)))^{-1} \in [0, 1]$ into a calibrated trial-level probability of ASD. For multi-trial subject evaluation, the system aggregates the out-of-fold predictions across all trials belonging to subject $s$ using majority voting ($\hat{y}_s = \mathbb{I}(\frac{1}{|T_s|}\sum_{i \in T_s} \hat{y}_i \ge 0.5)$) and mean probability averaging ($\bar{p}_s = \frac{1}{|T_s|}\sum_{i \in T_s} \hat{p}_i$). Finally, the TreeSHAP module decomposes $F(\mathbf{x}_i)$ into exact additive feature attributions $\phi_{i,1}, \dots, \phi_{i,86}$ to explain which specific joint movements contributed to the diagnostic decision.

---

## 5. Methodology

### 5.1 Parameter Selection

The Microsoft Kinect v2 sensor tracks $J = 25$ anatomical landmarks across the human body at a nominal frame rate of $30\text{ fps}$ [2, 3]. In the proposed methodology, all 25 skeletal joints are retained during preprocessing to preserve complete full-body kinematic chains (`src/data_preprocessing.py`). The canonical joint index assigns `SpineBase` (pelvis center) to index $j = 0$ and `SpineShoulder` (upper thoracic spine) to index $j = 1$, followed by the axial joints (`Midspain`, `Neck`, `Head`), bilateral upper-extremity joints (`Shoulder`, `Elbow`, `Wrist`, `Hand`, `HandTip`, `Thumb`), and bilateral lower-extremity joints (`Hip`, `Knee`, `Ankle`, `Foot`).

Choosing `SpineBase` ($j = 0$) as the anatomical root parameter is biomechanically justified because the pelvis represents the approximate center of mass during upright human locomotion [8, 12]. Similarly, selecting the 3D Euclidean segment between `SpineShoulder` ($j = 1$) and `SpineBase` ($j = 0$) as the anatomical scaling parameter provides a rigid axial skeletal reference that is directly proportional to standing height while remaining unaffected by knee flexion or arm movement during walking.

### 5.2 Dataset Preparation

The raw dataset directory (`Dataset-2/Dataset/`) contains two primary diagnostic groups:
- **Typically Developing (TD) Control Group**: 50 nominal subject folders (`Typical/1` through `Typical/50`), mapped to canonical subject identifiers `Child_01` through `Child_50` with binary class label $y_i = 0$.
- **Autism Spectrum Disorder (ASD) Group**: 50 nominal walking subject folders (`Autism/children with ASD/1` through `50`), mapped to canonical subject identifiers `Child_51` through `Child_100` with binary class label $y_i = 1$. An additional 9 folders in `Severe autism/1..9` contain only raw video files without `.xlsx` 3D skeletal coordinate spreadsheets and are therefore excluded from 3D coordinate analysis.

Within each of the 100 nominal subject folders, `video/8.xlsx` stores the original unaugmented camera recording, and `augmentation/1/1.xlsx` through `augmentation/7/7.xlsx` store 7 coordinate-augmented trials. Because the raw `.xlsx` files store the 25 joints in alphabetical column order starting at column offsets $1, 4, 7, \dots, 73$, the ingestion parser (`src/data_preprocessing.py`) maps each 3-column block $(X, Y, Z)$ to its canonical joint index $j \in \{0, \dots, 24\}$. Frame timestamps formatted as `H:M:S:MS` are converted into elapsed seconds $t_k$, with a deterministic fallback step of $\Delta t = 1/30\text{ s}$ when timestamp strings are missing. Any rows containing missing (`NaN`) or non-numeric coordinate values are removed prior to normalization.

### 5.3 Data Preprocessing

As documented in the demographic audit (Section 6.2), the ASD walking cohort is on average $1.96\text{ years}$ older and $16.6\text{ cm}$ taller than the TD control cohort ($p < 0.0001$). If features are extracted directly from raw meter-scale coordinates, a classifier can distinguish the two cohorts simply by measuring vertical head height or bone length. Furthermore, raw coordinates depend on where the child walks relative to the camera origin.

To eliminate both camera position offset and physical stature confounding, a two-step frame-by-frame 3D normalization procedure is applied to every trial sequence $\mathbf{P}_i \in \mathbb{R}^{T_i \times 25 \times 3}$:

**Step 1: Frame-Wise Pelvis Root-Centering (`SpineBase`)**  
At each frame $t \in \{1, \dots, T_i\}$, the 3D position vector of `SpineBase` ($\mathbf{P}_i(t, 0) \in \mathbb{R}^3$) is subtracted from all 25 joints:
$$\mathbf{P}_i^{\text{centered}}(t, j) = \mathbf{P}_i(t, j) - \mathbf{P}_i(t, 0), \quad \forall j \in \{0, \dots, 24\}$$
This transformation places the pelvis center at $(0, 0, 0)$ at every frame and makes the skeletal representation invariant to any global 3D translation offset $(\Delta X, \Delta Y, \Delta Z)$.

**Step 2: Frame-Wise 3D Trunk-Height Scaling (`SpineShoulder` to `SpineBase`)**  
Next, the instantaneous 3D Euclidean trunk length $L_i(t)$ between `SpineShoulder` ($j = 1$) and `SpineBase` ($j = 0$) is computed at each frame $t$:
$$L_i(t) = \left\| \mathbf{P}_i(t, 1) - \mathbf{P}_i(t, 0) \right\|_2 = \sqrt{(X_{i,t,1} - X_{i,t,0})^2 + (Y_{i,t,1} - Y_{i,t,0})^2 + (Z_{i,t,1} - Z_{i,t,0})^2}$$
To prevent division by zero during rare depth-sensor drop-outs, any frame with $L_i(t) \le \epsilon$ ($\epsilon = 10^{-6}\text{ m}$) is replaced by the trial's median trunk length $\tilde{L}_i = \text{median}_t L_i(t)$. Every centered joint coordinate is then divided by $L_i(t)$:
$$\tilde{\mathbf{P}}_i(t, j) = \frac{\mathbf{P}_i(t, j) - \mathbf{P}_i(t, 0)}{\max(L_i(t), \epsilon)}$$
After Step 2, all coordinates $\tilde{\mathbf{P}}_i(t, j)$ are expressed in dimensionless **Torso Units (TU)**, where $\tilde{\mathbf{P}}_i(t, 0) \equiv (0, 0, 0)$ and the normalized 3D trunk length $\|\tilde{\mathbf{P}}_i(t, 1)\|_2 \equiv 1.000\text{ TU}$ across all participants, completely removing the $16.6\text{ cm}$ height difference between groups (Figure 2).

![Figure 2: 3D Kinect v2 Skeleton Normalization and Stature Confound Elimination](../05_figures/fig02_kinect_skeleton_normalization.png)
*Figure 2. Illustration of 3D Kinect v2 skeleton normalization on real participant frames (`Typical/1/video/8.xlsx` vs. `Autism/1/video/8.xlsx`): (a) raw meter-scale Kinect v2 camera coordinates influenced by child height and sensor distance; (b) SpineBase-centered and 3D SpineShoulder–SpineBase trunk-scaled coordinates in dimensionless Torso Units (TU); and (c) elimination of the inter-group age ($+1.96\text{ years}, p < 0.001$) and standing height ($+16.6\text{ cm}, p < 0.001$) difference.*

### 5.4 Feature Selection and Extraction

Instead of using 1,259 unnormalized axis statistics [3] or black-box principal components that obscure joint anatomy, the feature extraction module (`src/feature_extraction.py`) computes $d = 86$ domain-specific biomechanical features from each normalized trajectory $\tilde{\mathbf{P}}_i(t, j)$. These 86 features are organized into five clinical movement families:

1. **Three-Dimensional Anatomical Joint Angles (20 features)**:
   For each anatomical joint triad $(a, b, c)$ with vertex at joint $b$, the instantaneous 3D interior angle $\theta_{abc}(t)$ (in degrees) is calculated using the dot product of the adjacent bone segment vectors $\mathbf{u}_{ba}(t) = \tilde{\mathbf{P}}(t, a) - \tilde{\mathbf{P}}(t, b)$ and $\mathbf{u}_{bc}(t) = \tilde{\mathbf{P}}(t, c) - \tilde{\mathbf{P}}(t, b)$:
   $$\theta_{abc}(t) = \arccos\!\left(\text{clip}\!\left(\frac{\mathbf{u}_{ba}(t) \cdot \mathbf{u}_{bc}(t)}{\|\mathbf{u}_{ba}(t)\|_2 \, \|\mathbf{u}_{bc}(t)\|_2}, -1.0, 1.0\right)\right) \cdot \frac{180^\circ}{\pi}$$
   Temporal mean and standard deviation (`std`) are computed across 10 joint angles: Left and Right Knee (`Hip–Knee–Ankle`), Left and Right Hip (`SpineBase–Hip–Knee`), Left and Right Elbow (`Shoulder–Elbow–Wrist`), Left and Right Shoulder (`SpineShoulder–Shoulder–Elbow`), Cervical Neck Flexion (`SpineShoulder–Neck–Head`), and Thoracic Spine Alignment (`SpineBase–Midspain–SpineShoulder`). These features capture sagittal joint stiffness and upper-limb flexion postures [8, 10].
2. **Joint Velocity and Acceleration Dynamics (26 features)**:
   Using consecutive frame time intervals $\Delta t_k = t_k - t_{k-1}$, instantaneous 3D joint speed $v_j(t) = \|\tilde{\mathbf{P}}(t, j) - \tilde{\mathbf{P}}(t-1, j)\|_2 / \Delta t_k$ (in $\text{TU/s}$) and acceleration magnitude $a_j(t) = |v_j(t) - v_j(t-1)| / \Delta t_k$ (in $\text{TU/s}^2$) are computed. Temporal mean speed, speed standard deviation, mean acceleration, and peak speed are extracted across 13 primary joints (`Head`, `Neck`, `SpineShoulder`, `Midspain`, and bilateral `Shoulder`, `Elbow`, `Wrist`, `Hand`, `Hip`, `Knee`, `Ankle`, and `Foot`) to quantify push-off dynamics and movement smoothness [6, 9, 10].
3. **Axial Postural Sway and Dynamic Base of Support (12 features)**:
   This group measures 3D, mediolateral ($X$), and anteroposterior ($Z$) oscillation of the `Head` and `SpineShoulder` relative to `SpineBase` (`posture_head_spine_sway_mean/std`, `posture_trunk_tilt_deg_mean/std`), together with 3D and mediolateral inter-ankle separation (`step_width_3d_mean/std`, `step_width_x_mean/std`) and cadence regularity. These metrics quantify axial balance control and widened gait base [8, 12, 15].
4. **Bilateral Asymmetry and Upper-Limb Stereotypy Distances (14 features)**:
   This group computes absolute left–right differences $|\bar{\theta}_L - \bar{\theta}_R|$ across knee, hip, elbow, and shoulder angles, Contralateral limb cross-distances (e.g., Left Elbow to Right Foot and Left Wrist to Right Ankle), and normalized 3D Euclidean distances from the left and right hands to the head and chest (`dist_hand_head_left/right_mean`, `dist_hand_chest_mean`). These features capture asymmetric arm swing and repetitive hand-to-head or arm-guarding postures associated with ASD [8, 25, 26].
5. **Distal Extremity Spatial Excursion (14 features)**:
   Peak-to-peak range and interquartile dispersion ($\text{IQR} = P_{75} - P_{25}$) of distal limb trajectories (`Wrist`, `HandTip`, `Ankle`, `Foot`) in normalized torso units are computed to measure overall movement amplitude.

For the historical baseline configuration (`CatBoost Hist-300`), an in-fold variance threshold filter (`VarianceThreshold(1e-5)`) is also applied strictly inside each training fold to verify that near-constant features do not affect tree splitting.

### 5.5 Model Selection

To classify the 86 biomechanical features on a cohort of $M \le 100$ subjects, **Ordered Gradient Boosting on Oblivious Decision Trees (`CatBoost`)** [19] is selected as the primary classification model. Standard gradient boosting algorithms (such as XGBoost [23] and LightGBM [24]) construct trees using greedy depthwise or leaf-wise splitting and compute residual gradients $r_{i,m} = -\left[\partial \mathcal{L}(y_i, F(\mathbf{x}_i)) / \partial F(\mathbf{x}_i)\right]_{F=F_{m-1}}$ on the exact same training samples used to fit the preceding trees $F_{m-1}$. On small biomedical datasets, this causes **prediction shift**, a form of internal target leakage where deeper trees progressively memorize noise in the training fold [18, 19].

CatBoost addresses this problem through two mathematical mechanisms:

1. **Ordered Boosting**: CatBoost generates random permutations $\sigma_1, \dots, \sigma_s$ of the training fold and computes the residual gradient for sample $i$ using a supporting model $M_{\sigma(i)-1}$ trained only on the preceding samples $\{k : \sigma(k) < \sigma(i)\}$. This ensures unbiased gradient estimation with conditional expectation $\mathbb{E}[r_i \mid \mathbf{x}_i] = 0$ [19].
2. **Symmetrical Oblivious Decision Trees**: In an oblivious tree of depth $D$, all $2^{l-1}$ nodes at level $l \in \{1, \dots, D\}$ are required to share the exact same feature and split threshold $\mathbb{I}(x_{i, f_l} > \tau_l)$. Each tree $h_m(\mathbf{x}_i)$ is therefore defined by only $D$ split conditions and $2^D$ leaf weights $w_{m,v}$:
   $$h_m(\mathbf{x}_i) = \sum_{v=0}^{2^D - 1} w_{m,v} \prod_{l=1}^{D} \mathbb{I}\!\left(\mathbb{I}(x_{i, f_l} > \tau_l) = b_l(v)\right)$$
   where $b_l(v) \in \{0, 1\}$ is the $l$-th binary bit of leaf index $v$, and leaf weights are regularized by an $L_2$ penalty $\lambda$ (`l2_leaf_reg`). Because a depth-5 oblivious tree contains only 5 split parameters rather than up to $2^5 - 1 = 31$ independent split parameters, it acts as a strong structural regularizer that models multi-joint interactions (such as simultaneous knee extension and head sway) without overfitting individual child recordings.

To ensure a fair and comprehensive comparison, two CatBoost configurations (`CatBoost Tuned` and `CatBoost Hist-300`) are benchmarked against eight standard machine learning baselines spanning gradient boosting (`XGBoost`, `LightGBM / HistGradientBoosting`), bagged and boosted ensembles (`Random Forest`, `AdaBoost`), kernel and linear margin classifiers (`SVM-RBF`, `SVM-Linear`, `Logistic Regression`), and a single `Decision Tree`.

### 5.6 Experimental Methodology

To quantify the exact effect of both within-folder augmentation leakage (Level-1) and cross-folder subject duplication (Level-2), all 10 classifiers are evaluated across a **four-regime (six-protocol) 5-fold cross-validation hierarchy** (`RANDOM_SEED = 42`):

1. **Regime 1 (`R1_Naive_Ungrouped_700`)**: 5-fold `StratifiedKFold` across $N = 700$ trials without subject grouping. This regime replicates the trial-level splitting used in several prior studies [3, 7] and measures the accuracy inflation caused by Level-1 and Level-2 leakage.
2. **Regime 2 (`R2_Nominal_100Subj_SGKF_700`)**: 5-fold `StratifiedGroupKFold` across $N = 700$ trials grouped by the $M = 100$ nominal subject folders (`Child_01` to `Child_100`), combined with subject-level majority voting ($M = 100$). This regime eliminates Level-1 within-folder leakage and provides a direct comparison with prior nominal subject-grouped studies [4, 5].
3. **Regime 3A (`R3A_87Cluster_SGKF_700`)**: 5-fold `StratifiedGroupKFold` across $N = 700$ trials grouped by the $M = 87$ metadata clusters, ensuring that all multi-folder duplicate clusters remain inside a single fold.
4. **Regime 3B (`R3B_Deduplicated_88Subj_SGKF_616`)**: Removes the 12 duplicate subject folders prior to cross-validation and performs 5-fold `StratifiedGroupKFold` across the **$M = 88$ strictly unique children** ($47\text{ ASD}, 41\text{ TD}$; $88 \times 7 = 616$ trials). This regime eliminates both Level-1 and Level-2 leakage while giving every unique child an equal weight of 7 trials.
5. **Regime 4A (`R4A_Original_100_GroupedBy87`)**: Evaluates only the $N = 100$ original unaugmented camera recordings (`video/8.xlsx`) grouped by the $M = 87$ clusters under 5-fold `StratifiedGroupKFold`.
6. **Regime 4B (`R4B_Deduplicated_Original_88`)**: Evaluates only the **$N = 88$ strictly unique, unaugmented original camera recordings (`video/8.xlsx`)** ($47\text{ ASD}, 41\text{ TD}$; 1 trial per child) under 5-fold `StratifiedKFold`, eliminating all synthetic augmentations and all duplicate folders.

### 5.7 System Implementation

The complete pipeline is implemented in modular Python scripts inside the project repository:
- `src/data_preprocessing.py`: Implements raw `.xlsx` coordinate parsing, canonical 25-joint ordering, timestamp conversion, `SpineBase` root-centering, and 3D `SpineShoulder–SpineBase` trunk-height normalization.
- `src/feature_extraction.py`: Computes the 86 biomechanical features for each trial and exports the processed dataset to `data/processed_features.csv` ($700 \times 88$) and `original_unaugmented_100_trials.csv` ($100 \times 88$).
- `src/train.py`: Executes the multi-regime 5-fold cross-validation benchmark across all 10 classifiers, encapsulates all feature scalers (`StandardScaler`) and variance filters (`VarianceThreshold`) inside `scikit-learn` `Pipeline` objects so that parameters are fit exclusively on training folds (`X_train`), and computes out-of-fold trial and subject predictions.
- `src/explainability.py`: Computes exact polynomial-time `TreeSHAP` log-odds attribution matrices [20], two-sided Mann–Whitney $U$ tests, and Cohen's $d$ effect sizes across all 86 features.
- `test_model.py`: Provides automated verification tests confirming that root-centered pelvis coordinates equal $(0, 0, 0)$, normalized trunk lengths equal $1.000\text{ TU}$, and zero subject or cluster overlap occurs between any training and validation split.

### 5.8 Evaluation Procedures

Model performance is evaluated at both the **trial level** ($N = 700, 616, 100,$ or $88$) and the **subject level** ($M = 100, 88,$ or $87$). To quantify uncertainty on small subject cohorts without relying on Gaussian assumptions, **2,000-iteration stratified subject-cluster bootstrap resampling** (`seed = 42`) is performed to compute 95% percentile confidence intervals ($[P_{2.5}, P_{97.5}]$) for subject-level accuracy and $\text{AUROC}$.

To test whether performance differences between `CatBoost (Tuned)` and each competing baseline are statistically significant, **two-sided exact binomial McNemar tests** are conducted on paired out-of-fold predictions using the discordant error counts $(b, c)$:
$$p_{\text{McNemar}} = 2 \sum_{k=\max(b,c)}^{b+c} \binom{b+c}{k} 0.5^{b+c}$$
For biomechanical biomarker verification, each feature's discriminative power is evaluated using its mean absolute TreeSHAP value $\mathbb{E}[|\phi_j|]$, standardized mean difference (**Cohen's $d$** using pooled standard deviation), and two-sided non-parametric **Mann–Whitney $U$ test** $p$-value.

---

## 6. Experimentation

### 6.1 Experimental Environment

All data preprocessing, forensic coordinate audits, multi-regime cross-validation experiments, and TreeSHAP analyses were executed deterministically on a Windows 10/11 (x86_64) workstation using Python 3.12 inside an isolated virtual environment (`.venv`). The software stack used pinned library versions: `numpy` (2.2.6 / 2.4.2), `pandas` (2.3.3), `scipy` (1.15.3 / 1.17.0), `scikit-learn` (1.7.2 / 1.8.0), `catboost` (1.2.10), `xgboost` (3.2.0), `shap` (0.49.1 / 0.51.0), `matplotlib` (3.10.9), `seaborn` (0.13.2), and `openpyxl` (3.1.5). Every random number generator across data splitting, classifier initialization, and bootstrap resampling was fixed to `seed = 42`.

### 6.2 Dataset and Forensic Audit Findings

The experimental evaluation uses the Al-Jubouri et al. (2020) 3D Kinect v2 dataset (`DOI: 10.5061/dryad.s7h44j150`) [2, 3]. Before running machine learning experiments, three forensic audits were conducted on the raw dataset files (`Dataset-2/Dataset/readme.docx`, the 800 raw `.xlsx` trajectory files, and `Final dataset.xlsx`).

#### 6.2.1 Demographic and Physical Height Audit

Table 2 presents the demographic and anthropometric comparison between the TD control cohort and the ASD walking cohort extracted from the official dataset registry (`readme.docx`) and raw 3D skeletal coordinates.

**Table 2. Demographic characteristics and physical stature audit of the Al-Jubouri (2020) Kinect v2 cohort.**

| Demographic / Anthropometric Variable | Typically Developing (TD) Cohort ($N = 50$ Nominal) | Autism Spectrum Disorder (ASD) Walking Cohort ($N = 50$ Nominal) | Severe ASD Cohort (Video Only, Excluded) ($N = 9$) | Two-Sample Statistical Comparison (ASD Walking vs. TD) |
| :--- | :--- | :--- | :--- | :--- |
| **Nominal Subject Folders** | `Typical/1`–`50` (`Child_01`–`Child_50`) | `Autism/children with ASD/1`–`50` (`Child_51`–`Child_100`) | `Severe autism/1`–`9` (No `.xlsx` 3D coordinates) | $100$ nominal 3D skeleton folders |
| **Strictly Unique Coordinate Subjects** | $41$ unique children ($9$ duplicate folders) | $47$ unique children ($3$ duplicate folders) | Not applicable | $88$ unique 3D skeleton children |
| **Sex Distribution (Male / Female)** | $26\text{ M } (52.0\%) \text{ / } 24\text{ F } (48.0\%)$ | $42\text{ M } (84.0\%) \text{ / } 8\text{ F } (16.0\%)$ | $5\text{ M } (55.6\%) \text{ / } 4\text{ F } (44.4\%)$ | $\chi^2 = 11.76, p = 0.0006$ |
| **Chronological Age (Years, Mean $\pm$ SD [Range])** | $6.66 \pm 3.08\text{ yrs } [2.0\text{--}14.0]$ | $8.62 \pm 2.19\text{ yrs } [5.0\text{--}15.0]$ | $8.89 \pm 2.93\text{ yrs } [5.0\text{--}14.0]$ | $\Delta = +1.96\text{ yrs}, t = 3.67, p = 0.0004$ |
| **Recorded Standing Height (Meters, Mean $\pm$ SD)** | $1.158 \pm 0.195\text{ m } (3.75 \pm 0.67\text{ ft})$ | $1.324 \pm 0.154\text{ m } [0.976\text{--}1.707\text{ m}]$ | $1.318 \pm 0.231\text{ m}$ | $\Delta = +0.166\text{ m } (+16.6\text{ cm}), t = 4.72, p < 0.0001$ |
| **Raw 3D Trunk Length $\|\text{SpineShoulder} - \text{SpineBase}\|_2$** | $0.281 \pm 0.041\text{ m}$ | $0.324 \pm 0.037\text{ m}$ | Not applicable | $\Delta = +0.043\text{ m } (+15.3\%), p < 0.0001$ |
| **Normalized 3D Trunk Length (After Step 2 Scaling)** | $1.000 \pm 0.000\text{ TU}$ | $1.000 \pm 0.000\text{ TU}$ | Not applicable | $\Delta = 0.000\text{ TU } (p = 1.000$, bias removed) |

As shown in Table 2, the ASD walking group is $1.96\text{ years}$ older ($p = 0.0004$) and $16.6\text{ cm}$ taller ($p < 0.0001$) than the TD group, corresponding to a 15.3% longer raw 3D trunk length ($0.324\text{ m}$ vs. $0.281\text{ m}$). After applying the proposed Step 2 normalization, the 3D trunk length is standardized to $1.000 \pm 0.000\text{ TU}$ in both groups ($p = 1.000$).

#### 6.2.2 Audit of the Seven Geometric Augmentations (Level-1 Leakage)

Direct algebraic comparison between `augmentation/1..7` and the original camera capture `video/8.xlsx` across all 100 folders identified the exact transformation formulas used to create the 7 augmentations:
- **Augmentation 1 (`1.xlsx`)**: Mediolateral translation $X' = X + 0.05\text{ m}, Y' = Y, Z' = Z$.
- **Augmentation 2 (`2.xlsx`)**: Mediolateral translation $X' = X + 0.35\text{ m}, Y' = Y, Z' = Z$.
- **Augmentation 3 (`3.xlsx`)**: Mediolateral translation $X' = X - 0.25\text{ m}, Y' = Y, Z' = Z$.
- **Augmentation 4 (`4.xlsx`)**: 3D camera pitch rotation and perspective perturbation.
- **Augmentation 5 (`5.xlsx`)**: Sagittal mirror reflection $X' = -X, Y' = Y, Z' = Z$.
- **Augmentation 6 (`6.xlsx`)**: Mediolateral scaling compression $X' = 0.7 X, Y' = Y, Z' = Z$.
- **Augmentation 7 (`7.xlsx`)**: Mediolateral scaling expansion $X' = 1.6 X, Y' = Y, Z' = Z$.

Because Step 1 root-centering subtracts $\mathbf{P}_i(t, \text{SpineBase})$ from all 25 joints at every frame, any constant translation vector $\mathbf{c} = (\Delta X, 0, 0)$ cancels out exactly:
$$(\mathbf{P}_i(t, j) + \mathbf{c}) - (\mathbf{P}_i(t, 0) + \mathbf{c}) = \mathbf{P}_i(t, j) - \mathbf{P}_i(t, 0)$$
Consequently, **Augmentations 1, 2, and 3 produce normalized 86-feature vectors that are identical to the original trial (`video/8.xlsx`) up to floating-point precision ($\max |\Delta| \le 1.92 \times 10^{-13}$)**, and **Augmentation 5 ($-X$ reflection)** produces identical values for all symmetric and magnitude features. Only Augmentations 4, 6, and 7 introduce non-trivial changes to 3D joint angles and mediolateral speeds.

#### 6.2.3 Audit of Cross-Folder Coordinate Duplicates (Level-2 Leakage)

Computing the $100 \times 100$ pairwise Pearson correlation matrix and maximum coordinate difference across all 100 subject folders (`Child_01` to `Child_100`) revealed that **12 subject folders are exact coordinate clones ($r = 1.0000$, $\max |\Delta| = 0.0$) of 7 primary folders within the same diagnostic class**. Table 3 catalogs these 7 multi-folder duplicate clusters and 1 metadata rounding collision.

**Table 3. Catalog of the 7 cross-folder coordinate duplicate clusters (12 clone folders) and 1 metadata collision in the Al-Jubouri (2020) Kinect v2 dataset.**

| Cluster Identifier | Diagnostic Class | Primary Subject Folder | Exact Duplicate (Clone) Subject Folders | Total Folders in Cluster | Coordinate Verification Details |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`TD_Cluster_00`** | TD ($0$) | `Child_01` (`Typical/1`) | `Child_07` (`Typical/7`), `Child_16` (`Typical/16`), `Child_20` (`Typical/20`), `Child_29` (`Typical/29`) | $5$ folders ($1$ primary + $4$ clones) | Identical raw 3D joint trajectories ($r = 1.0000, \max |\Delta| = 0.0$) across `video/8.xlsx` and `augmentation/1..7` |
| **`TD_Cluster_01`** | TD ($0$) | `Child_02` (`Typical/2`) | `Child_26` (`Typical/26`), `Child_40` (`Typical/40`) | $3$ folders ($1$ primary + $2$ clones) | Identical raw 3D joint trajectories ($r = 1.0000, \max |\Delta| = 0.0$) |
| **`TD_Cluster_03`** | TD ($0$) | `Child_04` (`Typical/4`) | `Child_23` (`Typical/23`), `Child_28` (`Typical/28`) | $3$ folders ($1$ primary + $2$ clones) | Identical raw 3D joint trajectories ($r = 1.0000, \max |\Delta| = 0.0$) |
| **`TD_Cluster_12`** | TD ($0$) | `Child_13` (`Typical/13`) | `Child_14` (`Typical/14`) | $2$ folders ($1$ primary + $1$ clone) | Identical raw 3D joint trajectories ($r = 1.0000, \max |\Delta| = 0.0$) |
| **`ASD_Cluster_50`** | ASD ($1$) | `Child_51` (`ASD Case 1`) | `Child_84` (`ASD Case 34`) | $2$ folders ($1$ primary + $1$ clone) | Identical raw 3D joint trajectories ($r = 1.0000, \max |\Delta| = 0.0$) |
| **`ASD_Cluster_56`** | ASD ($1$) | `Child_57` (`ASD Case 7`) | `Child_68` (`ASD Case 18`) | $2$ folders ($1$ primary + $1$ clone) | Identical raw 3D joint trajectories ($r = 1.0000, \max |\Delta| = 0.0$) |
| **`ASD_Cluster_79`** | ASD ($1$) | `Child_80` (`ASD Case 30`) | `Child_90` (`ASD Case 40`) | $2$ folders ($1$ primary + $1$ clone) | Identical raw 3D joint trajectories ($r = 1.0000, \max |\Delta| = 0.0$) |
| **Total Coordinate Clones** | **Both** | **$7$ Primary Folders** | **$12$ Exact Clone Folders ($9\text{ TD}, 3\text{ ASD}$)** | **$19$ Folders $\rightarrow 7$ Unique** | **$100 - 12 = 88$ Strictly Unique Coordinate Subjects ($47\text{ ASD}, 41\text{ TD}$)** |
| **`ASD_Metadata_Collision`** | ASD ($1$) | `Child_75` (`ASD Case 25`, Block 24) | `Child_93` (`ASD Case 43`, Block 42) | $2$ distinct recordings | Distinct 3D gait trajectories ($r = 0.41$), but share rounded `mean-x-Midspain = -0.09342` in `Final dataset.xlsx` ($\rightarrow 87$ metadata clusters) |

As shown in Table 3, removing the 12 duplicate folders leaves **88 strictly unique children** (47 ASD and 41 TD). In addition, two distinct ASD recordings (`Child_75` and `Child_93`) share the same 5-decimal rounded value of `mean-x-Midspain = -0.09342` in `Final dataset.xlsx`, which explains why grouping `Final dataset.xlsx` by that column produces **87 metadata clusters** (46 ASD and 41 TD).

### 6.3 Experimental Setup

Two feature tables were constructed for experimentation:
1. `data/processed_features.csv`: Contains $N = 700$ rows (100 nominal subject folders $\times 7$ augmentation trials per folder) and 86 normalized biomechanical features (`R1`, `R2`, `R3A`), which reduces to $N = 616$ rows ($88$ unique children $\times 7$ trials) after removing the 12 clone folders (`R3B`).
2. `original_unaugmented_100_trials.csv`: Contains the 86 normalized biomechanical features extracted exclusively from the $N = 100$ unaugmented camera recordings (`video/8.xlsx`, `R4A`), which reduces to $N = 88$ rows (1 original recording per unique child) after removing the 12 clone folders (`R4B`).

### 6.4 Models and Algorithms Evaluated

Ten machine learning classifiers were evaluated across identical cross-validation splits (`seed = 42`):
1. **`CatBoost (Tuned)`**: `CatBoostClassifier(iterations=500, depth=5, learning_rate=0.04, l2_leaf_reg=4.0, random_seed=42)`
2. **`CatBoost (Hist-300)`**: `Pipeline([("vt", VarianceThreshold(1e-5)), ("clf", CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=42))])`
3. **`XGBoost`**: `XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=4.0, random_state=42)`
4. **`LightGBM (HistGB)`**: `HistGradientBoostingClassifier(max_iter=300, max_depth=5, learning_rate=0.05, l2_regularization=4.0, random_state=42)`
5. **`Random Forest`**: `RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=4, random_state=42)`
6. **`AdaBoost`**: `AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)`
7. **`SVM (RBF)`**: `Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=42))])`
8. **`SVM (Linear)`**: `Pipeline([("scaler", StandardScaler()), ("clf", SVC(C=0.1, kernel="linear", probability=True, random_state=42))])`
9. **`Logistic Regression`**: `Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(C=0.1, penalty="l2", max_iter=1000, random_state=42))])`
10. **`Decision Tree`**: `DecisionTreeClassifier(max_depth=5, min_samples_leaf=4, random_state=42)`

### 6.5 Evaluation Metrics

To evaluate classification discrimination, class balance, and probability calibration, the following metrics were computed across all folds and out-of-fold (OOF) predictions:
- **Classification Accuracy**: $\text{Acc} = \frac{TP + TN}{TP + TN + FP + FN}$
- **Balanced Accuracy**: $\text{BalAcc} = \frac{1}{2}\left(\frac{TP}{TP + FN} + \frac{TN}{TN + FP}\right)$
- **Sensitivity (Recall)** and **Specificity**: $\text{Sens} = \frac{TP}{TP + FN}$, $\text{Spec} = \frac{TN}{TN + FP}$
- **Precision** and **F1-Score**: $\text{Prec} = \frac{TP}{TP + FP}$, $F_1 = \frac{2 \cdot \text{Prec} \cdot \text{Sens}}{\text{Prec} + \text{Sens}}$
- **Threshold-Independent Ranking Metrics**: Area Under the Receiver Operating Characteristic Curve ($\text{AUROC}$) and Area Under the Precision–Recall Curve ($\text{PR-AUC}$).
- **Agreement and Calibration Metrics**: Cohen's Kappa ($\kappa$), Matthews Correlation Coefficient ($\text{MCC}$), and Out-of-Fold Brier Calibration Score ($\text{Brier} = \frac{1}{N}\sum_{i=1}^N (\hat{p}_i - y_i)^2$, where lower values indicate better calibration).

### 6.6 Testing Procedure and Validation Strategy

For each of the six validation regimes (`R1`, `R2`, `R3A`, `R3B`, `R4A`, `R4B`), 5-fold cross-validation was executed such that every trial $i$ received an out-of-fold probability prediction $\hat{p}_i^{\text{OOF}}$ from a model trained without that trial's fold. In the grouped regimes (`R2`, `R3A`, `R3B`, `R4A`), `StratifiedGroupKFold(n_splits=5)` guaranteed that all trials from the same subject or cluster were assigned exclusively to either the training set or the validation set in every fold. In addition, an 8-seed sensitivity experiment (`seeds = [42, 0, 1, 7, 13, 21, 99, 123]`) was conducted on a 25% stratified cluster holdout ($N = 22$ clusters out of 87) to test the stability of single-split holdout evaluation.

---

## 7. Results and Discussion

### 7.1 Performance Under Nominal Subject-Grouped Cross-Validation (Regime 2)

Table 4 presents the quantitative performance of all 10 classifiers under `Regime 2` (5-fold `StratifiedGroupKFold` grouped by the 100 nominal subject folders, $N = 700$ trials across $M = 100$ nominal children), including 2,000-iteration subject-cluster bootstrap 95% confidence intervals and exact binomial McNemar test $p$-values against `CatBoost (Tuned)`.

**Table 4. Performance comparison of 10 classifiers under 5-fold nominal subject-grouped cross-validation (`Regime 2`, $N = 700$ trials across $M = 100$ nominal children, `seed = 42`).**

| Classifier | Trial Acc ($\text{Mean} \pm \text{SD}$ %) | Trial Bal. Acc (%) | Trial AUROC ($\text{Mean} \pm \text{SD}$) | Trial PR-AUC ($\text{Mean} \pm \text{SD}$) | Sens. (%) | Spec. (%) | F1-Score | OOF Brier Score | Subject Majority Acc ($M=100$) [95% CI] | Subject AUROC [95% CI] | McNemar $p$-value vs. `CatBoost (Tuned)` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CatBoost (Tuned)`** | **$92.57 \pm 2.70$** | **$92.57$** | **$0.9802 \pm 0.0251$** | **$0.9785 \pm 0.0284$** | **$90.29$** | $94.86$ | **$0.9233$** | **$0.0618$** | **$93.00\%$ ($93/100$) [$88.0, 97.0$]** | **$0.9804$ [$0.950, 0.999$]** | Reference ($1.000$) |
| **`CatBoost (Hist-300)`** | $91.43 \pm 2.35$ | $91.43$ | $0.9779 \pm 0.0211$ | $0.9765 \pm 0.0250$ | $88.29$ | $94.57$ | $0.9105$ | $0.0737$ | $92.00\%$ ($92/100$) [$86.0, 97.0$] | $0.9776$ [$0.951, 0.998$] | $p = 0.0574$ |
| **`AdaBoost`** | $91.00 \pm 3.82$ | $91.00$ | $0.9659 \pm 0.0230$ | $0.9657 \pm 0.0275$ | $86.57$ | **$95.43$** | $0.9035$ | $0.1496$ | $91.00\%$ ($91/100$) [$84.0, 96.0$] | $0.9652$ [$0.904, 0.990$] | $p = 3.24 \times 10^{-4}$** |
| **`LightGBM (HistGB)`** | $89.86 \pm 4.77$ | $89.86$ | $0.9644 \pm 0.0329$ | $0.9613 \pm 0.0438$ | $84.57$ | $95.14$ | $0.8919$ | $0.0810$ | $90.00\%$ ($90/100$) [$84.0, 95.0$] | $0.9652$ [$0.916, 0.990$] | $p = 5.46 \times 10^{-4}$** |
| **`Random Forest`** | $89.57 \pm 5.32$ | $89.57$ | $0.9635 \pm 0.0429$ | $0.9637 \pm 0.0440$ | $87.71$ | $91.43$ | $0.8933$ | $0.0868$ | $91.00\%$ ($91/100$) [$84.0, 95.0$] | $0.9692$ [$0.935, 0.992$] | $p = 1.02 \times 10^{-3}$* |
| **`XGBoost`** | $89.14 \pm 6.61$ | $89.14$ | $0.9694 \pm 0.0330$ | $0.9671 \pm 0.0398$ | $84.57$ | $93.71$ | $0.8863$ | $0.0756$ | $89.00\%$ ($89/100$) [$84.0, 96.0$] | $0.9712$ [$0.937, 0.995$] | $p = 5.46 \times 10^{-4}$** |
| **`SVM (RBF)`** | $89.00 \pm 3.82$ | $89.00$ | $0.9720 \pm 0.0139$ | $0.9735 \pm 0.0130$ | $86.29$ | $91.71$ | $0.8841$ | $0.0758$ | $88.00\%$ ($88/100$) [$81.0, 94.0$] | $0.9752$ [$0.950, 0.994$] | $p = 7.62 \times 10^{-3}$* |
| **`Logistic Regression`** | $88.29 \pm 3.88$ | $88.29$ | $0.9506 \pm 0.0310$ | $0.9601 \pm 0.0217$ | $82.57$ | $94.00$ | $0.8733$ | $0.0887$ | $88.00\%$ ($88/100$) [$81.0, 93.0$] | $0.9608$ [$0.894, 0.977$] | $p = 7.24 \times 10^{-6}$*** |
| **`SVM (Linear)`** | $86.86 \pm 4.62$ | $86.86$ | $0.9156 \pm 0.0685$ | $0.9347 \pm 0.0459$ | $82.00$ | $91.71$ | $0.8592$ | $0.1162$ | $87.00\%$ ($87/100$) [$77.0, 92.0$] | $0.9340$ [$0.862, 0.966$] | $p = 1.20 \times 10^{-10}$*** |
| **`Decision Tree`** | $79.86 \pm 10.86$ | $79.86$ | $0.8077 \pm 0.1154$ | $0.7641 \pm 0.1312$ | $76.86$ | $82.86$ | $0.7872$ | $0.1940$ | $80.00\%$ ($80/100$) [$72.0, 87.0$] | $0.8668$ [$0.789, 0.939$] | $p = 1.46 \times 10^{-19}$*** |

*\* $p < 0.01$, \*\* $p < 0.001$, \*\*\* $p < 0.0001$ by two-sided exact binomial McNemar test on paired out-of-fold predictions.*

The experimental results in Table 4 indicate that `CatBoost (Tuned)` achieves the highest performance across all primary evaluation metrics under `Regime 2`. Specifically, `CatBoost (Tuned)` reaches $92.57\% \pm 2.70\%$ trial-level accuracy ($\text{OOF}: TN = 332, FP = 18, FN = 34, TP = 316$; Cohen's $\kappa = 0.8514$, $\text{MCC} = 0.8523$), $93.00\%$ ($93/100$: $TN = 48, FP = 2, FN = 5, TP = 45$) subject-level majority-vote accuracy, and $\text{AUROC} = 0.9802 \pm 0.0251$ ($\text{Subject AUROC} = 0.9804$). Across the five subject-disjoint folds, `CatBoost (Tuned)` maintains consistent performance: Fold 1 ($\text{Acc} = 95.71\%, \text{AUROC} = 0.9971$), Fold 2 ($\text{Acc} = 93.57\%, \text{AUROC} = 0.9971$), Fold 3 ($\text{Acc} = 92.86\%, \text{AUROC} = 0.9906$), Fold 4 ($\text{Acc} = 88.57\%, \text{AUROC} = 0.9371$), and Fold 5 ($\text{Acc} = 92.14\%, \text{AUROC} = 0.9792$).

Furthermore, `CatBoost` exhibits substantially lower fold-to-fold accuracy variance ($\text{SD} = 2.70\%$ for `CatBoost Tuned` and $2.35\%$ for `CatBoost Hist-300`) than depthwise gradient boosting models such as `XGBoost` ($\text{SD} = 6.61\%$) and `LightGBM` ($\text{SD} = 4.77\%$), as well as `Random Forest` ($\text{SD} = 5.32\%$). In addition, `CatBoost (Tuned)` achieves the lowest out-of-fold Brier calibration error ($0.0618$), indicating well-calibrated probability estimates compared to `AdaBoost` ($\text{Brier} = 0.1496$) and `Decision Tree` ($\text{Brier} = 0.1940$). Exact binomial McNemar tests confirm that the accuracy improvement of `CatBoost (Tuned)` over every non-CatBoost classifier is statistically significant ($p \le 7.62 \times 10^{-3}$). Compared to prior studies using nominal 100-subject validation on the same dataset, the proposed pipeline improves subject-level classification accuracy by $+6.00\%$ over the AdaBoost model of Yazdi et al. (2024) [4] ($87.0\%$) and by $+6.64\%$ over the Random Forest model of Fırat (2026) [5] ($86.36\%$).

Figures 3 through 8 illustrate the comparative discrimination, precision-recall behavior, confusion matrices, multi-metric benchmarks, and probability calibration across the evaluated models.

![Figure 3: Multi-Regime Accuracy and ROC-AUC Comparison](../05_figures/fig03_leakage_and_regime_comparison.png)
*Figure 3. Quantitative impact of two-level data leakage across all 10 classifiers: (a) cross-validated classification accuracy across `R1` (Naive ungrouped SKF), `R2` (Nominal 100-subject SGKF), `R3B` (Strictly deduplicated 88-subject SGKF), and `R4B` (Unaugmented original-only $N=88$ recordings); and (b) complete 10-model by 6-regime ROC-AUC heatmap.*

![Figure 4: Cross-Validated ROC Curves](../05_figures/fig04_roc_curves_multimodel.png)
*Figure 4. Five-fold cross-validated Receiver Operating Characteristic (ROC) curves showing the true positive rate against the false positive rate under `Regime 2`.*

![Figure 5: Cross-Validated Precision-Recall Curves](../05_figures/fig05_precision_recall_curves.png)
*Figure 5. Five-fold cross-validated Precision–Recall (PR) curves under `Regime 2`, illustrating high precision across sensitivity thresholds.*

![Figure 6: Out-of-Fold Confusion Matrices Across Validation Regimes](../05_figures/fig06_confusion_matrices.png)
*Figure 6. Out-of-fold confusion matrices for CatBoost across four key evaluation levels: (a) `R2` Trial-level ($N=700$), (b) `R2` Subject majority vote ($M=100$), (c) `R3B` Strictly deduplicated children ($M=88$), and (d) `R4B` Unaugmented original recordings ($N=88$).*

![Figure 7: Multi-Metric Classifier Benchmark](../05_figures/fig07_metrics_benchmark_bar.png)
*Figure 7. Multi-metric bar chart comparing accuracy, balanced accuracy, sensitivity, specificity, F1-score, and ROC-AUC across classifier families under `Regime 2`.*

![Figure 8: Out-of-Fold Probability Distribution and Calibration](../05_figures/fig08_probability_calibration.png)
*Figure 8. Out-of-fold predicted probability distribution and calibration profile for `CatBoost (Tuned)`, demonstrating strong class separation and low Brier calibration error ($0.0618$).*

### 7.2 Multi-Regime Leakage Evaluation and Deduplicated Performance (Regimes 1, 3A, 3B, 4A, and 4B)

While `Regime 2` provides a direct comparison with prior 100-folder studies [4, 5], the forensic audit established that 12 of the 100 folders are exact coordinate clones (Table 3). Table 5 presents the classification accuracy and $\text{AUROC}$ of all 10 models across all six validation regimes (`R1` through `R4B`) to quantify the exact performance shift at each level of leakage control.

**Table 5. Multi-regime classification accuracy and AUROC comparison across all six validation protocols (5-fold cross-validation, `seed = 42`).**

| Classifier | `R1`: Naive Ungrouped ($N=700$, Leaked) Acc% / AUROC | `R2`: Nominal 100-Subj SGKF ($N=700, M=100$) Acc% / AUROC | `R3A`: 87-Cluster SGKF ($N=700, M=87$) Acc% / AUROC | `R3B`: Deduplicated 88-Subj SGKF ($N=616, M=88$) Acc% / AUROC | `R4A`: Original-Only 100 ($N=100, M=87$) Acc% / AUROC | `R4B`: Deduplicated Original-Only ($N=88, M=88$) Acc% / AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`CatBoost (Tuned)`** | $99.00\% \text{ / } 0.9997$ | **$92.57\% \text{ / } 0.9802$** | **$86.71\% \text{ / } 0.9320$** | $88.59\% \text{ / } 0.9717$ | $83.00\% \text{ / } 0.9120$ | $87.45\% \text{ / } 0.9243$ |
| **`CatBoost (Hist-300)`** | **$99.14\% \text{ / } 0.9995$** | $91.43\% \text{ / } 0.9779$ | $86.14\% \text{ / } 0.9258$ | $87.49\% \text{ / } 0.9689$ | $89.00\% \text{ / } 0.9360$ | $89.74\% \text{ / } 0.9502$ |
| **`SVM (RBF)`** | $97.71\% \text{ / } 0.9992$ | $89.00\% \text{ / } 0.9720$ | $86.57\% \text{ / } 0.9238$ | **$90.53\% \text{ / } 0.9751$** | $88.00\% \text{ / } 0.9040$ | $89.74\% \text{ / } \mathbf{0.9622}$ |
| **`XGBoost`** | $99.14\% \text{ / } 0.9997$ | $89.14\% \text{ / } 0.9694$ | $86.43\% \text{ / } 0.9076$ | $89.54\% \text{ / } 0.9683$ | **$94.00\% \text{ / } 0.9560$** | $90.85\% \text{ / } 0.9588$ |
| **`LightGBM (HistGB)`** | $98.71\% \text{ / } 0.9993$ | $89.86\% \text{ / } 0.9644$ | $85.57\% \text{ / } 0.9080$ | $87.41\% \text{ / } 0.9670$ | $92.00\% \text{ / } 0.9560$ | $90.92\% \text{ / } 0.9533$ |
| **`Random Forest`** | $97.71\% \text{ / } 0.9986$ | $89.57\% \text{ / } 0.9635$ | $84.43\% \text{ / } 0.9079$ | $88.20\% \text{ / } 0.9618$ | $87.00\% \text{ / } 0.9220$ | $91.90\% \text{ / } 0.9385$ |
| **`AdaBoost`** | $98.86\% \text{ / } 0.9996$ | $91.00\% \text{ / } 0.9659$ | $86.71\% \text{ / } 0.9216$ | $86.05\% \text{ / } 0.9618$ | $86.00\% \text{ / } 0.9400$ | $88.63\% \text{ / } 0.9560$ |
| **`Logistic Regression`** | $97.29\% \text{ / } 0.9927$ | $88.29\% \text{ / } 0.9506$ | $85.14\% \text{ / } 0.9256$ | $87.82\% \text{ / } 0.9656$ | $87.00\% \text{ / } 0.8920$ | **$92.09\% \text{ / } 0.9127$** |
| **`SVM (Linear)`** | $97.71\% \text{ / } 0.9923$ | $86.86\% \text{ / } 0.9156$ | $79.57\% \text{ / } 0.8996$ | $86.79\% \text{ / } 0.9472$ | $84.00\% \text{ / } 0.9040$ | $92.09\% \text{ / } 0.9336$ |
| **`Decision Tree`** | $96.29\% \text{ / } 0.9709$ | $79.86\% \text{ / } 0.8077$ | $69.71\% \text{ / } 0.7036$ | $82.18\% \text{ / } 0.8314$ | $80.00\% \text{ / } 0.7960$ | $81.90\% \text{ / } 0.8831$ |

Based on the evaluation in Table 5 and Figure 3, several important observations can be made regarding data leakage and model generalization:

1. **Effect of Level-1 Within-Folder Augmentation Leakage (`R1` vs. `R2`)**: When trials are partitioned randomly without subject grouping (`R1`), all 10 classifiers achieve between $96.29\%$ and $99.14\%$ accuracy ($\text{AUROC} = 0.9709\text{--}0.9997$). This near-perfect performance occurs because Augmentations 1, 2, 3, and 5 are translation- or reflection-invariant copies of the same child's walking trial. When subject grouping is enforced in `R2`, accuracy drops by $6.43\%$ for `CatBoost (Tuned)` ($99.00\% \rightarrow 92.57\%$), $10.00\%$ for `XGBoost` ($99.14\% \rightarrow 89.14\%$), and $16.43\%$ for `Decision Tree` ($96.29\% \rightarrow 79.86\%$). This result is further illustrated in Figure 11.
2. **Effect of Level-2 Cross-Folder Clone Leakage (`R2` vs. `R3A` and `R3B`)**:
   - In `R3A`, where all 700 trials are grouped by the 87 metadata clusters without removing duplicate folders, `CatBoost (Tuned)` ranks first among all 10 models in trial accuracy ($86.71\% \pm 13.99\%$), trial $\text{AUROC}$ ($0.9320 \pm 0.0811$), subject accuracy ($88.00\%$), and subject $\text{AUROC}$ ($0.9488$). However, keeping 5-folder clusters (`TD_Cluster_00`, 35 trials) and 3-folder clusters (`TD_Cluster_01` and `TD_Cluster_03`, 21 trials each) intact inside validation folds introduces large class-ratio and sample-size imbalances across the 5 folds, resulting in higher standard deviations ($\pm 13.99\%$).
   - When the 12 duplicate folders are removed prior to partitioning in **`R3B` ($M = 88$ unique children, $N = 616$ trials)**, every unique child is weighted equally (7 trials per child). Under `R3B`, fold variance decreases substantially ($\text{SD} = 6.14\%$), and `CatBoost (Tuned)` achieves $88.59\% \pm 6.14\%$ trial accuracy ($88.64\%$ [$78/88$] subject accuracy, $TN = 37, FP = 4, FN = 6, TP = 41$) and $\text{AUROC} = 0.9717 \pm 0.0237$ ($\text{PR-AUC} = 0.9751$). Under `R3B`, `SVM (RBF)` ($90.53\%, \text{AUROC} = 0.9751$) and `XGBoost` ($89.54\%, \text{AUROC} = 0.9683$) also achieve strong generalization.
3. **Performance on Unaugmented Camera Recordings (`R4A` and `R4B`)**: Evaluating strictly on the 88 unique original camera recordings (`video/8.xlsx` in `R4B`, with zero synthetic augmentation), the shallower depth-4 **`CatBoost (Hist-300)` achieves $89.74\% \pm 8.56\%$ accuracy ($89.77\%$ OOF: $TN = 35, FP = 6, FN = 3, TP = 44$) and $\text{AUROC} = 0.9502 \pm 0.0453$** ($\text{Subject OOF AUROC} = 0.9543$). In this small-sample unaugmented regime ($\sim 70$ training samples per fold), regularized linear and kernel classifiers (`Logistic Regression`: $92.09\%$, `SVM-Linear`: $92.09\%$, `SVM-RBF`: $\text{AUROC} = 0.9622$) and shallow ensembles (`Random Forest`: $91.90\%$, `XGBoost`: $90.85\%$) perform slightly better than the deeper depth-5 `CatBoost (Tuned)` ($87.45\%$). This finding indicates that stronger model constraints (`depth = 4` or linear/RBF margins) reduce variance when training on fewer than 75 unaugmented samples. Crucially, achieving $89.74\%\text{--}92.09\%$ accuracy on `R4B` contrasts sharply with the $60.0\%$ unaugmented Rough Set accuracy reported by Al-Jubouri et al. (2021) [3], demonstrating that the proposed 3D trunk normalization and 86 biomechanical features extract strong diagnostic signal directly from raw Kinect v2 recordings without requiring synthetic augmentation.

### 7.3 Reconciliation of Historical Holdout and Preliminary Metrics

During initial repository development, several preliminary metrics were recorded (`5-Fold SGKF ROC-AUC: 0.919 ± 0.072`, `Balanced Accuracy: ~80.3%`, `Holdout AUC: 1.000, N = 22`, and `Original-only AUC: 0.975`). Table 6 summarizes the forensic reconciliation of these values against the verified multi-regime benchmark.

**Table 6. Reconciliation of historical preliminary metrics against the verified multi-regime benchmark.**

| Historical Metric | Forensic Origin in Repository and Dataset Audit | Replicated Value in Present Study | Methodological Interpretation |
| :--- | :--- | :--- | :--- |
| **`5-Fold SGKF ROC-AUC: 0.919 ± 0.072`** & **`Balanced Accuracy: ~80.3%`** | `CatBoost (Hist-300)` (`d=4, iter=300, lr=0.05, l2=3, VT=1e-5`) evaluated under un-deduplicated 87/88-cluster 5-fold SGKF | **87-Cluster Raw `Final dataset.xlsx`**: $\text{AUROC} = 0.8968 \pm 0.1371$ ($\text{Seed 99}: 0.9237 \pm 0.0744$), $\text{BalAcc} = 79.88\%\text{--}82.01\%$; **88-Cluster Normalized**: $\text{AUROC} = 0.9322 \pm 0.0429$, $\text{BalAcc} = 79.43\%$ | Leaving 5-folder (`35`-trial) and 3-folder (`21`-trial) duplicate clusters un-deduplicated skews fold class balances and decision thresholds. Deduplicating first (`R3B`) restores $\text{BalAcc} = 88.80\%$ and $\text{AUROC} = 0.9717$. |
| **`Holdout AUC: 1.000, N = 22`** | 25% stratified cluster holdout of the 87 metadata clusters ($87 \times 0.25 = 21.75 \rightarrow 22\text{ clusters}$, $65$ train clusters) under `seed = 42` | **`seed = 42` ($N=22$ clusters)**: $\text{Holdout AUROC} = 1.0000$ ($\text{BalAcc} = 95.00\%$). Across 8 random seeds (`42, 0, 1, 7, 13, 21, 99, 123`), Holdout AUROC ranges from **$0.8750$ to $1.0000$** ($\text{Mean} = 0.9750 \pm 0.0430$) | A single holdout set of $N = 22$ clusters ($\sim 12\text{ ASD}, 10\text{ TD}$) is too small to guarantee stable evaluation (`seed = 123` drops to $0.8750$). Full 5-fold cross-validation across all 88 unique subjects is required. |
| **`Original-Only AUC: 0.975`** | Out-of-fold ROC-AUC evaluated on the 100 original `video/8.xlsx` recordings | **100 Original Trials (`video/8.xlsx`)**: OOF $\text{AUROC} = 0.9756$ ($5\text{-fold SKF Mean} = 0.9640 \pm 0.0575$); **88 Deduplicated Original Trials (`R4B`)**: $\text{AUROC} = 0.9502 \pm 0.0453$ | Confirms that unaugmented camera captures alone provide high discriminative accuracy ($\text{AUROC} = 0.9502\text{--}0.9756$). |

As shown in Table 6, a 25% holdout partition of the 87 metadata clusters contains exactly $N = 22$ test clusters. While `CatBoost (Hist-300)` achieves $\text{AUROC} = 1.0000$ on those 22 clusters under `seed = 42` (as well as seeds 13, 21, and 99), changing the random partition to `seed = 123` reduces the holdout $\text{AUROC}$ to $0.8750$ ($\text{BalAcc} = 73.33\%$). This finding demonstrates why reporting a single small holdout split can be misleading and why full 5-fold cross-validation (`R3B` and `R4B`) is necessary for honest reporting.

### 7.4 Feature Attribution and Biomechanical Interpretation Using TreeSHAP

To verify that the height-normalized `CatBoost` classifier relies on clinically meaningful movement patterns, exact `TreeSHAP` values $\phi_{i,j}$ [20] were computed across all $N = 700$ normalized trials and compared against two-sided Mann–Whitney $U$ tests and Cohen's $d$ effect sizes. Table 7 lists the top 15 biomechanical features ranked by mean absolute SHAP value $\mathbb{E}[|\phi_j|]$, and Figures 9 and 10 display the local beeswarm distributions and global effect size comparisons.

**Table 7. Top 15 3D kinematic biomarkers ranked by exact TreeSHAP attribution, with Mann–Whitney $U$ significance and biomechanical interpretation.**

| Rank | Feature Identifier | Anatomical / Kinematic Domain | Mean $\|\text{SHAP}\|$ (Log-Odds) | ASD Mean $\pm$ SD ($N=350$) | TD Mean $\pm$ SD ($N=350$) | Cohen's $d$ Effect Size | Mann–Whitney $U$ $p$-value | Biomechanical Interpretation |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | `angle_Right_Knee_mean` | Right Knee 3D Angle Mean ($^\circ$) | $0.5067$ | $165.43^\circ \pm 6.45^\circ$ | $152.71^\circ \pm 13.79^\circ$ | $+1.182$ | $2.77 \times 10^{-45}$ | **Reduced Sagittal Knee Flexion**: ASD gait remains $12.7^\circ$ closer to full $180^\circ$ extension during walking, indicating a stiffer knee pattern [8, 10]. |
| **2** | `posture_head_spine_sway_std` | 3D Head-to-Pelvis Sway SD (TU) | $0.4408$ | $0.0566 \pm 0.0365$ | $0.0320 \pm 0.0132$ | $+0.896$ | $2.94 \times 10^{-38}$ | **Axial Postural Instability**: $76.9\%$ higher dynamic oscillation of the head relative to `SpineBase` during locomotion [8, 25]. |
| **3** | `angle_Right_Elbow_mean` | Right Elbow 3D Angle Mean ($^\circ$) | $0.2966$ | $143.67^\circ \pm 16.58^\circ$ | $155.14^\circ \pm 13.09^\circ$ | $-0.768$ | $9.04 \times 10^{-25}$ | **Upper-Limb Arm Flexion Guarding**: Children with ASD hold the right elbow $11.5^\circ$ more flexed rather than swinging the arm freely [8, 26]. |
| **4** | `vel_Head_mean_speed` | Head 3D Mean Speed (TU/s) | $0.2943$ | $0.858 \pm 0.871$ | $0.684 \pm 1.298$ | $+0.157$ | $9.81 \times 10^{-11}$ | **Increased Cranial Velocity**: Reflects rapid, un-damped head movements and visual-vestibular stabilization differences [15, 25]. |
| **5** | `vel_HandRight_std_speed` | Right Hand Speed Variability (TU/s) | $0.2899$ | $4.971 \pm 4.531$ | $7.319 \pm 11.416$ | $-0.270$ | $1.32 \times 10^{-4}$ | **Reduced Distal Arm Rhythm**: Indicates restricted arm carriage compared to normal pendular arm swing [8, 13]. |
| **6** | `vel_KneeLeft_mean_accel` | Left Knee Mean Acceleration ($\text{TU/s}^2$) | $0.2314$ | $159.46 \pm 170.39$ | $228.88 \pm 366.85$ | $-0.243$ | $1.40 \times 10^{-4}$ | **Reduced Lower-Limb Push-Off Acceleration**: Lower joint acceleration during swing initiation [9, 10]. |
| **7** | `angle_Left_Elbow_mean` | Left Elbow 3D Angle Mean ($^\circ$) | $0.2072$ | $144.87^\circ \pm 15.96^\circ$ | $154.09^\circ \pm 14.37^\circ$ | $-0.607$ | $1.12 \times 10^{-19}$ | **Bilateral Elbow Flexion Guarding**: Confirms bilateral consistency of flexed elbow posture ($-9.2^\circ$ in ASD) [8, 26]. |
| **8** | `vel_SpineShoulder_mean_accel` | Upper Trunk Mean Acceleration ($\text{TU/s}^2$) | $0.2022$ | $96.81 \pm 101.67$ | $122.40 \pm 227.64$ | $-0.145$ | $2.92 \times 10^{-3}$ | **Altered Upper Trunk Acceleration**: Reflects rigid trunk movement during walking [8, 12]. |
| **9** | `vel_FootLeft_mean_speed` | Left Foot Mean 3D Speed (TU/s) | $0.1869$ | $2.248 \pm 1.714$ | $2.587 \pm 3.310$ | $-0.128$ | $2.10 \times 10^{-1}$ | **Swing-Phase Foot Speed Modulation**: Interacts non-linearly with knee flexion in symmetric oblivious trees [9, 10]. |
| **10** | `angle_Right_Hip_mean` | Right Hip 3D Angle Mean ($^\circ$) | $0.1846$ | $161.36^\circ \pm 4.30^\circ$ | $156.90^\circ \pm 9.10^\circ$ | $+0.627$ | $1.13 \times 10^{-14}$ | **Reduced Hip Flexion Excursion**: ASD participants maintain a more extended hip posture ($+4.5^\circ$), coupled with reduced knee flexion [8, 12]. |
| **11** | `vel_FootLeft_mean_accel` | Left Foot Mean Acceleration ($\text{TU/s}^2$) | $0.1598$ | $317.85 \pm 322.25$ | $426.31 \pm 673.48$ | $-0.206$ | $2.63 \times 10^{-4}$ | **Reduced Distal Foot Swing Acceleration** [9, 10]. |
| **12** | `vel_FootLeft_max_speed` | Left Foot Peak Speed (TU/s) | $0.1573$ | $15.72 \pm 14.37$ | $23.16 \pm 36.11$ | $-0.271$ | $1.68 \times 10^{-5}$ | **Reduced Peak Swing Velocity of the Distal Foot** [8, 10]. |
| **13** | `vel_AnkleRight_mean_speed` | Right Ankle Mean Speed (TU/s) | $0.1558$ | $2.126 \pm 1.751$ | $2.254 \pm 3.122$ | $-0.051$ | $5.25 \times 10^{-2}$ | **Right Ankle Pacing Feature** [10]. |
| **14** | `vel_WristLeft_std_speed` | Left Wrist Speed Variability (TU/s) | $0.1520$ | $3.621 \pm 3.457$ | $5.377 \pm 8.926$ | $-0.260$ | $3.08 \times 10^{-6}$ | **Constrained Left Wrist Speed Variability** [8, 13]. |
| **15** | `vel_AnkleLeft_mean_speed` | Left Ankle Mean Speed (TU/s) | $0.1469$ | $1.925 \pm 1.503$ | $2.212 \pm 2.931$ | $-0.123$ | $2.64 \times 10^{-1}$ | **Bilateral Ankle Kinematic Coordination** [8, 10]. |

![Figure 9: Exact TreeSHAP Beeswarm Plot](../05_figures/fig09_shap_beeswarm.png)
*Figure 9. Exact TreeSHAP beeswarm summary plot across the top 15 3D kinematic features. Each point represents a normalized gait trial ($N = 700$), colored by standardized feature value (red = high, blue = low), with horizontal position showing the additive log-odds contribution toward ASD classification.*

![Figure 10: Ranked Feature Importance and Cohen's d Effect Sizes](../05_figures/fig10_shap_importance_cohens_d.png)
*Figure 10. Comparison of (left) global mean $|\text{SHAP}|$ feature importance and (right) standardized group difference (Cohen's $d$) with two-sided Mann–Whitney $U$ significance markers.*

![Figure 11: Naive vs. Subject-Grouped Generalization Delta](../05_figures/fig11_naive_vs_grouped_delta.png)
*Figure 11. Diagnostic comparison showing the accuracy inflation between naive ungrouped cross-validation (`R1`) and subject-grouped cross-validation (`R2`), highlighting the necessity of subject-disjoint evaluation.*

The findings in Table 7 and Figures 9–10 show that the model's predictions are driven by three consistent biomechanical patterns that align with clinical gait literature:

1. **Lower-Extremity Sagittal Stiffness (`angle_Right_Knee_mean`, Rank 1, $d = +1.182, p = 2.77 \times 10^{-45}$; `angle_Right_Hip_mean`, Rank 10, $d = +0.627, p = 1.13 \times 10^{-14}$)**: During walking, TD children flex the right knee to a mean angle of $152.71^\circ \pm 13.79^\circ$, whereas children with ASD maintain a more extended knee angle of $165.43^\circ \pm 6.45^\circ$ and a more extended hip angle ($161.36^\circ$ vs. $156.90^\circ$). In clinical biomechanics, reduced knee flexion during stance and swing is known as a stiff-knee gait strategy, which children with ASD adopt to increase joint stability during forward locomotion [8, 10, 12].
2. **Axial Head-to-Pelvis Postural Sway (`posture_head_spine_sway_std`, Rank 2, $d = +0.896, p = 2.94 \times 10^{-38}$; `vel_Head_mean_speed`, Rank 4, $p = 9.81 \times 10^{-11}$)**: Children with ASD exhibit a 76.9% higher standard deviation in 3D head-to-pelvis sway ($0.0566$ vs. $0.0320\text{ TU}$) and higher mean head speed. This observation reflects impaired sensorimotor integration and reduced head-trunk stabilization during walking [8, 14, 15, 25].
3. **Bilateral Upper-Limb Flexion Guarding (`angle_Right_Elbow_mean`, Rank 3, $d = -0.768, p = 9.04 \times 10^{-25}$; `angle_Left_Elbow_mean`, Rank 7, $d = -0.607, p = 1.12 \times 10^{-19}$)**: While TD children walk with relatively extended, freely swinging arms ($154.1^\circ\text{--}155.1^\circ$), children with ASD hold both elbows in a more flexed posture ($143.7^\circ\text{--}144.9^\circ$) accompanied by lower hand and wrist speed variability (Ranks 5 and 14). This pattern matches clinical descriptions of upper-extremity "high-guard" arm carriage and motor stereotypy in pediatric ASD [8, 25, 26].

### 7.5 Practical Implications and Weaknesses in Obtained Results

From a practical perspective, these results demonstrate that a single markerless Kinect v2 camera combined with 3D trunk-height scaling and Ordered CatBoost can classify pediatric ASD gait with $\text{AUROC} = 0.9717$ (`R3B`) to $0.9802$ (`R2`) without requiring wearable sensors or reflective skin markers. Furthermore, the present audit establishes a practical quality-control protocol for future movement repositories: researchers must verify that synthetic geometric augmentations are not cancelled out by coordinate normalization and must run pairwise trajectory correlation checks to remove duplicate subject folders before splitting data.

At the same time, certain weaknesses in the obtained results should be noted. First, when moving from nominal 100-subject cross-validation (`R2`, $92.57\%$ accuracy) to the un-deduplicated 87-cluster split (`R3A`, $86.71\% \pm 13.99\%$ accuracy), fold-to-fold variance increases substantially because multi-folder clone clusters create uneven fold sizes. Second, on the smallest unaugmented original-only regime (`R4B`, $N = 88$ total samples, or $\sim 70$ training samples per fold), the deeper depth-5 `CatBoost (Tuned)` model ($87.45\%$ accuracy, $\text{AUROC} = 0.9243$) is outperformed by the shallower depth-4 `CatBoost (Hist-300)` ($89.74\%$ accuracy, $\text{AUROC} = 0.9502$) and by simpler linear or kernel classifiers (`Logistic Regression`: $92.09\%$, `SVM-RBF`: $\text{AUROC} = 0.9622$). This indicates that depth-5 gradient boosting requires either multi-view perturbations (such as Augmentations 4, 6, and 7 in `R3B`) or a larger number of unique subjects to reach its full advantage over linear models.

---

## 8. Limitations

In accordance with rigorous academic reporting standards, the limitations of this study are detailed below across six areas:

1. **Dataset Limitations**: The Al-Jubouri (2020) dataset provides aggregate group demographic statistics (`readme.docx`), but individual chronological age and biological sex labels are not linked to individual numbered folders (`1..50`). Although per-frame 3D trunk-height scaling (`SpineShoulder` to `SpineBase`) removes physical stature differences ($p = 1.000$), the ASD group has a higher proportion of males ($84.0\%$ vs. $52.0\%$ in TD, $p = 0.0006$) and a higher mean age ($8.62$ vs. $6.66\text{ years}$). Because folder-level age and sex covariates are unavailable in the public archive, age- and sex-stratified subgroup analyses could not be conducted.
2. **Sample-Size Constraints**: Although the repository contains 800 coordinate files across 100 nominal folders, the coordinate audit proved that 12 folders are exact duplicates and 4 of the 7 augmentations per folder are translation-invariant copies. Consequently, the true effective sample size is $M = 88$ unique children (47 ASD and 41 TD) from a single clinical center.
3. **Model Limitations**: The proposed feature representation summarizes each walking trial using temporal statistics (mean, standard deviation, maximum, and spectral entropy) across the walking sequence. While this tabular representation works well with CatBoost and prevents sequence-level overfitting on $M = 88$ subjects, it does not explicitly segment individual heel-strike to toe-off gait cycles or model phase-locked joint trajectories.
4. **Evaluation Limitations**: All evaluations were conducted using internal 5-fold grouped cross-validation and bootstrap resampling on a single-center cohort. No independent external test cohort from a second hospital or different geographical population was available for cross-site validation.
5. **Implementation Constraints**: The dataset was recorded using the Microsoft Kinect v2 time-of-flight camera at $30\text{ fps}$. Microsoft has discontinued commercial production of the Kinect v2 sensor, and depth accuracy can degrade under strong ambient infrared sunlight or when loose clothing obscures joint centers.
6. **Generalizability Concerns**: First, the 9 children in the `Severe autism/` folder lacked 3D skeletal coordinate files and were excluded; hence, the model is verified only for ambulatory children who can walk unassisted. Second, the control group consists exclusively of typically developing children rather than children with other neurodevelopmental motor conditions, such as Developmental Coordination Disorder (DCD), Attention-Deficit/Hyperactivity Disorder (ADHD), or cerebral palsy [8, 26]. Therefore, the model captures ASD-vs-TD motor differences and must be used as a supportive motor screening tool rather than a standalone differential diagnostic device.

---

## 9. Future Scope

Further investigation is required to extend this research toward clinical deployment:

1. **Multi-Center Data Collection with Modern RGB-D and Video Sensors**: Future studies should collect larger, multi-site cohorts ($M \ge 300$ unique children) using current depth sensors (such as the Azure Kinect DK or Orbbec Femto Bolt) as well as standard smartphone RGB video with 3D pose estimation backbones. Since modern sensors share the same 25-to-33 canonical joint topology, the proposed `SpineBase` root-centering and 3D trunk-height normalization pipeline can be transferred directly.
2. **Age- and Sex-Matched Longitudinal Cohorts**: Recording participant age, sex, standing height, and clinical severity scores (such as ADOS-2 calibrated severity scores) for every individual file will allow age-normed developmental trajectory modeling and sex-stratified fairness auditing.
3. **Inclusion of Non-ASD Neurodevelopmental Control Groups**: Evaluating the 86 biomechanical features across children with DCD, ADHD, and idiopathic toe-walking will clarify which kinematic markers (such as combined knee stiffness, head sway, and bilateral elbow guarding) are specific to ASD versus general motor delay.
4. **Automated Gait-Cycle Phase Segmentation**: Integrating automated heel-strike and toe-off event detection will enable separate quantification of stance-phase and swing-phase joint angles, improving biomechanical precision.

---

## 10. Conclusion

This study presented a leakage-audited machine learning framework for classifying Autism Spectrum Disorder in children from 3D Kinect v2 skeletal movement data using Ordered CatBoost and exact TreeSHAP explainability. The primary objective was to resolve the methodological issues of physical height confounding, synthetic augmentation redundancy, and cross-folder subject duplication in the widely used Al-Jubouri (2020) pediatric gait dataset.

The proposed methodology introduced frame-by-frame pelvis root-centering (`SpineBase`) and 3D trunk-height scaling (`SpineShoulder–SpineBase` distance) to eliminate a $+16.6\text{ cm}$ inter-group height confound, followed by the extraction of 86 biomechanical features across five movement domains. A coordinate-level audit revealed two critical sources of data leakage in the public archive: four of the seven coordinate augmentations per folder are mathematically invariant under root-centering ($\max |\Delta| \le 1.92 \times 10^{-13}$), and 12 of the 100 nominal subject folders are exact coordinate clones ($r = 1.0000$) of 7 other folders, leaving 88 strictly unique children (47 ASD and 41 TD) and 87 metadata clusters.

Across a four-regime 5-fold cross-validation benchmark of 10 classifiers, tuned CatBoost achieved $92.57\% \pm 2.70\%$ trial accuracy and $93.00\%$ ($93/100$) subject majority-vote accuracy ($\text{AUROC} = 0.9802$, Brier score $= 0.0618$) under nominal 100-subject grouped validation (`R2`), significantly outperforming all non-CatBoost baselines ($p \le 0.0076$). Under strict 88-child deduplicated grouped validation (`R3B`), tuned CatBoost achieved $88.59\% \pm 6.14\%$ accuracy and $\text{AUROC} = 0.9717$, while on the 88 unaugmented original camera recordings (`R4B`), depth-4 CatBoost achieved $89.74\% \pm 8.56\%$ accuracy and $\text{AUROC} = 0.9502$. Exact TreeSHAP attribution and Mann–Whitney $U$ testing demonstrated that classification decisions rely on three height-invariant motor markers: reduced right knee flexion ($d = +1.18$), increased 3D head-to-pelvis postural sway ($d = +0.90$), and bilateral elbow flexion guarding ($d = -0.77$). Although limited by a single-center sample size of 88 unique children and male sex skew, this work provides a reproducible, leakage-free baseline for markerless computational motor screening in pediatric ASD.

---

## Back Matter and Compliance Declarations

### Data and Code Availability Statement
The raw 3D Kinect v2 skeletal movement dataset analyzed in this study is publicly archived on the Dryad Digital Repository by Al-Jubouri (AbdulRahman), Hadi, and Rajihy (2020) at `https://doi.org/10.5061/dryad.s7h44j150` [2]. All preprocessing scripts (`src/data_preprocessing.py`), 86-feature biomechanical extraction pipelines (`src/feature_extraction.py`), multi-regime cross-validation benchmarks (`src/train.py`), exact TreeSHAP explainability modules (`src/explainability.py`), deduplicated unaugmented feature tables (`research_output/04_experiments/new_experiments/original_unaugmented_100_trials.csv`), and multi-regime evaluation logs (`master_multi_regime_benchmark.csv`) are preserved in the project repository with deterministic random seeds (`seed = 42`).

### Ethics Statement
This study conducts a secondary computational and statistical analysis of a publicly archived, de-identified human movement dataset (`DOI: 10.5061/dryad.s7h44j150`) [2, 3]. In the original data collection protocol [2, 3], institutional ethical approval and written informed parental consent were obtained for all participating children in accordance with the Declaration of Helsinki, and all 3D skeletal coordinate files were released without personally identifiable information. No new human participant recruitment or clinical intervention was performed in the present work.

### Author Contributions
**Conceptualization, Methodology, Software, Validation, Formal Analysis, Investigation, Data Curation, Writing—Original Draft, Writing—Review and Editing, Visualization**: Project Research Team.

### Funding
This research received no external grant funding.

### Conflicts of Interest
The authors declare no conflicts of interest.

---

## References

[1] H. Rakotomanana and G. Rouhafzay, "A Scoping Review of AI-Based Approaches for Detecting Autism Traits Using Voice and Behavioral Data," *Bioengineering*, vol. 12, no. 11, p. 1136, 2025, doi: 10.3390/bioengineering12111136.

[2] A. A. AbdulRahman (Al-Jubouri), I. H. Hadi, and Y. Rajihy, "Generating 3D dataset of Gait and Full body movement of children with Autism spectrum disorders collected by Kinect v2 camera," *COMPUSOFT: An International Journal of Advanced Computer Technology*, vol. 9, no. 8, pp. 3791–3797, 2020, doi: 10.5061/dryad.s7h44j150.

[3] A. A. Al-Jubouri, I. H. Ali, and Y. Rajihy, "Gait and Full Body Movement Dataset of Autistic Children Classified by Rough Set Classifier," *Journal of Physics: Conference Series*, vol. 1818, no. 1, p. 012201, 2021, doi: 10.1088/1742-6596/1818/1/012201.

[4] S. A. Yazdi, A. Janghorbani, and A. Maleki, "Diagnosis of Autism in Children Based on their Gait Pattern and Movement Signs Using the Kinect Sensor," *Journal of Medical Signals & Sensors*, vol. 14, no. 10, p. 29, 2024, doi: 10.4103/jmss.jmss_19_24.

[5] Y. Fırat, "SHAP-based explainable AI framework for autism severity classification using 3D motor biomarkers," *Frontiers in Psychiatry*, vol. 17, p. 1751654, 2026, doi: 10.3389/fpsyt.2026.1751654.

[6] N. Le-Thi-Thu, L. Pham-Viet-Chi, N. Tran-Ly, N. Le-Ba, T. Le-Dang-Van, and P. Nguyen-Van, "Effects of Skeleton Gait Normalization and Dynamic Motion Features on Deep Learning Performance," in *Integrated Uncertainty in Knowledge Modelling and Decision Making (IUKM 2026)*, Lecture Notes in Computer Science, Singapore: Springer, 2026, pp. 142–154, doi: 10.1007/978-981-92-5641-9_12.

[7] A. Amirbay, N. Baigabylov, A. Mukhanova, et al., "Autism detection using facial and motor analysis using machine learning," *Bulletin of Electrical Engineering and Informatics*, vol. 14, no. 5, pp. 3985–4000, 2025, doi: 10.11591/eei.v14i5.10319.

[8] A. Ioannidou, P. Vlotinou, A. Tsiakiri, N. Aggelousis, G. Labiris, and A. Serdari, "Gait patterns in children with autism spectrum disorder: A systematic review," *Sports Medicine and Health Science*, 2025, doi: 10.1016/j.smhs.2025.11.002.

[9] A. Pradhan, V. Chester, and K. Padhiar, "Classification of Autism and Control Gait in Children Using Multisegment Foot Kinematic Features," *Bioengineering*, vol. 9, no. 10, p. 552, 2022, doi: 10.3390/bioengineering9100552.

[10] D. N. Olivas, M. R. Kendall, A. Parada, R. A. Manning, and J. D. Eggleston, "Children with autism display altered ankle strategies when changing speed during over-ground gait," *Clinical Biomechanics*, vol. 100, p. 105804, 2022, doi: 10.1016/j.clinbiomech.2022.105804.

[11] S. M. S. Aljabiri and M. M. Hamdan, "Analyzing lower body movements using machine learning to classify autistic children," *Biomedical Signal Processing and Control*, vol. 94, p. 106288, 2024, doi: 10.1016/j.bspc.2024.106288.

[12] Y. Li, M. A. Mache, and T. A. Todd, "Age-related gait development in children with autism spectrum disorder," *Gait & Posture*, vol. 84, pp. 260–266, 2021, doi: 10.1016/j.gaitpost.2020.12.022.

[13] A. Anzulewicz, K. Sobota, and J. T. Delafield-Butt, "Toward the Autism Motor Signature: Gesture patterns during smart tablet gameplay identify children with autism," *Scientific Reports*, vol. 6, p. 31107, 2016, doi: 10.1038/srep31107.

[14] A. Ardalan, A. H. Assadi, O. J. Surgent, and B. G. Travers, "Whole-Body Movement during Videogame Play Distinguishes Youth with Autism from Youth with Typical Development," *Scientific Reports*, vol. 9, p. 20094, 2019, doi: 10.1038/s41598-019-56362-6.

[15] M. Alcañiz Raya, J. Marín-Morales, M. E. Minissi, G. Teruel Garcia, L. Abad, and I. A. Chicchi Giglioli, "Machine Learning and Virtual Reality on Body Movements' Behaviors to Classify Children with Autism Spectrum Disorder," *Journal of Clinical Medicine*, vol. 9, no. 5, p. 1260, 2020, doi: 10.3390/jcm9051260.

[16] R. Simeoli, N. Milano, A. Rega, and D. Marocco, "Using Technology to Identify Children With Autism Through Motor Abnormalities," *Frontiers in Psychology*, vol. 12, p. 635696, 2021, doi: 10.3389/fpsyg.2021.635696.

[17] H. Doi, N. Iijima, A. Furui, Z. Soh, R. Yonei, K. Shinohara, M. Iriguchi, K. Shimatani, and T. Tsuji, "Prediction of autistic tendencies at 18 months of age via markerless video analysis of spontaneous body movements in 4-month-old infants," *Scientific Reports*, vol. 12, p. 18045, 2022, doi: 10.1038/s41598-022-21308-y.

[18] U. J. Ganai, A. Ratne, B. Bhushan, and K. S. Venkatesh, "Early detection of autism spectrum disorder: gait deviations and machine learning," *Scientific Reports*, vol. 15, p. 873, 2025, doi: 10.1038/s41598-025-85348-w.

[19] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: unbiased boosting with categorical features," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 31, 2018, pp. 6638–6648, doi: 10.48550/arXiv.1706.09516.

[20] S. M. Lundberg, G. Erion, H. Chen, A. DeGrave, J. M. Prutkin, B. Nair, R. Katz, J. Himmelfarb, N. Bansal, and S.-I. Lee, "From local explanations to global understanding with explainable AI for trees," *Nature Machine Intelligence*, vol. 2, no. 1, pp. 56–67, 2020, doi: 10.1038/s42256-019-0138-9.

[21] S. Saeb, L. Lonini, A. Jayaraman, D. C. Mohr, and K. P. Kording, "The need to approximate the use-case in clinical machine learning," *GigaScience*, vol. 6, no. 5, p. gix019, 2017, doi: 10.1093/gigascience/gix019.

[22] S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, p. 100804, 2023, doi: 10.1016/j.patter.2023.100804.

[23] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," in *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, San Francisco, CA, USA: ACM, 2016, pp. 785–794, doi: 10.1145/2939672.2939785.

[24] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017, pp. 3146–3154.

[25] K. A. Fournier, C. J. Hass, S. K. Naik, N. Lodha, and J. H. Cauraugh, "Motor coordination in autism spectrum disorders: a synthesis and meta-analysis," *Journal of Autism and Developmental Disorders*, vol. 40, no. 10, pp. 1227–1240, 2010, doi: 10.1007/s10803-010-0981-3.

[26] C. J. Zampella, L. A. L. Wang, M. Haley, A. G. Hutchinson, and A. de Marchena, "Motor Skill Differences in Autism Spectrum Disorder: a Clinically Focused Review," *Current Psychiatry Reports*, vol. 23, no. 10, p. 64, 2021, doi: 10.1007/s11920-021-01280-6.
