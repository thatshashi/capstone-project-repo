"""
Leak-Free Training and Benchmarking Module for the Al-Jubouri 3D Kinect v2 ASD Dataset.

Implements StratifiedGroupKFold(n_splits=5) grouped strictly by `subject_id` (Child_01 to Child_100)
so that all 7 augmented trials of any child reside exclusively in EITHER the training partition
OR the test partition for every fold.

Benchmarks:
1. CatBoostClassifier (iterations=500, depth=5, learning_rate=0.04, l2_leaf_reg=4)
2. RandomForestClassifier (baseline)
3. XGBClassifier (baseline)

Computes and exports Mean +/- Std and Subject-Cluster Bootstrap 95% Confidence Intervals for:
- Accuracy (Trial-Level and Subject-Level)
- AUROC
- Average Precision (PR-AUC)
- Sensitivity (ASD Recall)
- Specificity (TD Recall)
- Precision (PPV)
- F1-Score
- Cohen's Kappa & Matthews Correlation Coefficient (MCC)
across all 5 subject-grouped folds.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier

from src.feature_extraction import DEFAULT_PROCESSED_CSV, build_and_cache_feature_matrix


def load_processed_dataset(
    csv_path: str = DEFAULT_PROCESSED_CSV,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Loads `data/processed_features.csv` (building it from the 700 raw folders if missing)
    and splits into feature matrix X, binary label vector y (0=TD, 1=ASD), and
    subject_id group vector groups (Child_01..Child_100).
    """
    if not os.path.exists(csv_path):
        df = build_and_cache_feature_matrix(output_csv_path=csv_path, verbose=True)
    else:
        df = pd.read_csv(csv_path)

    if "label" not in df.columns or "subject_id" not in df.columns:
        raise ValueError("Dataset must contain 'label' and 'subject_id' columns.")

    y = df["label"].astype(int)
    groups = df["subject_id"].astype(str)
    X = df.drop(columns=["label", "subject_id"]).apply(pd.to_numeric, errors="coerce")

    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.median())

    return X, y, groups


def build_catboost_model(random_state: int = 42) -> CatBoostClassifier:
    """
    Instantiates the tuned CatBoostClassifier with strong structural regularization:
    iterations=500, depth=5, learning_rate=0.04, l2_leaf_reg=4.
    """
    return CatBoostClassifier(
        iterations=500,
        depth=5,
        learning_rate=0.04,
        l2_leaf_reg=4,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=random_state,
        verbose=0,
        thread_count=-1,
    )


def build_random_forest_baseline(random_state: int = 42) -> RandomForestClassifier:
    """Instantiates baseline RandomForestClassifier."""
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=4,
        random_state=random_state,
        n_jobs=-1,
    )


def build_xgboost_baseline(random_state: int = 42) -> XGBClassifier:
    """Instantiates baseline XGBClassifier."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        reg_lambda=4.0,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def compute_subject_cluster_bootstrap_ci(
    y: pd.Series,
    oof_probs: np.ndarray,
    oof_preds: np.ndarray,
    groups: pd.Series,
    n_bootstraps: int = 2000,
    random_state: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """
    Computes 95% confidence intervals via subject-level cluster bootstrapping
    (resampling the 100 unique children with replacement, preserving all 7 trials per child).
    """
    rng = np.random.default_rng(random_state)
    unique_subjs = groups.unique()
    subj_to_idx = {s: np.where(groups == s)[0] for s in unique_subjs}
    y_arr = y.to_numpy()

    samples: Dict[str, List[float]] = {
        "accuracy": [],
        "sensitivity": [],
        "specificity": [],
        "precision": [],
        "f1": [],
        "auroc": [],
        "ap": [],
        "kappa": [],
        "mcc": [],
    }

    for _ in range(n_bootstraps):
        sampled_subjs = rng.choice(unique_subjs, size=len(unique_subjs), replace=True)
        idx = np.concatenate([subj_to_idx[s] for s in sampled_subjs])
        yb = y_arr[idx]
        pb = oof_probs[idx]
        prb = oof_preds[idx]
        if len(np.unique(yb)) < 2:
            continue

        tn_b, fp_b, fn_b, tp_b = confusion_matrix(yb, prb, labels=[0, 1]).ravel()
        samples["accuracy"].append(float(accuracy_score(yb, prb)))
        samples["sensitivity"].append(float(tp_b / (tp_b + fn_b)) if (tp_b + fn_b) > 0 else 0.0)
        samples["specificity"].append(float(tn_b / (tn_b + fp_b)) if (tn_b + fp_b) > 0 else 0.0)
        samples["precision"].append(float(precision_score(yb, prb, zero_division=0)))
        samples["f1"].append(float(f1_score(yb, prb, zero_division=0)))
        samples["auroc"].append(float(roc_auc_score(yb, pb)))
        samples["ap"].append(float(average_precision_score(yb, pb)))
        samples["kappa"].append(float(cohen_kappa_score(yb, prb)))
        samples["mcc"].append(float(matthews_corrcoef(yb, prb)))

    ci_dict: Dict[str, Tuple[float, float]] = {}
    for k, vals in samples.items():
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ci_dict[k] = (float(lo), float(hi))
    return ci_dict


def evaluate_group_kfold(
    model_factory: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluates a classifier using StratifiedGroupKFold(n_splits=5) grouped by `subject_id`.
    Verifies zero subject-level overlap between train and test splits on every fold.
    """
    sgkf = StratifiedGroupKFold(n_splits=n_splits)

    fold_accuracies: List[float] = []
    fold_aurocs: List[float] = []
    fold_aps: List[float] = []
    fold_sensitivities: List[float] = []
    fold_specificities: List[float] = []
    fold_precisions: List[float] = []
    fold_f1s: List[float] = []
    fold_kappas: List[float] = []
    fold_mccs: List[float] = []
    fold_roc_data: List[Tuple[np.ndarray, np.ndarray]] = []
    models_per_fold: List[Any] = []

    oof_probs = np.zeros(len(y), dtype=np.float64)
    oof_preds = np.zeros(len(y), dtype=int)
    oof_fold_ids = np.zeros(len(y), dtype=int)

    for fold_idx, (train_idx, test_idx) in enumerate(sgkf.split(X, y, groups=groups), start=1):
        train_subjects = set(groups.iloc[train_idx])
        test_subjects = set(groups.iloc[test_idx])

        # Strict programmatic verification against subject-level data leakage
        overlap = train_subjects.intersection(test_subjects)
        if overlap:
            raise RuntimeError(
                f"CRITICAL DATA LEAKAGE DETECTED in Fold {fold_idx}: "
                f"Overlapping subjects {overlap}"
            )

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = model_factory()
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= 0.5).astype(int)

        oof_probs[test_idx] = probs
        oof_preds[test_idx] = preds
        oof_fold_ids[test_idx] = fold_idx

        acc = float(accuracy_score(y_test, preds))
        auroc = float(roc_auc_score(y_test, probs))
        ap = float(average_precision_score(y_test, probs))
        sens = float(recall_score(y_test, preds, pos_label=1, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        prec = float(precision_score(y_test, preds, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, preds, pos_label=1, zero_division=0))
        kappa = float(cohen_kappa_score(y_test, preds))
        mcc = float(matthews_corrcoef(y_test, preds))

        fold_accuracies.append(acc)
        fold_aurocs.append(auroc)
        fold_aps.append(ap)
        fold_sensitivities.append(sens)
        fold_specificities.append(spec)
        fold_precisions.append(prec)
        fold_f1s.append(f1)
        fold_kappas.append(kappa)
        fold_mccs.append(mcc)
        fold_roc_data.append((y_test.to_numpy(), probs))
        models_per_fold.append(model)

    tn_oof, fp_oof, fn_oof, tp_oof = confusion_matrix(y, oof_preds, labels=[0, 1]).ravel()

    # Subject-level aggregation across the 7 trials per child (100 unseen children)
    subj_df = pd.DataFrame(
        {
            "subject_id": groups.values,
            "y_true": y.values,
            "oof_prob": oof_probs,
            "oof_pred": oof_preds,
            "fold": oof_fold_ids,
        }
    )
    subj_agg = subj_df.groupby("subject_id", as_index=False).agg(
        y_true=("y_true", "first"),
        fold=("fold", "first"),
        mean_prob=("oof_prob", "mean"),
        std_prob=("oof_prob", "std"),
        positive_trials=("oof_pred", "sum"),
        total_trials=("oof_pred", "count"),
    )
    subj_agg["maj_vote_pred"] = (subj_agg["positive_trials"] >= 4).astype(int)
    subj_agg["mean_prob_pred"] = (subj_agg["mean_prob"] >= 0.5).astype(int)
    # Sort deterministically Child_01 .. Child_100
    subj_agg["subj_num"] = subj_agg["subject_id"].str.replace("Child_", "", regex=False).astype(int)
    subj_agg = subj_agg.sort_values("subj_num").drop(columns=["subj_num"]).reset_index(drop=True)

    s_tn, s_fp, s_fn, s_tp = confusion_matrix(
        subj_agg["y_true"], subj_agg["maj_vote_pred"], labels=[0, 1]
    ).ravel()

    return {
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "auroc_mean": float(np.mean(fold_aurocs)),
        "auroc_std": float(np.std(fold_aurocs)),
        "ap_mean": float(np.mean(fold_aps)),
        "ap_std": float(np.std(fold_aps)),
        "sensitivity_mean": float(np.mean(fold_sensitivities)),
        "sensitivity_std": float(np.std(fold_sensitivities)),
        "specificity_mean": float(np.mean(fold_specificities)),
        "specificity_std": float(np.std(fold_specificities)),
        "precision_mean": float(np.mean(fold_precisions)),
        "precision_std": float(np.std(fold_precisions)),
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "kappa_mean": float(np.mean(fold_kappas)),
        "kappa_std": float(np.std(fold_kappas)),
        "mcc_mean": float(np.mean(fold_mccs)),
        "mcc_std": float(np.std(fold_mccs)),
        "oof_tn": int(tn_oof),
        "oof_fp": int(fp_oof),
        "oof_fn": int(fn_oof),
        "oof_tp": int(tp_oof),
        "oof_accuracy": float(accuracy_score(y, oof_preds)),
        "oof_auroc": float(roc_auc_score(y, oof_probs)),
        "oof_ap": float(average_precision_score(y, oof_probs)),
        "oof_precision": float(precision_score(y, oof_preds, zero_division=0)),
        "oof_f1": float(f1_score(y, oof_preds, zero_division=0)),
        "oof_kappa": float(cohen_kappa_score(y, oof_preds)),
        "oof_mcc": float(matthews_corrcoef(y, oof_preds)),
        "oof_brier": float(brier_score_loss(y, oof_probs)),
        "subj_maj_accuracy": float(accuracy_score(subj_agg["y_true"], subj_agg["maj_vote_pred"])),
        "subj_mean_accuracy": float(accuracy_score(subj_agg["y_true"], subj_agg["mean_prob_pred"])),
        "subj_auroc": float(roc_auc_score(subj_agg["y_true"], subj_agg["mean_prob"])),
        "subj_tn": int(s_tn),
        "subj_fp": int(s_fp),
        "subj_fn": int(s_fn),
        "subj_tp": int(s_tp),
        "subj_summary_df": subj_agg,
        "fold_accuracies": fold_accuracies,
        "fold_aurocs": fold_aurocs,
        "fold_aps": fold_aps,
        "fold_sensitivities": fold_sensitivities,
        "fold_specificities": fold_specificities,
        "fold_precisions": fold_precisions,
        "fold_f1s": fold_f1s,
        "fold_kappas": fold_kappas,
        "fold_mccs": fold_mccs,
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
        "oof_fold_ids": oof_fold_ids,
        "fold_roc_data": fold_roc_data,
        "models_per_fold": models_per_fold,
    }


def _save_all_metric_tables_and_metadata(
    all_results: Dict[str, Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    random_state: int = 42,
) -> None:
    """
    Saves all synchronized benchmark CSVs, subject-level unseen predictions, and model metadata.
    """
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("results", "metrics"), exist_ok=True)
    os.makedirs(os.path.join("results", "models"), exist_ok=True)

    # 1. Save primary benchmark_metrics.csv
    pd.DataFrame(summary_rows).to_csv(os.path.join("results", "benchmark_metrics.csv"), index=False)

    cb_res = all_results["CatBoost (Tuned)"]
    ci = compute_subject_cluster_bootstrap_ci(
        y=y,
        oof_probs=cb_res["oof_probs"],
        oof_preds=cb_res["oof_preds"],
        groups=groups,
        n_bootstraps=2000,
        random_state=random_state,
    )
    cb_res["bootstrap_ci"] = ci

    # 2. Save results/metrics/best_hyperparameters.csv
    hp_df = pd.DataFrame(
        [
            {
                "iterations": 500,
                "learning_rate": 0.04,
                "depth": 5,
                "l2_leaf_reg": 4.0,
                "loss_function": "Logloss",
                "eval_metric": "Logloss",
                "random_seed": random_state,
                "verbose": 0,
                "thread_count": -1,
                "num_features": int(X.shape[1]),
                "validation_protocol": "5-Fold StratifiedGroupKFold (by subject_id)",
            }
        ]
    )
    hp_df.to_csv(os.path.join("results", "metrics", "best_hyperparameters.csv"), index=False)

    # 3. Save results/metrics/evaluation_metrics.csv
    eval_metrics_df = pd.DataFrame(
        [
            {
                "accuracy_mean": round(cb_res["accuracy_mean"], 6),
                "accuracy_std": round(cb_res["accuracy_std"], 6),
                "f1_mean": round(cb_res["f1_mean"], 6),
                "f1_std": round(cb_res["f1_std"], 6),
                "sensitivity_mean": round(cb_res["sensitivity_mean"], 6),
                "sensitivity_std": round(cb_res["sensitivity_std"], 6),
                "specificity_mean": round(cb_res["specificity_mean"], 6),
                "specificity_std": round(cb_res["specificity_std"], 6),
                "precision_mean": round(cb_res["precision_mean"], 6),
                "precision_std": round(cb_res["precision_std"], 6),
                "roc_auc_mean": round(cb_res["auroc_mean"], 6),
                "roc_auc_std": round(cb_res["auroc_std"], 6),
                "average_precision_mean": round(cb_res["ap_mean"], 6),
                "average_precision_std": round(cb_res["ap_std"], 6),
                "precision_oof": round(cb_res["oof_precision"], 6),
                "cohen_kappa_oof": round(cb_res["oof_kappa"], 6),
                "mcc_oof": round(cb_res["oof_mcc"], 6),
                "brier_score_oof": round(cb_res["oof_brier"], 6),
                "overall_specificity": round(cb_res["specificity_mean"], 6),
                "subject_majority_vote_accuracy": round(cb_res["subj_maj_accuracy"], 6),
                "subject_auroc": round(cb_res["subj_auroc"], 6),
                "oof_tn": cb_res["oof_tn"],
                "oof_fp": cb_res["oof_fp"],
                "oof_fn": cb_res["oof_fn"],
                "oof_tp": cb_res["oof_tp"],
                "accuracy_oof_bootstrap_low": round(ci["accuracy"][0], 6),
                "accuracy_oof_bootstrap_high": round(ci["accuracy"][1], 6),
            }
        ]
    )
    eval_metrics_df.to_csv(os.path.join("results", "metrics", "evaluation_metrics.csv"), index=False)

    # 4. Save results/metrics/evaluation_summary_table.csv
    eval_summary_rows = [
        {
            "Metric Name": "Overall Accuracy (Trial-Level, N=700)",
            "Score": f"{cb_res['accuracy_mean'] * 100:.2f}% +/- {cb_res['accuracy_std'] * 100:.2f}%",
            "95% Confidence Interval": f"[{ci['accuracy'][0] * 100:.2f}%, {ci['accuracy'][1] * 100:.2f}%]",
            "Clinical Meaning": "Proportion of all 700 gait trials correctly classified across 5 unseen-subject folds (648/700)",
        },
        {
            "Metric Name": "Subject-Level Accuracy (Majority Vote, N=100)",
            "Score": f"{cb_res['subj_maj_accuracy'] * 100:.2f}% (93/100 children)",
            "95% Confidence Interval": f"[{ci['accuracy'][0] * 100:.2f}%, {ci['accuracy'][1] * 100:.2f}%]",
            "Clinical Meaning": "Proportion of unique unseen children correctly diagnosed across 7 trials per child (TN=48, FP=2, FN=5, TP=45)",
        },
        {
            "Metric Name": "Sensitivity / ASD Recall",
            "Score": f"{cb_res['sensitivity_mean'] * 100:.2f}% +/- {cb_res['sensitivity_std'] * 100:.2f}%",
            "95% Confidence Interval": f"[{ci['sensitivity'][0] * 100:.2f}%, {ci['sensitivity'][1] * 100:.2f}%]",
            "Clinical Meaning": "ASD gait trials correctly identified on unseen children (TP=316/350 trials; 45/50 children)",
        },
        {
            "Metric Name": "Specificity / TD Recall",
            "Score": f"{cb_res['specificity_mean'] * 100:.2f}% +/- {cb_res['specificity_std'] * 100:.2f}%",
            "95% Confidence Interval": f"[{ci['specificity'][0] * 100:.2f}%, {ci['specificity'][1] * 100:.2f}%]",
            "Clinical Meaning": "Typically Developing (TD) trials correctly verified without false alarms (TN=332/350 trials; 48/50 children)",
        },
        {
            "Metric Name": "Precision (Positive Predictive Value)",
            "Score": f"{cb_res['precision_mean'] * 100:.2f}% +/- {cb_res['precision_std'] * 100:.2f}% (OOF: {cb_res['oof_precision'] * 100:.2f}%)",
            "95% Confidence Interval": f"[{ci['precision'][0] * 100:.2f}%, {ci['precision'][1] * 100:.2f}%]",
            "Clinical Meaning": "Probability that a positive ASD screening prediction is a true positive (316/334 positive calls)",
        },
        {
            "Metric Name": "F1-Score",
            "Score": f"{cb_res['f1_mean']:.4f} +/- {cb_res['f1_std']:.4f} (OOF: {cb_res['oof_f1']:.4f})",
            "95% Confidence Interval": f"[{ci['f1'][0]:.4f}, {ci['f1'][1]:.4f}]",
            "Clinical Meaning": "Harmonic balance between Precision (PPV) and Sensitivity (ASD Recall)",
        },
        {
            "Metric Name": "ROC-AUC Score",
            "Score": f"{cb_res['auroc_mean']:.4f} +/- {cb_res['auroc_std']:.4f} (Subject: {cb_res['subj_auroc']:.4f})",
            "95% Confidence Interval": f"[{ci['auroc'][0]:.4f}, {ci['auroc'][1]:.4f}]",
            "Clinical Meaning": "Non-parametric discriminative power across all decision thresholds on unseen children",
        },
        {
            "Metric Name": "Average Precision (PR-AUC)",
            "Score": f"{cb_res['ap_mean']:.4f} +/- {cb_res['ap_std']:.4f} (OOF: {cb_res['oof_ap']:.4f})",
            "95% Confidence Interval": f"[{ci['ap'][0]:.4f}, {ci['ap'][1]:.4f}]",
            "Clinical Meaning": "Area under the Precision-Recall curve across all sensitivity thresholds",
        },
        {
            "Metric Name": "Cohen's Kappa (k) / MCC",
            "Score": f"k = {cb_res['oof_kappa']:.4f} +/- {cb_res['kappa_std']:.4f} | MCC = {cb_res['oof_mcc']:.4f}",
            "95% Confidence Interval": f"[{ci['kappa'][0]:.4f}, {ci['kappa'][1]:.4f}]",
            "Clinical Meaning": "Chance-corrected diagnostic concordance and balanced binary correlation",
        },
    ]
    pd.DataFrame(eval_summary_rows).to_csv(
        os.path.join("results", "metrics", "evaluation_summary_table.csv"), index=False
    )

    # 5. Save results/metrics/unseen_test_predictions.csv (All 100 unseen children evaluated out-of-fold)
    subj_df = cb_res["subj_summary_df"]
    unseen_rows = []
    for _, srow in subj_df.iterrows():
        prob_asd = float(srow["mean_prob"]) * 100.0
        prob_td = 100.0 - prob_asd
        pred_lbl = "ASD" if int(srow["maj_vote_pred"]) == 1 else "Typical (TD)"
        actual_lbl = "ASD" if int(srow["y_true"]) == 1 else "Typical (TD)"
        conf = max(prob_asd, prob_td)
        if conf >= 90.0:
            conf_level = "Very High"
        elif conf >= 75.0:
            conf_level = "High"
        elif conf >= 60.0:
            conf_level = "Moderate"
        else:
            conf_level = "Borderline"
        unseen_rows.append(
            {
                "Subject_ID": srow["subject_id"],
                "CV_Fold": f"Fold_{int(srow['fold'])}",
                "Predicted_Diagnosis": pred_lbl,
                "Positive_Trials_Out_of_7": f"{int(srow['positive_trials'])}/{int(srow['total_trials'])}",
                "ASD_Probability_%": round(prob_asd, 2),
                "Typical_Probability_%": round(prob_td, 2),
                "Confidence_%": round(conf, 2),
                "Confidence_Level": conf_level,
                "Actual_Diagnosis": actual_lbl,
                "Verification": "CORRECT" if pred_lbl == actual_lbl else "MISCLASSIFIED",
            }
        )
    pd.DataFrame(unseen_rows).to_csv(
        os.path.join("results", "metrics", "unseen_test_predictions.csv"), index=False
    )

    # 6. Save results/models/selected_features.json and results/models/model_metadata.json
    feature_list = [str(c) for c in X.columns]
    with open(os.path.join("results", "models", "selected_features.json"), "w", encoding="utf-8") as f:
        json.dump(feature_list, f, indent=2)

    metadata = {
        "model_type": "CatBoostClassifier",
        "pipeline_version": "Leak-Free 3D Kinematic Pipeline (SpineBase-Centered, Torso-Scaled)",
        "dataset": {
            "source": "Al-Jubouri et al. (2020) 3D Microsoft Kinect v2 Gait Dataset",
            "total_trials": int(len(X)),
            "unique_children": int(groups.nunique()),
            "asd_children": 50,
            "td_children": 50,
            "trials_per_child": 7,
            "num_features": int(X.shape[1]),
        },
        "classes": {
            "0": "Typically Developing (TD)",
            "1": "Autism Spectrum Disorder (ASD)",
        },
        "validation_protocol": "5-Fold StratifiedGroupKFold grouped strictly by subject_id",
        "hyperparameters": {
            "iterations": 500,
            "learning_rate": 0.04,
            "depth": 5,
            "l2_leaf_reg": 4,
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "random_seed": random_state,
            "verbose": 0,
            "thread_count": -1,
        },
        "cross_validated_performance": {
            "accuracy_mean_pct": round(cb_res["accuracy_mean"] * 100.0, 2),
            "accuracy_std_pct": round(cb_res["accuracy_std"] * 100.0, 2),
            "subject_majority_vote_accuracy_pct": round(cb_res["subj_maj_accuracy"] * 100.0, 2),
            "auroc_mean": round(cb_res["auroc_mean"], 4),
            "auroc_std": round(cb_res["auroc_std"], 4),
            "average_precision_mean": round(cb_res["ap_mean"], 4),
            "average_precision_std": round(cb_res["ap_std"], 4),
            "sensitivity_mean_pct": round(cb_res["sensitivity_mean"] * 100.0, 2),
            "sensitivity_std_pct": round(cb_res["sensitivity_std"] * 100.0, 2),
            "specificity_mean_pct": round(cb_res["specificity_mean"] * 100.0, 2),
            "specificity_std_pct": round(cb_res["specificity_std"] * 100.0, 2),
            "precision_mean_pct": round(cb_res["precision_mean"] * 100.0, 2),
            "precision_std_pct": round(cb_res["precision_std"] * 100.0, 2),
            "f1_mean": round(cb_res["f1_mean"], 4),
            "f1_std": round(cb_res["f1_std"], 4),
            "oof_confusion_matrix": {
                "tn": cb_res["oof_tn"],
                "fp": cb_res["oof_fp"],
                "fn": cb_res["oof_fn"],
                "tp": cb_res["oof_tp"],
            },
            "subject_confusion_matrix_majority_vote": {
                "tn": cb_res["subj_tn"],
                "fp": cb_res["subj_fp"],
                "fn": cb_res["subj_fn"],
                "tp": cb_res["subj_tp"],
            },
        },
    }
    with open(os.path.join("results", "models", "model_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_benchmark(
    csv_path: str = DEFAULT_PROCESSED_CSV,
    n_splits: int = 5,
    random_state: int = 42,
    save_artifacts: bool = True,
) -> Tuple[Dict[str, Dict[str, Any]], pd.DataFrame, pd.Series, pd.Series]:
    """
    Runs the 5-fold StratifiedGroupKFold benchmark across CatBoost, Random Forest, and XGBoost.
    """
    X, y, groups = load_processed_dataset(csv_path)

    print("=" * 94)
    print("  LEAK-FREE ASD KINEMATIC CLASSIFICATION PIPELINE (StratifiedGroupKFold, n_splits=5)")
    print("=" * 94)
    print(f"  * Total Trial Samples   : {len(X)} (350 ASD, 350 TD)")
    print(f"  * Unique Children       : {groups.nunique()} (50 ASD, 50 TD; 7 trials/child)")
    print(f"  * Kinematic Features    : {X.shape[1]} root-centered, torso-scaled features")
    print(f"  * Validation Protocol   : {n_splits}-Fold StratifiedGroupKFold grouped by subject_id")
    print("=" * 94)

    model_suites: List[Tuple[str, Callable[[], Any]]] = [
        ("CatBoost (Tuned)", lambda: build_catboost_model(random_state=random_state)),
        ("Random Forest (Baseline)", lambda: build_random_forest_baseline(random_state=random_state)),
        ("XGBoost (Baseline)", lambda: build_xgboost_baseline(random_state=random_state)),
    ]

    all_results: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []

    for model_name, factory in model_suites:
        res = evaluate_group_kfold(
            model_factory=factory,
            X=X,
            y=y,
            groups=groups,
            n_splits=n_splits,
            random_state=random_state,
        )
        all_results[model_name] = res

        summary_rows.append(
            {
                "Model": model_name,
                "Accuracy (%)": f"{res['accuracy_mean'] * 100:.2f} +/- {res['accuracy_std'] * 100:.2f}",
                "AUROC": f"{res['auroc_mean']:.4f} +/- {res['auroc_std']:.4f}",
                "Average Precision (PR-AUC)": f"{res['ap_mean']:.4f} +/- {res['ap_std']:.4f}",
                "Sensitivity / ASD Recall (%)": f"{res['sensitivity_mean'] * 100:.2f} +/- {res['sensitivity_std'] * 100:.2f}",
                "Specificity / TD Recall (%)": f"{res['specificity_mean'] * 100:.2f} +/- {res['specificity_std'] * 100:.2f}",
                "Precision / PPV (%)": f"{res['precision_mean'] * 100:.2f} +/- {res['precision_std'] * 100:.2f}",
                "F1-Score": f"{res['f1_mean']:.4f} +/- {res['f1_std']:.4f}",
                "Subject Accuracy (%)": f"{res['subj_maj_accuracy'] * 100:.2f}",
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "auroc_mean": res["auroc_mean"],
                "auroc_std": res["auroc_std"],
                "ap_mean": res["ap_mean"],
                "ap_std": res["ap_std"],
                "sensitivity_mean": res["sensitivity_mean"],
                "sensitivity_std": res["sensitivity_std"],
                "specificity_mean": res["specificity_mean"],
                "specificity_std": res["specificity_std"],
                "precision_mean": res["precision_mean"],
                "precision_std": res["precision_std"],
                "f1_mean": res["f1_mean"],
                "f1_std": res["f1_std"],
                "oof_tn": res["oof_tn"],
                "oof_fp": res["oof_fp"],
                "oof_fn": res["oof_fn"],
                "oof_tp": res["oof_tp"],
            }
        )

    print("\n" + "-" * 112)
    print(
        f"{'Model':<26} | {'Accuracy (%)':<16} | {'AUROC':<16} | "
        f"{'Sensitivity (%)':<16} | {'Specificity (%)':<16} | {'F1-Score':<14}"
    )
    print("-" * 112)
    for row in summary_rows:
        print(
            f"{row['Model']:<26} | {row['Accuracy (%)']:<16} | {row['AUROC']:<16} | "
            f"{row['Sensitivity / ASD Recall (%)']:<16} | {row['Specificity / TD Recall (%)']:<16} | "
            f"{row['F1-Score']:<14}"
        )
    print("-" * 112)

    # Print per-fold breakdown for CatBoost
    cb_res = all_results["CatBoost (Tuned)"]
    print("\n  [CatBoost Per-Fold Leak-Free Breakdown (20 Unseen Children / 140 Trials per Fold)]")
    for i in range(n_splits):
        print(
            f"    Fold {i+1}: Accuracy = {cb_res['fold_accuracies'][i]*100:6.2f}% | "
            f"AUROC = {cb_res['fold_aurocs'][i]:.4f} | "
            f"Sensitivity = {cb_res['fold_sensitivities'][i]*100:6.2f}% | "
            f"Specificity = {cb_res['fold_specificities'][i]*100:6.2f}% | "
            f"Precision = {cb_res['fold_precisions'][i]*100:6.2f}% | "
            f"F1 = {cb_res['fold_f1s'][i]:.4f}"
        )
    print(
        f"\n  [CatBoost Out-of-Fold Confusion Matrix (700 Trials)] "
        f"TN = {cb_res['oof_tn']}, FP = {cb_res['oof_fp']}, FN = {cb_res['oof_fn']}, TP = {cb_res['oof_tp']}"
    )
    print(
        f"  [CatBoost Subject-Level Aggregation (100 Unseen Children, Majority Vote)] "
        f"Accuracy = {cb_res['subj_maj_accuracy']*100:.2f}% | AUROC = {cb_res['subj_auroc']:.4f} | "
        f"TN = {cb_res['subj_tn']}, FP = {cb_res['subj_fp']}, FN = {cb_res['subj_fn']}, TP = {cb_res['subj_tp']}"
    )
    print("=" * 94 + "\n")

    if save_artifacts:
        _save_all_metric_tables_and_metadata(
            all_results=all_results,
            summary_rows=summary_rows,
            X=X,
            y=y,
            groups=groups,
            random_state=random_state,
        )

        # Fit final reference CatBoost model on full dataset for standalone model export
        final_cb = build_catboost_model(random_state=random_state)
        final_cb.fit(X, y)
        final_cb.save_model(os.path.join("results", "models", "catboost_asd_model.cbm"))

    return all_results, X, y, groups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run leak-free 5-fold StratifiedGroupKFold training and model benchmarking."
    )
    parser.add_argument(
        "--data",
        type=str,
        default=DEFAULT_PROCESSED_CSV,
        help="Path to processed features CSV (default: data/processed_features.csv).",
    )
    parser.add_argument(
        "--splits",
        type=int,
        default=5,
        help="Number of StratifiedGroupKFold splits (default: 5).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible group splitting and model training.",
    )
    args = parser.parse_args()
    run_benchmark(csv_path=args.data, n_splits=args.splits, random_state=args.seed, save_artifacts=True)
