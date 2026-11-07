"""
Leak-Free Training and Benchmarking Module for the Al-Jubouri 3D Kinect v2 ASD Dataset.

Implements StratifiedGroupKFold(n_splits=5) grouped strictly by `subject_id` (Child_01 to Child_100)
so that all 7 augmented trials of any child reside exclusively in EITHER the training partition
OR the test partition for every fold.

Benchmarks:
1. CatBoostClassifier (iterations=500, depth=5, learning_rate=0.04, l2_leaf_reg=4)
2. RandomForestClassifier (baseline)
3. XGBClassifier (baseline)

Computes and prints Mean +/- Std for:
- Accuracy
- AUROC
- Sensitivity (ASD Recall)
- Specificity (TD Recall)
- F1-Score
across all 5 subject-grouped folds.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable, Dict, List, Tuple
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
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
    fold_sensitivities: List[float] = []
    fold_specificities: List[float] = []
    fold_f1s: List[float] = []
    fold_roc_data: List[Tuple[np.ndarray, np.ndarray]] = []
    models_per_fold: List[Any] = []

    oof_probs = np.zeros(len(y), dtype=np.float64)
    oof_preds = np.zeros(len(y), dtype=int)

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

        acc = float(accuracy_score(y_test, preds))
        auroc = float(roc_auc_score(y_test, probs))
        sens = float(recall_score(y_test, preds, pos_label=1, zero_division=0))
        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        f1 = float(f1_score(y_test, preds, pos_label=1, zero_division=0))

        fold_accuracies.append(acc)
        fold_aurocs.append(auroc)
        fold_sensitivities.append(sens)
        fold_specificities.append(spec)
        fold_f1s.append(f1)
        fold_roc_data.append((y_test.to_numpy(), probs))
        models_per_fold.append(model)

    return {
        "accuracy_mean": float(np.mean(fold_accuracies)),
        "accuracy_std": float(np.std(fold_accuracies)),
        "auroc_mean": float(np.mean(fold_aurocs)),
        "auroc_std": float(np.std(fold_aurocs)),
        "sensitivity_mean": float(np.mean(fold_sensitivities)),
        "sensitivity_std": float(np.std(fold_sensitivities)),
        "specificity_mean": float(np.mean(fold_specificities)),
        "specificity_std": float(np.std(fold_specificities)),
        "f1_mean": float(np.mean(fold_f1s)),
        "f1_std": float(np.std(fold_f1s)),
        "fold_accuracies": fold_accuracies,
        "fold_aurocs": fold_aurocs,
        "fold_sensitivities": fold_sensitivities,
        "fold_specificities": fold_specificities,
        "fold_f1s": fold_f1s,
        "oof_probs": oof_probs,
        "oof_preds": oof_preds,
        "fold_roc_data": fold_roc_data,
        "models_per_fold": models_per_fold,
    }


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
                "Sensitivity / ASD Recall (%)": f"{res['sensitivity_mean'] * 100:.2f} +/- {res['sensitivity_std'] * 100:.2f}",
                "Specificity / TD Recall (%)": f"{res['specificity_mean'] * 100:.2f} +/- {res['specificity_std'] * 100:.2f}",
                "F1-Score": f"{res['f1_mean']:.4f} +/- {res['f1_std']:.4f}",
                "accuracy_mean": res["accuracy_mean"],
                "accuracy_std": res["accuracy_std"],
                "auroc_mean": res["auroc_mean"],
                "auroc_std": res["auroc_std"],
                "sensitivity_mean": res["sensitivity_mean"],
                "sensitivity_std": res["sensitivity_std"],
                "specificity_mean": res["specificity_mean"],
                "specificity_std": res["specificity_std"],
                "f1_mean": res["f1_mean"],
                "f1_std": res["f1_std"],
            }
        )

    print("\n" + "-" * 94)
    print(
        f"{'Model':<26} | {'Accuracy (%)':<16} | {'AUROC':<16} | "
        f"{'Sensitivity (%)':<16} | {'Specificity (%)':<16} | {'F1-Score':<14}"
    )
    print("-" * 94)
    for row in summary_rows:
        print(
            f"{row['Model']:<26} | {row['Accuracy (%)']:<16} | {row['AUROC']:<16} | "
            f"{row['Sensitivity / ASD Recall (%)']:<16} | {row['Specificity / TD Recall (%)']:<16} | "
            f"{row['F1-Score']:<14}"
        )
    print("-" * 94)

    # Print per-fold breakdown for CatBoost
    cb_res = all_results["CatBoost (Tuned)"]
    print("\n  [CatBoost Per-Fold Leak-Free Breakdown]")
    for i in range(n_splits):
        print(
            f"    Fold {i+1}: Accuracy = {cb_res['fold_accuracies'][i]*100:6.2f}% | "
            f"AUROC = {cb_res['fold_aurocs'][i]:.4f} | "
            f"Sensitivity = {cb_res['fold_sensitivities'][i]*100:6.2f}% | "
            f"Specificity = {cb_res['fold_specificities'][i]*100:6.2f}% | "
            f"F1 = {cb_res['fold_f1s'][i]:.4f}"
        )
    print("=" * 94 + "\n")

    if save_artifacts:
        os.makedirs("results", exist_ok=True)
        os.makedirs(os.path.join("results", "models"), exist_ok=True)
        pd.DataFrame(summary_rows).to_csv(os.path.join("results", "benchmark_metrics.csv"), index=False)

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
