#!/usr/bin/env python3
"""
Standalone Inference & Evaluation Script for the Leak-Free ASD Kinematic Pipeline.

Evaluates the saved CatBoost model (`results/models/catboost_asd_model.cbm`) on a tabular
feature CSV (`data/processed_features.csv` or external test CSV), reports both the 5-Fold
StratifiedGroupKFold out-of-fold generalization benchmark and full-cohort reference model
inference, and displays TreeSHAP motor biomarker attributions.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import shap
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score

from src.explainability import translate_feature_to_clinical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference and SHAP biomarker attribution with the trained ASD CatBoost model."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed_features.csv",
        help="Path to processed kinematic features CSV (default: data/processed_features.csv).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="results/models/catboost_asd_model.cbm",
        help="Path to trained CatBoost model (.cbm).",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Compute and display top TreeSHAP motor biomarkers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model):
        raise FileNotFoundError(
            f"Model file not found at '{args.model}'. Run 'python -m src.train' first."
        )
    if not os.path.exists(args.input):
        raise FileNotFoundError(
            f"Input feature file not found at '{args.input}'. Run 'python -m src.feature_extraction' first."
        )

    model = CatBoostClassifier()
    model.load_model(args.model)

    df = pd.read_csv(args.input)
    y_true = df["label"].astype(int) if "label" in df.columns else None
    subjects = df["subject_id"].astype(str) if "subject_id" in df.columns else None

    feature_cols = [c for c in df.columns if c not in ("label", "subject_id")]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    print("=" * 90)
    print("  ASD 3D KINEMATIC CLASSIFICATION — MODEL INFERENCE & BENCHMARK VERIFICATION")
    print("=" * 90)
    print(f"  * Input Feature Matrix      : {args.input} ({len(X)} trials, {len(feature_cols)} features)")
    if subjects is not None:
        print(f"  * Unique Children (Groups)  : {subjects.nunique()} (7 trials/child)")

    eval_csv = os.path.join("results", "metrics", "evaluation_metrics.csv")
    if os.path.exists(eval_csv):
        ev = pd.read_csv(eval_csv).iloc[0]
        print(
            f"  * 5-Fold GroupCV Benchmark  : Acc = {ev['accuracy_mean']*100:.2f}% +/- {ev['accuracy_std']*100:.2f}% "
            f"(Subject Acc = {ev['subject_majority_vote_accuracy']*100:.2f}%)"
        )
        print(
            f"  * 5-Fold GroupCV Discrimination : AUROC = {ev['roc_auc_mean']:.4f} +/- {ev['roc_auc_std']:.4f} | "
            f"Sens = {ev['sensitivity_mean']*100:.2f}% | Spec = {ev['specificity_mean']*100:.2f}% | "
            f"F1 = {ev['f1_mean']:.4f}"
        )

    if y_true is not None:
        acc = accuracy_score(y_true, preds)
        auroc = roc_auc_score(y_true, probs)
        sens = recall_score(y_true, preds, pos_label=1)
        f1 = f1_score(y_true, preds, pos_label=1)
        print(
            f"  * Full-Cohort Reference Fit : Acc = {acc * 100:.2f}% | AUROC = {auroc:.4f} | "
            f"Sens = {sens * 100:.2f}% | F1 = {f1:.4f}"
        )
    print("=" * 90)

    if args.explain:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        mean_abs = np.abs(shap_vals).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:15]
        print("\n  Top 15 Motor Biomarkers (TreeSHAP Attribution):")
        for rank, idx in enumerate(top_idx, start=1):
            fname = feature_cols[idx]
            print(f"    {rank:2d}. {fname:<42} (Mean |SHAP| = {mean_abs[idx]:.4f})")
            print(f"        -> {translate_feature_to_clinical(fname)}")
        print("=" * 90)


if __name__ == "__main__":
    main()
