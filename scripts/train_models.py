#!/usr/bin/env python3
"""
Train and evaluate baseline and ensemble models for fraud detection.

Features:
- Stratified train-test split (already in processed files) or fallback to raw
- Baseline: Logistic Regression with class_weight balancing
- Ensemble: XGBoost with basic hyperparameter tuning
- Metrics: AUC-PR (Average Precision), F1-score, Confusion Matrix
- StratifiedKFold CV (k=5): mean and std for metrics
- Model comparison and best model selection with justification
- Saves results and models under models/

Usage:
    python scripts/train_models.py --dataset creditcard
    python scripts/train_models.py --dataset ecommerce

Options:
    --test-size 0.2            Proportion for test split if using raw
    --random-state 42          RNG seed
    --ensemble xgboost         Ensemble choice: xgboost or random_forest
    --cv-folds 5               Number of folds for StratifiedKFold
    --output-dir models        Directory to save artifacts
    --sample-frac 0.0          Optional fraction to sample for speed (0 disables)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:
    HAS_XGB = False

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
MODELS_DIR = ROOT / "models"


def _load_processed(dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load processed train/test and return target column name."""
    if dataset == "creditcard":
        train_path = DATA_PROCESSED / "creditcard_train_processed.csv"
        test_path = DATA_PROCESSED / "creditcard_test_processed.csv"
        target = "Class"
    elif dataset in {"ecommerce", "fraud_data"}:
        train_path = DATA_PROCESSED / "ecommerce_train_processed.csv"
        test_path = DATA_PROCESSED / "ecommerce_test_processed.csv"
        target = "class"
    else:
        raise ValueError("dataset must be 'creditcard' or 'ecommerce'/'fraud_data'")

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Processed files not found: {train_path}, {test_path}")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test, target


def _load_raw(
    dataset: str,
    test_size: float,
    random_state: int,
    sample_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load raw data and perform stratified split with minimal preprocessing.
    This path is a fallback when processed files are missing."""
    if dataset == "creditcard":
        path = DATA_RAW / "creditcard.csv"
        target = "Class"
    elif dataset in {"ecommerce", "fraud_data"}:
        path = DATA_RAW / "Fraud_Data.csv"
        target = "class"
    else:
        raise ValueError("dataset must be 'creditcard' or 'ecommerce'/'fraud_data'")

    df = pd.read_csv(path)
    if sample_frac and 0.0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=random_state)

    y = df[target]
    X = df.drop(columns=[target])

    # Minimal numeric scaling; one-hots will remain stable
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    train = pd.concat([X_train, y_train.rename(target)], axis=1)
    test = pd.concat([X_test, y_test.rename(target)], axis=1)
    return train, test, target


def _scale_pos_weight(y: pd.Series) -> float:
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    return (neg / pos) if pos > 0 else 1.0


def build_logreg(random_state: int) -> Pipeline:
    # Use saga solver with stronger convergence settings to avoid warnings
    model = LogisticRegression(
        max_iter=5000,
        tol=1e-3,
        penalty="l2",
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
        random_state=random_state,
    )
    return Pipeline([("clf", model)])


def build_xgb(random_state: int, scale_pos_weight: float) -> XGBClassifier:
    if not HAS_XGB:
        raise RuntimeError("xgboost not installed; please install or choose random_forest")
    return XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        tree_method="hist",
    )


def build_rf(random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=random_state,
        class_weight="balanced_subsample",
    )


def evaluate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float]:
    return {
        "auc_pr": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred)),
    }


def run_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int,
    random_state: int,
) -> dict[str, dict[str, float]]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    aucs: list[float] = []
    f1s: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]
        clf = model
        # Rebuild XGB with fold-specific scale_pos_weight
        if isinstance(model, XGBClassifier):
            clf = build_xgb(random_state, _scale_pos_weight(y_train))
        clf.fit(X_train, y_train)
        y_proba = clf.predict_proba(X_val)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        aucs.append(average_precision_score(y_val, y_proba))
        f1s.append(f1_score(y_val, y_pred))
    return {
        "auc_pr": {
            "mean": float(np.mean(aucs)),
            "std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
        },
        "f1": {
            "mean": float(np.mean(f1s)),
            "std": float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0,
        },
    }


def tune_ensemble(model_name: str, X_train: pd.DataFrame, y_train: pd.Series, random_state: int):
    if model_name == "xgboost":
        if not HAS_XGB:
            raise RuntimeError("xgboost not available")
        base = build_xgb(random_state, _scale_pos_weight(y_train))
        param_grid = {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        }
        grid = GridSearchCV(
            base,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            scoring="average_precision",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_, float(grid.best_score_)
    elif model_name == "random_forest":
        base = build_rf(random_state)
        param_grid = {
            "n_estimators": [200, 400, 600],
            "max_depth": [None, 10, 20],
            "max_features": ["sqrt", "log2", 0.8],
        }
        grid = GridSearchCV(
            base,
            param_grid,
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state),
            scoring="average_precision",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        return grid.best_estimator_, grid.best_params_, float(grid.best_score_)
    else:
        raise ValueError("Unsupported ensemble model")


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def select_best(
    baseline_cv: dict[str, dict[str, float]],
    ensemble_cv: dict[str, dict[str, float]],
    interpretability_priority: bool = True,
) -> str:
    # Prefer ensemble if AUC-PR improves by >2 points; otherwise LR for interpretability
    base_auc = baseline_cv["auc_pr"]["mean"]
    ens_auc = ensemble_cv["auc_pr"]["mean"]
    improvement = ens_auc - base_auc
    if improvement > 0.02:  # 2 points improvement threshold
        return "ensemble"
    return "baseline"


def main():
    parser = argparse.ArgumentParser(description="Fraud detection modeling")
    parser.add_argument(
        "--dataset",
        choices=["creditcard", "ecommerce", "fraud_data"],
        required=True,
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--ensemble", choices=["xgboost", "random_forest"], default="xgboost")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR))
    parser.add_argument("--sample-frac", type=float, default=0.0)
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    try:
        train_df, test_df, target = _load_processed(args.dataset)
    except FileNotFoundError:
        print("Processed files missing; falling back to raw with minimal preprocessing.")
        train_df, test_df, target = _load_raw(
            args.dataset,
            args.test_size,
            args.random_state,
            args.sample_frac,
        )

    # Optional sampling for speed
    if args.sample_frac and 0.0 < args.sample_frac < 1.0:
        train_df = train_df.sample(frac=args.sample_frac, random_state=args.random_state)
        # keep a bit more for test
        test_df = test_df.sample(
            frac=min(args.sample_frac * 2, 1.0),
            random_state=args.random_state,
        )

    X_train = train_df.drop(columns=[target])
    y_train = train_df[target].astype(int)
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target].astype(int)

    # Baseline: Logistic Regression
    lr = build_logreg(args.random_state)
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    lr_pred = (lr_proba >= 0.5).astype(int)
    lr_test_metrics = evaluate_metrics(y_test.values, lr_pred, lr_proba)
    lr_conf = confusion(y_test.values, lr_pred)
    lr_cv = run_cv(
        build_logreg(args.random_state),
        X_train,
        y_train,
        args.cv_folds,
        args.random_state,
    )

    # Ensemble: XGBoost or RandomForest, with basic tuning
    if args.ensemble == "xgboost":
        if not HAS_XGB:
            print("xgboost not installed; switching to random_forest.")
            args.ensemble = "random_forest"
    tuned_model, best_params, best_cv_score = tune_ensemble(
        args.ensemble,
        X_train,
        y_train,
        args.random_state,
    )
    tuned_model.fit(X_train, y_train)
    ens_proba = tuned_model.predict_proba(X_test)[:, 1]
    ens_pred = (ens_proba >= 0.5).astype(int)
    ens_test_metrics = evaluate_metrics(y_test.values, ens_pred, ens_proba)
    ens_conf = confusion(y_test.values, ens_pred)
    # CV for the tuned model (retrain per fold)
    if args.ensemble == "xgboost":
        ens_cv = run_cv(
            build_xgb(args.random_state, _scale_pos_weight(y_train)),
            X_train,
            y_train,
            args.cv_folds,
            args.random_state,
        )
    else:
        ens_cv = run_cv(
            build_rf(args.random_state),
            X_train,
            y_train,
            args.cv_folds,
            args.random_state,
        )

    # Compare and select best
    choice = select_best(lr_cv, ens_cv, interpretability_priority=True)
    best_model_name = "baseline" if choice == "baseline" else args.ensemble

    # Save artifacts
    results = {
        "dataset": args.dataset,
        "target": target,
        "baseline": {
            "model": "LogisticRegression(class_weight=balanced)",
            "test": {"metrics": lr_test_metrics, "confusion": lr_conf},
            "cv": lr_cv,
        },
        "ensemble": {
            "model": args.ensemble,
            "best_params": best_params,
            "cv_best_score_average_precision": best_cv_score,
            "test": {"metrics": ens_test_metrics, "confusion": ens_conf},
            "cv": ens_cv,
        },
        "selection": {
            "best": best_model_name,
            "justification": (
                "Selected ensemble due to >2 point AUC-PR improvement over baseline"
                if best_model_name != "baseline"
                else (
                    "Selected logistic regression for interpretability; "
                    "ensemble improvement < 2 points"
                )
            ),
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / f"results_{args.dataset}.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    # Persist models
    dump(lr, out_dir / f"lr_{args.dataset}.joblib")
    dump(tuned_model, out_dir / f"{args.ensemble}_{args.dataset}.joblib")

    # Friendly print
    print("\nModeling complete. Summary:")
    print(
        json.dumps(
            {
                "baseline_auc_pr": results["baseline"]["cv"]["auc_pr"]["mean"],
                "ensemble_auc_pr": results["ensemble"]["cv"]["auc_pr"]["mean"],
                "baseline_f1": results["baseline"]["cv"]["f1"]["mean"],
                "ensemble_f1": results["ensemble"]["cv"]["f1"]["mean"],
                "selected": results["selection"]["best"],
            },
            indent=2,
        )
    )
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
