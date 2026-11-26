"""
Train and evaluate two churn models, then write metrics and figures to reports/.

    uv run python -m churn.train
"""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from churn.data import REPORTS_DIR, load_players
from churn.features import FEATURES, make_preprocessor, split_xy

SEED = 42
TEST_SIZE = 0.2


def build_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("prep", make_preprocessor(scale=True)),
                ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", make_preprocessor(scale=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=300,
                        early_stopping=True,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
    }


def evaluate(y_true: pd.Series, proba, threshold: float = 0.5) -> dict:
    """Ranking quality (ROC-AUC, PR-AUC) plus what happens at one decision threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "roc_auc": round(roc_auc_score(y_true, proba), 3),
        "pr_auc": round(average_precision_score(y_true, proba), 3),
        "threshold": threshold,
        "precision": round(precision_score(y_true, pred), 3),
        "recall": round(recall_score(y_true, pred), 3),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def top_k_lift(y_true: pd.Series, proba, share: float = 0.2) -> dict:
    """If the team can only contact the riskiest `share` of players, how many churners do they reach?"""
    df = pd.DataFrame({"y": y_true.to_numpy(), "p": proba}).sort_values("p", ascending=False)
    k = int(len(df) * share)
    top = df.head(k)
    base_rate = df["y"].mean()
    return {
        "share_contacted": share,
        "churn_rate_in_top": round(top["y"].mean(), 3),
        "base_churn_rate": round(base_rate, 3),
        "lift": round(top["y"].mean() / base_rate, 2),
        "share_of_all_churners_reached": round(top["y"].sum() / df["y"].sum(), 3),
    }


def main() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    df = load_players()
    X, y = split_xy(df)

    # Stratified: train and test keep the same churn rate.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    results = {
        "players": len(df),
        "churn_rate": round(y.mean(), 3),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "models": {},
    }
    fitted, probas = {}, {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for name, pipe in build_models().items():
        # 5-fold cross-validation on the training set: is the score stable, or luck of one split?
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        fitted[name], probas[name] = pipe, proba
        results["models"][name] = (
            {"cv_roc_auc_mean": round(cv_auc.mean(), 3), "cv_roc_auc_std": round(cv_auc.std(), 3)}
            | evaluate(y_test, proba)
            | {"top_20pct": top_k_lift(y_test, proba)}
        )

    best = max(results["models"], key=lambda m: results["models"][m]["roc_auc"])
    results["best_model"] = best

    # Which inputs matter: shuffle one column at a time on the test set and see
    # how much ROC-AUC drops. Works for any model, in the original column names.
    imp = permutation_importance(
        fitted[best], X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=SEED
    )
    importance = (
        pd.DataFrame({"feature": FEATURES, "auc_drop": imp.importances_mean, "std": imp.importances_std})
        .sort_values("auc_drop", ascending=False)
        .round(4)
    )
    results["feature_importance"] = importance.to_dict(orient="records")

    # Direction of the effect, from the (interpretable) logistic regression.
    logit = fitted["logistic_regression"]
    coefs = pd.Series(
        logit.named_steps["model"].coef_[0],
        index=logit.named_steps["prep"].get_feature_names_out(),
    )
    results["logistic_coefficients_behaviour"] = (
        coefs[coefs.index.str.startswith("behaviour__")].round(3).sort_values().to_dict()
    )

    (REPORTS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))
    plot_roc(y_test, probas)
    plot_importance(importance)
    plot_calibration(y_test, probas)
    print(json.dumps({k: v for k, v in results.items() if k != "feature_importance"}, indent=2))


def plot_roc(y_test, probas) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, proba in probas.items():
        RocCurveDisplay.from_predictions(y_test, proba, name=name.replace("_", " "), ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="random guess")
    ax.set_title("ROC curve, test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "roc_curve.png", dpi=120)
    plt.close(fig)


def plot_importance(importance: pd.DataFrame) -> None:
    top = importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(top["feature"], top["auc_drop"], xerr=top["std"], color="#4C72B0")
    ax.set_xlabel("drop in ROC-AUC when the feature is shuffled")
    ax.set_title("Permutation importance (best model)")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "feature_importance.png", dpi=120)
    plt.close(fig)


def plot_calibration(y_test, probas) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, proba in probas.items():
        CalibrationDisplay.from_predictions(y_test, proba, n_bins=10, name=name.replace("_", " "), ax=ax)
    ax.set_title("Calibration: predicted vs. actual churn rate")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "calibration.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
