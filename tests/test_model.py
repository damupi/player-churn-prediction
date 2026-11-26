import numpy as np
import pandas as pd

from churn.features import FEATURES, split_xy
from churn.train import build_models, evaluate, top_k_lift


def fake_players(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Synthetic players where churn depends on engagement, so a model can learn it."""
    rng = np.random.default_rng(seed)
    engagement = rng.exponential(300, n)
    churned = (rng.random(n) < 1 / (1 + np.exp(-(engagement - 300) / 150))).astype(int)
    df = pd.DataFrame({
        "churned": churned,
        "sessions": rng.integers(0, 4, n),
        "engagement_seconds": engagement,
        "screen_views": rng.integers(0, 50, n),
        "levels_started": rng.integers(0, 20, n),
        "levels_completed": rng.integers(0, 5, n),
        "levels_failed": rng.integers(0, 5, n),
        "levels_reset": rng.integers(0, 5, n),
        "scores_posted": rng.integers(0, 5, n),
        "content_selected": rng.integers(0, 5, n),
        "first_hour": rng.integers(0, 24, n),
        "country": rng.choice(["United States", "India", None], n),
        "operating_system": rng.choice(["IOS", "ANDROID"], n),
        "first_day_of_week": rng.integers(1, 8, n),
    })
    return df


def test_split_xy_fills_missing_categories():
    X, y = split_xy(fake_players())
    assert list(X.columns) == FEATURES
    assert X["country"].isna().sum() == 0
    assert set(y.unique()) <= {0, 1}


def test_models_learn_a_real_signal():
    X, y = split_xy(fake_players(800))
    for pipe in build_models().values():
        pipe.fit(X[:600], y[:600])
        proba = pipe.predict_proba(X[600:])[:, 1]
        assert evaluate(y[600:], proba)["roc_auc"] > 0.65


def test_top_k_lift_perfect_ranking():
    y = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    proba = np.linspace(1, 0, 10)
    r = top_k_lift(y, proba, share=0.2)
    assert r["churn_rate_in_top"] == 1.0
    assert r["share_of_all_churners_reached"] == 1.0
    assert r["lift"] == 5.0
