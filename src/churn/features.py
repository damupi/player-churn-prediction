"""Which columns the model uses, and how they are prepared."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "churned"

# Behaviour in the player's first 24 hours (counts and time).
BEHAVIOUR = [
    "sessions",
    "engagement_seconds",
    "screen_views",
    "levels_started",
    "levels_completed",
    "levels_failed",
    "levels_reset",
    "scores_posted",
    "content_selected",
]

# Context of the first visit.
CATEGORICAL = ["country", "operating_system", "first_day_of_week"]
NUMERIC_CONTEXT = ["first_hour"]

FEATURES = BEHAVIOUR + NUMERIC_CONTEXT + CATEGORICAL

# Countries with fewer players than this are grouped into one "infrequent" bucket,
# so the model does not learn from a handful of players in a rare country.
MIN_COUNTRY_PLAYERS = 50


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Features (X) and label (y). Missing categories become the string 'unknown'."""
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = X[col].fillna("unknown").astype(str)
    y = df[TARGET].astype(int)
    return X, y


def make_preprocessor(scale: bool) -> ColumnTransformer:
    """
    Behaviour counts are very skewed (most players do little, a few do a lot),
    so they go through log(1 + x). The linear model also needs them on a common
    scale; tree models do not care, so scaling is optional.
    """
    numeric_steps = [FunctionTransformer(np.log1p, feature_names_out="one-to-one")]
    if scale:
        numeric_steps.append(StandardScaler())

    return ColumnTransformer(
        [
            ("behaviour", make_pipeline(*numeric_steps), BEHAVIOUR),
            ("hour", StandardScaler() if scale else "passthrough", NUMERIC_CONTEXT),
            (
                "context",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=MIN_COUNTRY_PLAYERS,
                    sparse_output=False,
                ),
                CATEGORICAL,
            ),
        ]
    )
