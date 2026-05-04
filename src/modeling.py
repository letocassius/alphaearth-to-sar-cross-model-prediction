"""Model construction, prediction, and persistence."""

from pathlib import Path
import warnings

import joblib
import lightgbm as lgb
import numpy as np
from sklearn.multioutput import MultiOutputRegressor

from .config import RANDOM_STATE


def build_model(n_estimators: int = 700, learning_rate: float = 0.03, num_leaves: int = 31) -> MultiOutputRegressor:
    """Create the LightGBM multi-output regressor F(AlphaEarth embedding) = SAR."""
    base = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        subsample=0.85,
        subsample_freq=1,
        colsample_bytree=0.85,
        min_child_samples=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )
    return MultiOutputRegressor(base, n_jobs=1)


def predict_sar(model: MultiOutputRegressor, X: np.ndarray) -> np.ndarray:
    """Predict SAR bands from AlphaEarth embeddings."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
            category=UserWarning,
        )
        return model.predict(X)


def save_model(model: MultiOutputRegressor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: Path) -> MultiOutputRegressor:
    return joblib.load(path)
