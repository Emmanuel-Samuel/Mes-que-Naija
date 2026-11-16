"""The per-frame judge: a gradient-boosted classifier from clues to log P(H), P(A), P(D)."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

PARAMS = dict(max_iter=150, learning_rate=0.08, max_leaf_nodes=24, min_samples_leaf=100,
              l2_regularization=1.0, early_stopping=False, random_state=0)
LOG_FLOOR = 1e-4


def train_judge(X_list, y_list, every=5):
    """Fit on every `every`-th frame of each (X, y). Frames with y outside 0..2 are skipped."""
    X = np.concatenate([x[::every] for x in X_list])
    y = np.concatenate([v[::every] for v in y_list])
    keep = y <= 2
    model = HistGradientBoostingClassifier(**PARAMS)
    return model.fit(X[keep], y[keep])


def predict_logp(model, X):
    """Log-probabilities [T,3] in label order H, A, D (floored so no class is ever impossible)."""
    return np.log(np.clip(model.predict_proba(X), LOG_FLOOR, 1.0))
