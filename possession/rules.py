"""No-training fallback: simple football evidence turned into log P(H), P(A), P(D) per frame."""

import numpy as np

MARGIN_SCALE = 2.0        # metres of nearest-player margin that gives about 73% confidence
GAP_SOFTEN_SECONDS = 4.0  # confidence halves after roughly this long without a ball
DEAD_PRIOR = -0.5         # logit of dead ball when nothing else is known
OUT_BONUS = 4.0           # ball outside the lines
GAP_BONUS = 2.5           # ball missing for a long time
GAP_START, GAP_RAMP = 4.0, 8.0   # seconds before the long-gap evidence starts, and how fast it grows
SLOW_BONUS = 2.5          # everybody nearly standing still
SLOW_SPEED, SLOW_RAMP = 1.5, 1.0  # m/s speed below which "slow" starts, and its ramp
LOG_FLOOR = 1e-4


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rule_logp(X, names):
    """Log-probabilities [T,3] in order H, A, D from the clue matrix and its names."""
    col = {n: X[:, i] for i, n in enumerate(names)}
    margin = np.nan_to_num(col["dd_fill"], nan=0.0)          # home nearest minus away nearest
    gap = col["gap_len"]
    scale = MARGIN_SCALE * (1.0 + gap / GAP_SOFTEN_SECONDS)  # carry-over: less sure the longer the gap
    p_home = _sigmoid(-margin / scale)

    out = np.where(col["ball_vis"] > 0, col["ball_out"],
                   0.5 * (col["ball_out_last"] + col["ball_out_next"]))
    out = np.nan_to_num(out, nan=0.0)
    long_gap = np.clip((gap - GAP_START) / GAP_RAMP, 0.0, 1.0)
    slow = np.clip((SLOW_SPEED - np.nan_to_num(col["all_speed_m1s"], nan=SLOW_SPEED)) / SLOW_RAMP,
                   0.0, 1.0)
    dead = _sigmoid(DEAD_PRIOR + OUT_BONUS * (out > 0.5) + GAP_BONUS * long_gap + SLOW_BONUS * slow)

    prob = np.stack([(1 - dead) * p_home, (1 - dead) * (1 - p_home), dead], axis=1)
    return np.log(np.clip(prob, LOG_FLOOR, 1.0))
