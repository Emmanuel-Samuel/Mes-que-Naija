"""Scoring helpers around the official possession metric (evaluation/possession_metrics.py)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
import possession_metrics as pm  # noqa: E402

H, A, D, X = pm.H, pm.A, pm.D, pm.X


def runs_from_half(half):
    """Half-open (start, stop) spans, one per half value above 0, first to last frame of that half."""
    half = np.asarray(half)
    runs = []
    for value in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == value)
        runs.append((int(idx[0]), int(idx[-1]) + 1))
    return runs


def spell_f1(gt, pred, half):
    """Official spellF1 plus counts and durations. Frames with half == 0 are not scored."""
    gt = np.asarray(gt, dtype=np.int8).copy()
    pred = np.asarray(pred, dtype=np.int8)
    gt[np.asarray(half) == 0] = X
    runs = runs_from_half(half)
    filled = pm.fill_abstentions(pred, runs)

    spells = pm.spell_scores(gt, filled, runs)
    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    gt_stats = pm.possession_stats(gt, runs)
    pred_stats = pm.possession_stats(filled, runs)
    return {
        "spell_f1": spells["mean_f1"],
        "per_iou": {iou: v["f1"] for iou, v in spells["per_iou"].items()},
        "macro_f1": pm.macro_f1(cm3),
        "n_gt": spells["n_gt"],
        "n_pred": spells["n_pred"],
        "gt_mean_duration_s": gt_stats["mean_duration_s"],
        "pred_mean_duration_s": pred_stats["mean_duration_s"],
    }
