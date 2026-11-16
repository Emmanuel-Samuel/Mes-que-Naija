"""Leave-one-match-out check on the 7 German matches, degraded to look like the competition match.

    .venv/bin/python scripts/cv_german.py
Decoder settings are chosen on the 6 training matches only, using predictions that those matches
got from models which never saw them (a nested inner loop), then scored once on the held-out match
against the translated labels. Writes the settings chosen on all 7 out-of-fold predictions to
possession/decoder_settings.json for make_submission.py. Only whole-match aggregates are printed.
"""

import json
import os
import sys
import time

import numpy as np
from sklearn.inspection import permutation_importance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession.canonical import load_cached  # noqa: E402
from possession.decoder import decode, grid_search  # noqa: E402
from possession.degrade import compare, degrade  # noqa: E402
from possession.features import build_features  # noqa: E402
from possession.judge import predict_logp, train_judge  # noqa: E402
from possession.load_idsse import MATCH_IDS  # noqa: E402
from possession.load_idsse_events import load_restarts  # noqa: E402
from possession.tagging import make_pseudo_tags, tag_features  # noqa: E402
from possession.load_metrica import NAME as REFERENCE  # noqa: E402
from possession.scoring import spell_f1  # noqa: E402
from possession.translate_labels import translate  # noqa: E402

SEED = 0
CONTROL_SECONDS = 2.0
N_JOBS = 18
TRAIN_SEEDS = (0, 1, 2)   # each training match is degraded three ways; the held-out one with SEED
TRAIN_EVERY = 10
GRID = {
    "emission_weight": [0.1, 0.3, 0.6, 1.0],
    "switch_cost": [2.0, 4.0, 8.0, 16.0],
    "dead_in_cost": [8.0, 14.0],
    "dead_out_cost": [8.0, 14.0],
    "min_control_frames": [25, 50, 75],
    "min_dead_frames": [12, 25, 50],
}
DECODER_PATH = os.path.join(ROOT, "possession", "decoder_settings.json")


def baseline_labels(X, names, window=50):
    """Starter method: nearest player's team, majority vote over +/- window frames, D if no votes."""
    col = {n: X[:, i] for i, n in enumerate(names)}
    dh, da = col["dh1"], col["da1"]
    home = np.nan_to_num(dh, nan=np.inf) < np.nan_to_num(da, nan=np.inf)
    away = np.nan_to_num(da, nan=np.inf) < np.nan_to_num(dh, nan=np.inf)
    n = len(X)
    csum_h = np.concatenate(([0], np.cumsum(home)))
    csum_a = np.concatenate(([0], np.cumsum(away)))
    idx = np.arange(n)
    lo, hi = np.maximum(0, idx - window), np.minimum(n, idx + window + 1)
    h, a = csum_h[hi] - csum_h[lo], csum_a[hi] - csum_a[lo]
    return np.where(h + a == 0, 2, np.where(h >= a, 0, 1)).astype(np.int8)


def score(gt, pred, half):
    r = spell_f1(gt, pred, half)
    g = half > 0
    dead = (int(np.sum((pred == 2) & (gt == 2) & g)), int(np.sum((pred == 2) & g)),
            int(np.sum((gt == 2) & g)))
    return r["spell_f1"], r["macro_f1"], r["n_pred"], r["n_gt"], dead


def tune(logps, halves, gts, idx):
    """Best decoder params on the matches in idx. Returns (params, mean spellF1)."""
    data = [(logps[i], halves[i], gts[i]) for i in idx]
    return grid_search(data, GRID, n_jobs=N_JOBS)[0]


def main():
    start = time.time()
    reference = load_cached(REFERENCE)
    raw = [load_cached(n) for n in MATCH_IDS]
    restarts = [load_restarts(n) for n in MATCH_IDS]
    k = len(raw)
    deg = {(m, sd): degrade(raw[m], reference, sd) for m in range(k) for sd in TRAIN_SEEDS}
    base, tagx = {}, {}
    for (m, sd), d in deg.items():
        X, names = build_features(d)
        tx, tnames = tag_features(make_pseudo_tags(restarts[m], 1000 * m + sd), d.n_frames, d.fps)
        base[m, sd], tagx[m, sd] = X, np.hstack([X, tx])
    variants = {"no_tags": (base, names), "tags": (tagx, names + tnames)}
    gts = [translate(m.label, m.half, m.fps, CONTROL_SECONDS) for m in raw]
    halves = [m.half for m in raw]
    print("degraded and featurised: {:.0f}s".format(time.time() - start), flush=True)

    print("degrade check, in-play frames (reference vs each degraded German match, seed {})".format(SEED))
    cols = ["missing_rate", "gaps", "gap_median_s", "gap_p90_s", "players_mean", "v_slow05",
            "v_slow2", "v_speed_med", "v_outside", "slow_given_seen", "slow_given_missing"]
    print("  {:<8}".format("") + "".join("{:>11}".format(c[:10]) for c in cols))
    for m in range(k):
        c = compare(reference, deg[m, SEED])
        if m == 0:
            print("  {:<8}".format("BAR-ATH") + "".join("{:>11.3f}".format(c["reference"][c_]) for c_ in cols))
        print("  {:<8}".format(raw[m].name) + "".join("{:>11.3f}".format(c["degraded"][c_]) for c_ in cols))

    results = {"baseline": [score(gts[m], baseline_labels(base[m, SEED], names), halves[m])
                            for m in range(k)]}
    final, importances = {}, {}
    for vname, (feats, vnames) in variants.items():
        def fit(idx):
            sets = [(feats[i, sd], gts[i]) for i in idx for sd in TRAIN_SEEDS]
            return train_judge([x for x, _ in sets], [y for _, y in sets], every=TRAIN_EVERY)

        models, outer, inner = {}, {}, {}
        for m in range(k):
            models[m] = fit([i for i in range(k) if i != m])
            outer[m] = predict_logp(models[m], feats[m, SEED])
            for i in range(k):
                if i != m:
                    inner[m, i] = predict_logp(fit([j for j in range(k) if j not in (m, i)]),
                                               feats[i, SEED])
        print("[{}] judge models trained: {:.0f}s".format(vname, time.time() - start), flush=True)
        results[vname], chosen = [], []
        for m in range(k):
            train = [i for i in range(k) if i != m]
            p, _ = tune({i: inner[m, i] for i in train}, halves, gts, train)
            chosen.append(p)
            results[vname].append(score(gts[m], decode(outer[m], halves[m], p), halves[m]))
        params, best = grid_search([(outer[i], halves[i], gts[i]) for i in range(k)], GRID,
                                   n_jobs=N_JOBS)[0]
        final[vname] = params.__dict__
        print("[{}] decoder settings (best mean spellF1 {:.3f} on all 7 out-of-fold sets): {}".format(
            vname, best, params))
        for key, values in GRID.items():
            v = getattr(params, key)
            print("    {} = {} {}".format(key, v, "(GRID EDGE)" if v in (min(values), max(values)) else ""))
        imp = np.zeros(len(vnames))
        for m in range(k):
            rows = np.arange(0, len(feats[m, SEED]), 10)
            r = permutation_importance(models[m], feats[m, SEED][rows], gts[m][rows],
                                       scoring="neg_log_loss", n_repeats=2, random_state=0)
            imp += r.importances_mean / k
        importances[vname] = [(vnames[i], imp[i]) for i in np.argsort(-imp)[:12]]

    print("\nper match: spellF1 / macroF1 / possessions predicted vs true")
    print("  {:<8}".format("match") + "".join("{:>24}".format(n) for n in results))
    for m in range(k):
        print("  {:<8}".format(raw[m].name) + "".join(
            "{:>24}".format("{:.3f} {:.3f} {:>3d}/{:<3d}".format(*r[m][:4])) for r in results.values()))
    print("  {:<8}".format("MEAN") + "".join(
        "{:>24}".format("{:.3f} {:.3f} {:>3.0f}/{:<3.0f}".format(*np.mean([x[:4] for x in r], axis=0)))
        for r in results.values()))
    for name, r in results.items():
        tp, pred, true = np.sum([x[4] for x in r], axis=0)
        print("  D class, folds pooled, {:<9} precision {:.3f} recall {:.3f}".format(
            name, tp / max(pred, 1), tp / max(true, 1)))
    final["variant"] = max(("no_tags", "tags"), key=lambda v: np.mean([x[0] for x in results[v]]))
    print("\nbest variant by mean held-out spellF1: {}".format(final["variant"]))
    with open(DECODER_PATH, "w") as f:
        json.dump(final, f, indent=1)
    for vname, top in importances.items():
        print("\ntop clues [{}] (mean rise in log-loss when shuffled)".format(vname))
        for n, v in top:
            print("  {:<20} {:.4f}".format(n, v))
    print("\ntotal {:.0f}s".format(time.time() - start))


if __name__ == "__main__":
    main()
