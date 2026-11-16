"""Experiment: fair head-to-head of model families, same folds / degraded data / labels as cv_german.py.

    .venv/bin/python scripts/compare_models.py --smoke              # 2 folds, tiny settings
    .venv/bin/python scripts/compare_models.py                      # fit + determinism + report
    .venv/bin/python scripts/compare_models.py --stage report       # only rebuild the table

Smoother settings: for each model, its 7 leave-one-match-out predictions are decoded with every
settings combination; fold m uses the combination with the best mean spellF1 on the OTHER 6
matches' predictions (not the nested inner loop of cv_german.py, which is too costly for the
neural models). Same scheme for every model. Only whole-match aggregates are printed.
Nothing here writes to possession/ or submission/.
"""

import argparse
import itertools
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from cv_german import CONTROL_SECONDS, GRID, N_JOBS, TRAIN_EVERY, TRAIN_SEEDS  # noqa: E402
from possession.canonical import load_cached  # noqa: E402
from possession.decoder import DecoderParams, decode  # noqa: E402
from possession.degrade import degrade  # noqa: E402
from possession.features import build_features  # noqa: E402
from possession.judge import predict_logp, train_judge  # noqa: E402
from possession.load_idsse import MATCH_IDS  # noqa: E402
from possession.load_metrica import NAME as REFERENCE  # noqa: E402
from possession.scoring import spell_f1  # noqa: E402
from possession.seq_models import LOG_FLOOR, Prep, predict_logp_seq, train_seq  # noqa: E402
from possession.translate_labels import translate  # noqa: E402

SEED = 0
K = len(MATCH_IDS)
CACHE = os.path.join(ROOT, ".cache", "compare")
MODELS = ["hgb", "logreg", "extratrees", "gru", "tcn", "stack"]
FULL = dict(every=TRAIN_EVERY, grid=GRID, seq=dict(hidden=64, lr=2e-3, window=200, batch=32,
                                                   steps=150, epochs=40, patience=6))
SMOKE = dict(every=50, grid={"emission_weight": [0.1, 0.3], "switch_cost": [4.0, 16.0]},
             seq=dict(hidden=16, lr=2e-3, window=100, batch=8, steps=3, epochs=2, patience=2))


# ---------------------------------------------------------------- data
def load_data():
    path = os.path.join(CACHE, "data.npz")
    if not os.path.exists(path):
        os.makedirs(CACHE, exist_ok=True)
        reference = load_cached(REFERENCE)
        out = {}
        for m, name in enumerate(MATCH_IDS):
            raw = load_cached(name)
            out["y%d" % m] = translate(raw.label, raw.half, raw.fps, CONTROL_SECONDS)
            out["half%d" % m] = np.asarray(raw.half)
            for sd in TRAIN_SEEDS:
                X, names = build_features(degrade(raw, reference, sd))
                out["X%d_%d" % (m, sd)] = X
            print("featurised", name, flush=True)
        out["names"] = np.array(names)
        np.savez(path, **out)
    z = np.load(path)
    return {"X": {(m, sd): z["X%d_%d" % (m, sd)] for m in range(K) for sd in TRAIN_SEEDS},
            "y": [z["y%d" % m] for m in range(K)], "half": [z["half%d" % m] for m in range(K)]}


def segments(half):
    """Index arrays, one per half, of the in-play frames (all frames here)."""
    return [np.flatnonzero(half == v) for v in np.unique(half[half > 0])]


# ---------------------------------------------------------------- models
def inner_hgb(m, D, cfg, tag):
    """HGB log-probs for each training match i from a model that saw neither i nor m."""
    path = os.path.join(CACHE, "inner_%s_%d.npz" % (tag, m))
    if not os.path.exists(path):
        out, t0 = {}, time.time()
        for i in range(K):
            if i == m:
                continue
            idx = [j for j in range(K) if j not in (m, i)]
            mod = train_judge([D["X"][j, sd] for j in idx for sd in TRAIN_SEEDS],
                              [D["y"][j] for j in idx for sd in TRAIN_SEEDS], every=cfg["every"])
            for sd in TRAIN_SEEDS:
                out["%d_%d" % (i, sd)] = predict_logp(mod, D["X"][i, sd]).astype(np.float32)
        np.savez(path, seconds=time.time() - t0, **out)
    z = np.load(path)
    return {(int(k.split("_")[0]), int(k.split("_")[1])): z[k] for k in z.files if "_" in k}, \
        float(z["seconds"])


def _seq_sets(arrs, ys, halves):
    sets = []
    for Z, y, h in zip(arrs, ys, halves, strict=True):
        sets += [(Z[idx], y[idx]) for idx in segments(h)]
    return sets


def _predict_segments(net, Z, half, window):
    out = np.full((len(Z), 3), np.log(1 / 3), dtype=np.float32)
    for idx in segments(half):
        out[idx] = predict_logp_seq(net, Z[idx], window)
    return out


def fit_predict(name, m, D, cfg, extra):
    """Train on the 6 other matches (3 degrade seeds each); log-probs for match m, seed 0."""
    train = [i for i in range(K) if i != m]
    t0 = time.time()
    if name == "hgb":
        mod = train_judge([D["X"][i, sd] for i in train for sd in TRAIN_SEEDS],
                          [D["y"][i] for i in train for sd in TRAIN_SEEDS], every=cfg["every"])
        lp = predict_logp(mod, D["X"][m, SEED])
    elif name in ("logreg", "extratrees"):
        from sklearn.ensemble import ExtraTreesClassifier
        from sklearn.linear_model import LogisticRegression
        e = cfg["every"]
        prep = Prep().fit([D["X"][i, sd] for i in train for sd in TRAIN_SEEDS], every=e)
        Xs = np.concatenate([prep.transform(D["X"][i, sd][::e]) for i in train for sd in TRAIN_SEEDS])
        ys = np.concatenate([D["y"][i][::e] for i in train for sd in TRAIN_SEEDS])
        if name == "logreg":
            mod = LogisticRegression(C=1.0, max_iter=500)
        else:
            mod = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=N_JOBS,
                                       random_state=0)
        mod.fit(Xs, ys)
        p = mod.predict_proba(prep.transform(D["X"][m, SEED]))
        lp = np.log(np.clip(p, LOG_FLOOR, 1.0))
    else:  # gru, tcn, stack
        val = (m + 1) % K     # one training match is held out inside the fold for early stopping
        fit_ids = [i for i in train if i != val]
        if name == "stack":
            inner = extra["inner"]
            feat = lambda i, sd: inner[i, sd] / 4.0  # noqa: E731
            test_in = extra["outer_hgb"] / 4.0
            kind = "gru"
        else:
            prep = Prep().fit([D["X"][i, sd] for i in train for sd in TRAIN_SEEDS],
                              every=cfg["every"])
            feat = lambda i, sd: prep.transform(D["X"][i, sd])  # noqa: E731
            test_in = prep.transform(D["X"][m, SEED])
            kind = name
        tr = _seq_sets([feat(i, sd) for i in fit_ids for sd in TRAIN_SEEDS],
                       [D["y"][i] for i in fit_ids for sd in TRAIN_SEEDS],
                       [D["half"][i] for i in fit_ids for sd in TRAIN_SEEDS])
        va = _seq_sets([feat(val, SEED)], [D["y"][val]], [D["half"][val]])
        net = train_seq(kind, tr, va, cfg["seq"], seed=0)
        lp = _predict_segments(net, test_in, D["half"][m], cfg["seq"]["window"])
    return np.maximum(lp, np.log(LOG_FLOOR)).astype(np.float32), time.time() - t0


def oof_path(tag, name):
    return os.path.join(CACHE, "oof_%s_%s.npz" % (tag, name))


def run_fit(models, D, cfg, folds, tag):
    for name in models:
        path = oof_path(tag, name)
        if os.path.exists(path):
            continue
        res, secs = {}, {}
        for m in folds:
            extra = {}
            if name == "stack":
                extra["inner"], inner_s = inner_hgb(m, D, cfg, tag)
                extra["outer_hgb"] = np.load(oof_path(tag, "hgb"))["lp%d" % m]
                hgb_s = float(np.load(oof_path(tag, "hgb"))["sec%d" % m])
            lp, s = fit_predict(name, m, D, cfg, extra)
            if name == "stack":
                s += inner_s + hgb_s   # the stack needs the inner HGB fits and the outer one too
            res["lp%d" % m], secs["sec%d" % m] = lp, s
            print("[{}] fold {} done, {:.0f}s".format(name, m, s), flush=True)
        np.savez(path, **res, **secs)


def run_det(models, D, cfg, tag):
    path = os.path.join(CACHE, "det_%s.json" % tag)
    det = json.load(open(path)) if os.path.exists(path) else {}
    for name in models:
        if name in det:
            continue
        extra = {}
        if name == "stack":
            extra["inner"], _ = inner_hgb(0, D, cfg, tag)
            extra["outer_hgb"] = np.load(oof_path(tag, "hgb"))["lp0"]
        lp, _ = fit_predict(name, 0, D, cfg, extra)
        ref = np.load(oof_path(tag, name))["lp0"]
        det[name] = {"identical": bool(np.array_equal(lp, ref)),
                     "max_abs_diff": float(np.abs(lp - ref).max())}
        print("[{}] determinism: {}".format(name, det[name]), flush=True)
        json.dump(det, open(path, "w"))
    return det


# ---------------------------------------------------------------- scoring
_W = {}


def _init(logps, halves, gts, plist):
    _W.update(logps=logps, halves=halves, gts=gts, plist=plist)


def _one(task):
    i, p = task
    return spell_f1(_W["gts"][i], decode(_W["logps"][i], _W["halves"][i], _W["plist"][p]),
                    _W["halves"][i])["spell_f1"]


def smoother_table(logps, halves, gts, grid, folds):
    """S[match, setting] = spellF1 of that match's out-of-fold predictions under that setting."""
    names = list(grid)
    plist = [replace(DecoderParams(), **dict(zip(names, v, strict=True)))
             for v in itertools.product(*(grid[k] for k in names))]
    tasks = [(i, p) for i in folds for p in range(len(plist))]
    with ProcessPoolExecutor(N_JOBS, initializer=_init,
                             initargs=(logps, halves, gts, plist)) as pool:
        vals = list(pool.map(_one, tasks, chunksize=8))
    S = np.zeros((K, len(plist)))
    for (i, p), v in zip(tasks, vals, strict=True):
        S[i, p] = v
    return S, plist


def score_model(logps, D, grid, folds):
    """Per fold: smoothed score (settings chosen on the other folds) and argmax-only score."""
    S, plist = smoother_table(logps, D["half"], D["y"], grid, folds)
    smooth, raw = {}, {}
    for m in folds:
        others = [i for i in folds if i != m]
        p = int(np.argmax(S[others].mean(axis=0)))   # first best, like grid_search
        pred = decode(logps[m], D["half"][m], plist[p])
        smooth[m] = (spell_f1(D["y"][m], pred, D["half"][m]), plist[p])
        arg = np.argmax(logps[m], axis=1).astype(np.int8)
        raw[m] = (spell_f1(D["y"][m], arg, D["half"][m]), None)
    return smooth, raw


def run_report(models, D, cfg, folds, tag, out_path):
    rows, perfold, chosen = [], {}, {}
    det = json.load(open(os.path.join(CACHE, "det_%s.json" % tag))) \
        if os.path.exists(os.path.join(CACHE, "det_%s.json" % tag)) else {}
    for name in models:
        z = np.load(oof_path(tag, name))
        logps = {m: z["lp%d" % m] for m in folds}
        logps = [logps.get(m) for m in range(K)]
        secs = [float(z["sec%d" % m]) for m in folds]
        smooth, raw = score_model(logps, D, cfg["grid"], folds)
        for label, res in (("smoothed", smooth), ("argmax", raw)):
            f = [res[m][0]["spell_f1"] for m in folds]
            rows.append((name, label, np.mean(f), min(f), max(f),
                         np.mean([res[m][0]["macro_f1"] for m in folds]),
                         np.mean([res[m][0]["n_pred"] for m in folds]),
                         np.mean([res[m][0]["n_gt"] for m in folds]),
                         np.mean(secs), sum(secs), det.get(name, {})))
            perfold[name, label] = f
        chosen[name] = {m: smooth[m][1].__dict__ for m in folds}
        print("scored", name, flush=True)
    base = np.array(perfold["hgb", "smoothed"])
    lines = ["# Model comparison (leave-one-match-out, 7 German matches, no_tags features)", "",
             "Smoother settings for each fold: best mean spellF1 over the other matches' "
             "out-of-fold predictions (same scheme for all models; not nested).", "",
             "| model | decoding | mean spellF1 | min-max | macroF1 | possessions pred/true | "
             "train s/fold (total) | deterministic? | folds > hgb+0.04 | folds > hgb |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for (name, label, mean, lo, hi, mac, npred, ngt, msec, tsec, d) in rows:
        f = np.array(perfold[name, label])
        dtxt = ("yes" if d.get("identical") else "NO (max diff %.2g)" % d["max_abs_diff"]) if d else "n/a"
        lines.append("| {} | {} | {:.3f} | {:.3f}-{:.3f} | {:.3f} | {:.0f}/{:.0f} | {:.0f} ({:.0f}) "
                     "| {} | {}/{} | {}/{} |".format(
                         name, label, mean, lo, hi, mac, npred, ngt, msec, tsec, dtxt,
                         int(np.sum(f - base > 0.04)), len(folds), int(np.sum(f > base)), len(folds)))
    lines += ["", "Per-fold spellF1 (rows: model/decoding, columns: held-out match):", "",
              "| model | " + " | ".join(MATCH_IDS[m] for m in folds) + " |",
              "|---|" + "---|" * len(folds)]
    for (name, label), f in perfold.items():
        lines.append("| {} {} | ".format(name, label) + " | ".join("%.3f" % v for v in f) + " |")
    lines += ["", "Chosen smoother settings (smoothed runs), per model and fold:", ""]
    for name, per in chosen.items():
        lines.append("- {}: ".format(name) + "; ".join(
            "f{} ew={emission_weight} sw={switch_cost} min_ctrl={min_control_frames}"
            .format(m, **{k: v for k, v in p.items()}) if "min_control_frames" in p else
            "f{} ew={emission_weight} sw={switch_cost}".format(m, **p) for m, p in per.items()))
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:4 + 2 * len(models) + 2]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--stage", default="all", choices=["all", "fit", "det", "report"])
    ap.add_argument("--models", default=",".join(MODELS))
    a = ap.parse_args()
    cfg, tag = (SMOKE, "smoke") if a.smoke else (FULL, "full")
    folds = [0, 1] if a.smoke else list(range(K))
    models = a.models.split(",")
    out = os.path.join(ROOT, ".scratch", "compare_models%s.md" % ("_smoke" if a.smoke else ""))
    t0 = time.time()
    D = load_data()
    print("data ready {:.0f}s".format(time.time() - t0), flush=True)
    if a.stage in ("all", "fit"):
        run_fit(models, D, cfg, folds, tag)
    if a.stage in ("all", "det"):
        run_det(models, D, cfg, tag)
    if a.stage in ("all", "report"):
        run_report(models, D, cfg, folds, tag, out)
    print("total {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    main()
