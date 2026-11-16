"""Make a clean match look like the competition tracking, using only the unlabelled reference.

What is copied from the reference: how often the ball is missing and how long the gaps last
(through a model that guesses "ball seen" from player clues alone), how many players are tracked
per team and how that count drifts over time, and how jittery the positions are.
Track swaps and duplicate tracks are not simulated.
"""

import zlib
from dataclasses import replace

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingClassifier

from possession.features import ball_speed, build_features, outside_lines, player_features

_FIT_CACHE = {}
TAU_GRID = (25, 100, 200, 400, 800, 1600, 3200)    # frames: how slowly the visibility noise drifts
SIGMA_GRID = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0)       # noise size in logit units
RUN_QUANTILES = (25, 50, 75, 90)
SMOOTH_FRAMES = 251
# Ways to hide a still or parked ball: speed below a threshold for at least a run length, and
# low-speed frames outside the lines. Each such spell is hidden with some probability. All are
# picked per match so the visible-ball aggregates match the reference (no labels involved).
STILL_SPEEDS = (0.5, 1.0, 2.0, 4.0)
STILL_RUNS = (12, 25, 50, 100)
OUT_SPEEDS = (2.0, 1e9)
HIDE_PROBS = (0.4, 0.6, 0.8, 1.0)
OUT_HIDE_PROBS = (0.5, 0.75, 0.9, 1.0)
# Share of hidden still spells where the ball is instead kept, carried by the nearest player.
CARRY_SHARES = (0.0, 0.3, 0.6)
CARRY_NOISE_M = 0.3
MAX_BASE_SHARE = 0.95 # the still-ball rule may hide at most this share of the reference missing rate
PLAYER_SCORE_TAU = 75                         # frames: how long one player's dropout tends to last


def _ar1(white, tau):
    """Unit-variance AR(1) noise along axis 0 with correlation time `tau` frames."""
    rho = np.exp(-1.0 / tau)
    return lfilter([np.sqrt(1 - rho ** 2)], [1.0, -rho], white, axis=0)


def _smooth_noise(white, tau):
    """Slow noise without frame-to-frame roughness, so a threshold crossing does not chatter."""
    z = uniform_filter1d(_ar1(white, tau), SMOOTH_FRAMES // 3, mode="nearest")
    return z / z.std()


def _gap_lengths(visible, runs):
    out = []
    for start, stop in runs:
        miss = np.concatenate(([0], (~visible[start:stop]).astype(np.int8), [0]))
        edges = np.flatnonzero(np.diff(miss))
        out.extend((edges[1::2] - edges[::2]).tolist())
    return np.array(out, dtype=np.int64)


def _quantiles(runs, fps):
    if not len(runs):
        return np.full(len(RUN_QUANTILES), 1e-3)
    return np.maximum(np.percentile(runs, RUN_QUANTILES) / fps, 1e-3)


def _gap_stats(visible, match):
    inplay = match.half > 0
    gaps = _gap_lengths(visible, match.halves_runs())
    return {
        "missing_rate": float(np.mean(~visible[inplay])),
        "gaps": int(len(gaps)),
        "gap_median_s": float(np.median(gaps) / match.fps) if len(gaps) else 0.0,
        "gap_p90_s": float(np.percentile(gaps, 90) / match.fps) if len(gaps) else 0.0,
        "gap_q": _quantiles(gaps, match.fps),
        "seen_q": _quantiles(_gap_lengths(~visible, match.halves_runs()), match.fps),
    }


def _ball_visible_stats(ball, fps, mask=None, speed=None):
    """Aggregates over frames where the ball is seen and its speed is known."""
    speed = ball_speed(ball, fps) if speed is None else speed
    ok = ~np.isnan(speed)
    if mask is not None:
        ok &= mask
    sp = speed[ok]
    out = outside_lines(ball)[ok]
    return {"v_slow05": float(np.mean(sp < 0.5)), "v_slow2": float(np.mean(sp < 2.0)),
            "v_speed_med": float(np.median(sp)), "v_outside": float(np.mean(out > 0))}


def _spells(flag, min_run):
    """[(start, stop)] runs of `flag` at least min_run long."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], flag.astype(np.int8), [0]))))
    return [(a, b) for a, b in zip(edges[::2], edges[1::2]) if b - a >= min_run]


def _hide_spells(spells, u, prob, n):
    """Mask hiding each spell whose uniform draw u[start] is below prob."""
    out = np.zeros(n, dtype=bool)
    for a, b in spells:
        if u[a] < prob:
            out[a:b] = True
    return out


def _carried_ball(match, rng):
    """A plausible tracked ball for a stoppage: the nearest present player, plus a small wobble."""
    both = np.concatenate([match.home_xy, match.away_xy], axis=1)
    d = np.hypot(both[:, :, 0] - match.ball_xy[:, None, 0], both[:, :, 1] - match.ball_xy[:, None, 1])
    idx = np.argmin(np.where(np.isnan(d), np.inf, d), axis=1)
    pos = both[np.arange(len(both)), idx]
    pos[np.isnan(d).all(axis=1)] = np.nan
    wobble = _ar1(rng.standard_normal((len(pos), 2)), 10) * CARRY_NOISE_M
    return (pos + wobble).astype(np.float32)


def _still_ball_mask(match, ref_stats, rng, slow):
    """Frames where a tracker would not report a still or parked ball, and which of those are
    instead tracked as a ball carried by a player. Chosen without labels. `slow` flags frames where
    the players are nearly still."""
    ball = match.ball_xy
    n = len(ball)
    speed = ball_speed(ball, match.fps)
    carried_ball = _carried_ball(match, rng)
    carried_speed = ball_speed(carried_ball, match.fps)
    carried_out = outside_lines(carried_ball) > 0
    slow_fill = np.nan_to_num(speed, nan=1e9)
    outside = outside_lines(ball) > 0
    inplay = match.half > 0
    u_still, u_out, u_carry = rng.random(n), rng.random(n), rng.random(n)
    target = ref_stats["missing_rate"]
    still_sets = {(th, run): _spells((slow_fill < th) & inplay, run)
                  for th in STILL_SPEEDS for run in STILL_RUNS}
    out_sets = {th: _spells(outside & (slow_fill < th) & inplay, 1) for th in OUT_SPEEDS}
    best = None
    for (th, run), spells in still_sets.items():
        for q in HIDE_PROBS:
            still = _hide_spells(spells, u_still, q, n)
            kept_still = {r: _hide_spells([sp for sp in spells if u_still[sp[0]] < q], u_carry, r, n)
                          for r in CARRY_SHARES}
            for out_th, ospells in out_sets.items():
                for qo in OUT_HIDE_PROBS:
                    out_hide = _hide_spells(ospells, u_out, qo, n)
                    hide = still | out_hide
                    for r in CARRY_SHARES:
                        carried = kept_still[r] | _hide_spells(
                            [sp for sp in ospells if u_out[sp[0]] < qo], u_carry, r, n)
                        shown = (~hide | carried) & inplay
                        sp_all = np.where(carried, carried_speed, speed)
                        ok = shown & ~np.isnan(sp_all)
                        v = sp_all[ok]
                        out_flag = np.where(carried, carried_out, outside)[ok]
                        st = {"v_slow05": float(np.mean(v < 0.5)), "v_slow2": float(np.mean(v < 2.0)),
                              "v_speed_med": float(np.median(v)), "v_outside": float(np.mean(out_flag))}
                        cost = sum(abs(np.log((st[k] + 0.005) / (ref_stats[k] + 0.005)))
                                   for k in ("v_slow05", "v_slow2", "v_outside"))
                        cost += abs(np.log(st["v_speed_med"] / ref_stats["v_speed_med"]))
                        cost += abs(np.log((np.mean(slow[shown]) + 0.01) / (ref_stats["slow_given_seen"] + 0.01)))
                        cost += 20 * max(0.0, np.mean((hide & ~carried)[inplay]) - MAX_BASE_SHARE * target)
                        if best is None or cost < best[0]:
                            best = (cost, hide & ~carried, carried)
    return best[1], best[2], carried_ball


def _team_counts(match):
    return (np.sum(~np.isnan(match.home_xy[:, :, 0]), axis=1),
            np.sum(~np.isnan(match.away_xy[:, :, 0]), axis=1))


def _jitter(match):
    """Robust size of frame-to-frame noise in metres, from second differences of player tracks."""
    inplay = match.half > 0
    parts = []
    for xy in (match.home_xy, match.away_xy):
        d2 = xy[2:] - 2 * xy[1:-1] + xy[:-2]
        ok = inplay[2:] & inplay[:-2]
        parts.append(d2[ok].ravel())
    d2 = np.concatenate(parts)
    d2 = d2[~np.isnan(d2)]
    return float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6.0))


def _autocorr_tau(series, mask, fps):
    """Frames until the autocorrelation of a count series first drops below 1/e."""
    x = series[mask].astype(np.float64)
    x = x - x.mean()
    var = float(np.mean(x * x))
    if var == 0:
        return 1.0
    for lag in (5, 10, 25, 50, 100, 200, 400, 800):
        if float(np.mean(x[lag:] * x[:-lag])) / var < np.exp(-1):
            return float(lag)
    return 800.0


def _slow_flags(match):
    """Frames where the players are nearly still (3 s mean of all-player speed under 1.5 m/s)."""
    blind = replace(match, ball_xy=np.full_like(match.ball_xy, np.nan))
    X, names = build_features(blind)
    return X[:, names.index("all_speed_m3s")] < 1.5


def _slow_given_seen(match, mask):
    return float(np.mean(_slow_flags(match)[mask]))


def _fit_reference(reference):
    key = (reference.name, reference.n_frames)
    if key in _FIT_CACHE:
        return _FIT_CACHE[key]
    inplay = reference.half > 0
    seen = ~np.isnan(reference.ball_xy[:, 0])
    X, _ = player_features(reference)
    X = X[:, :-1]  # drop the half clock
    rows = np.flatnonzero(inplay)[::3]
    clf = HistGradientBoostingClassifier(max_iter=100, max_depth=4, early_stopping=False,
                                         random_state=0)
    clf.fit(X[rows], seen[rows])
    home_n, away_n = _team_counts(reference)
    counts = np.concatenate([home_n[inplay], away_n[inplay]])
    fit = {
        "clf": clf,
        "stats": _gap_stats(seen, reference),
        "count_quantiles": np.quantile(counts, np.linspace(0, 1, 201)),
        "count_tau": _autocorr_tau(home_n + away_n, inplay, reference.fps),
        "jitter": _jitter(reference),
        "ball": {**_ball_visible_stats(reference.ball_xy, reference.fps, inplay & seen),
                 "missing_rate": float(np.mean(~seen[inplay])),
                 "slow_given_seen": _slow_given_seen(reference, inplay & seen)},
    }
    _FIT_CACHE[key] = fit
    return fit


def _drop_players(xy, target, score):
    """Blank the lowest-scored present players until only target[t] remain in each frame."""
    present = ~np.isnan(xy[:, :, 0])
    n_drop = np.maximum(present.sum(axis=1) - target, 0)
    ranked = np.where(present, score, np.inf)
    rank = np.argsort(np.argsort(ranked, axis=1, kind="stable"), axis=1, kind="stable")
    out = xy.copy()
    out[rank < n_drop[:, None]] = np.nan
    return out


def _visibility(match, fit, rng, hidden):
    """Sample a ball-seen mask: `hidden` frames stay hidden, then a learned mask hides more so the
    total missing rate and gap lengths match the reference."""
    X, _ = player_features(match)
    p = np.clip(fit["clf"].predict_proba(X[:, :-1])[:, 1], 0.01, 0.99)
    # Smooth over ~10 s: the clue is the slow trend; fast flicker would chop the gaps up.
    logit = uniform_filter1d(np.log(p / (1 - p)), SMOOTH_FRAMES, mode="nearest")
    inplay = match.half > 0
    white = rng.standard_normal(match.n_frames)
    ref = fit["stats"]
    free = inplay & ~hidden
    extra = (ref["missing_rate"] - np.mean(hidden[inplay] & inplay[inplay])) / max(np.mean(free[inplay]), 1e-9)
    extra = float(np.clip(extra, 0.0, 1.0))
    best = None
    for tau in TAU_GRID:
        z = _smooth_noise(white, tau)
        for sigma in SIGMA_GRID:
            score = logit + sigma * z
            cut = np.quantile(score[free], extra)
            visible = (score > cut) & ~hidden
            s = _gap_stats(visible, match)
            if not s["gaps"]:
                continue
            miss = (np.abs(np.log(s["gap_q"] / ref["gap_q"])).sum()
                    + np.abs(np.log(s["seen_q"] / ref["seen_q"])).sum())
            if best is None or miss < best[0]:
                best = (miss, visible)
    return best[1]


def degrade(match, reference, seed):
    """Return a copy of `match` with competition-like ball loss, player dropout and jitter."""
    fit = _fit_reference(reference)
    rng = np.random.default_rng([int(seed), zlib.crc32(match.name.encode())])
    T = match.n_frames

    # Players first, because the ball-loss model reads the degraded player clues.
    quant = fit["count_quantiles"]
    teams = []
    for xy in (match.home_xy, match.away_xy):
        u = ndtr(_ar1(rng.standard_normal(T), fit["count_tau"]))
        target = np.round(np.interp(u, np.linspace(0, 1, len(quant)), quant)).astype(int)
        score = _ar1(rng.standard_normal((T, xy.shape[1])), PLAYER_SCORE_TAU)
        teams.append(_drop_players(xy, target, score))
    # Only add noise if the reference is noisier than this match already is.
    sigma = float(np.sqrt(max(fit["jitter"] ** 2 - _jitter(match) ** 2, 0.0)))
    home, away = (xy + (rng.standard_normal(xy.shape) * sigma).astype(np.float32) for xy in teams)

    out = replace(match, name=match.name, home_xy=home.astype(np.float32),
                  away_xy=away.astype(np.float32))
    hidden, carried, carried_ball = _still_ball_mask(match, fit["ball"], rng, _slow_flags(out))
    seen = _visibility(out, fit, rng, hidden)
    ball = match.ball_xy + (rng.standard_normal(match.ball_xy.shape) * sigma).astype(np.float32)
    ball[carried] = carried_ball[carried]
    ball[~seen] = np.nan
    return replace(out, ball_xy=ball.astype(np.float32))


def compare(reference, degraded):
    """Whole-match aggregates of reference vs degraded tracking (no per-frame values)."""
    rows = {}
    for tag, m in (("reference", reference), ("degraded", degraded)):
        inplay = m.half > 0
        seen = ~np.isnan(m.ball_xy[:, 0])
        stats = _gap_stats(seen, m)
        stats = {k: v for k, v in stats.items() if not k.endswith("_q")}
        hn, an = _team_counts(m)
        both = np.concatenate([hn[inplay], an[inplay]])
        stats["players_mean"], stats["players_std"] = float(both.mean()), float(both.std())
        stats["jitter_m"] = _jitter(m)
        stats.update(_ball_visible_stats(m.ball_xy, m.fps, inplay & seen))
        X, names = build_features(m)
        slow = X[:, names.index("all_speed_m3s")] < 1.5
        stats["slow_given_seen"] = float(np.mean(slow[inplay & seen]))
        stats["slow_given_missing"] = float(np.mean(slow[inplay & ~seen]))
        rows[tag] = stats
    return rows
