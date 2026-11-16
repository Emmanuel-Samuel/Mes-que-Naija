"""Sequence decoder ("smoother") for per-frame home / away / dead-ball probabilities.

Hidden states: D, H_control, H_loose, A_control, A_loose. Loose states emit the same label as
their control state but with a flattened emission, so a stretch of weak or contrary evidence does
not flip the team. Changing team must go through the other team's control state.
"""

import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from possession.scoring import runs_from_half, spell_f1

H, A, D = 0, 1, 2
# state index -> output label; states: D, H_control, H_loose, A_control, A_loose
_STATE_LABEL = np.array([D, H, H, A, A], dtype=np.int8)
_NEG = -1e18


@dataclass(frozen=True)
class DecoderParams:
    """All costs are in log-probability units, after emission_weight scaling is applied to logp."""
    # Keep dead_in + dead_out above switch_cost, or a one-frame D becomes a cheaper team change.
    emission_weight: float = 0.1   # scales logp; neighbouring frames are highly correlated
    switch_cost: float = 16.0      # H <-> A change (always lands in the new team's control state)
    dead_in_cost: float = 10.0     # possession -> dead ball
    dead_out_cost: float = 10.0    # dead ball -> possession
    loose_cost: float = 3.0        # control -> loose within the same team
    loose_mix: float = 0.5         # loose emission = (1 - mix) * own logp + mix * other team's logp
    min_control_frames: int = 50   # shorter team-change spells are absorbed into the previous team
    min_dead_frames: int = 25      # shorter dead runs are filled with the surrounding team


def _transition_matrix(p):
    t = np.full((5, 5), _NEG)
    for i in range(5):
        t[i, i] = 0.0
    for control, loose in ((1, 2), (3, 4)):
        t[control, loose] = -p.loose_cost
        t[loose, control] = 0.0
        t[0, control] = -p.dead_out_cost
        for src in (control, loose):
            t[src, 0] = -p.dead_in_cost
    for src in (1, 2):
        t[src, 3] = -p.switch_cost
    for src in (3, 4):
        t[src, 1] = -p.switch_cost
    return t


def _viterbi(logp, p):
    n = len(logp)
    lp = p.emission_weight * logp
    m = p.loose_mix
    em = np.empty((n, 5))
    em[:, 0] = lp[:, D]
    em[:, 1] = lp[:, H]
    em[:, 2] = (1 - m) * lp[:, H] + m * lp[:, A]
    em[:, 3] = lp[:, A]
    em[:, 4] = (1 - m) * lp[:, A] + m * lp[:, H]

    trans = _transition_matrix(p)
    cols = np.arange(5)
    back = np.zeros((n, 5), dtype=np.int8)
    score = em[0] + np.array([0.0, 0.0, -p.loose_cost, 0.0, -p.loose_cost])
    for t in range(1, n):
        cand = score[:, None] + trans
        best = cand.argmax(axis=0)  # ties go to the lowest state index
        back[t] = best
        score = cand[best, cols] + em[t]

    state = int(score.argmax())
    states = np.empty(n, dtype=np.int8)
    for t in range(n - 1, -1, -1):
        states[t] = state
        state = back[t, state]
    return _STATE_LABEL[states]


def _runs(labels):
    """[[start, stop, label]] maximal constant runs."""
    breaks = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    bounds = [0, *breaks.tolist(), len(labels)]
    return [[bounds[i], bounds[i + 1], int(labels[bounds[i]])] for i in range(len(bounds) - 1)]


def _drop_short_dead(labels, min_frames):
    runs = _runs(labels)
    out = labels.copy()
    for i, (start, stop, code) in enumerate(runs):
        if code != D or stop - start >= min_frames:
            continue
        if i > 0:
            out[start:stop] = runs[i - 1][2]
        elif i + 1 < len(runs):
            out[start:stop] = runs[i + 1][2]
    return out


def _absorb_short_spells(labels, min_frames):
    out = []
    for start, stop, code in _runs(labels):
        if (code != D and out and out[-1][2] not in (D, code)
                and stop - start < min_frames):
            code = out[-1][2]
        if out and out[-1][2] == code:
            out[-1][1] = stop
        else:
            out.append([start, stop, code])
    result = np.empty(len(labels), dtype=np.int8)
    for start, stop, code in out:
        result[start:stop] = code
    return result


def _decode_half(logp, p):
    labels = _viterbi(logp, p)
    labels = _drop_short_dead(labels, p.min_dead_frames)
    return _absorb_short_spells(labels, p.min_control_frames)


def decode(logp, half, params=None):
    """Return int8 labels [T] (0 H, 1 A, 2 D). Each half is decoded alone; half == 0 gives D."""
    p = params or DecoderParams()
    logp = np.asarray(logp, dtype=np.float64)
    out = np.full(len(logp), D, dtype=np.int8)
    for start, stop in runs_from_half(half):
        seg = np.asarray(half[start:stop]) > 0
        if seg.all():
            out[start:stop] = _decode_half(logp[start:stop], p)
        else:
            idx = np.flatnonzero(seg) + start
            out[idx] = _decode_half(logp[idx], p)
    return out


def _score_params(params, matches):
    scores = [spell_f1(gt, decode(logp, half, params), half)["spell_f1"]
              for logp, half, gt in matches]
    return float(np.mean(scores))


_WORKER_MATCHES = None


def _init_worker(matches):
    global _WORKER_MATCHES
    _WORKER_MATCHES = matches


def _worker_score(params):
    return _score_params(params, _WORKER_MATCHES)


def grid_search(matches, grid, base=None, n_jobs=1):
    """Try every combination in grid ({param name: [values]}) on matches [(logp, half, gt)].

    Returns [(params, mean spellF1)] best first; ties keep grid order, so results are the same
    for any n_jobs.
    """
    base = base or DecoderParams()
    names = list(grid)
    all_params = [replace(base, **dict(zip(names, values, strict=True)))
                  for values in itertools.product(*(grid[k] for k in names))]
    if n_jobs > 1:
        with ProcessPoolExecutor(n_jobs, initializer=_init_worker, initargs=(matches,)) as pool:
            scores = list(pool.map(_worker_score, all_params, chunksize=1))
    else:
        scores = [_score_params(p, matches) for p in all_params]
    order = sorted(range(len(all_params)), key=lambda i: (-scores[i], i))
    return [(all_params[i], scores[i]) for i in order]
