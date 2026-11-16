import time

import numpy as np
import pytest

from possession.decoder import DecoderParams, decode, grid_search
from possession.scoring import runs_from_half, spell_f1

FPS = 25


def make_gt(rng, n):
    """Random H/A/D sequence: possessions 3-40 s, dead runs 5-40 s."""
    out, team = [], int(rng.integers(2))
    while len(out) < n:
        out.extend([team] * int(rng.uniform(3, 40) * FPS))
        if rng.random() < 0.35:
            out.extend([2] * int(rng.uniform(5, 40) * FPS))
            team = int(rng.integers(2))
        else:
            team = 1 - team
    return np.array(out[:n], dtype=np.int8)


def ar1(rng, n, rho, std):
    noise = rng.normal(size=n)
    out = np.empty(n)
    out[0] = noise[0]
    scale = np.sqrt(1 - rho ** 2)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + scale * noise[t]
    return out * std


def make_logp(rng, gt):
    """Noisy log-probs: correlated noise plus short bursts of wrong-team evidence."""
    n = len(gt)
    logits = np.stack([ar1(rng, n, 0.97, 1.2) for _ in range(3)], axis=1)
    logits[np.arange(n), gt] += 1.5
    t = 0
    while t < n:
        t += int(rng.exponential(15 * FPS))
        length = int(rng.uniform(1, 4) * FPS)
        if t < n and gt[t] != 2:
            wrong = 1 - gt[t]
            logits[t:t + length, wrong] += 4.0
        t += length
    logits -= logits.max(axis=1, keepdims=True)
    return logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))


def make_match(seed, half_len=20000):
    rng = np.random.default_rng(seed)
    gt = np.concatenate([make_gt(rng, half_len), make_gt(rng, half_len)])
    half = np.repeat(np.array([1, 2], dtype=np.int8), half_len)
    return make_logp(rng, gt), half, gt


def majority(logp, window=50):
    n = len(logp)
    onehot = np.eye(3)[logp.argmax(axis=1)]
    csum = np.vstack([np.zeros(3), np.cumsum(onehot, axis=0)])
    lo = np.clip(np.arange(n) - window, 0, n)
    hi = np.clip(np.arange(n) + window + 1, 0, n)
    return (csum[hi] - csum[lo]).argmax(axis=1).astype(np.int8)


def perfect_logp(gt):
    logp = np.full((len(gt), 3), -20.0)
    logp[np.arange(len(gt)), gt] = -1e-6
    return logp


def mean_score(matches, fn):
    return np.mean([spell_f1(gt, fn(logp, half), half)["spell_f1"] for logp, half, gt in matches])


@pytest.fixture(scope="module")
def matches():
    return [make_match(seed) for seed in range(4)]


def test_decoder_beats_argmax_and_majority(matches):
    argmax = mean_score(matches, lambda lp, h: lp.argmax(axis=1).astype(np.int8))
    vote = mean_score(matches, lambda lp, h: majority(lp))
    smooth = mean_score(matches, decode)
    print(f"argmax {argmax:.3f} majority {vote:.3f} decoder {smooth:.3f}")
    assert smooth > argmax + 0.2
    assert smooth > vote + 0.1


def test_perfect_logp_gives_one(matches):
    _, half, gt = matches[0]
    pred = decode(perfect_logp(gt), half)
    assert np.array_equal(pred, gt)
    assert spell_f1(gt, pred, half)["spell_f1"] == 1.0


def test_never_switches_across_half_boundary():
    n = 3000
    logp = perfect_logp(np.array([0] * n + [1] * n, dtype=np.int8))
    half = np.repeat(np.array([1, 2], dtype=np.int8), n)
    pred = decode(logp, half)
    assert (pred[:n] == 0).all() and (pred[n:] == 1).all()
    # Both halves start with the same team: the boundary must not merge them into one spell.
    logp = perfect_logp(np.zeros(2 * n, dtype=np.int8))
    pred = decode(logp, half)
    assert (pred == 0).all()
    # Weak evidence for the other team at the start of half 2 must not leak from half 1.
    logp[n:n + 10] = perfect_logp(np.ones(10, dtype=np.int8))
    assert (decode(logp, half)[:n] == 0).all()


def test_half_zero_frames_are_dead():
    gt = np.array([0] * 2000 + [1] * 2000 + [0] * 2000, dtype=np.int8)
    half = np.array([1] * 2000 + [0] * 2000 + [2] * 2000, dtype=np.int8)
    pred = decode(perfect_logp(gt), half)
    assert (pred[2000:4000] == 2).all()
    assert (pred[:2000] == 0).all() and (pred[4000:] == 0).all()
    assert runs_from_half(half) == [(0, 2000), (4000, 6000)]


def test_deterministic(matches):
    logp, half, _ = matches[1]
    assert np.array_equal(decode(logp, half), decode(logp, half))


def test_short_wrong_team_burst_is_absorbed():
    n = 6000
    gt = np.zeros(n, dtype=np.int8)
    logp = np.tile(np.log([0.7, 0.2, 0.1]), (n, 1))
    logp[3000:3050] = np.log([0.1, 0.8, 0.1])
    pred = decode(logp, np.ones(n, dtype=np.int8))
    assert (pred == 0).all()
    assert spell_f1(gt, pred, np.ones(n, dtype=np.int8))["n_pred"] == 1


def test_real_dead_run_is_kept():
    gt = np.array([0] * 1500 + [2] * 250 + [1] * 1500, dtype=np.int8)
    pred = decode(perfect_logp(gt), np.ones(len(gt), dtype=np.int8))
    assert np.array_equal(pred, gt)


def test_speed_80000_frames():
    rng = np.random.default_rng(0)
    gt = make_gt(rng, 80000)
    logp = make_logp(rng, gt)
    half = np.ones(80000, dtype=np.int8)
    start = time.perf_counter()
    decode(logp, half)
    elapsed = time.perf_counter() - start
    print(f"80000 frames: {elapsed:.2f}s")
    assert elapsed < 5.0


def test_grid_search_is_deterministic_and_pool_matches(matches):
    small = [(lp[:8000], h[:8000], gt[:8000]) for lp, h, gt in matches[:2]]
    grid = {"switch_cost": [4.0, 8.0], "emission_weight": [0.05, 0.1]}
    serial = grid_search(small, grid)
    assert serial == grid_search(small, grid)
    assert serial == grid_search(small, grid, n_jobs=2)
    assert len(serial) == 4
    assert serial[0][1] >= serial[-1][1]
    assert isinstance(serial[0][0], DecoderParams)
