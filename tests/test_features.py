import os
from dataclasses import replace

import numpy as np
import pytest

from possession.canonical import cache_path, load_cached
from possession.degrade import degrade
from possession.features import build_features

pytestmark = pytest.mark.skipif(
    not (os.path.isfile(cache_path("J03WMX")) and os.path.isfile(cache_path("BAR-ATH"))),
    reason="match cache not built (scripts/build_cache.py)")

N = 12000


def clip(match, start=20000):
    s = slice(start, start + N)
    return replace(match, home_xy=match.home_xy[s], away_xy=match.away_xy[s],
                   ball_xy=match.ball_xy[s], half=match.half[s], label=match.label[s])


@pytest.fixture(scope="module")
def clean():
    return clip(load_cached("J03WMX"))


@pytest.fixture(scope="module")
def degraded(clean):
    return degrade(clean, load_cached("BAR-ATH"), 0)


def test_no_nan_only_columns(degraded):
    X, names = build_features(degraded)
    dead = [n for i, n in enumerate(names) if np.isnan(X[:, i]).all()]
    assert not dead
    assert X.shape == (N, len(names)) and X.dtype == np.float32


def test_slot_shuffle_changes_nothing(degraded):
    rng = np.random.default_rng(1)
    h, a = rng.permutation(degraded.home_xy.shape[1]), rng.permutation(degraded.away_xy.shape[1])
    shuffled = replace(degraded, home_xy=degraded.home_xy[:, h], away_xy=degraded.away_xy[:, a])
    x1, _ = build_features(degraded)
    x2, _ = build_features(shuffled)
    np.testing.assert_allclose(x1, x2, rtol=1e-4, atol=1e-4, equal_nan=True)


def test_labels_are_never_read(degraded):
    rng = np.random.default_rng(2)
    scrambled = replace(degraded, label=rng.integers(0, 4, len(degraded.label)).astype(np.int8))
    x1, _ = build_features(degraded)
    x2, _ = build_features(scrambled)
    np.testing.assert_array_equal(x1, x2)


def test_degrade_is_deterministic_per_seed(clean):
    ref = load_cached("BAR-ATH")
    a, b, c = degrade(clean, ref, 3), degrade(clean, ref, 3), degrade(clean, ref, 4)
    for field in ("home_xy", "away_xy", "ball_xy"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    assert not np.array_equal(a.ball_xy, c.ball_xy, equal_nan=True)


def test_degrade_removes_ball_and_players(clean, degraded):
    assert np.isnan(degraded.ball_xy[:, 0]).mean() > 0.1
    assert np.isnan(degraded.home_xy[:, :, 0]).mean() > np.isnan(clean.home_xy[:, :, 0]).mean()
