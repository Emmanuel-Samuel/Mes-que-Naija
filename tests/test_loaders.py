import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession.canonical import Match, cache_path, direction_check  # noqa: E402
from possession.load_idsse import MATCH_IDS, load_idsse  # noqa: E402
from possession.load_metrica import NAME as METRICA, load_metrica  # noqa: E402


def get(name):
    """Cached match if present, otherwise build and cache it."""
    path = cache_path(name)
    if os.path.isfile(path):
        return Match.load(path)
    match = load_metrica() if name == METRICA else load_idsse(name)
    match.save()
    return match


def spells(match):
    return pm.possession_spells(match.label, match.halves_runs())


def test_idsse_wpy_label_counts_and_possessions():
    m = get("J03WPY")
    counts = [int((m.label == k).sum()) for k in range(3)]
    assert counts == [51192, 30268, 64751]
    assert len(spells(m)) == 421


def test_idsse_wmx_possessions():
    assert len(spells(get("J03WMX"))) == 511


@pytest.mark.parametrize("name", MATCH_IDS)
def test_idsse_shapes_and_direction(name):
    m = get(name)
    assert m.home_xy.dtype == np.float32 and m.ball_xy.shape == (m.n_frames, 2)
    assert set(np.unique(m.half)) == {1, 2}
    assert (m.label != 3).all()
    check_direction(m)
    # At kick-off each IDSSE team stands in its own half, so the 5 s check is sound there.
    assert all(diff < 0 for _, _, diff in direction_check(m, seconds=5))


def check_direction(m):
    # Deepest home player (keeper) sits near -x, deepest away player near +x, in both halves.
    for home_deep, away_deep, diff in direction_check(m):
        assert home_deep < -25 and away_deep > 25
        assert diff < 0  # home sits on the left over the half, away on the right


def test_metrica_counts_and_direction():
    m = get(METRICA)
    assert m.n_frames == 199319
    assert int((~np.isnan(m.ball_xy[:, 0])).sum()) == 113488
    assert int((m.half > 0).sum()) == 158857
    assert (m.label == 3).all()
    check_direction(m)
