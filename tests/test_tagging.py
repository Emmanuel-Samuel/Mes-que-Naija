import numpy as np
import pytest

from possession.load_idsse import MATCH_IDS
from possession.load_idsse_events import load_restarts
from possession.tagging import make_pseudo_tags, phase_possession_sign, tag_features


def test_tag_features_distances():
    sp = np.array([[250, 1150, 0, 1, 0], [2500, 3400, 1, -1, 0]])
    X, names = tag_features(sp, 5000, 25)
    col = {n: X[:, i] for i, n in enumerate(names)}
    assert col["sp_until_start"][100] == pytest.approx(6.0)
    assert col["sp_since_start"][500] == pytest.approx(10.0)
    assert col["sp_inside"][300] == 1 and col["sp_inside"][1500] == 0
    assert col["sp_next_team"][100] == 1 and col["sp_last_team"][3000] == -1


def test_no_tags_gives_constant_columns():
    X, _ = tag_features(np.zeros((0, 5), dtype=np.int64), 100, 25)
    assert X.shape[0] == 100 and np.isfinite(X[:, :3]).all()


def test_phase_sign():
    phases = [(0, 100, "build up", 1), (100, 200, "defending in middle third", 1),
              (200, 300, "build up", 0)]
    s = phase_possession_sign(phases, 300)
    assert s[50] == 1 and s[150] == -1 and s[250] == 0


def test_pseudo_tags_deterministic_and_imperfect():
    try:
        restarts = load_restarts(MATCH_IDS[0])
    except FileNotFoundError:
        pytest.skip("open data not present")
    a, b, c = (make_pseudo_tags(restarts, s) for s in (1, 1, 2))
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert 0 < len(a) < len(restarts)
    assert (a[a[:, 2] == 0][:, 3] == 0).all()  # throw-in team unknown
