import numpy as np

from possession.translate_labels import translate

H, A, D = 0, 1, 2


def run(labels, half=None, control_seconds=2.0):
    labels = np.array(labels, dtype=np.int8)
    half = np.ones(len(labels), dtype=np.int8) if half is None else np.array(half, dtype=np.int8)
    # fps=1 so control_seconds is a frame count
    return translate(labels, half, fps=1, control_seconds=control_seconds)


def test_short_spell_absorbed():
    out = run([H, H, H, A, H, H, H], control_seconds=2)
    assert out.tolist() == [H] * 7


def test_long_spell_switches():
    labels = [H, H, H, A, A, A]
    assert run(labels).tolist() == labels


def test_first_spell_after_dead_ball_kept():
    out = run([H, H, H, D, A, D, A, A, A], control_seconds=2)
    assert out.tolist() == [H, H, H, D, A, D, A, A, A]


def test_dead_untouched():
    labels = [H, D, D, A, H, H, D]
    out = run(labels, control_seconds=3)
    assert out[[1, 2, 6]].tolist() == [D, D, D]


def test_chain_of_short_spells_collapses():
    out = run([H, H, H, A, H, A, H, A, A, A], control_seconds=2)
    assert out.tolist() == [H] * 7 + [A] * 3


def test_never_crosses_half_boundary():
    labels = [H, H, H, A, A, A]
    half = [1, 1, 1, 2, 2, 2]
    assert run(labels, half, control_seconds=5).tolist() == labels


def test_half_boundary_first_spell_kept():
    labels = [H, H, H, H, A, H]
    half = [1, 1, 1, 1, 2, 2]
    out = run(labels, half, control_seconds=2)
    assert out.tolist() == [H, H, H, H, A, A]


def test_unknown_and_input_untouched():
    labels = np.array([3, 3, 3], dtype=np.int8)
    out = translate(labels, np.ones(3, dtype=np.int8))
    assert out.tolist() == [3, 3, 3]
    src = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.int8)
    translate(src, np.ones(7, dtype=np.int8), fps=1)
    assert src.tolist() == [0, 0, 0, 1, 0, 0, 0]
