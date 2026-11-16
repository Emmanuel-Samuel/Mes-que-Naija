"""Turn fine-grained possession labels into the competition's coarser possession rule.

A possession only changes hands when the other team gains controlled possession, so short
spells (touches, deflections) are absorbed into the team that had the ball before them.
"""

import numpy as np

H, A = 0, 1


def translate(label, half, fps=25, control_seconds=2.0):
    """Return a copy of `label` where short team spells are merged into the previous holder.

    Works inside each half and inside each stretch of H/A frames (D and unknown frames end a
    stretch and are never changed). The first spell of a stretch is always kept; any later spell
    shorter than control_seconds takes the label of the last controlled team.
    """
    label = np.asarray(label)
    half = np.asarray(half)
    out = label.copy()
    min_frames = int(round(control_seconds * fps))

    team = (label == H) | (label == A)
    # A stretch breaks wherever the frame is not H/A or the half number changes.
    breaks = np.flatnonzero(np.diff(np.concatenate(([0], team.astype(np.int8), [0]))) != 0)
    for start, stop in zip(breaks[::2], breaks[1::2]):
        for h_start, h_stop in _split_by_half(half, start, stop):
            _absorb_short_spells(out, label, h_start, h_stop, min_frames)
    return out


def _split_by_half(half, start, stop):
    window = half[start:stop]
    cuts = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = [0, *cuts.tolist(), stop - start]
    return [(start + bounds[i], start + bounds[i + 1]) for i in range(len(bounds) - 1)]


def _absorb_short_spells(out, label, start, stop, min_frames):
    window = label[start:stop]
    cuts = np.flatnonzero(window[1:] != window[:-1]) + 1
    bounds = [0, *cuts.tolist(), stop - start]
    holder = window[0]
    for i in range(1, len(bounds) - 1):
        a, b = bounds[i], bounds[i + 1]
        if b - a >= min_frames:
            holder = window[a]
        else:
            out[start + a:start + b] = holder
