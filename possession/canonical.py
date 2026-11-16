"""The one in-memory match format shared by every loader, with an on-disk cache.

Positions are in metres, origin at the pitch centre, x along the length. The home team always
attacks towards +x in both halves (loaders flip a half when needed and record it in `flipped`).
Label codes: 0 = H, 1 = A, 2 = D (dead ball), 3 = unknown.
"""

import os
import warnings
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, ".cache", "matches")
FPS = 25
H, A, D, UNKNOWN = 0, 1, 2, 3


@dataclass
class Match:
    name: str
    home_xy: np.ndarray   # float32 [T, P_home, 2]
    away_xy: np.ndarray   # float32 [T, P_away, 2]
    ball_xy: np.ndarray   # float32 [T, 2]
    half: np.ndarray      # int8 [T], 1 / 2 in play, 0 otherwise
    label: np.ndarray     # int8 [T]
    flipped: np.ndarray   # int8 [2], 1 if that half was mirrored to make home attack +x
    fps: int = FPS

    @property
    def n_frames(self):
        return len(self.half)

    def halves_runs(self):
        """[(start, stop)] frame ranges of half 1 and half 2 (for possession_spells)."""
        runs = []
        for h in (1, 2):
            idx = np.flatnonzero(self.half == h)
            if len(idx):
                runs.append((int(idx[0]), int(idx[-1]) + 1))
        return runs

    def save(self, path=None):
        path = path or cache_path(self.name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, name=np.array(self.name), fps=np.array(self.fps),
                            home_xy=self.home_xy, away_xy=self.away_xy, ball_xy=self.ball_xy,
                            half=self.half, label=self.label, flipped=self.flipped)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path) as z:
            return cls(name=str(z["name"]), fps=int(z["fps"]), home_xy=z["home_xy"],
                       away_xy=z["away_xy"], ball_xy=z["ball_xy"], half=z["half"],
                       label=z["label"], flipped=z["flipped"])


def cache_path(name):
    return os.path.join(CACHE_DIR, name + ".npz")


def load_cached(name):
    return Match.load(cache_path(name))


def _extremes_score(xy):
    """Mean over frames of (leftmost + rightmost player x); negative = the team sits left.

    The keeper is the extreme player on the defended side, so the sum leans towards that side.
    """
    x = xy[:, :, 0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
        return float(np.nanmean(np.nanmin(x, axis=1) + np.nanmax(x, axis=1)))


def normalise_direction(home_xy, away_xy, ball_xy, half):
    """Mirror (x and y) each half in place so that home attacks +x. Returns the flipped flags.

    Decided over the whole half from both teams' keeper side. A kick-off window is not used
    because the first seconds of the Metrica second half are mostly untracked.
    """
    flipped = np.zeros(2, dtype=np.int8)
    for h in (1, 2):
        rows = np.flatnonzero(half == h)
        if not len(rows):
            continue
        # Home defends right (+x) when its extremes lean right more than the away team's do.
        if _extremes_score(home_xy[rows]) > _extremes_score(away_xy[rows]):
            for arr in (home_xy, away_xy, ball_xy):
                arr[rows] *= -1
            flipped[h - 1] = 1
    return flipped


def direction_check(match, seconds=None):
    """Per half: (home deepest-player mean x, away deepest-player mean x, home-minus-away mean x).

    After normalisation the home keeper should sit near -x and the away keeper near +x. With
    `seconds` the mean-x difference uses only that many seconds from the start of the half.
    """
    out = []
    for h in (1, 2):
        rows = np.flatnonzero(match.half == h)
        if seconds:
            rows = rows[:int(seconds * match.fps)]
        home_x = match.home_xy[rows, :, 0]
        away_x = match.away_xy[rows, :, 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN rows
            out.append((float(np.nanmean(np.nanmin(home_x, axis=1))),
                        float(np.nanmean(np.nanmax(away_x, axis=1))),
                        float(np.nanmean(home_x) - np.nanmean(away_x))))
    return out
