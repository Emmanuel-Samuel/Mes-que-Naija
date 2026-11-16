"""Loader for the competition (Metrica ATD) tracking of BAR-ATH."""

import xml.etree.ElementTree as ET

import numpy as np

import games
from possession.canonical import Match, normalise_direction, UNKNOWN

NAME = "BAR-ATH"


def _slot_teams(meta_path):
    """Per track slot, 0 for the first team in the file (home) and 1 for the other."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")])


def load_metrica(game=games.GAME):
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], game)
    n_frames = games.source_frames("tactical", game)
    length, width = games.GAMES[game]["pitch"]
    team = _slot_teams(meta_path)
    n_slots = len(team)

    slots = np.full((n_frames, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.split(":")
            if len(parts) != 3:
                continue
            row = int(parts[0]) - 1
            values = np.fromstring(parts[1].replace(";", ","), sep=",", dtype=np.float32)
            slots[row] = values.reshape(-1, 3)[:, :2]
            ball[row] = np.fromstring(parts[2], sep=",", dtype=np.float32)[:2]

    # Normalised 0-1 to metres, origin at the pitch centre.
    scale = np.array([length, width], dtype=np.float32)
    slots = (slots - 0.5) * scale
    ball = (ball - 0.5) * scale

    home = np.ascontiguousarray(slots[:, team == 0])
    away = np.ascontiguousarray(slots[:, team == 1])
    half = np.zeros(n_frames, dtype=np.int8)
    for h, first, last in games.half_video_blocks(game):
        half[first:last + 1] = h
    label = np.full(n_frames, UNKNOWN, dtype=np.int8)

    flipped = normalise_direction(home, away, ball, half)
    return Match(name=NAME, home_xy=home, away_xy=away, ball_xy=ball, half=half,
                 label=label, flipped=flipped)
