"""Loader for the open DFL/IDSSE matches (raw observed positions, streamed line by line)."""

import os
import re
import xml.etree.ElementTree as ET

import numpy as np

from possession.canonical import ROOT, Match, normalise_direction, H, A, D

IDSSE_DIR = os.path.join(ROOT, "external_data", "idsse")
MATCH_IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]

SET_RE = re.compile(r'<FrameSet GameSection="(\w+)".*? TeamId="([^"]*)" PersonId="([^"]*)"')
FRAME_RE = re.compile(
    r'<Frame N="(\d+)"[^>]*? X="([^"]*)" Y="([^"]*)"(?:[^>]*BallPossession="(\d)" BallStatus="(\d)")?')


def _find(kind, match_id):
    prefix = "DFL_{}_".format(kind)
    for f in sorted(os.listdir(IDSSE_DIR)):
        if f.startswith(prefix) and match_id in f:
            return os.path.join(IDSSE_DIR, f)
    raise FileNotFoundError("{} for {}".format(kind, match_id))


def _team_ids(match_id):
    """(home_id, guest_id) from the match information file."""
    root = ET.parse(_find("02_01_matchinformation", match_id)).getroot()
    roles = {t.get("Role"): t.get("TeamId") for t in root.iter("Team")}
    return roles["home"], roles["guest"]


def _floats(strings):
    return np.array([float(s) if s else np.nan for s in strings], dtype=np.float32)


def _parse_positions(path, team_ids):
    """{(section, team, person): (N, xy)} plus {section: (N, xy, possession, status)} for the ball."""
    players, balls = {}, {}
    key = None
    cur = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("<Frame "):
                m = FRAME_RE.match(line)
                cur.append(m.groups())
            elif line.startswith("<FrameSet"):
                m = SET_RE.match(line)
                key = m.groups()
                cur = []
                if key[1] == "BALL":
                    balls[key[0]] = cur
                elif key[1] in team_ids:  # skip referees
                    players[key] = cur
    return players, balls


def _arrays(rows):
    n = np.array([r[0] for r in rows], dtype=np.int64)
    xy = np.stack([_floats([r[1] for r in rows]), _floats([r[2] for r in rows])], axis=1)
    return n, xy


def load_idsse(match_id):
    home_id, guest_id = _team_ids(match_id)
    players, balls = _parse_positions(_find("04_03_positions_raw_observed", match_id),
                                       (home_id, guest_id))

    sections = ["firstHalf", "secondHalf"]
    persons = {home_id: [], guest_id: []}
    for section, team, person in players:
        if person not in persons[team]:
            persons[team].append(person)
    column = {t: {p: i for i, p in enumerate(ps)} for t, ps in persons.items()}

    lengths = []
    ball_data = {}
    for section in sections:
        n, xy = _arrays(balls[section])
        poss = np.array([int(r[3]) for r in balls[section]], dtype=np.int8)
        status = np.array([int(r[4]) for r in balls[section]], dtype=np.int8)
        ball_data[section] = (n, xy, poss, status)
        lengths.append(len(n))
    total = sum(lengths)
    offsets = [0, lengths[0]]

    home = np.full((total, len(persons[home_id]), 2), np.nan, dtype=np.float32)
    away = np.full((total, len(persons[guest_id]), 2), np.nan, dtype=np.float32)
    ball = np.full((total, 2), np.nan, dtype=np.float32)
    half = np.zeros(total, dtype=np.int8)
    label = np.zeros(total, dtype=np.int8)

    for i, section in enumerate(sections):
        n, xy, poss, status = ball_data[section]
        sl = slice(offsets[i], offsets[i] + lengths[i])
        ball[sl] = xy
        half[sl] = i + 1
        label[sl] = np.where(status == 0, D, np.where(poss == 1, H, A))
        for (sec, team, person), rows in players.items():
            if sec != section:
                continue
            pn, pxy = _arrays(rows)
            idx = np.searchsorted(n, pn)
            ok = (idx < len(n)) & (n[np.minimum(idx, len(n) - 1)] == pn)
            target = home if team == home_id else away
            target[offsets[i] + idx[ok], column[team][person]] = pxy[ok]

    flipped = normalise_direction(home, away, ball, half)
    return Match(name=match_id, home_xy=home, away_xy=away, ball_xy=ball, half=half,
                 label=label, flipped=flipped)
