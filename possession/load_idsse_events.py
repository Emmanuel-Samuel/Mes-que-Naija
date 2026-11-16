"""Restarts (throw-ins, corners, goal kicks, free kicks, kick-offs) from the open DFL event files.

The frame is the moment the restart is taken (EventTime), on our canonical frame index.
"""

import xml.etree.ElementTree as ET
from datetime import datetime

import numpy as np

from possession.load_idsse import _find, _team_ids

FRAME_SECONDS = 0.04
KINDS = {"ThrowIn": 0, "CornerKick": 1, "GoalKick": 2, "FreeKick": 3, "KickOff": 4}
# Same order as possession.tagging.SET_PIECE_TYPES: throw-in, corner kick, goal kick, free kick, kick off.
SECTIONS = ("firstHalf", "secondHalf")


def _ts(text):
    return datetime.fromisoformat(text).timestamp()


def _section_clock(match_id):
    """Per section: (timestamp of its first ball frame, frame count). Only the ball track is read."""
    clock, section, in_ball, first, count = {}, None, False, None, 0
    with open(_find("04_03_positions_raw_observed", match_id), encoding="utf-8") as f:
        for line in f:
            if line.startswith("<FrameSet"):
                if in_ball:
                    clock[section] = (first, count)
                in_ball = 'TeamId="BALL"' in line
                section = line.split('GameSection="')[1].split('"')[0]
                first, count = None, 0
            elif in_ball and line.startswith("<Frame "):
                if first is None:
                    first = _ts(line.split(' T="')[1].split('"')[0])
                count += 1
    if in_ball:
        clock[section] = (first, count)
    return clock


def load_restarts(match_id):
    """int array [N,3] of (frame, type_index, team_sign), sorted by frame; team_sign +1 home, -1 away.

    Throw-in team is returned as it is in the file; callers make it unknown if they want.
    """
    home_id, _ = _team_ids(match_id)
    clock = _section_clock(match_id)
    offsets, total = {}, 0
    for s in SECTIONS:
        offsets[s] = total
        total += clock[s][1]
    rows = []
    for ev in ET.parse(_find("03_02_events_raw", match_id)).getroot().iter("Event"):
        for child in ev:
            if child.tag not in KINDS:
                continue
            t = _ts(ev.get("EventTime"))
            team = child.get("Team") or (child.find("Play").get("Team") if child.find("Play") is not None else None)
            for s in SECTIONS:
                first, n = clock[s]
                idx = int(round((t - first) / FRAME_SECONDS))
                if 0 <= idx < n:
                    sign = 0 if team is None else (1 if team == home_id else -1)
                    rows.append((offsets[s] + idx, KINDS[child.tag], sign))
                    break
    return np.array(sorted(rows), dtype=np.int64).reshape(-1, 3)
