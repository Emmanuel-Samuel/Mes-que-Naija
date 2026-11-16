"""Smart Tagging set pieces and phases as per-frame clues.

A set piece is a restart: the ball is dead just before it and the tagged team has it just after.
Real tags and pseudo-tags built from the German event files go through the same feature code.
Only whole-match aggregates from the competition tags should ever be printed.
"""

import json
import os
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
OFFSETS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tag_offsets.json")
HOME, AWAY = 1, -1
SET_PIECE_TYPES = ("throw-in", "corner kick", "goal kick", "free kick", "kick off")
CAP_SECONDS = 120.0
# Phase codes where the tagged team is the one with the ball, and where it is defending.
ATTACK_PHASES = ("build up", "progression", "creating chances", "attacking transition",
                 "long ball", "ball in final third", "ball in the box", "players in the box")
DEFEND_PHASES = ("defending in attacking third", "defending in middle third",
                 "defending in defensive third", "defensive transition")


def _type_index(text):
    text = (text or "").strip().lower()
    for i, name in enumerate(SET_PIECE_TYPES):
        if text.startswith(name):
            return i
    return -1


def load_tag_file(path, home_name, fps=FPS):
    """Return (set_pieces, phases).

    set_pieces: int array [N,5] of (start_frame, end_frame, type_index, team_sign, 0).
    phases: list of (start_frame, end_frame, code, team_sign); team_sign is 0 when the tag is N/A.
    """
    sets, phases = [], []
    for inst in ET.parse(path).iter("instance"):
        code = (inst.findtext("code") or "").strip().lower()
        tags = {label.findtext("group"): label.findtext("text") for label in inst.findall("label")}
        team = (tags.get("Team") or "").strip()
        sign = HOME if team == home_name else AWAY if team and team.lower() != "n/a" else 0
        start = int(round(float(inst.findtext("start")) * fps))
        end = int(round(float(inst.findtext("end")) * fps))
        if code == "set pieces":
            sets.append((start, end, _type_index(tags.get("Type")), sign, 0))
        else:
            phases.append((start, end, code, sign))
    return np.array(sorted(sets), dtype=np.int64).reshape(-1, 5), phases


def _nearest_before_after(starts, n):
    """For every frame: index of the last start <= t (or -1) and of the first start > t (or len)."""
    t = np.arange(n)
    return np.searchsorted(starts, t, side="right") - 1, np.searchsorted(starts, t, side="right")


def tag_features(set_pieces, n_frames, fps=FPS):
    """Per-frame clues from set-piece tags. Returns (X float32 [T,F], names).

    No tags at all gives constant "nothing known" columns, so a model can be trained with tags
    switched off by passing an empty array.
    """
    sp = np.asarray(set_pieces, dtype=np.int64).reshape(-1, 5)
    sp = sp[np.argsort(sp[:, 0], kind="stable")]
    starts, ends = sp[:, 0], sp[:, 1]
    types, teams = sp[:, 2], sp[:, 3]
    last, nxt = _nearest_before_after(starts, n_frames)
    t = np.arange(n_frames)
    has_last, has_next = last >= 0, nxt < len(starts)
    li, ni = np.clip(last, 0, max(len(starts) - 1, 0)), np.clip(nxt, 0, max(len(starts) - 1, 0))
    nan = np.float32(np.nan)

    def pick(arr, ok, idx):
        return np.where(ok, arr[idx], nan) if len(arr) else np.full(n_frames, nan)

    since_start = np.minimum(np.where(has_last, (t - pick(starts, has_last, li)) / fps, CAP_SECONDS), CAP_SECONDS)
    since_end = np.minimum(np.where(has_last, (t - pick(ends, has_last, li)) / fps, CAP_SECONDS), CAP_SECONDS)
    until_start = np.minimum(np.where(has_next, (pick(starts, has_next, ni) - t) / fps, CAP_SECONDS), CAP_SECONDS)
    # Inside a tagged interval: the window around the restart.
    inside = np.zeros(n_frames, dtype=np.float32)
    for a, b in zip(starts, ends):
        inside[max(a, 0):max(min(b, n_frames), 0)] = 1.0
    cols = {
        "sp_since_start": since_start, "sp_since_end": since_end, "sp_until_start": until_start,
        "sp_inside": inside,
        "sp_last_type": pick(types.astype(np.float32), has_last, li),
        "sp_next_type": pick(types.astype(np.float32), has_next, ni),
        "sp_last_team": pick(teams.astype(np.float32), has_last, li),
        "sp_next_team": pick(teams.astype(np.float32), has_next, ni),
    }
    names = list(cols)
    return np.stack([np.asarray(cols[n], dtype=np.float32) for n in names], axis=1), names


def phase_possession_sign(phases, n_frames):
    """Per frame: +1 if tags say home has the ball, -1 away, 0 unknown (attack/defend phases only)."""
    vote = np.zeros(n_frames, dtype=np.float32)
    for start, end, code, sign in phases:
        if sign == 0:
            continue
        if code in ATTACK_PHASES:
            vote[max(start, 0):max(end, 0)] += sign
        elif code in DEFEND_PHASES:
            vote[max(start, 0):max(end, 0)] -= sign
    return np.sign(vote)


def make_pseudo_tags(restarts, seed, fps=FPS, offsets_path=OFFSETS_PATH):
    """Turn exact German restarts [N,3] (frame, type, team) into imperfect tags like the real ones.

    Keeps each type with the rate that matches the real tag counts, starts the tag window before
    the restart by an amount drawn from the real offset distribution, hides the team for
    throw-ins and gets the team wrong one time in ten. Returns the [M,5] array tag_features uses.
    """
    with open(offsets_path) as f:
        agg = json.load(f)
    real_counts = {int(k): v for k, v in agg["counts"].items()}
    windows = {int(k): v for k, v in agg["window_s"].items()}
    quant = np.array(agg["start_before_restart_s"])
    rng = np.random.default_rng(seed)
    restarts = np.asarray(restarts).reshape(-1, 3)
    out = []
    for t in range(len(SET_PIECE_TYPES)):
        rows = restarts[restarts[:, 1] == t]
        if not len(rows):
            continue
        # Real tags per type vs restarts of that type in one match, as a keep rate.
        keep_rate = min(1.0, real_counts.get(t, 0) / max(len(rows), 1))
        keep = rng.random(len(rows)) < keep_rate
        for frame, _, team in rows[keep]:
            before = np.interp(rng.random(), np.linspace(0, 1, len(quant)), quant)
            start = int(frame - round(before * fps))
            end = start + int(round(windows.get(t, 35.0) * fps))
            if t == 0:
                team = 0
            elif rng.random() < 0.1:
                team = -team
            out.append((start, end, t, int(team), 0))
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 5)
