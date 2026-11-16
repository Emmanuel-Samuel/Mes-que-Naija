"""Label-free aggregates of the Smart Tagging against our ball-missing gaps and the v2 labels.

    .venv/bin/python scripts/tag_stats.py
Prints percentiles and shares only, never per-frame values.
"""

import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import games  # noqa: E402
from possession.canonical import load_cached  # noqa: E402
from possession.load_idsse import MATCH_IDS  # noqa: E402
from possession.load_idsse_events import load_restarts  # noqa: E402
from possession.tagging import (ATTACK_PHASES, DEFEND_PHASES, HOME, OFFSETS_PATH,  # noqa: E402
                                SET_PIECE_TYPES, load_tag_file)

LABEL_CSV = os.path.join(ROOT, "evaluation", "results", "v2_judge.csv")


def home_team_name():
    meta = games.require(["atd_meta"])[0]
    return ET.parse(meta).getroot().find(".//Team/Name").text


def pct(x, qs=(5, 25, 50, 75, 95)):
    x = np.asarray(x, dtype=float)
    return " ".join("p{}={:+.1f}".format(q, v) for q, v in zip(qs, np.percentile(x, qs))) if len(x) else "none"


def nearest_offsets(points, anchors, fps):
    """Signed seconds from each point to the nearest anchor (anchor minus point)."""
    anchors = np.sort(anchors)
    idx = np.clip(np.searchsorted(anchors, points), 1, len(anchors) - 1)
    left, right = anchors[idx - 1], anchors[idx]
    pick = np.where(np.abs(points - left) <= np.abs(right - points), left, right)
    return (pick - points) / fps


def gap_ends(visible):
    """First visible frame after each ball-missing gap, and first missing frame of each gap."""
    d = np.diff(visible.astype(np.int8))
    return np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1


def save_offsets(sets, alive_in, fps):
    """Store the aggregates that make German pseudo-tags as imperfect as the real ones."""
    off = nearest_offsets(sets[:, 0], alive_in, fps)
    dur = {}
    counts = {}
    for t in range(len(SET_PIECE_TYPES)):
        sel = sets[:, 2] == t
        counts[t] = int(sel.sum())
        dur[t] = float(np.median((sets[sel, 1] - sets[sel, 0]) / fps)) if sel.any() else 35.0
    qs = np.arange(5, 100, 5)
    out = {"start_before_restart_s": np.percentile(off, qs).round(2).tolist(),
           "window_s": dur, "counts": counts, "n_switches": int(len(alive_in))}
    with open(OFFSETS_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved tag aggregates to {}".format(os.path.relpath(OFFSETS_PATH, ROOT)))


def german_alignment():
    """Do open-data restart events line up with the dead-to-alive switches of the German labels?"""
    print("\nGerman restarts vs dead-to-alive label switches (event time minus switch, s)")
    allv, per_type = [], {}
    for name in MATCH_IDS:
        r = load_restarts(name)
        lab = load_cached(name).label
        sw = np.flatnonzero((lab[1:] != 2) & (lab[:-1] == 2)) + 1
        off = -nearest_offsets(r[:, 0], sw, 25)
        allv.append(off)
        for t in range(len(SET_PIECE_TYPES)):
            per_type.setdefault(t, []).extend(off[r[:, 1] == t])
        print("  {} restarts {:>3} switches {:>3}  {}".format(name, len(r), len(sw), pct(off)))
    print("  all     " + pct(np.concatenate(allv)))
    for t, v in per_type.items():
        print("  {:<12} n={:>3}  {}".format(SET_PIECE_TYPES[t], len(v), pct(v)))


def main():
    m = load_cached("BAR-ATH")
    fps = m.fps
    tag_path = games.require(["smart_tagging"])[0]
    sets, phases = load_tag_file(tag_path, home_team_name())
    labels = pd.read_csv(LABEL_CSV)["label"].values
    visible = ~np.isnan(m.ball_xy[:, 0])
    ends, starts_gap = gap_ends(visible)
    dead = labels == "D"
    alive_in = np.flatnonzero(np.diff(dead.astype(np.int8)) == -1) + 1  # D to alive switches
    alive_out = np.flatnonzero(np.diff(dead.astype(np.int8)) == 1) + 1

    print("set pieces: {} (types {})".format(len(sets), {SET_PIECE_TYPES[t] if t >= 0 else "?": int((sets[:, 2] == t).sum()) for t in set(sets[:, 2])}))
    dur = (sets[:, 1] - sets[:, 0]) / fps
    print("tag window length s: " + pct(dur, (5, 50, 95)))
    for tag, pts in (("start", sets[:, 0]), ("end", sets[:, 1]), ("mid", (sets[:, 0] + sets[:, 1]) // 2)):
        print("\nset piece {}:".format(tag))
        print("  offset to nearest ball-gap end (s):    " + pct(nearest_offsets(pts, ends, fps)))
        print("  offset to nearest ball-gap start (s):  " + pct(nearest_offsets(pts, starts_gap, fps)))
        print("  offset to nearest v2 D-to-alive (s):   " + pct(nearest_offsets(pts, alive_in, fps)))
        print("  offset to nearest v2 alive-to-D (s):   " + pct(nearest_offsets(pts, alive_out, fps)))
        print("  inside a v2 alive stretch: {:.2f}   ball visible there: {:.2f}".format(
            float(np.mean(~dead[np.clip(pts, 0, len(dead) - 1)])),
            float(np.mean(visible[np.clip(pts, 0, len(dead) - 1)]))))
    near = np.abs(nearest_offsets(sets[:, 0], alive_in, fps)) < 10
    print("\nset piece starts within 10 s of a v2 D-to-alive switch: {:.2f}".format(near.mean()))
    # v2 D-to-alive switches that have any set-piece start within 10 s (how complete the tags are)
    print("v2 D-to-alive switches near a set piece start (10 s): {:.2f} of {}".format(
        float(np.mean(np.abs(nearest_offsets(alive_in, sets[:, 0], fps)) < 10)), len(alive_in)))

    save_offsets(sets, alive_in, fps)
    german_alignment()

    print("\nphase tags with a team: share of tagged frames where v2 agrees on the team")
    home_v2 = labels == "H"
    away_v2 = labels == "A"
    rows = {}
    for start, end, code, sign in phases:
        if sign == 0:
            continue
        sl = slice(max(start, 0), min(end, len(labels)))
        if code in ATTACK_PHASES:
            want = sign
        elif code in DEFEND_PHASES:
            want = -sign
        else:
            continue
        agree = (home_v2[sl] if want == HOME else away_v2[sl]).sum()
        alive = (home_v2[sl] | away_v2[sl]).sum()
        r = rows.setdefault(code, [0, 0, 0])
        r[0] += agree
        r[1] += alive
        r[2] += sl.stop - sl.start
    tot = np.zeros(3)
    for code, (a, al, n) in sorted(rows.items()):
        tot += (a, al, n)
        print("  {:<30} agree {:.2f} of alive frames, dead {:.2f}  ({} frames)".format(
            code, a / max(al, 1), 1 - al / max(n, 1), n))
    print("  ALL                            agree {:.2f} of alive frames, dead {:.2f}".format(
        tot[0] / tot[1], 1 - tot[1] / tot[2]))


if __name__ == "__main__":
    main()
