"""Build the match cache (Metrica BAR-ATH + 7 IDSSE matches) and print one aggregate table.

    .venv/bin/python scripts/build_cache.py [--rebuild]
Only whole-match aggregates are printed.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))

import possession_metrics as pm  # noqa: E402
from possession.canonical import Match, cache_path, direction_check  # noqa: E402
from possession.load_idsse import MATCH_IDS, load_idsse  # noqa: E402
from possession.load_metrica import NAME as METRICA, load_metrica  # noqa: E402
from possession.translate_labels import translate  # noqa: E402

CONTROL_SECONDS = (1, 2, 3)


def build(name):
    start = time.time()
    match = load_metrica() if name == METRICA else load_idsse(name)
    match.save()
    return name, time.time() - start


def spell_stats(label, runs, fps):
    spells = pm.possession_spells(label, runs)
    n = len(spells)
    mean = sum(b - a for a, b, _ in spells) / n / fps if n else 0.0
    return n, mean


def summary_row(match, seconds):
    runs = match.halves_runs()
    row = {
        "name": match.name,
        "T": match.n_frames,
        "P": "{}/{}".format(match.home_xy.shape[1], match.away_xy.shape[1]),
        "ball%": 100 * float(np.mean(~np.isnan(match.ball_xy[:, 0]))),
        "H/A/D/U": "/".join(str(int((match.label == k).sum())) for k in range(4)),
        "build_s": seconds,
    }
    row["orig"] = "{} / {:.1f}s".format(*spell_stats(match.label, runs, match.fps))
    for c in CONTROL_SECONDS:
        translated = translate(match.label, match.half, match.fps, c)
        row["ctl{}".format(c)] = "{} / {:.1f}s".format(*spell_stats(translated, runs, match.fps))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    names = [METRICA] + MATCH_IDS
    todo = [n for n in names if args.rebuild or not os.path.isfile(cache_path(n))]
    seconds = {}
    with ProcessPoolExecutor(max_workers=3) as pool:
        for name, took in pool.map(build, todo):
            seconds[name] = took
            print("built {} in {:.0f}s".format(name, took), flush=True)

    rows = [summary_row(Match.load(cache_path(n)), seconds.get(n, float("nan"))) for n in names]
    columns = list(rows[0])
    print("\nspells columns are: possessions / mean possession seconds")
    print("  ".join("{:>14}".format(c) for c in columns))
    for row in rows:
        print("  ".join("{:>14}".format(row[c] if isinstance(row[c], (str, int)) else
                                      "{:.1f}".format(row[c])) for c in columns))

    print("\ndirection check per half 1/2: (home deepest mean x, away deepest mean x, "
          "home-minus-away mean x)")
    for n in names:
        checks = direction_check(Match.load(cache_path(n)))
        print("  {:<8} ".format(n) + "  ".join("({:.1f}, {:.1f}, {:.1f})".format(*c) for c in checks))


if __name__ == "__main__":
    main()
