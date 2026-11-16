"""Write evaluation/results/v3_tags.csv and v3_notags.csv for the competition match.

    .venv/bin/python scripts/make_submission.py
Needs possession/decoder_settings.json from scripts/cv_german.py. The judge is trained on all 7
German matches, each degraded with three seeds. Only whole-match aggregates are printed.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from possession.canonical import load_cached  # noqa: E402
from possession.decoder import DecoderParams, decode  # noqa: E402
from possession.degrade import degrade  # noqa: E402
from possession.features import build_features  # noqa: E402
from possession.judge import predict_logp, train_judge  # noqa: E402
from possession.load_idsse import MATCH_IDS  # noqa: E402
from possession.load_idsse_events import load_restarts  # noqa: E402
from possession.tagging import load_tag_file, make_pseudo_tags  # noqa: E402
from possession.load_metrica import NAME as REFERENCE  # noqa: E402
from possession.translate_labels import translate  # noqa: E402

TRAIN_SEEDS = (0, 1, 2)
TRAIN_EVERY = 10
CONTROL_SECONDS = 2.0
OUT_DIR = os.path.join(ROOT, "evaluation", "results")
DECODER_PATH = os.path.join(ROOT, "possession", "decoder_settings.json")
LETTERS = np.array(["H", "A", "D"])


def real_tags():
    """Set-piece tags of the competition match (home = first team in the tracking metadata)."""
    import xml.etree.ElementTree as ET
    import games
    home = ET.parse(games.require(["atd_meta"])[0]).getroot().find(".//Team/Name").text
    return load_tag_file(games.require(["smart_tagging"])[0], home)[0]


def write(path, labels):
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, c) for i, c in enumerate(LETTERS[labels]))


def build_variant(variant, reference):
    """Train the judge on all 7 degraded German matches and predict the competition match."""
    Xs, ys = [], []
    for m, name in enumerate(MATCH_IDS):
        raw = load_cached(name)
        restarts = load_restarts(name)
        for seed in TRAIN_SEEDS:
            tags = make_pseudo_tags(restarts, 1000 * m + seed) if variant == "tags" else None
            Xs.append(build_features(degrade(raw, reference, seed), tags)[0])
            ys.append(translate(raw.label, raw.half, raw.fps, CONTROL_SECONDS))
    model = train_judge(Xs, ys, every=TRAIN_EVERY)
    X, _ = build_features(reference, real_tags() if variant == "tags" else None)
    return predict_logp(model, X)


def main():
    with open(DECODER_PATH) as f:
        saved = json.load(f)
    reference = load_cached(REFERENCE)
    for variant, file_tag in (("tags", "v3_tags"), ("no_tags", "v3_notags")):
        params = DecoderParams(**saved[variant])
        labels = decode(build_variant(variant, reference), reference.half, params)
        write(os.path.join(OUT_DIR, file_tag + ".csv"), labels)
        counts = [int(np.sum(labels == k)) for k in range(3)]
        print("{}: {} rows, H/A/D = {}/{}/{}, params {}".format(file_tag, len(labels), *counts, params))


if __name__ == "__main__":
    main()
