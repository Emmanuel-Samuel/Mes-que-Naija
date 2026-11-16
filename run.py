"""One command from the provided data to the submission file.

    python run.py --out "Mes que Naija.csv"
Downloads the 7 open German matches if missing, builds the match caches, trains the judge on them
(degraded to look like the competition tracking), decodes with possession/decoder_settings.json
(variant no_tags) and writes the CSV, then runs the official validator. Set MTAH_DATA_DIR if the
competition data is not in ./data. The cross-validation (scripts/cv_german.py) is not needed.
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
VARIANT = "no_tags"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="Mes que Naija.csv")
    parser.add_argument("--rebuild-cache", action="store_true", help="ignore existing .cache files")
    args = parser.parse_args()
    start = time.time()

    import download_idsse
    download_idsse.main()

    from concurrent.futures import ProcessPoolExecutor
    import build_cache
    from possession.canonical import cache_path
    from possession.load_idsse import MATCH_IDS
    from possession.load_metrica import NAME as REFERENCE

    names = [REFERENCE] + MATCH_IDS
    todo = [n for n in names if args.rebuild_cache or not os.path.isfile(cache_path(n))]
    if todo:
        with ProcessPoolExecutor(max_workers=3) as pool:
            for name, took in pool.map(build_cache.build, todo):
                print("cache {} built in {:.0f}s".format(name, took), flush=True)

    import make_submission
    make_submission.make(VARIANT, args.out)

    print("validating {}".format(args.out), flush=True)
    check = subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "validate_submission.py"),
                            args.out])
    print("total {:.0f}s".format(time.time() - start))
    return check.returncode


if __name__ == "__main__":
    sys.exit(main())
