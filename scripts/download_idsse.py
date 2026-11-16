"""Download the 7 open IDSSE/DFL Bundesliga matches into external_data/idsse/.

    python scripts/download_idsse.py
Standard library only. Files already present with the right size are skipped (about 2.6 GB total).
Source: https://huggingface.co/datasets/pysport/idsse-data (Bassek et al. 2025, CC-BY 4.0).
"""

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "external_data", "idsse")
BASE = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
COM = {"J03WMX": "000001", "J03WN1": "000001"}  # every other match is COM-000002
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
KINDS = ["02_01_matchinformation", "03_02_events_raw", "04_03_positions_raw_observed"]


def file_names():
    return ["DFL_{}_DFL-COM-{}_DFL-MAT-{}.xml".format(k, COM.get(m, "000002"), m)
            for m in MATCHES for k in KINDS]


def fetch(name):
    path = os.path.join(DEST, name)
    with urllib.request.urlopen(BASE + name, timeout=60) as response:
        expected = int(response.headers["Content-Length"])
        if os.path.isfile(path) and os.path.getsize(path) == expected:
            print("have     {}".format(name), flush=True)
            return
        print("download {} ({:.0f} MB)".format(name, expected / 1e6), flush=True)
        part = path + ".part"
        with open(part, "wb") as out:
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    if os.path.getsize(part) != expected:
        os.remove(part)
        raise RuntimeError("{}: size mismatch, try again".format(name))
    os.replace(part, path)


def main():
    os.makedirs(DEST, exist_ok=True)
    for name in file_names():
        fetch(name)
    print("all {} files present in {}".format(len(file_names()), DEST))


if __name__ == "__main__":
    sys.exit(main())
