import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import download_idsse  # noqa: E402
from possession.load_idsse import MATCH_IDS  # noqa: E402


def test_file_names_cover_every_match_and_kind():
    names = download_idsse.file_names()
    assert len(names) == 21 and len(set(names)) == 21
    for match in MATCH_IDS:
        assert sum(match in n for n in names) == 3
    assert "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml" in names
