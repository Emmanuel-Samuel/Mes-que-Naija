# Team Mes que Naija: how to rebuild the submission

This branch holds our solution to More than a Hack 2027 (ball possession). The technical document
is [`Mes que Naija.md`](Mes%20que%20Naija.md). The submitted file is the `v3_notags` variant.

## Install and run

Needs Python 3.11, about 6 GB of free disk (2.6 GB of open German match files plus caches) and
the competition data bundle (not in this repository).

```
uv venv --python 3.11
uv pip install -r requirements-solution.txt
export MTAH_DATA_DIR=/path/to/data        # skip if the data is in ./data
.venv/bin/python run.py --out "Mes que Naija.csv"
```

`run.py` does everything in order: downloads the 7 open IDSSE matches into `external_data/idsse/`
(skips files already there), builds the caches in `.cache/matches/`, trains the judge on the 7
German matches, decodes with the committed `possession/decoder_settings.json` (variant
`no_tags`), writes the CSV and runs `evaluation/validate_submission.py` on it. The cross-validation
is not needed. The output must be byte-identical to the file we submitted.

## Hardware and runtime

Built and tested on an Apple-silicon Mac with 18 cores and 48 GB of memory. A full run of `run.py` from a fresh copy (German files already downloaded, caches built from scratch) took 139 seconds. Downloading the 2.6 GB of German files adds a few minutes, depending on the connection.
Memory use stays below about 10 GB. No GPU.

## Folder map

| path | what it is |
|---|---|
| `run.py` | the one-command entry point |
| `possession/` | the pipeline: loaders (`load_*.py`), common match format and cache (`canonical.py`), label translator (`translate_labels.py`), tracker-matching step (`degrade.py`), clues (`features.py`), judge (`judge.py`), smoother (`decoder.py`), fallback rules (`rules.py`), set-piece tags experiment (`tagging.py`), `decoder_settings.json` |
| `scripts/download_idsse.py` | downloads the open German matches (standard library only) |
| `scripts/build_cache.py` | builds the match caches and prints one summary table |
| `scripts/make_submission.py` | trains and writes the CSV(s); `--variant no_tags --out file.csv` |
| `scripts/cv_german.py` | leave-one-match-out cross-validation and decoder-setting search |
| `tests/` | unit tests (`python -m pytest tests -q`) |
| `evaluation/` | the organisers' scorer and validator |

Not in git, by design: `data/`, `.cache/`, `external_data/`, `submission/`, `.scratch/`, result CSVs.

## Re-running the cross-validation (optional)

```
.venv/bin/python scripts/cv_german.py
```

Takes about 20 minutes on 18 cores (`N_JOBS` at the top of the file). It rewrites
`possession/decoder_settings.json`; the committed version is the one behind our file, so only run
it if you want to check the numbers in the technical document.

## What is random, and how it is seeded

- The degrade step uses `numpy.random.default_rng([seed, crc32(match name)])`, with seeds 0, 1, 2
  for each training match. Nothing depends on the clock.
- The judge (`HistGradientBoostingClassifier`) has `random_state=0` and no early stopping.
- The tags variant (not submitted) seeds its stand-in tags from the match index and seed. It
  needs `possession/tag_offsets.json`, a few summary numbers about the competition's Smart
  Tagging. That file is not in the repository; `python scripts/tag_stats.py` rebuilds it from the
  data bundle. `run.py` does not need it.
- Everything else (clues, translator, smoother) is deterministic.
- Library versions are pinned in `requirements-solution.txt`; other versions of numpy or
  scikit-learn may change the last decimals and so the file.
