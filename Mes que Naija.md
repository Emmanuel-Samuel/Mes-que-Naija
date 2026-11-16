# Mes que Naija: technical document

Team Mes que Naija. More than a Hack 2027, Phase 1 (ball possession). The file we submit is the
`v3_notags` variant. To rebuild it, see "Reproduction" at the end and `SOLUTION.md`.

## 1. The approach, and why

The task is to say, for every video frame, whether the home team (H), the away team (A) or nobody
(D, dead ball) has the ball. We built three small steps in a row.

1. **Clues.** For every frame we compute simple football facts from the tracking: how far the
   nearest player of each team is from the ball, how fast the ball moves, whether it is outside the
   lines, whether it is missing, how long it has been missing, whether everybody is standing
   still, and the same facts averaged over the last and next 1 and 3 seconds. The clues describe
   teams, never named players, so no player identity is needed.
2. **Judge.** A scikit-learn `HistGradientBoostingClassifier` reads the clues and gives, for each
   frame, a probability for H, A and D.
3. **Smoother.** A Viterbi decoder (a standard way to pick the best sequence of states) turns the
   frame-by-frame probabilities into clean runs. It has "control" and "loose" states for each team,
   a cost for switching team, a cost for entering and leaving a dead ball, and a minimum control
   time of 2 seconds. It never writes `X` on a frame that could be graded.

Why this and not something else. We have no labels from the competition match and we chose not to
make any, so the judge has to learn from other football. The only large open source of per-frame
possession labels is the German Bundesliga data set (section 3). That data differs from ours in two
ways, and most of our work went into closing those two gaps rather than into a bigger model.

**Gap 1, the labels mean something different.** The German labels come from the league's own rules
and give 370 to 511 possessions per match, lasting 5.5 to 9.2 seconds on average. The competition
asks for *controlled* possession. We wrote a translator: a new team only takes the ball if it holds
it for 2 seconds (CHALLENGE.md says "a few seconds"); a shorter spell goes back to the previous
team; the first team after a restart keeps the ball. After translation the German matches have 267
to 315 possessions lasting 8.7 to 12.8 seconds on average.

**Gap 2, the trackers behave differently.** We measured this without any labels. Among frames
where the ball is visible, the ball stands still on 27 to 33% of frames in the German feed (after
naively hiding the ball where it is missing) but on only 5% in ours. Outside the lines it is 15 to
23% against 4%. The German feed keeps a frozen ball during stoppages; ours loses it. If we trained
on the German data as it is, the judge would learn "ball still means dead", which is false for
our feed. So we added a "degrade" step. It learns from our match (no labels, only the
whole-match behaviour of the tracker) when the ball is visible, then hides still and parked balls,
shows a carried ball in about 29% of those moments, drops players in bursts, and matches the
whole-match totals: 40.7% of ball frames missing in both; ball still 3 to 6% against 5%; ball
outside the lines 3 to 6% against 4%. Each training match is degraded with 3 random seeds.

Deep set-transformer models (Kim et al. 2023, arXiv 2306.08206; TranSPORTmer, arXiv 2410.17785;
PathCRF, arXiv 2602.12080) were considered and not used: we have 7 training matches and no public
weights.

## 2. Assets used and dropped

**Used**
- The tracking positions and the ball track of the competition match (`atd_raw` and its metadata),
  only as input to the clues and to the degrade step. They carry no possession labels.
- The 7 open German matches (section 3), both their tracking and their possession labels.
- The German event data, only to know where restarts happen, in the tags experiment (dropped, below).

**Tried and dropped**
- *Rules-only version* (no training, hand-written football rules): 0.239 in an earlier
  cross-validation round, below the baseline method (0.276 in that round) and far below the
  trained judge (0.679 in that round).
- *Smart Tagging set pieces.* The file has 59 tags that cover 39% of restarts and start about 4.4
  seconds before the restart. We tested the idea with stand-in tags built from German event data:
  no gain (0.719 against 0.719). So the submitted variant does not use tags.
- *Smart Tagging phase tags.* They agree with our output 75% of the time and we found no fair way to
  validate them, so we left them out.
- *Homography, broadcast feed and video.* Not needed. The video was never downloaded.
- *LightGBM.* It needs an extra system library on macOS. scikit-learn's model of the same family
  installs with one command and gave identical output on repeated runs, so we used that. We did
  not compare the two on score.

## 3. Training data

Only the 7 open IDSSE / DFL Bundesliga matches (match ids J03WMX, J03WN1, J03WOH, J03WOY, J03WPY,
J03WQQ, J03WR9): Bassek et al. 2025, *Scientific Data*, licence CC-BY 4.0. Positions are recorded
for every player and the ball at 25 frames per second, and the league itself recorded the ball
status (alive or dead) and which team had possession on every frame. We downloaded them from
https://huggingface.co/datasets/pysport/idsse-data.

- No labels from the competition match were used.
- We made no hand labels at all.
- The 5 public minutes were used only as a format and sanity check and never to choose a setting.
- The competition data was never given to a hosted AI model.

## 4. The possession rules we implemented

- **Dead ball (D).** A stoppage is never a possession. The judge learns "dead" from the league's
  ball-status labels in the German matches. In the smoother, entering or leaving a dead ball is
  costly (14 points each way) and dead runs shorter than 12 frames (about half a second) are removed,
  so stoppages do not flicker.
- **Controlled possession.** A team gains possession only after holding the ball for 2 seconds.
  Shorter touches, deflections and aerial duels are given back to the team that had it before.
  The training labels follow the same rule through the translator in section 1.
- **Loose balls and challenges.** When the evidence is weak or points the other way, the smoother
  stays in a "loose" state of the same team rather than switching. A change of team always passes
  through the other team's "control" state, so a contested ball stays with the last team that
  clearly had it until the other team has held it for the minimum time.
- **Restarts.** In the training labels, the first team after a restart keeps the ball. At
  prediction time we give the smoother no restart information (the tags were dropped); it picks the
  team the clues favour once play starts.
- **No X.** Every frame gets H, A or D.

## 5. Results

Leave-one-match-out on the 7 German matches, each degraded to look like our tracking, scored with
spellF1 (the ranking metric) against the translated labels:

| method | spellF1 |
|---|---|
| baseline method (nearest player, majority vote over ±50 frames) | 0.300 |
| ours (submitted, no tags) | 0.719 |

An earlier variant scored 0.747, but on easier practice data in which a stopped ball always
vanished, so the two numbers cannot be compared. On the 5 public minutes ours reaches 0.680 against 0.326 for the baseline, but that is
only 17 possessions, so it is noisy. Our final file has 225 possessions averaging 15.7 seconds,
and 44.3% of graded frames are dead (the German matches have 41 to 52%).

### Which model is best? Our own comparison

We tested six kinds of judge on the same seven degraded German matches, with the same clues,
labels and scoring. For every model the smoother settings for a test match were chosen on the
other six matches' predictions (a simpler scheme than the nested one above, which is why our own
judge reads 0.718 here and 0.719 there).

| judge | spellF1, mean | lowest to highest match | without the smoother |
|---|---|---|---|
| boosted trees (submitted) | 0.718 | 0.681 to 0.759 | 0.266 |
| small recurrent network (BiGRU) | 0.715 | 0.676 to 0.748 | 0.337 |
| temporal convolution network (TCN) | 0.709 | 0.666 to 0.736 | 0.318 |
| boosted trees feeding a recurrent network | 0.702 | 0.682 to 0.725 | 0.468 |
| logistic regression | 0.660 | 0.604 to 0.707 | 0.211 |
| extra trees | 0.654 | 0.614 to 0.679 | 0.280 |

The top three sit within 0.01 of each other, far inside the 0.04 noise band the organisers give
for this metric, so the ranking among them is not settled. No model beat ours on average; the
recurrent network was ahead by more than 0.04 on one match of seven. We kept boosted trees: equal
score, about ten seconds to train, no extra dependency. Two things stand out. Every model gave
byte-identical output on a second run, including the neural networks on CPU with fixed seeds. And
every model loses most of its score without the smoother, so the smoother matters more than the
choice of judge. The comparison code is `scripts/compare_models.py` (it needs PyTorch, which
`run.py` does not).

## 6. Limitations and what we would do next

- Stoppages where the ball is still tracked are only partly found: 14.7% of visible-ball frames
  are called dead.
- Three smoother settings (dead entry cost, dead exit cost, minimum dead length) sit at the edge of
  the range we searched. A wider search could help or could overfit.
- German football and clean German tracking differ from ours in ways whole-match totals cannot
  show.
- Track swaps and duplicate tracks in our feed are not simulated by the degrade step.
- The 2-second control rule is our reading of the written rules. We did not tune it on the
  5 public minutes. With more time we would test it on labelled matches from other leagues.

## 7. Reproduction

Python 3.11. In a fresh copy of the repository:

```
uv venv --python 3.11 && uv pip install -r requirements-solution.txt
export MTAH_DATA_DIR=/path/to/data      # only if the data is not in ./data
python run.py --out "Mes que Naija.csv"
```

`run.py` downloads the 7 German matches (about 2.6 GB) if they are missing, builds the caches,
trains the judge, decodes with the committed settings (`possession/decoder_settings.json`, variant
`no_tags`), writes the CSV and runs the official validator. The settings were chosen once by
`scripts/cv_german.py` and are committed, so the cross-validation does not need to be repeated.
Random choices are all seeded, so the same file comes out each time (details in `SOLUTION.md`).

**Use of AI.** The code was written with an AI coding assistant. No competition data was given to
any hosted AI model; only whole-match summary numbers (counts and percentages) were shared.
