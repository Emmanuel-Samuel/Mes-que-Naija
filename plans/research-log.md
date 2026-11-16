# Research log: team Mes que Naija

How we got to the submitted method: what we read, what we measured ourselves, what we tried and
dropped. The short version for the organisers is in [`Mes que Naija.md`](../Mes%20que%20Naija.md);
how to run the code is in [`SOLUTION.md`](../SOLUTION.md).

Rules we worked under: no hand labels on the competition match, no training or tuning on the
5 public minutes, and no competition data given to a hosted AI model. Every number about the
competition match below is a whole-match total. Every setting was chosen on other matches.

## 1. What we read

We read papers first to decide what to build. None of them compares models on this exact task, so
the reading set the direction and our own tests (section 4) made the decisions.

| Source | What it is | What we took from it |
|---|---|---|
| Bassek et al. 2025, *Scientific Data*, [PMC11787359](https://pmc.ncbi.nlm.nih.gov/articles/PMC11787359/) | The open German dataset: 7 Bundesliga matches, 25 frames per second, with the league's own ball status and possession on every frame. It publishes data only; it trains no possession model. | Our only training data. Its paper says possession is "defined by the player in control of the ball". |
| Kim et al. 2023, BallRadar, [arXiv 2306.08206](https://arxiv.org/abs/2306.08206) | Guesses the ball from player movement with a set transformer and a recurrent network. | Player movement alone carries a lot of information about who has the ball. |
| Capellera et al. 2024, TranSPORTmer, [arXiv 2410.17785](https://arxiv.org/abs/2410.17785) | One transformer for missing players, missing ball and game state, trained on 283 LaLiga matches. | The frontier needs hundreds of matches. We have 7, so we did not go this way. |
| [arXiv 2512.19528](https://arxiv.org/abs/2512.19528) (WACV 2026) | Adds image crops to the trajectories. | Not usable: we work from tracking only. |
| Vidal-Codina et al. 2022, [arXiv 2202.00804](https://arxiv.org/abs/2202.00804) | Hand-written rules for events from tracking data. | The idea behind our rules-only fallback. |
| Kim et al. 2026, PathCRF, [arXiv 2602.12080](https://arxiv.org/abs/2602.12080) | Possession path from player movement, decoded with rules that forbid impossible switches. | The closest published idea to our smoother. We did not use its code. |

We searched for an open Spanish tracking dataset with per-frame labels and found none. We also
found no published possession model trained on the seven German matches. "Found none" is what we
can say; we cannot say none exists.

## 2. Finding 1: the practice labels mean something different

We measured the possession labels of all seven German matches.

| | German matches, as published | Competition match, public 5-minute sample |
|---|---|---|
| Average possession length | 5.5 to 9.2 s | 14.2 s |
| Possessions per match | 370 to 511 | 17 in 5 minutes |
| Possessions shorter than 2 s | 25% to 37% | not measured |

The German labels hand the ball over about twice as easily. Both definitions talk about control on
paper; we do not know why they differ in practice. A judge trained on the German labels as they
are would split possessions too often.

**What we did.** A label translator, written from the wording of `CHALLENGE.md`: a new team takes
the ball only if it holds it for 2 seconds; a shorter spell goes back to the previous team; the
first team after a restart keeps the ball. After translation the German matches have 267 to 315
possessions averaging 8.7 to 12.8 seconds (with a 3-second rule: 220 to 266, averaging 10.3 to
15.4 seconds). We kept 2 seconds as our reading of "a few seconds" and did not tune it on the
public sample.

## 3. Finding 2: the two trackers treat a stopped ball differently

Measured from whole-match totals, with no answers used:

| Among frames where the ball is visible | German matches, ball hidden naively | Competition match |
|---|---|---|
| Ball standing still | 27% to 33% | 5% |
| Ball outside the pitch lines | 15% to 23% | 4% |

In the German feed a stopped ball is recorded as frozen (99% of its dead-ball frames have a still
ball). The competition tracker mostly loses a stopped ball. A judge trained on the German data
learns "still ball means stoppage", a sign that barely exists in the competition match: our first
version called only 35% of that match dead, against 41% to 52% in the German matches.

**What we did.** A "degrade" step that makes each German match look like the competition
tracking. It learns from the competition match, without labels, when the ball tends to be visible;
hides still and parked balls; shows a carried ball in about 29% of those moments; drops players in
bursts; and is tuned only so whole-match totals agree (ball missing 40.7% in both; ball still 3%
to 6% against 5%; outside the lines 3% to 6% against 4%). After this the submitted version calls
44.3% of the competition match dead.

## 4. What we tested

All scores are spellF1, the ranking metric, from leave-one-match-out on the seven degraded German
matches: train on six, test on the seventh, in rotation.

| Step | Score | Note |
|---|---|---|
| Baseline method (nearest player, majority vote) | 0.300 | Same practice data as the submitted version |
| Rules only, no training | 0.239 | Earlier round; the baseline scored 0.276 in that round |
| Version 1: trained judge and smoother | 0.679 | Earlier, less realistic practice data |
| Version 2: still ball hidden | 0.747 | Easier practice data, so not comparable with the next row |
| Version 3 (submitted): carried ball, player-behaviour clues | 0.719 | Harder, more realistic practice data |
| Version 3 with Smart Tagging set pieces | 0.719 | No gain, so dropped |

**Which model is best.** Six judges on the same data, clues, labels and scoring:

| Judge | Mean | Lowest to highest match | Without the smoother |
|---|---|---|---|
| Boosted trees (submitted) | 0.718 | 0.681 to 0.759 | 0.266 |
| Small recurrent network (BiGRU) | 0.715 | 0.676 to 0.748 | 0.337 |
| Temporal convolution network | 0.709 | 0.666 to 0.736 | 0.318 |
| Boosted trees feeding a recurrent network | 0.702 | 0.682 to 0.725 | 0.468 |
| Logistic regression | 0.660 | 0.604 to 0.707 | 0.211 |
| Extra trees | 0.654 | 0.614 to 0.679 | 0.280 |

The top three are within 0.01 of each other, inside the 0.04 noise band the organisers give for
this metric, so the order among them is not settled. We kept boosted trees: equal score, about ten
seconds to train, no extra dependency. Every model loses most of its score without the smoother,
so the smoother matters more than the choice of judge. All six gave byte-identical output on a
second run.

**Frame-level view.** On the held-out German matches the submitted pipeline labels 928,259 of
1,002,644 frames correctly (92.6%); 92.1% before the smoother. The smoother adds half a point of
frame accuracy and 0.45 of spellF1, which is the argument for the ranking metric in one line.

**Smart Tagging.** The 59 set-piece tags cover about 39% of restarts and start about 4.4 seconds
before play restarts. To test them fairly we built stand-in tags for the German matches from their
open event lists, thinned and shifted to match. They added nothing. Phase tags agree with our
output on 75% of frames (99% for build-up, 60% to 70% near the box); we had no fair way to test
them, so they are not used.

## 5. Sanity check on the public 5 minutes

Run once per version with the official scorer, and never used to choose a setting. It covers 17
possessions, and the scorer itself says gaps below about 0.40 are within noise here.

| File | spellF1 | macroF1 |
|---|---|---|
| Submitted | 0.680 | 0.860 |
| Version 2 | 0.653 | 0.712 |
| Organisers' `baseline_smooth50` | 0.326 | 0.621 |

Bootstrap range for the submitted file: 0.47 to 0.85. In this sample every frame we call dead is
dead, but we find only 63% of the dead frames, which merges possessions (13 found, 17 true).

## 6. What we did not solve

- Stoppages where the ball stays tracked are only partly found.
- The practice matches are German football on clean tracking; whole-match totals cannot show every
  difference from the competition match.
- Track swaps and duplicate tracks are not simulated.
- The 2-second control rule is our reading of the written rules.
- The top three judges are tied within noise; seven matches cannot separate them.
