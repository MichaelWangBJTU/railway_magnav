# 2026-06-09 Research Progress: Axis Calibration, Uniqueness Gate, and No-Wheel HMM

## Current Technical Route

The route is now organized as:

1. Data coordinate trust: diagnose and correct cross-day magnetic axis inconsistency.
2. Magnetic feature layer: build total-field high-pass features and axis-calibrated X/Y high-pass features.
3. Matching layer: compare global MSD/NCC, uniqueness-gated window matching, and no-wheel HMM/Viterbi.
4. Constraint layer: test information uniqueness gate, endpoint prior, and INSPVAX speed as a soft transition prior.
5. Method layer: avoid a blind one-size-fits-all fusion; build confidence-aware method selection/rejection.

## Key Data Finding

The 5.13 body axes are not directly comparable with 4.14 body axes. For forward segments, the useful empirical relation is:

```text
X_4.14 ~= Z_5.13
Y_4.14 ~= Y_5.13
Z_4.14 ~= X_5.13
```

After axis remapping and per-channel bias removal, local cross-day X/Y curves can reach nT-level shape similarity. This explains why local plots can look very good while global localization still fails.

## Window Matching Results

Global matching remains dominated by repeated magnetic signatures:

- Best global 150 m total high-pass NCC median absolute error: about 76 m.
- Axis-calibrated X/Y MSD at 150 m: about 84 m median absolute error.
- If a true-position search window of +/-20 m is given, total high-pass NCC can reach about 2 m median error for 50-100 m windows. This is an upper-bound analysis only, not a deployable result.

Uniqueness gate with score gap is useful but insufficient alone:

- For 150 m total high-pass NCC, accepting the top 30-40% by score gap gives median error around 1.5 m and P75 around 2 m.
- However RMSE remains high because a small number of high-confidence false peaks survive.
- Therefore score gap should be used as a confidence/weight in HMM, not as an independent final decision.

## No-Wheel HMM/Viterbi Results

Main comparable old baseline:

- Previous `Proposed_RobustTotalHP_Viterbi`: median absolute error 62.8 m, mean absolute error 109.2 m, RMSE 123.1 m.

New experiments:

- `AxisCal_XY_TotalHP_InfoGate_Viterbi` at 4 s sampling:
  - median absolute error: 17.6 m
  - mean absolute error: 91.7 m
  - RMSE: 108.6 m
  - strong improvement in median, but still has two failing segments.

- `AxisCal_XY_TotalHP_MidGate_Viterbi` at 4 s sampling:
  - median absolute error: 17.6 m
  - mean absolute error: 90.4 m
  - RMSE: 107.3 m
  - slightly better mean/RMSE than hard gate.

- `SpeedPrior_TotalHP_Viterbi` using INSPVAX speed as a soft transition prior:
  - median absolute error: 38.5 m
  - mean absolute error: 71.5 m
  - RMSE: 78.9 m
  - more stable mean/RMSE, but not as sharp as axis-calibrated multi-feature HMM in median.

Endpoint prior did not materially improve results.

## Segment-Level Diagnosis

The best axis-calibrated information-gated HMM performs well on:

- `BMAW15230010L_1_seg03`: median error about 13.8 m
- `BMAW15230010L_9_seg01`: median error about 17.6 m
- `BMAW15230010L_9_seg02`: median error about 17.2 m

It fails on:

- `BMAW15230010L_1_seg01`: median error about 195 m
- `BMAW15230010L_1_seg04`: median error about 176 m

The `1_seg04` trajectory plot shows the information-gated axis method stays near the start for too long, then catches up late. This indicates weak early magnetic uniqueness plus insufficient progress/speed constraint.

## Negative Results

These paths are not worth prioritizing right now:

- Hard-deleting known bad 4.14 reference passes: did not fix global matching.
- Alternative backward axis permutation: worsened HMM median error.
- Endpoint-start prior: no meaningful improvement.
- Naive low-weight fusion of total + axis features + speed prior: did not improve stability.

## Current Best Interpretation

The valuable method is not "axis correction alone". It is:

```text
Cross-day axis calibration
+ robust high-pass magnetic signatures
+ information-gated multi-feature likelihood
+ no-wheel HMM/Viterbi motion continuity
+ optional INSPVAX speed soft transition prior
+ confidence-aware method selection/rejection
```

This is a real, literature-compatible innovation direction:

- It addresses ICCP/MSD repeated-peak failure through temporal inference.
- It addresses cross-day vector inconsistency through axis self-calibration.
- It avoids wheel odometry by using only direction, maximum speed, and optional IMU/INS speed as weak priors.

## Next Experiments

1. Build a confidence-aware selector between:
   - `AxisCal_XY_TotalHP_MidGate_Viterbi`
   - `SpeedPrior_TotalHP_Viterbi`
   - reject/unknown

2. Use method disagreement, final score margin, measurement uniqueness margin, and endpoint/stop consistency as selection features.

3. Test the selector under leave-one-segment-out rules to avoid overfitting the five 5.13 segments.

4. Implement a sequence-window HMM likelihood so each observation is a short magnetic signature rather than a single magnetic sample. This should reduce repeated single-point ambiguity without requiring a wheel encoder.

## Latest Literature Alignment: 2026-06-09

A new literature pass was added after checking recent railway magnetic-localization work:

- FUSION 2018 train magnetic particle filter remains the closest fair no-wheel baseline family.
- FUSION 2024 railway Graph SLAM is a strong rail SOTA reference, but it depends on odometer constraints and is not a fair baseline for the present no-wheel setup.
- arXiv 2025 rail-vehicle localization combines heavy-tailed particle filtering with stateless spatial sequence alignment. This supports using top-k short-sequence matching as an initializer, not as the final estimator.
- EUSIPCO 2025 snapshot localization with uncalibrated magnetometers supports a SLAC-style idea: jointly estimate local sensor calibration and position. This is relevant because 4.14 and 5.13 vector axes are empirically inconsistent.
- A 2026 weak-mileage geomagnetic feature-matching paper supports using weak mileage/speed information, but it needs full-paper access before being treated as a reproducible baseline.

The detailed table is saved at:

`C:\Users\m1352\Documents\railway_magnav\latest_literature_aligned_experiments\latest_literature_matrix_20260609.md`

## Literature-triggered Experiment: Short-sequence Top-k and SLAC-lite

Script:

`C:\Users\m1352\Documents\railway_magnav\latest_literature_aligned_experiments.py`

Output folder:

`C:\Users\m1352\Documents\railway_magnav\latest_literature_aligned_experiments`

Main 5.13-on-4.14 results:

- 150 m `TotalHP_NCC`: top-1 median start error 87.5 m; top-3 within 25 m rate 0.500.
- 150 m `AxisCal_XY_MSD`: top-1 median start error 83.75 m; top-3 within 25 m rate 0.261.
- 150 m `SLAC_Affine_OldXYZ_to_RefXYZ`: top-1 median start error 95.0 m; top-3 within 25 m rate 0.207.

Interpretation:

- Recent top-k sequence-alignment ideas do not directly solve our dataset. Correct positions are sometimes retained among candidates, but repeated magnetic signatures still dominate top-1.
- SLAC-lite local affine calibration is not a standalone solution; its score gap is very small, meaning wrong candidates can be fitted too easily.
- The next serious method should combine top-k candidate generation, no-wheel temporal inference, axis/SLAC-adaptive likelihood, and reliability rejection.

## Confidence Selector Diagnostic

A small selector diagnostic was added:

`C:\Users\m1352\Documents\railway_magnav\axis_calibrated_hmm_diagnostics\confidence_selector_diagnostic.csv`

Result:

- An oracle selector among `TotalHP_InfoGate_Viterbi`, `AxisCal_XY_TotalHP_MidGate_Viterbi`, `SpeedPrior_TotalHP_Viterbi`, and `Baseline_TotalHP_Viterbi` would have segment-level median errors around 4.6-28.3 m, with an aggregate median of about 15.3 m.
- Selecting the method with the largest final Viterbi score margin fails; aggregate median becomes about 40.0 m and one segment is selected badly.
- Adding the current measurement-margin heuristic does not fix this.

Interpretation:

- Viterbi final margin alone is not a reliable confidence score on this short railway section.
- Repeated magnetic signatures can produce high-confidence false paths.
- A publishable reliability layer should use richer evidence: top-k candidate cluster geometry, multi-method trajectory disagreement, speed/progress consistency, and local magnetic information content.

## Additional Experiments: 2026-06-09 Late Pass

### Sequence-likelihood HMM

Script:

`C:\Users\m1352\Documents\railway_magnav\sequence_hmm_experiment.py`

Result:

- `Seq150TruthUB_TotalHP_Viterbi`: median absolute error 33.2 m, mean absolute error 64.4 m, RMSE 74.2 m.
- `Seq100Speed_TotalHP_Viterbi`: median absolute error 46.7 m, mean absolute error 140.2 m, RMSE 147.1 m.
- `Seq100Speed_AxisXYTotal_Viterbi`: median absolute error 72.6 m, mean absolute error 102.7 m, RMSE 112.2 m.

Interpretation:

- Even when the sequence geometry is given by SPAN truth as a non-deployable upper bound, short-sequence likelihood does not beat the current best axis-calibrated HMM median of 17.6 m.
- The deployable speed-based version is worse because INSPVAX speed does not integrate to the SPAN/GPGGA projected distance consistently.
- Therefore, top-k / sequence matching alone should not be the main innovation unless the weak-mileage source is improved.

### Adaptive Weak-mileage Scale

Script:

`C:\Users\m1352\Documents\railway_magnav\adaptive_speed_scale_experiment.py`

Result:

- Oracle per-segment speed-scale selection: median absolute error 35.6 m.
- Posterior automatic scale selection with strong scale regularization: median absolute error 42.3 m.

Interpretation:

- Weak-mileage scale adaptivity has limited potential, but the current posterior score cannot select scale reliably.
- This should remain an auxiliary diagnostic, not the main paper claim.

### Reliability Learning / Rejection

Script:

`C:\Users\m1352\Documents\railway_magnav\reliability_learning_experiment.py`

Result:

- Oracle method selection among four HMM variants gives sample-level median absolute error 9.8 m and P90 42.9 m.
- Leave-one-segment-out random-forest reliability selection fails to reproduce that upper bound; even at 41% coverage, median error is about 59.7 m and P90 is about 224 m.

Interpretation:

- There is strong complementarity among methods, but current confidence features are not sufficient to learn a reliable selector from only five 5.13 segments.
- A publishable reliability learner would require either more days or training on 4.14 leave-one-pass-out with a carefully matched feature distribution.

## Key Pivot: Distance-axis / Ground-truth Diagnostic

Script:

`C:\Users\m1352\Documents\railway_magnav\distance_warp_diagnostic.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\distance_warp_diagnostic`

Main result:

- Direct identity-distance comparison between 5.13 segment total-field high-pass and the 4.14 reference has low correlation, about 0.11-0.29.
- If the full 5.13 segment is allowed to align monotonically to the 4.14 map with DTW, the correlation rises to about 0.74-0.91.
- The implied distance-axis correction is large: median absolute correction ranges from about 6.8 m to 95 m, and P90 can exceed 200 m for one segment.
- A simple affine relation between query SPAN distance and DTW-mapped reference distance often leaves modest residuals: median residual about 3.7-15.7 m.

After adding a stricter distance-banded DTW constrained to +/-60 m around the original projected distance axis:

- 5.13 cross-day identity-distance correlation remains low: about 0.11-0.29 for most usable segments.
- 5.13 cross-day banded DTW correlation improves to about 0.77-0.89.
- The required banded correction is moderate but non-negligible: median about 8-28 m, P90 about 34-47 m.

4.14 leave-one-pass-out control:

Script:

`C:\Users\m1352\Documents\railway_magnav\distance_warp_lopo_4_14.py`

Result:

- Same-day 4.14 identity-distance correlation is usually higher than cross-day, about 0.5-0.7 for normal passes, but two passes are poor.
- Same-day banded DTW improves correlation to about 0.82-0.94.
- Typical same-day banded correction is smaller for many passes, with median around 1-22 m and P90 around 13-55 m.

Interpretation:

- The magnetic shape is probably much more repeatable than the raw cross-day localization metrics suggested.
- The main blocker is not only a 5.13 truth problem; even same-day map construction benefits from a constrained distance-axis alignment layer.
- The unbounded DTW result was too optimistic; the publishable direction should use physically constrained DTW / piecewise-affine alignment, not arbitrary warping.
- This changes the research route: before claiming or optimizing a localization algorithm, build and validate a no-wheel magnetic map-alignment / distance-axis self-calibration layer.

## Revised Publishable Direction

The current top-k + HMM + weak-speed route alone is not yet strong enough for a publishable claim on this dataset.

A stronger and more defensible paper direction is:

```text
No-wheel railway magnetic mapping and localization
with cross-run distance-axis self-calibration,
axis-adaptive magnetic features,
and integrity-aware HMM/PF matching.
```

Core hypothesis:

- In short railway sections without wheel encoders, the dominant error may be distance-axis inconsistency rather than magnetic non-repeatability.
- Multi-pass magnetic sequences can be used to estimate per-run affine or piecewise-monotonic distance corrections.
- After map self-calibration, the downstream HMM/PF localization should be evaluated again on a corrected reference/query distance frame.

Next required experiments:

1. Build a corrected 4.14 reference map by aligning individual passes with banded DTW or regularized piecewise-affine corrections before averaging.
2. Compare raw-map vs corrected-map repeatability using leave-one-pass-out and 5.13 cross-day tests.
3. Align 5.13 passes to the corrected 4.14 map with constrained affine / piecewise-linear corrections and measure map similarity.
4. Re-run HMM/PF localization on the corrected map frame.
5. Separate deployable online localization from offline magnetic map construction. Full-pass DTW can be valid for map construction; online localization should use short-window HMM/PF against the corrected map.

## Follow-up: Constrained Map Alignment and Anchor-map HMM

### Full-pass and selective constrained map alignment

Script:

`C:\Users\m1352\Documents\railway_magnav\constrained_map_alignment_experiment.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\constrained_map_alignment_experiment`

Result:

- Naively aligning every 4.14 pass to the current mean map with +/-60 m banded DTW worsened map spread: median standardized spread increased from 0.903 to 2.234.
- Adding an alignment-quality gate improved map spread slightly: raw median spread 0.903, selective-aligned median spread 0.794.
- However, selective alignment did not improve downstream LOPO or cross-day map metrics. 5.13 cross-day banded-DTW median correlation changed only from 0.754 to 0.756, and same-day LOPO median banded correlation dropped from 0.822 to 0.734.
- The accepted selective alignments were mainly the three forward passes plus one backward pass; most backward passes were rejected because their alignment correlation was low or their correction hit the band edge.

Interpretation:

- Banded DTW is useful as a diagnostic, but direct full-pass self-alignment to a mean map can overfit or drift.
- Selective alignment is safer than aligning all passes, but map smoothing alone is not enough; the reference-pass choice matters.
- This is a negative but useful result: the paper method should not claim "DTW-correct every pass then average" as the final solution.

### Anchor reference selection

Script:

`C:\Users\m1352\Documents\railway_magnav\anchor_reference_selection_experiment.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\anchor_reference_selection_experiment`

Result:

- Several 4.14 anchor reference maps were compared: all passes, quality-good, forward-only, backward-only, top4/top6 same-day LOPO identity-correlation passes, and selective-gate accepted passes.
- For 5.13 cross-day map-quality diagnostics, `backward_only` had the best total-field banded-DTW median correlation: 0.789 versus 0.754 for the all-pass reference.
- `backward_only` also required a smaller median banded distance correction, 15.5 m versus 30.0 m for all-pass reference.
- Direct identity-distance correlation remained low for all candidates, which means raw SPAN/GPGGA distance axes are still inconsistent across days.

Interpretation:

- Some 4.14 passes do appear to dilute or distort the averaged magnetic map.
- Offline map-shape repeatability and online localization performance are not identical metrics; the best map-quality candidate still needs HMM/PF validation.
- Direction-aware or quality-aware map construction is a promising preprocessing contribution, but it must be tied to deployable localization metrics.

### Anchor reference HMM validation

Script:

`C:\Users\m1352\Documents\railway_magnav\anchor_reference_hmm_experiment.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\anchor_reference_hmm_experiment`

Result:

- The best deployable-style localization combination was `forward_only` reference map + `TotalHP_Viterbi`.
- It achieved median absolute error 27.3 m, mean absolute error 64.2 m, and RMSE 73.5 m on five usable 5.13 query segments.
- This improves the previous all-pass `Baseline_TotalHP_Viterbi` median of 75.6 m and mean of 77.4 m.
- It also improves the previous `SpeedPrior_TotalHP_Viterbi` mean/RMSE tradeoff, though the old axis-calibrated HMM still had a lower median of 17.6 m but much worse mean/RMSE around 90.4/107.3 m.
- `backward_only` was best in offline banded-DTW map quality but not in online HMM, so it should not be selected based on map-quality correlation alone.

Interpretation:

- A clean anchor map can materially improve no-wheel HMM localization.
- Total-field high-pass is currently more robust than vector-axis fusion for online HMM on this dataset; axis features remain useful for diagnostics and calibration but are not yet stable enough as high-weight online likelihoods.
- This result supports a publishable method component: quality/direction-aware anchor-map construction before HMM/PF localization.

### Forward-anchor HMM tuning

Script:

`C:\Users\m1352\Documents\railway_magnav\forward_anchor_hmm_tuning.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\forward_anchor_hmm_tuning`

Best result:

- `forward_only` reference map + total-field high-pass HMM + `vmax = 1.2 m/s`.
- Median absolute error: 24.6 m.
- Mean absolute error: 46.0 m.
- RMSE: 51.2 m.
- P90 absolute error: 71.8 m.

Segment-level behavior:

- `BMAW15230010L_1_seg01`: median 3.1 m, mean 9.2 m after tuning. This segment was previously a major error source.
- `BMAW15230010L_9_seg01`: median 24.6 m, mean 22.1 m.
- `BMAW15230010L_9_seg02`: median 13.8 m, mean 14.9 m.
- `BMAW15230010L_1_seg04`: median 44.7 m, mean 37.9 m.
- `BMAW15230010L_1_seg03`: still bad, median about 130.1 m and mean about 145.7 m under the best global setting.

Interpretation:

- The main improvement comes from a physically reasonable speed bound (`vmax=1.2 m/s`), not from endpoint prior; uniform and endpoint-start priors produced the same aggregate result.
- The current INSPVAX speed prior worsens many settings, confirming that speed scale is not reliable enough without per-pass calibration.
- Information-gating helps some individual segments but worsens aggregate mean/RMSE in the current simple form.
- The remaining critical outlier is `BMAW15230010L_1_seg03`. Before claiming stronger accuracy, inspect this segment for time alignment, magnetic coverage, and SPAN/GPGGA distance-axis anomalies.

Updated current best deployable result:

```text
Quality/direction-aware forward-anchor magnetic map
+ total-field high-pass observation likelihood
+ no-wheel monotonic HMM/Viterbi
+ physically tuned maximum speed bound, vmax = 1.2 m/s
= median 24.6 m, mean 46.0 m, RMSE 51.2 m on five usable 5.13 segments.
```

This does not yet reach the meter-level odometer-assisted railway Graph-SLAM papers, but it is much more defensible for the no-wheel-encoder condition. The next research question is whether the remaining outlier can be detected/rejected or corrected without using truth.

### Truth-axis anomaly diagnostic

Script:

`C:\Users\m1352\Documents\railway_magnav\truth_axis_anomaly_diagnostic.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\truth_axis_anomaly_diagnostic`

Result:

- `BMAW15230010L_1_seg03` has severe SPAN/GPGGA projected-distance jumps: examples include jumps of about 192 m, 197 m, 292 m, and 271 m between adjacent 4 s samples.
- This segment's distance truth is not physically monotonic for a backward railway pass, so using it as ordinary truth unfairly penalizes online monotonic HMM estimates.
- `BMAW15230010L_1_seg01` has mild distance-axis warnings, with two jumps around 64 m, but the tuned HMM still achieves median 3.1 m and mean 9.2 m; therefore mild warnings should not automatically cause exclusion.
- `BMAW15230010L_9_seg01` has many small opposite-direction steps but no large jumps; it remains usable and the tuned HMM achieves median 24.6 m.

Best current method, `forward_only` + `TotalHP_vmax1.2_uniform`:

- All five usable 5.13 segments: median 24.6 m, mean 46.0 m, RMSE 51.2 m, P90 71.8 m.
- Excluding only severe truth-axis anomaly (`BMAW15230010L_1_seg03`): median 19.2 m, mean 21.0 m, RMSE 25.9 m, P90 37.0 m.
- Strict no-warning-only segments: median 29.2 m, mean 26.4 m, RMSE 30.2 m, P90 42.4 m.

Interpretation:

- The severe outlier is largely an evaluation-truth problem, not just a magnetic matching problem.
- A follow-up check excluding only the jump neighborhoods in `BMAW15230010L_1_seg03` still left median error above 100 m. Therefore, this segment is not only bad at the jump samples; it also exposes a real no-wheel initialization / repeated-signature false-peak failure.
- The segment starts around the middle of the route rather than at a clean endpoint, so endpoint-direction priors do not help. This is an important boundary condition for a deployable algorithm.
- Future reports should present both "all usable segments" and "exclude severe truth-axis anomaly" metrics.
- The algorithm should not use truth-axis flags at runtime; a deployable integrity module should detect analogous failures from GNSS quality, magnetic likelihood ambiguity, and HMM residuals.

## Cold-start / Multi-hypothesis Follow-up

### Delayed multi-hypothesis HMM

Script:

`C:\Users\m1352\Documents\railway_magnav\delayed_multihypothesis_hmm_experiment.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\delayed_multihypothesis_hmm_experiment`

Motivation:

- Recent rail magnetic localization literature supports sequence-alignment initialization followed by particle-filter tracking.
- The 2025 rail-vehicle localization paper explicitly reports that cold starts and low speeds are hard for particle-filter localization, and that sequence alignment can help initialization.
- Our `BMAW15230010L_1_seg03` failure looked like a no-wheel cold-start false-peak problem.

Implemented method:

1. Run a warm-up Viterbi pass using the first W samples.
2. Keep top-K delayed endpoint hypotheses separated by at least 50 m.
3. For each hypothesis, continue a suffix HMM with a narrow prior around the delayed endpoint.
4. Select the hypothesis with the largest combined score.

Result:

- Best delayed multi-hypothesis setting, `W=90`, achieved median 28.6 m, mean 69.1 m, RMSE 73.7 m.
- This is worse than the tuned forward-anchor total-field HMM, which had median 24.6 m, mean 46.0 m, RMSE 51.2 m.
- Score-gap rejection did not reliably isolate `1_seg03`; it also rejected good segments or kept bad ones depending on the threshold.

Interpretation:

- This is a negative result.
- The top-K candidate set often contains better candidates for `1_seg03`, but cumulative likelihood ranks them too low.
- Therefore, candidate generation is not the main blocker; candidate scoring and integrity selection are the real problem.

### Robust candidate scoring

Script:

`C:\Users\m1352\Documents\railway_magnav\robust_candidate_scoring_experiment.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\robust_candidate_scoring_experiment`

Tested scoring rules:

- cumulative likelihood sum;
- per-sample likelihood;
- median path likelihood;
- trimmed mean after removing the worst 20% / 35% samples;
- 10th percentile likelihood;
- a simple robust hybrid score.

Result:

- None of the robust total-field scoring rules exceeded the tuned forward-anchor HMM.
- Best robust score was still around median 28.6 m and RMSE 73.7 m.

Interpretation:

- The correct `1_seg03` candidate cannot be reliably recovered using only robust statistics of the same total-field likelihood.
- A stronger selector needs independent evidence, not just a different aggregation of the same score.

## Current Best: IMU Progress-gated Magnetic Ensemble

Script:

`C:\Users\m1352\Documents\railway_magnav\imu_progress_gated_ensemble.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\imu_progress_gated_ensemble`

Literature support:

- FUSION 2018 train magnetic localization uses magnetic map likelihood in a particle-filter framework.
- 2025 rail-vehicle localization combines sequence alignment and particle-filter tracking, supporting hybrid candidate-generation and tracking ideas.
- WM-GFM 2026 proposes geomagnetic matching with weak mileage aid, supporting the idea that low-precision mileage information can constrain sequence matching without a wheel odometer.

Method:

1. Generate `TotalForwardAnchor`: forward-only anchor map + total-field high-pass HMM + `vmax=1.2 m/s`.
2. Generate `AxisAllMidGate`: all-pass axis-calibrated X/Y/total HMM with information gate.
3. Compute each candidate trajectory's total progress:
   - `candidate_progress = |s_final - s_start|`.
4. Compute weak IMU progress by integrating INSPVAX horizontal speed over the segment.
5. Select the candidate with smaller log-ratio mismatch:

```text
compatibility = |log((candidate_progress + 10) / (imu_progress + 10))|
```

Boundary:

- The method does not use SPAN/GPGGA truth for selection.
- INSPVAX speed is not treated as an accurate wheel odometer.
- The IMU is only used as a whole-segment progress consistency cue between two magnetic candidates.

Result:

- All five usable 5.13 segments:
  - `TotalForwardAnchor`: median 24.6 m, mean 46.0 m, RMSE 51.2 m.
  - `AxisAllMidGate`: median 17.6 m, mean 90.4 m, RMSE 107.3 m.
  - `IMUProgressClosest_TotalVsAxis`: median 13.8 m, mean 27.5 m, RMSE 42.6 m.
- Excluding only the severe truth-axis anomaly `1_seg03`:
  - `IMUProgressClosest_TotalVsAxis`: median 15.7 m, mean 22.9 m, RMSE 29.4 m.
  - `TotalForwardAnchor`: median 19.2 m, mean 21.0 m, RMSE 25.9 m.

Segment-level selection:

- `1_seg01`: selected total-field anchor path, median 3.1 m.
- `1_seg03`: selected axis-calibrated path, median 13.8 m.
- `1_seg04`: selected total-field anchor path, median 44.7 m.
- `9_seg01`: selected axis-calibrated path, median 17.6 m.
- `9_seg02`: selected total-field anchor path, median 13.8 m.

Interpretation:

- This is the strongest current all-segment result and the first method in this run that improves median, mean, and RMSE together over the tuned total-field HMM.
- The method is scientifically defensible because it uses independent weak progress evidence to arbitrate between two complementary magnetic matchers.
- It is not yet final SOTA: the dataset has only five 5.13 usable segments and needs another day or a stricter leave-one-day validation.
- It is, however, a strong candidate innovation for this no-wheel railway magnetic-navigation scenario.

Updated best deployable result:

```text
Direction/quality-aware magnetic candidate generation
+ total-field invariant HMM candidate
+ axis-calibrated vector HMM candidate
+ IMU weak-progress candidate selection
= median 13.8 m, mean 27.5 m, RMSE 42.6 m on five usable 5.13 segments.
```

### Complete-segment endpoint evaluation

Script:

`C:\Users\m1352\Documents\railway_magnav\endpoint_error_evaluation.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\imu_progress_gated_ensemble\endpoint_evaluation`

Reason:

- Median sample error is not enough to judge whether the method is usable for navigation.
- A complete segment should also be evaluated by final endpoint error, maximum error, and time coverage within error bands.

Result for `IMUProgressClosest_TotalVsAxis`:

- Median final endpoint error: 29.6 m.
- Mean final endpoint error: 36.7 m.
- Maximum final endpoint error: 84.6 m.
- Mean time coverage within 25 m: 64.2%.
- Mean time coverage within 50 m: 82.0%.

Segment endpoint errors:

- `1_seg01`: final error 8.5 m, selected total-field anchor candidate.
- `1_seg03`: final error 10.4 m, selected axis candidate, but this segment has severe truth-axis jumps.
- `1_seg04`: final error 29.6 m, selected total-field anchor candidate.
- `9_seg01`: final error 84.6 m, selected axis candidate; median error is good but terminal drift remains large.
- `9_seg02`: final error 50.3 m, selected total-field anchor candidate.

Interpretation:

- Endpoint metrics are less flattering than median error and should be included in every serious report.
- The current method is promising but not yet robust enough for a "continuous navigation" claim.
- `9_seg01` and `9_seg02` show that terminal drift / endpoint consistency should be the next optimization target.

## Output Locations

- `C:\Users\m1352\Documents\railway_magnav\axis_calibrated_experiment`
- `C:\Users\m1352\Documents\railway_magnav\axis_calibrated_hmm`
- `C:\Users\m1352\Documents\railway_magnav\axis_calibrated_hmm_gate_sweep`
- `C:\Users\m1352\Documents\railway_magnav\axis_calibrated_hmm_speed_prior`
- `C:\Users\m1352\Documents\railway_magnav\axis_calibrated_hmm_diagnostics`

## Endpoint-error Follow-up: 2026-06-09

### Why endpoint error can remain large

Magnetic matching only prevents drift accumulation when the magnetic data association remains correct. On this short railway section, large final errors came from data-association and motion-model failures rather than from ordinary dead-reckoning integration drift:

- `9_seg01` is not a clean monotonic backward pass. INSPVAX projected along-track velocity detects a sustained reverse tail near the end, with an estimated reverse distance of about 78.6 m. A fixed backward HMM naturally stays near the low-distance endpoint, while the SPAN/GPGGA truth moves back toward higher distance, producing a large final error.
- `9_seg02` is mostly monotonic, but `vmax=1.2 m/s` under-tracks the last part. The existing `vmax=1.4` total-field candidate reduces the final error from about 50.3 m to about 3.2 m, while keeping a similar sample median.
- `1_seg03` remains a special case because SPAN/GPGGA projected distance has severe jumps. It should be reported separately as a truth-axis anomaly, while still keeping an all-segment metric for honesty.

### Negative experiments

Script:

`C:\Users\m1352\Documents\railway_magnav\switching_direction_hmm_experiment.py`

Result:

- A free direction-switch HMM worsened results. Best switch variants had much larger mean/RMSE than the fixed total-field HMM.
- Interpretation: allowing direction changes without independent motion evidence creates new false-peak jumps.

Script:

`C:\Users\m1352\Documents\railway_magnav\signed_imu_prior_hmm_experiment.py`

Result:

- A global signed-INSPVAX motion-prior HMM also worsened results. The best tested signed-IMU variant had median error about 99.2 m, far worse than fixed total-field HMM.
- Interpretation: signed IMU velocity should not replace the strong known-pass-direction constraint from the start of the segment.

Script:

`C:\Users\m1352\Documents\railway_magnav\imu_switch_signed_suffix_experiment.py`

Result:

- IMU reverse-tail detection was successful and found `9_seg01` only.
- Applying signed-IMU HMM only to the detected tail improved `9_seg01` sample median but did not solve the final endpoint error robustly.
- Interpretation: tail handling needs a stronger multi-hypothesis/PF update, not a simple Viterbi restart.

### Effective endpoint-oriented selector

Script:

`C:\Users\m1352\Documents\railway_magnav\progress_margin_selector_experiment.py`

Method:

1. Generate total-field forward-anchor HMM candidates with `vmax = 1.0, 1.2, 1.4 m/s`.
2. For each total candidate, compute weak IMU progress mismatch:

```text
progress_compat = |log((candidate_progress + 10) / (imu_progress + 10))|
```

3. Keep total candidates within 0.04 log-progress mismatch of the best one.
4. Prefer candidates whose final Viterbi score margin is within 0.20 of the best margin; if still tied, choose the higher `vmax` to avoid endpoint lag.
5. Switch to the axis-calibrated candidate only when:

```text
axis_final_score_margin >= 5
and
axis_progress_compat + 0.04 < selected_total_progress_compat
```

This selector does not use SPAN/GPGGA truth for method selection. Truth is used only for evaluation.

Result on five usable 5.13 segments:

- Previous `IMUProgressClosest_TotalVsAxis`: median sample error 14.2 m, mean 27.9 m, RMSE 42.3 m; median final endpoint error 29.6 m, mean final endpoint error 36.7 m, max final endpoint error 84.6 m.
- New `ProgressMarginSelector`: median sample error 13.8 m, mean 25.2 m, RMSE 40.1 m; median final endpoint error 10.4 m, mean final endpoint error 24.6 m, max final endpoint error 71.1 m.

Selected candidates:

- `1_seg01`: `TotalHP_vmax1_uniform`
- `1_seg03`: `AxisMidGate`
- `1_seg04`: `TotalHP_vmax1.4_uniform`
- `9_seg01`: `TotalHP_vmax1.4_uniform`
- `9_seg02`: `TotalHP_vmax1.4_uniform`

Interpretation:

- This is the best current all-segment result and directly improves the endpoint metric the user identified as critical.
- It remains a small-data heuristic and must be validated with additional days or a strict leave-one-run protocol before being claimed as a final paper method.
- The scientifically defensible contribution is not merely "add direction"; it is confidence-aware selection among complementary magnetic HMM candidates using weak IMU progress and Viterbi margin evidence.

### Dataset QC: monotonic-pass interval definition

Script:

`C:\Users\m1352\Documents\railway_magnav\verify_turnaround_and_trim.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\turnaround_and_trim_diagnostic`

GPGGA / SPAN tail finding:

- `9_seg01` reached the low-distance endpoint at about `s = 9.13 m`, then moved back to about `s = 86.60 m`.
- The reverse tail length is about `77.5-78.5 m`, lasting about `492 s` in the 4 s resampled sequence.
- Raw aligned-sample GPGGA quality during the tail is good: median fix quality `4`, median satellites `21`, median HDOP `1.0`.
- Therefore the tail is not an IMU drift artifact contaminating GPGGA. It is a real recorded motion after reaching the endpoint.

Dataset-QC rule, not a method contribution:

```text
For a nominally monotonic pass:
if the GPGGA along-track distance reaches an endpoint and then reverses by more than 20 m,
evaluate the pass endpoint at the first endpoint-reaching time.
```

This is only used to define the valid one-way evaluation interval. It should be written in the data preprocessing / evaluation-protocol section, not in the proposed method. A deployable online system would instead need an operator/end-of-run flag or an independent motion-state detector, but that is outside the current method claim.

Effect on the current `ProgressMarginSelector`:

- Full five segments: median sample error `13.8 m`, mean `25.2 m`, RMSE `40.1 m`, median final endpoint error `10.4 m`, mean final endpoint error `24.6 m`, max final endpoint error `71.1 m`.
- After trimming only the detected reversal tail: median sample error `13.8 m`, mean `24.3 m`, RMSE `39.0 m`, median final endpoint error `10.4 m`, mean final endpoint error `16.7 m`, max final endpoint error `31.9 m`.
- Excluding only the severe truth-axis anomaly `1_seg03` and trimming the reversal tail: median sample error `14.1 m`, mean `18.9 m`, RMSE `24.9 m`, median final endpoint error `19.0 m`, mean final endpoint error `18.3 m`, max final endpoint error `31.9 m`.

This supports reporting both:

1. complete raw-collection segment metrics, and
2. cleaned monotonic-pass metrics with a clear independent trim rule.

Paper-writing note:

- Do not present this reverse-tail handling as an algorithmic innovation.
- In the method section, keep the contribution focused on magnetic map construction, HMM/PF-style candidate tracking, weak IMU progress constraints, and reliability/integrity scoring.
- In the experiment section, report both full raw-collection metrics and monotonic-pass metrics so reviewers can see the effect transparently.

### 4.14 / 5.13 cart-turn hypothesis check

Question checked:

- Senior-student hypothesis: 4.14 may have been pushed backward without turning the cart, while 5.13 was definitely turned around.

Evidence from raw body-axis forward/backward correlations:

- Direction-mean raw-axis correlation:
  - 4.14: x `-0.23`, y `-0.12`, z `0.16`, total `0.66`.
  - 5.13: x `-0.33`, y `0.03`, z `-0.15`, total `0.30`.
- Pairwise forward/backward pass summary:
  - 4.14: x median correlation `-0.26`, y `-0.24`, z `0.56`, total `0.45`.
  - 5.13: x median correlation `-0.28`, y `0.00`, z `-0.03`, total `0.24`.
- The yaw column does not show a clean 180 degree separation between directions, even for 5.13 where the cart was believed to be turned, so yaw is not reliable enough as a decisive orientation label here.

Interpretation:

- The current data do not support a clean rule saying "4.14 no turn, 5.13 turn" at algorithm level.
- 5.13 does show somewhat stronger x-axis sign reversal, but 4.14 also shows mixed/negative axis correlations.
- Therefore the algorithm should not hard-code this hypothesis.
- Continue prioritizing total-field features, empirical axis calibration, and confidence-aware candidate selection.

## Real-time-style Follow-up: Fixed-lag HMM and Weak-mileage Filtering

### Fixed-lag HMM

Script:

`C:\Users\m1352\Documents\railway_magnav\fixed_lag_online_hmm_experiment.py`

Purpose:

- Test whether the offline HMM can be converted into a real-time or near-real-time estimator by allowing only limited future information.
- Lags tested: `0 s`, `20 s`, `60 s`.

Result:

- Without any start-position prior, fixed-lag HMM performs much worse than the offline full-pass HMM.
- Best trimmed monotonic-pass fixed-lag result is around:
  - median sample error `108.0 m`
  - mean error `128.6 m`
  - RMSE `157.3 m`
  - median final endpoint error `29.6 m`
- Increasing lag from `0 s` to `60 s` helps slightly but does not solve the cold-start false-peak problem.

Interpretation:

- The good offline HMM/selector results rely heavily on full-pass retrospective information.
- A real-time claim is not currently justified unless initialization and multi-hypothesis management are improved.
- This aligns with rail magnetic localization literature: particle filtering can track continuously, but initialization and ambiguous magnetic signatures are critical.

### Weak-mileage sequence filter

Script:

`C:\Users\m1352\Documents\railway_magnav\weak_mileage_sequence_filter.py`

Method:

- Maintain hypotheses over:

```text
initial_position + scale * integrated_INSPVAX_speed
```

- Tested scale bank:

```text
0.45, 0.60, 0.75, 0.90, 1.05, 1.25, 1.50, 1.80, 2.20, 2.70, 3.20
```

- Tested sequence windows:
  - `120 s`
  - `240 s`
  - full history

Result:

- Best trimmed monotonic-pass result is still poor:
  - `WeakMileage_W120s`: median sample error `91.0 m`, mean `162.8 m`, RMSE `201.7 m`, median final endpoint error `91.8 m`.

Interpretation:

- A WM-GFM-like weak-mileage sequence method cannot be directly applied to the current dataset.
- INSPVAX speed scale varies too strongly between segments, and repeated magnetic signatures still dominate the sequence score.
- Weak mileage is useful as a candidate-selection cue, but not sufficient as the main online estimator here.

### Coarse-start fixed-lag HMM

Script:

`C:\Users\m1352\Documents\railway_magnav\coarse_start_online_hmm_experiment.py`

Purpose:

- Test a practical boundary condition: the vehicle starts from a known/coarsely known location, e.g. station endpoint or initial GNSS.
- Start prior was centered at the first GPGGA along-track coordinate with sigma `30 m`, `60 m`, or `100 m`.
- This is not a no-initial-position method; it is an upper-bound/practical-initialization test.

Result:

- Best trimmed monotonic-pass result:
  - `Total_vmax1_start30m_lag20s`
  - median sample error `27.5 m`
  - mean error `58.7 m`
  - RMSE `72.3 m`
  - median final endpoint error `50.3 m`
- Coarse start helps compared with uniform-start fixed-lag HMM, but still does not approach the offline `ProgressMarginSelector`.

Interpretation:

- Coarse initialization is necessary but not sufficient.
- The next serious online method should keep multiple start/scale/feature hypotheses alive and perform reliability-aware pruning, closer to a particle filter than a single Viterbi path.
- For the current dataset, the publishable near-term claim is stronger as an offline or delayed post-processing magnetic matching method than as a fully real-time navigation system.

### Endpoint-prior HMM check

Script:

`C:\Users\m1352\Documents\railway_magnav\endpoint_prior_hmm_experiment.py`

Output:

`C:\Users\m1352\Documents\railway_magnav\endpoint_prior_hmm_experiment`

Purpose:

- Test a practical railway assumption: a vehicle starts near the endpoint/station corresponding to its nominal pass direction.
- This does not use along-route SPAN/GPGGA truth for navigation and is therefore more defensible than a truth-centered start prior.

Result:

- Endpoint prior alone did not improve the uniform-start HMM; `start-endpoint_by_direction_vmax1.2` and `start-uniform_vmax1.2` both produced median error about `23.6 m`, mean error about `46.5 m`, RMSE about `51.8 m`, and mean endpoint error about `50.6 m`.
- Adding a light INSPVAX speed prior to the endpoint-start HMM improved endpoint robustness for `vmax=1.4`: median error about `29.0 m`, mean error about `53.5 m`, RMSE about `61.1 m`, median endpoint error about `29.6 m`, and mean endpoint error about `31.6 m`.
- This is still worse than the current `ProgressMarginSelector` offline/delayed result: median error about `13.8 m`, mean error about `25.2 m`, RMSE about `40.1 m`, and mean endpoint error about `24.6 m`.

Interpretation:

- A station/endpoint prior is not enough to solve the repeated-signature ambiguity on this short rail section.
- IMU speed is useful as a weak constraint, especially for endpoint consistency, but it should not be treated as a wheel odometer.
- The next publishable online method should keep multiple hypotheses over position and progress scale, using IMU only as a soft consistency term.
