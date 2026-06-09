# Railway Magnetic Navigation Publication Strategy - 2026-06-09

## Current status

Current best offline / delayed-postprocessing method:

```text
Progress-Margin Selector
= total-field HMM candidates with multiple vmax
  + axis-calibrated HMM candidate
  + weak IMU progress and Viterbi-margin candidate selection
```

Current metrics on 5 usable 5.13 query segments against the 4.14 reference:

| Evaluation set | Median error | Mean error | RMSE | Median endpoint error | Mean endpoint error | Max endpoint error |
|---|---:|---:|---:|---:|---:|---:|
| Full raw segments | 13.8 m | 25.2 m | 40.1 m | 10.4 m | 24.6 m | 71.1 m |
| Clean monotonic-pass evaluation | 13.8 m | 24.3 m | 39.0 m | 10.4 m | 16.7 m | 31.9 m |
| Exclude severe truth-axis anomaly + clean monotonic-pass evaluation | 14.1 m | 18.9 m | 24.9 m | 19.0 m | 18.3 m | 31.9 m |

Important boundary:

- These are offline / full-pass matching results, not proven real-time navigation results.
- Fixed-lag and weak-mileage online-style experiments are currently much worse.
- Therefore the paper should not claim fully real-time magnetic navigation yet.

## Is this enough for a CAS Q3 paper now?

My current judgment: **not safely enough yet**.

It is promising for an internal report or conference abstract, but risky for a CAS Q3 journal because:

1. Data volume is too small: one reference day and one query day, with only 5 usable cross-day query segments.
2. The strongest result is offline / postprocessing, while the title topic "navigation" naturally implies online operation.
3. The comparison to rail magnetic SOTA is still weak. Recent rail magnetic localization reports sub-5 m performance over long operational train data when particle filtering and stronger motion information are available.
4. The method has several heuristic thresholds. They are defensible, but need ablation and validation on more days.
5. The severe `1_seg03` truth-axis anomaly and the monotonic-pass cleaning rule are acceptable as data quality control, but reviewers may challenge them if the dataset is tiny.

My confidence:

- Conference/workshop style paper: **possible now** if framed as preliminary cross-day railway magnetic map matching.
- CAS Q3 journal: **possible after more data + stronger validation**, not ideal right now.
- CAS Q2 / high-level navigation journal: **not yet**.

## Candidate venues

### Safer near-term targets

1. **IEEE Sensors Letters**
   - Pros: short paper, sensor/magnetic localization topic fit.
   - Cons: likely not CAS Q3 in some Chinese lists; one indexed source reports it as CAS 4 block, so it may not satisfy the user's target.
   - Suitable if we want a compact method note after adding one more acquisition day.

2. **Journal of Navigation**
   - Pros: navigation and PNT fit; current public metrics show JCR Q2 and CAS block B4-like records in some journal-metric pages.
   - Cons: navigation reviewers will care about online validity and operational evaluation. Current offline result is probably not enough.

3. **Sensors**
   - Pros: broad sensor-fusion and magnetic positioning scope; public current metrics show JCR Q2.
   - Cons: broad journal; reviewers may expect more complete experiments, open dataset, and ablations.
   - This is a practical target if additional data are collected.

### Harder but better aligned after method upgrade

1. **NAVIGATION: Journal of the Institute of Navigation**
   - Pros: very strong PNT fit.
   - Cons: stronger novelty and validation requirements; current real-time weakness is a major issue.

2. **Measurement / Measurement Science and Technology**
   - Pros: sensor measurement + matching methods fit; recent weak-mileage geomagnetic matching paper appeared in Measurement.
   - Cons: public current JCR records suggest these are Q1-level venues, so the current dataset is likely not enough.

3. **IEEE Sensors Journal**
   - Pros: sensors and magnetic localization fit.
   - Cons: public metrics show JCR Q1; needs stronger method and data.

## Recommended paper positioning

Do **not** position as:

```text
Real-time railway magnetic navigation without odometer.
```

This is too strong for the current experiments.

Better positioning:

```text
Cross-day railway magnetic map matching without wheel odometry:
anchor-map construction, axis-adaptive magnetic signatures,
and weak-IMU-progress-guided candidate selection.
```

Main claim:

```text
Without a wheel encoder, cross-day railway magnetic matching is still feasible
on a short rail segment if total-field invariant signatures, empirical vector-axis
calibration, and weak IMU progress are used to arbitrate among HMM candidates.
```

Secondary claim:

```text
Strict online localization remains difficult because repeated railway magnetic
signatures cause cold-start false peaks. This motivates a future particle-filter
or multi-hypothesis integrity layer.
```

## What must be improved before a Q3 journal submission

### Metric targets

For the current 700 m short-line dataset, I would use the following targets before trying a CAS Q3 journal:

| Claim level | Required evidence | Suggested target |
|---|---|---|
| Preliminary conference / workshop | cross-day matching works on the present two days | median error < 20 m, endpoint mean < 30 m |
| Q3 journal, offline/postprocessing map matching | multiple held-out days, transparent full/raw and clean-pass metrics | RMSE < 20 m, P90 < 40 m, endpoint mean < 20 m |
| Q3 journal, online navigation claim | fixed-lag or causal PF/HMM, no future data, no truth-based trimming | RMSE < 10-15 m, endpoint error usually < 20 m |
| Strong rail-magnetic SOTA comparison | long-track or richer multi-day data | approach 5 m-level RMSE, or clearly justify why no-wheel trolley data are harder |

The current best result is close to the offline/postprocessing median target, but the RMSE and endpoint robustness still need work. It is not yet close enough for a serious online-navigation claim.

### Minimum additional data

I recommend collecting at least:

- 3 additional days;
- each day at least 3 round trips, preferably 6+ one-way passes;
- keep one day as reference-map construction, one day for validation, and one or two days as held-out tests;
- keep consistent metadata: whether the cart turned around, sensor mounting direction, start/end operation, stops, and manual events.

Minimum target dataset:

```text
>= 4 days
>= 20 one-way passes
>= 3 cross-day validation pairs
```

This would make the method credible enough for a Q3 attempt.

### Method optimization priorities

1. **Particle filter / multi-hypothesis HMM**
   - State should include:

```text
s position, velocity/progress scale, direction mode, feature-model candidate
```

   - The filter should keep multiple hypotheses alive instead of collapsing to one Viterbi path early.
   - A new endpoint-prior HMM check showed that merely assuming the vehicle starts near the correct rail endpoint is not enough; it improves the story operationally, but does not solve repeated-signature ambiguity.
   - Light IMU speed priors improved endpoint consistency but still did not beat the current offline/delayed selector, so IMU should be used as a soft progress constraint rather than as a wheel-odometer substitute.

2. **Integrity / reliability score**
   - Need a deployable confidence score:

```text
top-k separation, likelihood margin, method disagreement,
IMU-progress residual, local magnetic uniqueness, endpoint consistency
```

   - Report accepted and rejected segments separately, but define rejection without using truth.

3. **Map construction**
   - Build a cleaner anchor map:

```text
quality-aware pass selection
direction-aware but not hard-coded cart-turn assumptions
robust averaging or Gaussian-process / spline smoothing
```

4. **Evaluation protocol**
   - Always report:

```text
full raw collection
clean monotonic-pass evaluation
held-out day evaluation
endpoint error
time coverage within 10/25/50 m
```

5. **Ablation**
   - Required ablations:

```text
total-only HMM
axis-only HMM
fixed vmax vs multi-vmax
without IMU progress
without Viterbi margin
with/without quality-aware map construction
```

## Handling the reverse-tail issue in the paper

Do not write this as a method contribution.

Write it only in data preprocessing / dataset quality control:

```text
For each nominal one-way pass, the evaluation interval was restricted to the
monotonic traversal of the mapped rail section. Samples recorded after the
vehicle reached the endpoint and then moved back were excluded from monotonic-pass
evaluation. Full raw-collection metrics are also reported for transparency.
```

This is not weak if both full and cleaned metrics are reported. It becomes weak only if we hide the full metric.

## Current recommendation

Best next action:

1. collect more data first;
2. in parallel, upgrade the current selector into a particle-filter / multi-hypothesis method;
3. keep the current `ProgressMarginSelector` as a strong offline baseline;
4. target a Q3-friendly sensor/navigation journal only after cross-day validation across multiple days.

My preferred near-term target after adding data:

```text
Sensors or Journal of Navigation, depending on the final story.
```

If the method remains offline/postprocessing, Sensors is easier.
If the PF becomes credible online navigation, Journal of Navigation is a better fit.
