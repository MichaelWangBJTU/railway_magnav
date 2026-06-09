# Latest Literature Matrix for No-wheel Railway Magnetic Navigation

Date checked: 2026-06-09

## Why This Pass Was Needed

The project currently has no wheel encoder. Therefore, the fair comparison set must separate:

- Magnetic-only or magnetometer + weak motion-model methods.
- Magnetometer + IMU/INS speed-prior methods.
- Odometer / wheel-speed / Doppler-radar assisted methods, which are strong references but not directly fair baselines.

## Literature Matrix

| Year | Paper | Scenario | Sensors / assumptions | Core method | Reported performance | Relevance to our dataset |
|---:|---|---|---|---|---|---|
| 2018 | Siebler, Heirich, Sand, *Train Localization with Particle Filter and Magnetic Field Measurements*, FUSION 2018 | Railway along-track localization | Prior magnetic map, train magnetometer, simple train motion model; no wheel encoder required in the proposed baseline | SIR particle filter over along-track position, speed, and train orientation | Overall along-track RMSE 3.84 m on their train dataset; max 43.48 m. They also note weak magnetic variation causes error growth and additional speed/acceleration sensors can help. | Best historical fair baseline family for our no-wheel condition. Our reproduced no-wheel PF/HMM results are much worse mainly because our line is short, low-speed trolley data, cross-day axis inconsistency, and repeated signatures. |
| 2022 | Siebler et al., *Robust Particle Filter for Magnetic field-based Train Localization*, ION GNSS+ 2022 | Railway localization with faults/outliers | Prior magnetic map, magnetometer, railway motion model | Particle filter with fault detection / exclusion using likelihood-ratio style checks | Shows robustness improvement under sensor faults and measurement errors | Directly supports our use of heavy-tailed likelihood and confidence / uniqueness gating instead of trusting every magnetic observation. |
| 2024 | Siebler et al., *Magnetic Field Mapping of Railway Lines with Graph SLAM*, FUSION 2024 | Railway magnetic mapping over multiple runs | Magnetometer + odometer; local map per node; pose graph | Magnetic loop closures from correlation of local magnetic signatures, added as pose-graph constraints | Magnetic loop-closure relative-position RMSE 0.45 m; odometer drift bounded to meter-level in their experiments | Strong rail SOTA reference, but not directly fair because it depends on odometer edges. Useful idea: local 50 m signatures, correlation threshold, loop-closure confidence, and graph constraints. |
| 2025 | Dieckow et al., *Real-time rail vehicle localisation using spatially resolved magnetic field measurements*, arXiv 2507.19327 | Operational rail localization | Spatially resolved magnetic measurements, pre-recorded magnetic map; PF for warm start and sequence alignment for cold start | Heavy-tailed particle filter + stateless spatial sequence alignment; top-k initialization followed by PF | PF achieves sub-5 m warm-start accuracy over 21.6 km; sequence alignment localizes within 30 m in 92% top-1 and 100% top-3 tests | Most relevant latest method family. We reproduced its top-k idea on our data: 150 m TotalHP_NCC top-3/25 m is only 0.50, so our short line has stronger repeated-signature ambiguity than their dataset. |
| 2025 | Siebler et al., *Snapshot Estimator for Magnetic Field-based Train Localization with Uncalibrated Magnetometers*, EUSIPCO 2025 | Railway localization with uncalibrated magnetometers | Odometer-derived virtual magnetometer array, magnetic map, uncalibrated triad | ML snapshot estimator / SLAC: jointly estimate position plus calibration matrix and bias; calibration is conditionally linear | RMSE below 1 m at 2 Hz on an 8 km Berlin test track; about 17% false alarms for one outlier detector threshold | Very relevant to our cross-day axis inconsistency. Our SLAC-lite test shows local affine calibration alone reduces some mean error but creates very small score gaps, so it needs a motion prior and uniqueness checks. |
| 2026 | WM-GFM, *A novel geomagnetic feature matching positioning method with weak mileage aid*, Measurement 257:118850 | General geomagnetic matching | Weak mileage information plus magnetic feature matching | Uses weak mileage to constrain matching and improve geomagnetic feature matching | Public abstract metadata indicates a weak-mileage-assisted geomagnetic feature matching method; full details need library access | Conceptually supports our use of INSPVAX speed as weak mileage. Not rail-specific and not yet a fair baseline until the full paper is obtained. |

## Literature-constrained Technical Route

The most defensible route for our no-wheel data is:

1. **Magnetic-only baseline**: total-field high-pass NCC / PF / HMM.
2. **Latest rail-inspired candidate generator**: 50-150 m short spatial sequence matching, reporting top-k not just top-1.
3. **Cross-day sensor adaptation**: local affine calibration or SLAC-inspired likelihood, because 4.14 and 5.13 vector axes are not directly comparable.
4. **No-wheel temporal inference**: HMM/Viterbi or particle filter with direction, max-speed, and optional INSPVAX speed prior.
5. **Reliability layer**: uniqueness gap, method disagreement, and outlier rejection. The 2018 and 2025 railway papers both show magnetic features are not uniformly informative along the track.

## New Experiment Triggered by This Pass

Output folder:

`C:\Users\m1352\Documents\railway_magnav\latest_literature_aligned_experiments`

Short-sequence global initialization on 5.13 against the 4.14 map:

| Method | Window | Top-1 median abs error | Top-3 within 25 m |
|---|---:|---:|---:|
| TotalHP_NCC | 150 m | 87.5 m | 0.500 |
| AxisCal_XY_MSD | 150 m | 83.75 m | 0.261 |
| SLAC-lite affine XYZ | 150 m | 95.0 m | 0.207 |

Interpretation:

- The correct place often exists among candidates but is not consistently the top-1 candidate.
- SLAC-lite is not a standalone solution on this dataset because its score gap is tiny; local affine calibration can make wrong repeated signatures look plausible.
- The next useful method should be a **top-k candidate + no-wheel HMM/PF + SLAC/axis-adaptive likelihood + confidence rejection** framework.

## References and Links

- Siebler, Heirich, Sand, 2018 FUSION: https://elib.dlr.de/119898/1/FUSION_2018.pdf
- Siebler et al., 2022 ION GNSS+: https://www.ion.org/publications/abstract.cfm?articleID=18536
- Siebler et al., 2024 FUSION Graph SLAM: https://isas.iar.kit.edu/pdf/FUSION24_Siebler.pdf
- Dieckow et al., 2025 arXiv: https://arxiv.org/pdf/2507.19327
- Siebler et al., 2025 EUSIPCO snapshot estimator: https://eusipco2025.org/wp-content/uploads/pdfs/0002142.pdf
- WM-GFM weak mileage article metadata: https://www.sciencedirect.com/science/article/abs/pii/S026322412600268X
