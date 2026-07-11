# Beamforming Report Artifacts

Use this reference when creating or reviewing beamforming report packs. It defines the expected artifacts, their signal-processing meaning, and the checks required before using the report for adoption decisions.

## Core Rule

A report pack must not leave plot semantics implicit. For every figure, CSV, and NPZ array, state:

1. the source array or formula,
2. shape and axis meanings,
3. units and dB reference,
4. what decision the artifact supports,
5. what decision the artifact must not be used for.

Do not decide pass/fail from a single BL, FRAZ, or BTR plot. Use the evaluation pattern and the SLC role rules in `evaluation-standards.md`.


## Report Language

For Japanese projects, write final report text artifacts in Japanese by default. This includes:

- `review_index.md`,
- scenario summaries written as Markdown,
- sweep summaries such as `frequency_offset_sweep.md`,
- design documents under `doc/`,
- final analysis text delivered to the user.

Figure titles, axes, legends, colorbars, CSV column names, NPZ keys, and method IDs may remain in English when that reduces ambiguity. Examples include `Azimuth [deg]`, `Frequency [Hz]`, `RMS Level [dB re input RMS]`, `dB re frame max`, `source_frequency_bl_overlay.png`, and `diff_mvdr_fir512`.

Do not mix languages in a way that changes the technical meaning. If a Japanese report references an English figure label or CSV column, explain the meaning in Japanese at least once in the report or artifact-definition document.

## Standard Report Pack Layout

Use this layout unless the project has a stricter local convention:

```text
artifacts/beamforming/<case_id>/review_pack/
  review_index.md
  scenario_summary.csv
  worst_cases.csv
  figures/<scenario>/
    bl_overlay.png
    source_frequency_bl_overlay.png
    bl_delta.png
    fraz_delta.png
    btr_panel.png
  data/<scenario>.npz
```

Optional sweeps should live outside the review pack when they answer a separate design question:

```text
artifacts/beamforming/<case_id>/frequency_offset_sweep/
  frequency_offset_sweep.csv
  frequency_offset_sweep.md
  frequency_offset_sweep.png
```

## Scenario, Method, and Mask Metadata

Each scenario should define:

```text
scenario_id: unique stable name
purpose: why this scenario exists
evaluation_pattern: one of the Beamforming Evaluation pattern IDs
target: azimuth [deg], frequency [Hz], level [dB re input RMS or calibrated reference]
interferers: zero or more source definitions with azimuth [deg], frequency [Hz], level, phase
mask_type: source/non-source rule, including guard width and units
```

Each method row should define the implementation role. Keep a fixed fallback method, for example `fixed_baseline`, in every report pack.

`source_mask` marks known source mainlobes and guard regions. `non_source_mask` is its complement. For source-preserving scans, known source mainlobes are not false peaks.

## review_index.md

`review_index.md` is the report table of contents. It should include:

- the role of each method,
- the dB reference for BL/FRAZ and BTR,
- the mask type and guard width,
- each scenario's purpose and source truth,
- method-level status summaries,
- relative paths to every figure and NPZ file.

This file is not the primary numerical evidence. Use `scenario_summary.csv` and `worst_cases.csv` for metric values.

## scenario_summary.csv

`scenario_summary.csv` is the primary scenario-by-method metric table. Recommended columns are:

```text
scenario, method, mask_type, candidate, status
source_peak_delta_db
source_azimuth_error_deg
non_source_global_peak_delta_db
non_source_p95_level_delta_db
non_source_p99_level_delta_db
non_source_integrated_level_delta_db
source_to_non_source_margin_delta_db
false_peak_count_delta
max_local_worsening_db_gated
fallback_required
fallback_reason
runtime_factor
evaluation_pattern
target_azimuth_deg
target_frequency_hz
interferer_azimuth_deg
interferer_frequency_hz
target_mainlobe_delta_db
target_peak_azimuth_error_deg
interferer_leakage_delta_db
interferer_leakage_reduction_db
mixed_target_beam_delta_db
q_reconstruction_rms_error
loaded_condition_number_max
source_count_expected
source_count_detected
```

Interpretation rules:

- `source_peak_delta_db` may be target-frequency BL based. Do not use it alone for different-frequency interferer visibility.
- `interferer_leakage_reduction_db` applies to local leakage cancellation into the protected target beam.
- `source_visibility_preservation` must be checked with source-frequency BL and FRAZ when sources have different frequencies.
- Runtime fields are ratios or seconds, not dB.

## worst_cases.csv

`worst_cases.csv` is a review-priority extraction, not an automatic adoption decision. Include:

- each metric's worst top 10,
- detected source-count mismatches,
- fallback rows,
- negative or watch rows,
- rows where the practical method differs strongly from the oracle/reference method.

## Figure Definitions

### bl_overlay.png

Definition:

```text
input:  FRAZ[:, target_frequency_index]
shape:  [n_beam]
x-axis: azimuth [deg]
y-axis: RMS level [dB re input RMS] or calibrated RMS reference
```

Use for:

- protected target-frequency mainlobe preservation,
- local target-beam level changes,
- same-frequency mixed response inspection.

Do not use for:

- interferer visibility when target and interferer frequencies differ,
- BTR track continuity,
- broadband suppression claims.

If an interferer is shifted from 1536.0 Hz to 1536.1 Hz, a 1536.0 Hz target-frequency BL slice will not show the 1536.1 Hz interferer peak. That is a plot-definition effect, not evidence of suppression.

### source_frequency_bl_overlay.png

Generate this figure for every scenario.

Definition:

```text
input:  FRAZ[:, source_frequency_indices]
shape before reduction: [n_beam, n_source_frequency]
reduction: max over source-frequency axis
shape after reduction:  [n_beam]
x-axis: azimuth [deg]
y-axis: RMS level [dB re input RMS] or calibrated RMS reference
```

Formula for method `m`:

```text
L_m(theta) = max_{f in F_source} FRAZ_m(theta, f)
F_source = {f_target, f_interferer_1, ...}
```

Use for:

- checking that target and interferers remain visible at their own frequencies,
- source-preserving scan review,
- near-frequency or different-frequency source visibility.

Do not use for:

- identifying which frequency produced the peak; use FRAZ for that,
- broadband noise-floor or unknown-frequency false-peak evaluation,
- BTR time-continuity checks,
- integrated suppression claims.

### bl_delta.png

Definition:

```text
delta(theta) = BL_method(theta) - BL_fixed(theta)
unit: dB re fixed BL level
```

Use for target-frequency local worsening and target mainlobe preservation. Do not use it as the sole visibility check for different-frequency interferers.

### fraz_delta.png

Definition:

```text
delta(theta, f) = FRAZ_method(theta, f) - FRAZ_fixed(theta, f)
shape: [n_beam, n_freq]
x-axis: azimuth [deg]
y-axis: frequency [Hz]
unit: dB re fixed FRAZ level
```

Use for frequency-ridge preservation, frequency-local nulls, and consistency with source-frequency BL. Preserve nonuniform equal-cos azimuth axes with cell edges when applicable.

### btr_panel.png

Definition:

```text
input: beam-time response after per-frame normalization
shape: [n_time, n_beam]
x-axis: azimuth [deg]
y-axis: time [s]
unit: dB re frame max
```

Use BTR for source-track continuity. Do not use BTR for quantitative suppression or absolute level comparisons because each frame is normalized independently.

## NPZ Plot Data

Save the arrays used to draw figures before plotting. Recommended keys:

```text
azimuth_deg: [n_beam], deg
frequency_hz: [n_freq], Hz
time_sec: [n_time], s
source_mask: [n_beam], bool
non_source_mask: [n_beam], bool

fixed_level_db: [n_beam]
<method>_level_db: [n_beam]
fixed_source_frequency_level_db: [n_beam]
<method>_source_frequency_level_db: [n_beam]

fixed_fraz_level_db: [n_beam, n_freq]
<method>_fraz_level_db: [n_beam, n_freq]

fixed_btr_level_db: [n_time, n_beam]
<method>_btr_level_db: [n_time, n_beam]
```

Label `*_level_db` and `*_fraz_level_db` with the BL/FRAZ reference, such as `dB re input RMS`. Label `*_btr_level_db` as `dB re frame max`.

## Frequency-Offset Sweep Reports

Use a frequency-offset sweep when the design question is: "how far must source frequencies differ before they are separable?"

Recommended outputs:

```text
frequency_offset_sweep.csv: primary sweep data
frequency_offset_sweep.md: conditions and conclusion
frequency_offset_sweep.png: visibility/leakage versus offset
```

Recommended columns:

```text
base_frequency_hz
offset_hz
target_frequency_hz
interferer_frequency_hz
target_visibility_delta_db
interferer_visibility_delta_db
leakage_reduction_db
target_beam_azimuth_error_deg
interferer_beam_azimuth_error_deg
analytical_two_frequency_separable
observation_bin_separable
both_source_peaks_visible
conclusion
```

State the observation duration or bin width. If the bin width is 1 Hz, offsets below 1 Hz can be analytically distinct only when the frequency axis explicitly includes both frequencies; they should not be called separable for a 1-second STFT-style observation.

## Report Creation Workflow

1. Select the evaluation pattern for each scenario.
2. Define target/interferer truth, including azimuth [deg], frequency [Hz], level, and phase.
3. Define masks and guard widths in physical units.
4. Generate method outputs and fixed fallback output.
5. Compute BL, source-frequency BL, FRAZ, and BTR arrays with explicit dB references.
6. Save NPZ plot data before rendering figures.
7. Generate figures with source masks and non-source sectors shown where useful.
8. Write `scenario_summary.csv` from effective outputs, not raw debug outputs.
9. Write `worst_cases.csv` for review prioritization.
10. Write `review_index.md` linking every figure and data file.
11. Run static/type checks and any signal-processing tests required by the repository.
12. In the final analysis, state skipped recommended checks and why.
