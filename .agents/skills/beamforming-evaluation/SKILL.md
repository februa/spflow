---
name: beamforming-evaluation
description: Beamforming evaluation standards for array signal processing work. Use when Codex evaluates or designs fixed beamforming, fractional/integer delay beamforming, sparse arrays, channel shading, SLC/GSC, BL/FRAZ/BTR plots, SNR improvement, waveform integrity, sidelobe/mainlobe metrics, SL/NL level normalization, spectrum-based level checks, or dB reference labeling such as dB re uPa and dB re input RMS.
---

# Beamforming Evaluation

Use this skill to choose and document evaluation criteria for beamforming and SLC work, especially in ocean-acoustic array processing code.

## Workflow

1. Identify the evaluation pattern before choosing metrics:
   - `fixed_beam_single_source`
   - `fixed_beam_multi_source`
   - `sparse_array_design`
   - `shading_design`
   - `slc_scan_multi_source_display`
   - `slc_target_only`
   - `slc_same_frequency_interference`
   - `slc_different_frequency_interference`
   - `slc_runtime`

2. Read [references/evaluation-standards.md](references/evaluation-standards.md) when you need the full criteria list, dB reference rules, SL/NL amplitude conversion rules, spectrum-based level checks, or pattern-to-metric mapping.

3. Read [references/report-artifacts.md](references/report-artifacts.md) when you create or review a report/review pack with `review_index.md`, `scenario_summary.csv`, `worst_cases.csv`, BL/FRAZ/BTR figures, source-frequency BL overlays, NPZ plot arrays, or frequency-offset sweeps.

4. Do not rely on a single BL/FRAZ/BTR figure. Check at least the required metrics for the selected pattern and state any skipped recommended checks. For BL sweeps, aim to use numerical metrics, but first quantify how well candidate metrics reproduce controlled human visual rankings or decisions. Validated metrics may drive sweeps and adoption; unvalidated metrics remain observations.

5. Treat `dB` as a ratio, not a unit. Every absolute-like level in figures, JSON, or docs must state its reference, such as `dB re 1 uPa RMS`, `dB re uPa/sqrt(Hz)`, `dB re input RMS`, or `dB re frame max`.

6. Keep relative metrics explicit. Examples: `dB re mainlobe peak`, `dB re before level`, `dB re fixed beamformer output`.

7. For BL/FRAZ/BTR plotting, preserve nonuniform equal-cos azimuth axes with cell edges, not linear imshow extents.

8. For SLC, separate the evaluation role before judging pass/fail: `source-preserving scan` keeps target and interferer visible as separate sources; `local leakage canceller` reduces interferer leakage into a protected target beam; `BL sidelobe reducer` lowers the guard-outside sidelobe envelope. Do not require interferer cancellation for source-preserving scan, and do not judge SLC only by global sidelobe reduction.

9. For SNR level checks, compare observed SNR gain against the effective channel count and analysis width. Use `20log10(sqrt(N_eff)) = 10log10(N_eff)` and label the displayed RMS levels with an explicit `dB re ...` reference.

10. For SL/NL input normalization, define whether each level is RMS tone level or one-sided noise amplitude spectral density before generating signals. Verify both in the frequency spectrum, not only in time-domain RMS.

   - `10^(SL/20)` is tone RMS amplitude. A real cosine requires peak amplitude `sqrt(2)*10^(SL/20)`.
   - Noise RMS in a one-sided bandwidth `B` Hz is `10^(NL/20)*sqrt(B)`.
   - One FFT bin uses `B=delta_f`; dividing by `sqrt(M)` means selecting one of `M` equal-width partitions of a previously integrated bandwidth, not setting a resolution of `M` Hz.

11. For narrowband and broadband output-level checks, use the same band-integrated RMS-power rule: convert the one-sided FFT bins to RMS power, sum the bins over the input signal band, and compare that sum with the input RMS power. Do not treat narrowband and broadband as different level conventions once the input band is integrated.

12. For BL metric calibration, render every method with the same azimuth axis, y-axis limits, dB reference, dynamic range, source markers, and mask display. Record numerical features and structured human rankings, pairwise preferences, or labeled decisions. Validate candidate metrics with rank correlation, pairwise agreement, classification performance, and held-out scenarios. Once a metric reaches the documented agreement target, use it for sweeps. Preserve disagreement cases as counterexamples for metric revision.

13. Decompose BL evidence into `target-only`, `noise-only`, and `target+noise` outputs before interpreting the curve:
   - Target-only: peak azimuth error, mainlobe level error relative to input SL, first-null mainlobe boundaries, first sidelobe level relative to the peak, remaining sidelobe peaks, and grating lobes.
   - Noise-only: observed output noise level versus `w^H R_n w`; for spatially white equal-variance noise use `sigma_n^2*sum(abs(w)^2)`. A rectangular normalized CBF preserves target level and improves SNR by `10log10(N)` dB.
   - Target+noise: confirm that the predicted target and noise components explain the displayed source visibility.
   - A uniform finite ULA has a Dirichlet-type array factor, approximating sinc for many sensors. Its first sidelobe is near `-13 dB re mainlobe peak`; do not require an exact sinc curve.
   - Grating-lobe onset is governed primarily by sensor spacing relative to wavelength and steering direction. Aperture length primarily controls mainlobe width and null spacing.

## spflow-Specific Hooks

When working in the `spflow` repository:

- Prefer `spflow.beamforming.get_evaluation_criteria_for_pattern(pattern_id)` for selecting required/recommended criteria.
- Use `spflow.beamforming.write_beamforming_evaluation_criteria_markdown(path)` to regenerate the design catalog.
- Use BL/FRAZ/BTR plotting helpers in `spflow.beamforming.diagnostic_plotting`; pass `level_unit_label` when physical calibration is known.
- When building report packs, include `source_frequency_bl_overlay.png` for every scenario and save the corresponding source-frequency BL arrays in the NPZ plot data.
- Follow `AGENTS.md`: Japanese comments/docstrings, shape/axis/unit comments, and explicit signal-processing assumptions are mandatory.
