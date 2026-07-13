# Beamforming Evaluation Standards

## Beamforming Evaluation Criteria

The criteria in this section begin as quantitative observations. For BL sweeps, combine or select them into a numerical decision metric only after controlled comparisons quantify agreement with structured human visual review for intended and held-out operational scenarios. A validated numerical metric should be preferred for large sweeps because it is reproducible and scalable.

- `beam_peak_position`: Check BL/FRAZ/BTR peak azimuth and frequency against source truth. Use deg and Hz. Any level must state `dB re ...`.
- `mainlobe_preservation`: Check target mainlobe level delta, peak shift, and target power delta before/after SLC or shading. These are relative dB unless an absolute reference is stated.
- `sidelobe_peak_margin`: Check mainlobe peak versus guard-outside sidelobe peak. Margin is `dB re mainlobe peak`; absolute sidelobe levels require a reference such as `dB re input RMS`.
- `grating_lobe_and_ambiguity`: Check mirror/outside peaks and alias limits, especially sparse high-frequency active subsets.
- `three_db_overlap`: Check adjacent waiting-beam -3 dB mainlobe overlap for beam interpolation. The -3 dB point is relative to local peak RMS.
- `fraz_btr_consistency`: Confirm BL peaks are consistent with FRAZ and BTR. BTR normalized per frame must be labeled `dB re frame max`.
- `source_visibility_preservation`: For SLC source-preserving scan, confirm target and interferer remain visible as separate source peaks; do not require interferer cancellation.
- `target_leakage_components`: For local leakage canceller SLC, separate mixed, target-only, and interferer-only outputs.
- `slc_covariance_health`: Check reference beam count, capacity, weight norm, and condition number. These are counts/ratios, not dB.
- `waveform_integrity`: Check output RMS/peak, NaN/inf, and power delta. Absolute levels require `dB re ...`; deltas are relative.
- `input_output_level_consistency`: Check whether output signal/noise levels and SNR gain are plausible for input level, active channel count, shading, and analysis width.
- `input_band_rms_consistency`: Convert one-sided FFT bins to RMS power, sum the bins over the input signal band, and verify that the sum equals the input RMS power. This applies equally to narrowband and broadband signals; the distinction disappears after integrating over the input band.
- `array_file_consistency`: Check channel count, active indices, aperture, spacing, and frequency table. Units are count, m, and Hz.
- `runtime_budget`: Check CPU real-time feasibility. Units are s, ratio, and count.

## Pattern Mapping

- `fixed_beam_single_source`: Required: peak position, sidelobe margin, FRAZ/BTR consistency, input/output level consistency. Recommended: grating/ambiguity, array consistency, waveform integrity.
- `fixed_beam_multi_source`: Required: peak position, FRAZ/BTR consistency, grating/ambiguity. Recommended: sidelobe margin, waveform integrity, input/output level consistency.
- `sparse_array_design`: Required: array consistency, sidelobe margin, grating/ambiguity. Recommended: peak position, input/output level consistency, runtime.
- `shading_design`: Required: -3 dB overlap, sidelobe margin, array consistency. Recommended: grating/ambiguity, input/output level consistency.
- `slc_scan_multi_source_display`: Required: source visibility preservation, mainlobe preservation, FRAZ/BTR consistency, waveform integrity. Recommended: sidelobe margin, array consistency, input/output level consistency.
- `slc_target_only`: Required: mainlobe preservation, target leakage components, waveform integrity, input/output level consistency. Recommended: covariance health.
- `slc_same_frequency_interference`: For local leakage canceller use. Required: target leakage components, mainlobe preservation, covariance health. Recommended: waveform integrity, FRAZ/BTR consistency, input/output level consistency. For source-preserving scan use `slc_scan_multi_source_display` instead.
- `slc_different_frequency_interference`: For local leakage canceller use. Required: target leakage components, mainlobe preservation, waveform integrity. Recommended: covariance health, FRAZ/BTR consistency, input/output level consistency. For source-preserving scan use `slc_scan_multi_source_display` instead.
- `slc_runtime`: Required: runtime budget, covariance health, array consistency. Recommended: waveform integrity.

## SLC Role Rules

- `source-preserving scan`: Interferers are observed sources. Preserve their visibility and do not count known-source mainlobes as sidelobes. Interferer reduction is not a pass/fail requirement.
- `local leakage canceller`: The protected target beam is the product. Require target-only preservation, interferer-only leakage reduction into that target beam, mixed-output sanity, fallback behavior, and runtime/covariance health.
- `BL sidelobe reducer`: Observe guard-outside peak, first sidelobe, integrated/percentile sidelobe envelope, maximum local worsening, peak width, and source-separation valley. Compare these observations with structured visual review under fixed display conditions. Marker-only nulling is insufficient, but no unvalidated single metric or weighted score may determine adoption.

## BL Numerical and Visual Agreement

For method comparisons, use identical azimuth axes, y-axis limits, dB references, dynamic ranges, line styles, source markers, and mask displays. A visual comparison is invalid when display conditions differ.

Record numerical and visual observations separately:

- Numerical observations: peak position, peak width, guard-outside peak, local peaks, percentiles, integrated level, source-separation valley, and maximum local worsening.
- Structured visual observations: mainlobe identifiability, source separation, conspicuous unwanted peaks, skirt width, asymmetry, and local artifacts.

Use structured visual data to calibrate candidate numerical metrics. Suitable validation measures include Spearman rank correlation, pairwise preference agreement, classification precision/recall, false-negative rate, and performance on held-out frequencies, source layouts, SNRs, and methods. When a metric meets a documented target, it may drive parameter sweeps and adoption decisions.

When numerical and visual rankings disagree, preserve the scenario, plot arrays, rendered figures, numerical rankings, visual rankings, and reasons given by reviewers. Treat the case as a counterexample for redesigning the evaluation method. Do not force agreement by changing plot limits per method or by selecting only the metric that matches the preferred conclusion.

## BL Component Decomposition

Do not infer target sidelobes and noise floor from one mixed BL curve. Generate or retain three component views with the same waiting-beam axis and level reference.

### Target-only BL

Check:

- peak azimuth error relative to source truth,
- mainlobe peak level error relative to input source RMS level,
- left and right first-null azimuths,
- -3 dB width and first-null width,
- first sidelobe peak level in each side region, expressed as `dB re mainlobe peak`,
- remaining local sidelobe peaks and maximum local worsening,
- grating-lobe candidates outside the mainlobe.

For a uniformly weighted finite ULA, the exact array factor is Dirichlet-type rather than an exact sinc. With enough uniformly spaced sensors, the first sidelobe approaches approximately `-13.26 dB re mainlobe peak`. Use this as a CBF sanity reference, not as a universal threshold for small arrays, shading, sparse arrays, endfire steering, or nonuniform grids. ABF should reduce the intended sidelobes, but it must also preserve the mainlobe and avoid moving energy into other azimuth sectors.

### Noise-only BL

Noise floor exists for CBF and ABF. Do not interpret it as an ABF-specific feature.

For beam weight `w` and channel-noise covariance `R_n`:

```text
P_noise,out = w^H R_n w
```

For spatially white, channel-uncorrelated noise with equal per-channel variance `sigma_n^2`:

```text
P_noise,out = sigma_n^2 * sum_ch |w_ch|^2
```

A normalized rectangular CBF uses `w_ch=1/N`, preserves a distortionless target level, reduces output noise power by `1/N`, and improves SNR by `10log10(N)` dB. This is spatial array gain; keep it separate from FFT-bin width, band integration, and temporal averaging gains.

For ABF, compare the observed noise-only BL with `w^H R_n w` per waiting beam. Report both the improvement relative to the fixed CBF and any azimuth where noise is amplified.

### Target-plus-noise BL

Use the mixed curve to judge operational source visibility only after target-only and noise-only components explain its level. The mixed power should be consistent with the component powers when target and noise are uncorrelated. Do not label a target sidelobe as noise floor or a noise fluctuation as a deterministic sidelobe.

### Grating Lobes

For a ULA, grating-lobe existence is governed primarily by spacing-to-wavelength ratio `d/lambda`, steering direction, and visible direction-cosine range. Aperture length `D` primarily determines mainlobe width and null spacing. When a grating-lobe candidate is detected, compare its direction with the spatial phase-alias condition; do not explain it from aperture length alone.

Perform this check before inspecting the computed BL, not only after an unexpected peak appears. For steering direction `theta_0`, solve the ULA alias equation

```text
d * (cos(theta_g) - cos(theta_0)) = m * lambda
```

for every nonzero integer `m` whose `cos(theta_g)` lies in `[-1, 1]`. Record the predicted `theta_g` values in scenario metadata and draw them on narrowband BL figures. For broadband input, solve the equation across occupied frequencies and record or shade the resulting direction ranges; broadband power integration may smear a frequency-dependent alias but does not make the array unaliased.

Classify every relevant observed peak into one of these roles before diagnosing the method:

- `target_mainlobe`: consistent with the source direction and expected beam-grid resolution,
- `predicted_grating_lobe`: consistent with a geometry/frequency prediction,
- `sidelobe`: consistent with the finite-aperture reference pattern,
- `unexplained_artifact`: inconsistent with all declared physical predictions and therefore a candidate FIR, sign, conjugation, normalization, boundary, or evaluation error.

When finite-length FIR realization suppresses the target enough that an already predicted grating lobe becomes the global maximum, report two separate observations: target-level loss caused by FIR realization, and peak-order reversal caused by the pre-existing spatial alias. Do not state that FIR truncation generated the grating lobe.

## dB Reference Rules

- Never write dB as if it were a standalone physical unit.
- Absolute acoustic RMS level: use `dB re 1 uPa RMS` when calibrated.
- Amplitude spectral density: use `dB re uPa/sqrt(Hz)`.
- Power spectral density: use `dB re uPa^2/Hz`; use `dB re uPa/Hz@ch` only if that convention is explicitly defined in the project.
- Simulation normalized level: use `dB re input RMS` or another explicit simulation reference.
- BTR normalized per time frame: use `dB re frame max`.
- Mainlobe/sidelobe margin: use `dB re mainlobe peak` or state that it is a level difference.
- Before/after reduction: use `dB re before level` or state that it is a before-after difference.

## SL/NL Input Normalization Rules

Use these rules when creating synthetic scene-renderer inputs, checking input/output level consistency, or debugging why a 0 dB source does not appear at 0 dB.

Definitions:

- `SL` is a real tone RMS level in `dB re input RMS`.
- `NL` is a one-sided white-noise amplitude spectral-density level in `dB re input RMS/sqrt(Hz)`.
- `fs` is sampling frequency in Hz.
- `N_FFT` is the FFT length used for spectrum checks.

Amplitude conversion:

```text
Amp_SL = sqrt(2) * 10^(SL / 20)
Amp_NL = 10^(NL / 20) * sqrt(fs / 2)
```

Use `Amp_SL` as the peak amplitude of a real sinusoid. Use `Amp_NL` as the time-domain sample standard deviation of real channel-uncorrelated white noise. Do not pass `10^(NL/20)` directly as the time-domain noise RMS when `NL` is an amplitude spectral-density level.

Keep these amplitude meanings separate:

```text
tone RMS amplitude                  = 10^(SL/20)
real-cosine peak amplitude          = sqrt(2) * 10^(SL/20)
noise RMS in one-sided band B [Hz]  = 10^(NL/20) * sqrt(B)
noise RMS in one FFT bin            = 10^(NL/20) * sqrt(delta_f)
delta_f                             = fs / N_FFT
```

Do not call `10^(NL/20)*sqrt(fs/2)/sqrt(M)` an `M Hz resolution` amplitude. It is the RMS in one of `M` equal-width partitions of the full one-sided bandwidth, whose width is `(fs/2)/M` Hz. It equals the RMS of a `delta_f`-wide FFT bin only when `delta_f=(fs/2)/M`.

Tone spectrum check:

```text
SL_observed = 10log10(2 * |X[k] / N_FFT|^2)
```

For an integer-bin real tone generated with `Amp_SL`, `SL_observed` should match `SL` at the tone bin. Exclude DC and Nyquist from the one-sided 2x correction unless the convention is explicitly different.

When plotting a single-tone spectrum, do not let the y-axis extend down to numerical tiny or floating-point residue can look like multiple narrowband sources. Use a documented display floor such as `max(source SL) - 120 dB`, keep bins below that floor hidden or clipped, and report the maximum non-source-bin level and visible false-peak count. If a non-source bin exceeds the display floor, treat it as a real diagnostic finding rather than hiding it.

White-noise spectrum check:

```text
NL_observed[k] = 10log10(2 * |X[k]|^2 / (N_FFT * fs))
NL_mean = 10log10(mean_k(2 * |X[k]|^2 / (N_FFT * fs)))
```

Compute `NL_mean` over non-DC, non-Nyquist positive-frequency bins, preferably averaging over channels or frames. `NL_mean` should match `NL` within statistical tolerance. Plot the per-bin `NL_observed[k]` curve with a horizontal target line at `NL`.

PNG/report expectation:

- For scene-renderer input checks, the primary frequency-spectrum PNG must show the rendered `signal + noise` waveform because that is the signal actually passed to beamforming.
- Its y-axis is per-bin RMS level `[dB re input RMS]`, not ASD. State the exact FFT normalization in the report or figure caption.
- Figure titles or captions must state the processing stage: pre-beamforming rendered signal+noise or post-beamforming output.
- Do not draw vertical source-frequency lines where they hide a narrowband peak. Put exact source frequencies in the caption or a text note when the peak itself must remain visible.
- A clean/noise-separated diagnostic is optional and must not be used as the only evidence for a scene-renderer beamforming evaluation.
- When showing post-beamforming spectra, include BL at the nearest source-frequency bin, FL at the nearest source waiting beam, and FRAZ with azimuth x-axis, frequency y-axis, and level color.
- Set line-plot y-axis limits and FRAZ color limits from finite data with documented padding/dynamic range so that mainlobes, sidelobes, and nulls remain visually readable.

## Input Band RMS Consistency

Use the same RMS-power accounting for narrowband and broadband signals. The signal bandwidth is not a separate level convention: after converting one-sided FFT bins to RMS power and summing over the bins occupied by the input signal, the result must equal the input time-domain RMS power squared.

For an `N_FFT` point real-signal rFFT spectrum `X[k]`, define per-bin RMS power as:

```text
P[k] = |X[k]|^2 / N_FFT^2              for DC and Nyquist
P[k] = 2 * |X[k]|^2 / N_FFT^2          for interior one-sided bins
```

For an input signal band `B`:

```text
sum_{k in B} P[k] == input_band_RMS^2
```

Evaluation rules:

- A narrowband integer-bin tone is just the special case where `B` contains one positive-frequency bin.
- A broadband signal uses the same rule with `B` containing all bins in the occupied input band.
- Do not compare broadband level by the height of an individual bin unless the figure is explicitly a per-bin spectrum.
- Beam-response or BL figures that claim source level preservation should plot or report band-integrated RMS level, not per-bin level.
- Spectrum figures may plot per-bin RMS level, but their caption or y-axis label must distinguish it from band-integrated RMS level.
- For multiple non-overlapping sources, check both source-specific band sums and the total band sum. If each of two sources has RMS 1, the total RMS reference is `sqrt(2)`, and each isolated source contributes `-3.01 dB re input total RMS`.
- The dB label must state the reference, for example `dB re input RMS` for a single-source band sum or `dB re input total RMS` for a multi-source total-band sum.
## SNR Gain Rules

For uncorrelated, equal-variance channel noise and a distortionless target beam:

```text
sigma_out^2 = sigma_in^2 sum(|w_ch|^2)
SNR gain = -10log10(sum(|w_ch|^2))
```

For RMS dB20 displays the same value is:

```text
SNR gain = 20log10(1 / sqrt(sum(|w_ch|^2)))
```

For rectangular delay-and-sum:

```text
w_ch = 1/N
SNR gain = 20log10(sqrt(N)) = 10log10(N)
```

For channel shading:

```text
N_eff = (sum(g_ch))^2 / sum(g_ch^2)
SNR gain = 20log10(sqrt(N_eff)) = 10log10(N_eff)
```

Keep analysis-width gain separate from spatial gain. BL/FRAZ averaging can reduce noise-floor variance; instantaneous time-domain waveform output should not assume that gain unless averaging, band limitation, or STFT integration is explicit.

## Plotting Rules

### Beam Response vs Beam Pattern

- Beam response fixes the input source condition and evaluates the output over the configured waiting-beam azimuths. Its x-axis is the waiting/steering beam axis, so the number of samples is the beam count.
- Beam pattern fixes one steering weight and sweeps the input source azimuth. Its x-axis is the input azimuth sweep, and it is not constrained to the number of configured beams.
- For MVDR beam patterns, first define the protected steering direction and the covariance used to design the weight, then freeze that weight before sweeping input azimuth. Do not redesign the MVDR weight for each input azimuth when plotting one beam pattern.
- Label beam-response and beam-pattern figures explicitly. Do not call both BL unless the axis definition is stated in the caption.


- BL y-axis examples: `RMS Level [dB re input RMS]`, `RMS Level [dB re 1 uPa RMS]`.
- FRAZ colorbar examples: `RMS Level [dB re input RMS]`, `Amplitude Spectral Density [dB re uPa/sqrt(Hz)]`.
- BTR colorbar example: `Relative Level [dB re frame max]`.
- For equal-cos beam grids, create pcolormesh cell edges from beam centers; do not use linear imshow extents.
