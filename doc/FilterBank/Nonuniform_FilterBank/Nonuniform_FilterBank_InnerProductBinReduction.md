# Nonuniform FilterBank Inner-Product Bin Reduction

> This document is kept as a historical comparison note.
> The current leaf-structure specification is `doc/Nonuniform_FilterBank_leaf処理構造.md`.

## 1. Purpose

This document records:

- why the current leaf beamforming path requires `6656` inner-product bins
- what can be reduced while keeping the current structure
- what structural change is required to approach the ideal target of `1672` bins

Here, `bin` means the number of frequency bins where real-time channel inner products are computed.

The 50% overlap of overlap-save affects frame update rate, but it does not directly change how many bins are multiplied within one frame.

---

## 2. Ideal Target

For each leaf, if we count:

- positive-frequency bins
- plus the upper-edge bin

then the ideal physical beamforming bins are:

| band | target resolution [Hz] | ideal bins |
|---|---:|---:|
| `0 - 128` | `1` | `129` |
| `128 - 256` | `1` | `129` |
| `256 - 512` | `2` | `129` |
| `512 - 1024` | `2` | `257` |
| `1024 - 2048` | `4` | `257` |
| `2048 - 4096` | `8` | `257` |
| `4096 - 8192` | `16` | `257` |
| `8192 - 16384` | `32` | `257` |

Total:

```text
129 * 3 + 257 * 5 = 1672
```

This is the ideal per-leaf physical-bin target.

---

## 3. Current Implementation Counts

The current `src/spflow/filterbank/nonuniform_leaf.py` uses:

- long path: `frame_size = 2 * valid_size`
- short path: `short_fft_size = valid_size`

Under the current default configuration:

| band | valid size | long frame size |
|---|---:|---:|
| `0 - 128` | `256` | `512` |
| `128 - 256` | `256` | `512` |
| `256 - 512` | `256` | `512` |
| `512 - 1024` | `512` | `1024` |
| `1024 - 2048` | `512` | `1024` |
| `2048 - 4096` | `512` | `1024` |
| `4096 - 8192` | `512` | `1024` |
| `8192 - 16384` | `512` | `1024` |

### 3.1 Short Path

The current short path updates all FFT bins:

```text
256 * 3 + 512 * 5 = 3328
```

### 3.2 Long Path

The current long path performs beamforming inner products on all FFT bins:

```text
512 * 3 + 1024 * 5 = 6656
```

Therefore, the current real-time channel inner-product bin count is:

```text
6656
```

---

## 4. What Can Be Reduced Without Changing the Overall Structure

Each leaf packet is a lower-edge shifted complex band signal. Ideally, its occupied spectrum is limited to:

- `0 ... Fs_leaf / 2`

Therefore the current structure can still be improved in two steps.

### 4.1 One-Sided Long Path

The present long FFT path computes channel inner products on the full complex spectrum. Since the leaf signal is one-sided in local baseband, the long path can be reduced to:

- `frame_size / 2 + 1`

Under the current defaults this becomes:

```text
257 * 3 + 513 * 5 = 3336
```

This is about a 2x reduction from `6656`, while keeping the present overlap-save skeleton.

### 4.2 One-Sided Short Path

The short path can also be reduced from all FFT bins to:

- `short_fft_size / 2 + 1`

Under the current defaults this becomes:

```text
129 * 3 + 257 * 5 = 1672
```

So the short path can already match the ideal physical-bin target without changing its basic structure.

---

## 5. Why One-Sided Long Path Still Does Not Reach 1672

Even after one-sided reduction, the long path stops at `3336`, not `1672`.

The reason is that the current long path uses:

- `frame_size = 2 * valid_size`
- full overlap-save frame FFT

as the beamforming frequency grid itself.

That means the long-path frequency spacing is:

- `Fs_leaf / frame_size`

which is 2x finer than the physical target spacing:

- `Fs_leaf / valid_size`

So bands that only need `129` or `257` physical bins are still being beamformed on `257` or `513` long-path bins.

Therefore, one-sided packing alone is not enough. Reaching `1672` requires changing the long-path beamforming grid itself from the `frame_size` grid to the `valid_size` grid.

---

## 6. Required Measures

The practical roadmap should be split into two phases.

## 6.1 Phase A: Current-Structure Reduction `6656 -> 3336`

Goal:

- reduce the long-path inner-product bins from full-spectrum to one-sided bins

Implementation:

1. restrict long-frame beamforming to bins `0 .. frame_size/2`
2. store steering, weights, and filter responses in one-sided form
3. handle the negative side by the leaf band convention rather than explicit beamforming

Effect:

- long path: `6656 -> 3336`
- short path can also be reduced from `3328 -> 1672`

Properties:

- keeps the current overlap-save skeleton mostly intact
- moderate implementation difficulty
- does not yet reach the ideal target

## 6.2 Phase B: Long-Path Grid Change `3336 -> 1672`

Goal:

- perform long-path beamforming on the `valid_size` physical-bin grid instead of the `frame_size` overlap-save grid

Required changes:

1. unify the long-path and short-path beamforming grids
2. remove `resample_frequency_response(weights_short, long_fft_frame_size)` from the standard path
3. stop using the overlap-save frame FFT itself as the beamforming grid
4. re-establish time-domain leaf output reconstruction from the `valid_size` beamforming grid

There are two candidate directions.

### Candidate B-1: Complex One-Sided WOLA / STFT Synthesis

- perform beamforming only on the one-sided `valid_size` grid
- reconstruct the leaf output by WOLA
- use the same grid for long-path and short-path statistics

Advantages:

- aligns the inner-product bins with the ideal target `1672`
- eliminates the current weight resampling step

Challenges:

- requires a new leaf output reconstruction rule
- requires time-origin, delay, and valid-region contract review

### Candidate B-2: Physical-Bin Beamforming Plus Dedicated Time-Domain Realization

- perform beamforming only on the `valid_size` physical bins
- realize the output with a dedicated time-domain FIR or block structure

Advantages:

- inner-product bins can match the ideal target
- long-path beamforming cost is tied directly to physical bins

Challenges:

- needs a new FIR/block realization path
- the current `apply_beamformer_filter_fft()` path cannot be reused as-is

---

## 7. Recommended Order

A practical order is:

1. reduce the short path to one-sided bins
2. reduce the long path to one-sided bins and achieve `6656 -> 3336`
3. freeze the formal design for the `valid_size` long-path beamforming grid
4. implement the leaf output reconstruction needed for `3336 -> 1672`

Reasoning:

- Phase A stays close to the current structure
- Phase B is a true architectural change because it affects leaf output synthesis
- Phase A gives an intermediate result before the full redesign is complete

---

## 7.1 Current Status on 2026-07-02

What is already implemented:

- short path one-sided bin processing
- one-sided steering storage for short-path MVDR/CBF updates
- one-sided covariance update on `short_fft_size / 2 + 1` bins
- one-sided MVDR weight update on `short_fft_size / 2 + 1` bins

As a result, the default leaf set now uses:

```text
129 * 3 + 257 * 5 = 1672
```

for the short-path beamforming/statistics bins.

What is not yet implemented:

- long-path one-sided inner products

The current long path still expands the short-path response back to the full FFT grid and performs overlap-save beamforming on all long-frame bins.
This is kept intentionally because the current leaf packet signal is not sufficiently one-sided to allow an exact drop-in replacement of the long path.

A quick inspection on analyzed random analytic input showed that the negative-side energy of current leaf packets is not negligible relative to the positive side. Therefore, reducing long-path inner products to one-sided bins while keeping the current overlap-save synthesis contract would change the output rather than just lower the bin count.

So the present status is:

- short path: Phase A completed
- long path: Phase A pending, blocked by the current leaf packet / overlap-save contract

## 7.2 Phase B Prototype Status on 2026-07-02

The codebase now also contains an experimental Phase B output path:

- `output_path_mode = "leaf_independent_one_sided"`
- output frame size = `short_fft_size`
- output valid size = `short_fft_hop_size`
- runtime channel inner products are computed only on `short_fft_size / 2 + 1` bins
- the default 8-leaf set therefore reaches the target `1672 bin` on the output path

Confirmed so far:

- leaf-local one-sided output processing matches its direct physical-bin reference implementation
- full-tree offline and streaming executions match each other in this mode
- the existing `full_overlap_save` path remains available as the baseline and default path

What is still blocking promotion to the default path:

- the current formal tree leaf packets are not yet sufficiently one-sided
- direct one-sided leaf reconstruction still shows large relative RMS error versus the original leaf packet
- measured relative RMS error was about `0.86` to `1.01` across the current default leaves

Interpretation:

- the Phase B processing structure itself is now implementable
- however, the present leaf packet contract still carries substantial negative-side energy
- therefore Phase B should remain an experimental candidate until the analytic leaf packet quality is improved enough to justify default adoption

## 8. Conclusion

To approach the ideal target of `1672` bins, the work should be separated into:

1. reductions possible within the current structure
   - `6656 -> 3336`
2. structural change of the long-path beamforming grid itself
   - `3336 -> 1672`

Therefore, the next measures should be:

- first, formalize one-sided long/short bin processing
- then, remove the `frame_size`-grid dependency from the long path and move to a `valid_size` physical-bin beamforming grid
