# Beamforming Artifacts

`beamforming/` 配下には、複数方式の検討結果と一時診断出力が混在している。
現在の固定整相 + 差分 MVDR 評価は次の階層にある。

```text
beamforming/
  fixed_delay_diff_mvdr/
    low_frequency_128sample_mvdr/
    low_frequency_128sample_mvdr.zip
```

## 今回の評価

- 対象: x 軸 ULA、固定整相 + 差分 MVDR、128 sample 共分散
- 方位軸: 0-180 deg の x 方向余弦軸
- 広帯域: 256-1024 Hz、8500-9500 Hz
- 狭帯域: 512 Hz、768 Hz、8960 Hz の FFT bin-aligned tone

## 見ないでよいもの

- `_debug_*`
- `_op_slc_probe_*`
- `_slc_probe*`
- `_target_slc_probe*`
- `diagnostic_plotting_test`
- `evaluation_criteria_test`

これらは過去の診断・試験出力であり、今回の固定整相 + 差分 MVDR レポートの主成果物ではない。
