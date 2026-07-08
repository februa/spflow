# Fixed Delay Diff MVDR Artifacts

この階層には、固定整相 + 差分 MVDR に関する複数の検討出力がある。
現在見るべき成果物は `low_frequency_128sample_mvdr/` である。

## Current Report

```text
low_frequency_128sample_mvdr/
  review_index.md
  figures/
  data/
  broadband_scenario_summary.csv
  narrowband_scenario_summary.csv
  metadata.json
low_frequency_128sample_mvdr.zip
```

## 主要ファイル

- `low_frequency_128sample_mvdr/review_index.md`: 評価条件、成果物一覧、解釈メモ。
- `low_frequency_128sample_mvdr/figures/beam_response_band_integrated_low_256_1024hz.png`: 低周波広帯域の帯域加算 beam response。
- `low_frequency_128sample_mvdr/figures/beam_response_band_integrated_high_8500_9500hz.png`: 高周波広帯域の帯域加算 beam response。
- `low_frequency_128sample_mvdr/figures/beam_response_band_integrated_narrow_low_bin_512hz.png`: 低周波狭帯域 512 Hz。1 bin を選んだ帯域加算。
- `low_frequency_128sample_mvdr/figures/beam_response_band_integrated_narrow_low_bin_768hz.png`: 低周波狭帯域 768 Hz。1 bin を選んだ帯域加算。
- `low_frequency_128sample_mvdr/figures/beam_response_band_integrated_narrow_high_bin_8960hz.png`: 高周波狭帯域 8960 Hz。1 bin を選んだ帯域加算。
- `low_frequency_128sample_mvdr.zip`: AI 向け・共有向けパッケージ。

## 判断に使わないもの

`external_*`、`review_pack*`、`single_tone_noise_sweep*`、`tap_runtime_tradeoff` などは別検討の出力である。
今回の 128 sample 共分散・広帯域/狭帯域比較とは条件が異なるため、混ぜて読まない。
