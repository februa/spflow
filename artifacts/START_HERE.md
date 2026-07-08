# Artifacts Start Here

このディレクトリには、過去の方式検討・一時検証・テスト出力が混在している。
現在の固定整相 + 差分 MVDR の 128 sample 共分散評価を見る場合は、まず次だけを確認する。

## 今回見る成果物

- `beamforming/fixed_delay_diff_mvdr/low_frequency_128sample_mvdr/review_index.md`
- `beamforming/fixed_delay_diff_mvdr/low_frequency_128sample_mvdr/figures/`
- `beamforming/fixed_delay_diff_mvdr/low_frequency_128sample_mvdr/broadband_scenario_summary.csv`
- `beamforming/fixed_delay_diff_mvdr/low_frequency_128sample_mvdr/narrowband_scenario_summary.csv`
- `beamforming/fixed_delay_diff_mvdr/low_frequency_128sample_mvdr.zip`

## 読み方

1. `review_index.md` で評価条件と図の意味を確認する。
2. `figures/beam_response_band_integrated_*.png` で広帯域・狭帯域の beam response を見る。
3. `*_scenario_summary.csv` で peak 方位、target beam level、条件数を確認する。
4. AI や別環境へ渡す場合は `low_frequency_128sample_mvdr.zip` を使う。

## 注意

- `_debug_*`、`*_probe*`、`*_test*` は過去の診断またはテスト出力であり、今回の判断用ではない。
- `pyright_*_temp.json` は型チェック用の一時設定であり、評価結果ではない。
- 640 Hz / 9000 Hz の off-bin 自己 null 図は成果物から除外している。狭帯域 sanity check は 128-point FFT bin に一致する tone だけを見る。
