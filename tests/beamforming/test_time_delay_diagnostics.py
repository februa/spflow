"""時間領域固定整相の BL/FRAZ/BTR 診断に関する回帰試験。"""

# 以前の非均一帯域フィルタバンク検証で代表条件だった単一音源条件に加え、
# 実運用に近い複数方位・複数周波数かつスパース片舷アレイ条件、さらに
# 同一方位に異なる周波数が重畳する条件でも、固定整相のピーク位置が大きく崩れないことを保存画像つきで固定する。

from __future__ import annotations

import json
from pathlib import Path

from evaluations.beamforming.scenarios.time_delay_diagnostics import (
    TimeDelayDiagnosticConfig,
    TimeDelayDiagnosticSource,
    run_integer_delay_diagnostics,
)


def test_integer_delay_diagnostics_save_bl_fraz_btr_and_report_expected_peaks():
    """整数遅延診断について BL/FRAZ/BTR を保存し target のピーク位置とレベルを要約できることを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "time_delay_integer_diagnostics_test"
    summary = run_integer_delay_diagnostics(
        TimeDelayDiagnosticConfig(
            output_dir=output_dir,
            fs_hz=32768.0,
            duration_s=1.0,
            sound_speed_m_s=1500.0,
            source_frequency_hz=1536.0,
            source_level_db20=0.0,
            source_azimuth_deg=20.0,
            source_elevation_deg=0.0,
            noise_level_db20=-40.0,
            random_seed=1234,
            array_n_ch=160,
            array_sensor_spacing_m=0.05,
            n_beam_az_real=241,
            btr_block_size=1024,
        )
    )

    bl_path = output_dir / "bl.png"
    fraz_path = output_dir / "fraz.png"
    btr_path = output_dir / "btr.png"
    notes_path = output_dir / "plot_usage_notes.md"
    summary_path = output_dir / "summary.json"

    assert bl_path.exists()
    assert fraz_path.exists()
    assert btr_path.exists()
    assert notes_path.exists()
    assert summary_path.exists()

    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert saved_summary["bl_png_path"] == summary["bl_png_path"]
    assert saved_summary["fraz_png_path"] == summary["fraz_png_path"]
    assert saved_summary["btr_png_path"] == summary["btr_png_path"]
    assert saved_summary["plot_usage_notes_path"] == summary["plot_usage_notes_path"]

    notes_text = notes_path.read_text(encoding="utf-8")
    assert "等 cos 空間" in notes_text
    assert "imshow(extent=...)" in notes_text
    assert "BTR は各時刻で最大ビームを 0 dB に正規化" in notes_text

    # ビーム軸は等角ではなく cos 空間で配置しているため、
    # target 方位との差は nearest beam 1 本分程度までは許容する。
    assert abs(float(summary["bl_peak_azimuth_deg"]) - 20.0) <= 2.0
    assert abs(float(summary["bl_peak_level_db20"]) - 0.0) <= 0.2

    # FRAZ は RFFT の 1 Hz 分解能条件に固定しているため、
    # target 周波数ピークは整数 Hz bin に一致することを期待する。
    assert abs(float(summary["fraz_peak_azimuth_deg"]) - 20.0) <= 2.0
    assert abs(float(summary["fraz_peak_frequency_hz"]) - 1536.0) <= 0.5
    assert abs(float(summary["fraz_peak_level_db20"]) - 0.0) <= 0.2
    assert abs(float(summary["fraz_level_at_nearest_source_grid_db20"]) - 0.0) <= 0.2

    # 単一音源条件の BTR では、各時間ブロックで同じ方位ビームが最大になることを確認する。
    assert abs(float(summary["btr_mean_peak_azimuth_deg"]) - 20.0) <= 2.0
    assert float(summary["btr_peak_azimuth_std_deg"]) <= 1e-9

    # 1 列片舷アレイを x 軸へ正しく置けていれば、鏡像方位 180-az では強いピークにならない。
    assert abs(float(summary["mirror_azimuth_deg"]) - 160.0) <= 2.0
    assert float(summary["mirror_level_db20"]) <= float(summary["bl_peak_level_db20"]) - 20.0


def test_integer_delay_diagnostics_handle_sparse_array_with_multiple_azimuths_and_multiple_frequencies():
    """スパース片舷アレイ上で複数方位・複数周波数 source を同時に与えても source ごとのピークが概ね正しいことを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "time_delay_sparse_multi_source_diagnostics_test"
    summary = run_integer_delay_diagnostics(
        TimeDelayDiagnosticConfig(
            output_dir=output_dir,
            fs_hz=32768.0,
            duration_s=1.0,
            sound_speed_m_s=1500.0,
            noise_level_db20=-45.0,
            random_seed=1234,
            array_n_ch=61,
            array_sensor_spacing_m=0.05,
            sparse_stride_pattern=(1, 2, 1, 3, 1, 4, 2, 5),
            n_beam_az_real=241,
            btr_block_size=1024,
            source_specs=(
                TimeDelayDiagnosticSource(azimuth_deg=20.0, frequency_hz=1024.0, level_db20=0.0, label="A"),
                TimeDelayDiagnosticSource(azimuth_deg=58.0, frequency_hz=1536.0, level_db20=0.0, label="B"),
                TimeDelayDiagnosticSource(azimuth_deg=112.0, frequency_hz=2304.0, level_db20=0.0, label="C"),
            ),
        )
    )

    summary_path = output_dir / "summary.json"
    fraz_path = output_dir / "fraz.png"
    btr_path = output_dir / "btr.png"
    notes_path = output_dir / "plot_usage_notes.md"

    assert summary_path.exists()
    assert fraz_path.exists()
    assert btr_path.exists()
    assert notes_path.exists()
    assert bool(summary["array_is_sparse"])
    assert int(summary["n_source"]) == 3
    assert float(summary["array_max_sensor_spacing_m"]) > float(summary["array_min_sensor_spacing_m"])

    source_metrics = summary["source_metrics"]
    assert isinstance(source_metrics, list)
    assert len(source_metrics) == 3

    # スパース化で等間隔 ULA の周期性を崩しても、各 source 周波数で見た BL の主ピークが
    # 対応する到来方位近傍へ残っていることを確認する。
    for source_metric in source_metrics:
        assert Path(str(source_metric["bl_png_path"])).exists()
        assert abs(float(source_metric["bl_peak_azimuth_deg"]) - float(source_metric["source_azimuth_deg"])) <= 3.0
        assert abs(float(source_metric["bl_peak_level_db20"]) - 0.0) <= 0.8
        assert abs(float(source_metric["bl_level_at_nearest_source_grid_db20"]) - 0.0) <= 1.0
        assert float(source_metric["mirror_level_db20"]) <= float(source_metric["bl_peak_level_db20"]) - 8.0

    # 複数同時音源の BTR では単一 peak track ではなく各 target ridge の見え方が重要になるため、
    # 各 target beam の平均相対レベルが極端に落ち込んでいないことを確認する。
    assert min(float(source_metric["btr_mean_relative_level_db"]) for source_metric in source_metrics) >= -6.0


def test_integer_delay_diagnostics_handle_same_azimuth_with_multiple_frequencies():
    """同一方位に異なる周波数が重畳しても、各周波数別 BL の主ピークが同じ target 方位へ残ることを確認する。"""
    output_dir = Path.cwd() / "artifacts" / "beamforming" / "time_delay_same_azimuth_multi_frequency_diagnostics_test"
    summary = run_integer_delay_diagnostics(
        TimeDelayDiagnosticConfig(
            output_dir=output_dir,
            fs_hz=32768.0,
            duration_s=1.0,
            sound_speed_m_s=1500.0,
            noise_level_db20=-45.0,
            random_seed=1234,
            array_n_ch=61,
            array_sensor_spacing_m=0.05,
            sparse_stride_pattern=(1, 2, 1, 3, 1, 4, 2, 5),
            n_beam_az_real=241,
            btr_block_size=1024,
            source_specs=(
                TimeDelayDiagnosticSource(azimuth_deg=58.0, frequency_hz=1024.0, level_db20=0.0, label="F1"),
                TimeDelayDiagnosticSource(azimuth_deg=58.0, frequency_hz=1536.0, level_db20=0.0, label="F2"),
                TimeDelayDiagnosticSource(azimuth_deg=58.0, frequency_hz=2304.0, level_db20=0.0, label="F3"),
            ),
        )
    )

    summary_path = output_dir / "summary.json"
    fraz_path = output_dir / "fraz.png"
    btr_path = output_dir / "btr.png"

    assert summary_path.exists()
    assert fraz_path.exists()
    assert btr_path.exists()
    assert bool(summary["array_is_sparse"])
    assert int(summary["n_source"]) == 3

    source_metrics = summary["source_metrics"]
    assert isinstance(source_metrics, list)
    assert len(source_metrics) == 3

    # 同一方位に周波数だけ異なる source を重ねた条件では、各周波数別 BL のピーク方位が
    # 同じ target 方位へ揃うことが最重要である。周波数軸上では 3 本の ridge が分かれ、
    # BTR では 1 本の方位 ridge として見えることを期待する。
    for source_metric in source_metrics:
        assert Path(str(source_metric["bl_png_path"])).exists()
        assert abs(float(source_metric["source_azimuth_deg"]) - 58.0) <= 1e-9
        assert abs(float(source_metric["bl_peak_azimuth_deg"]) - 58.0) <= 3.0
        assert abs(float(source_metric["bl_level_at_nearest_source_grid_db20"]) - 0.0) <= 1.0
        assert float(source_metric["btr_mean_relative_level_db"]) >= -1.0

    # 同方位なので、周波数ごとに求めた peak azimuth は互いにほぼ一致しているべきである。
    peak_azimuths_deg = [float(source_metric["bl_peak_azimuth_deg"]) for source_metric in source_metrics]
    assert max(peak_azimuths_deg) - min(peak_azimuths_deg) <= 1.0
    assert abs(float(summary["btr_global_peak_azimuth_mean_deg"]) - 58.0) <= 3.0
