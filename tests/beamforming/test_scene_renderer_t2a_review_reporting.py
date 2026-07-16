"""T2a評価結果とreview pack直列化の責務境界を検証する。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluations.beamforming.scene_renderer_t2a_review_reporting import (
    ScenarioSummaryRow,
    T2aReviewContext,
    T2aReviewData,
    write_t2a_review_pack,
)
from evaluations.beamforming.scene_renderer_t2a_streaming import T2aScenarioConfig
from evaluations.beamforming.scene_renderer_t2a_waveform_reporting import (
    WaveformIntegrityResult,
)


def test_review_reporting_writes_completed_arrays_without_signal_processing(tmp_path: Path) -> None:
    """固定型の完成結果だけからCSV、NPZ、JSON、Markdown、PNGを生成する。

    2 beam、5 rFFT bin、32 sampleの小さい決定論配列を使う。reporting側が重み設計や
    streaming処理を必要とせず、NPZへPNG再描画元の軸、FRAZ、source-frequency BL、
    block境界参照波形を保存する責務境界を固定する。
    """
    method_id = "fixed_baseline"
    n_beam = 2
    n_frequency = 5
    n_sample = 32
    config = T2aScenarioConfig(
        fs_hz=64.0,
        duration_s=2.0,
        # 32 sampleの診断信号内に、training除外後も16 sampleのspectrum区間を残す。
        training_duration_s=0.25,
        target_azimuth_deg=45.0,
        target_frequency_hz=8.0,
        interferer_azimuth_deg=90.0,
        interferer_frequency_hz=16.0,
        noise_band_hz=(2.0, 24.0),
        analysis_fft_size=8,
        analysis_hop_size=8,
        residual_fir_tap_count=4,
        runtime_block_size=7,
    )
    frequency_hz = np.fft.rfftfreq(config.analysis_fft_size, d=1.0 / config.fs_hz)
    beam_azimuth_deg = np.array([45.0, 90.0], dtype=np.float64)
    fraz_level = np.array(
        [[-40.0, 0.0, -20.0, -50.0, -60.0], [-45.0, -15.0, 3.0, -48.0, -62.0]],
        dtype=np.float64,
    )
    fraz_by_component = {
        component_id: {method_id: fraz_level.copy()}
        for component_id in ("target", "interferer", "noise", "mixed")
    }
    time_sample = np.arange(n_sample, dtype=np.float64)
    reference = np.cos(2.0 * np.pi * config.target_frequency_hz * time_sample / config.fs_hz)
    beam_output = np.vstack((reference, 0.5 * reference)).astype(np.complex128)
    valid = np.ones((n_beam, n_sample), dtype=np.bool_)
    integrity = WaveformIntegrityResult(
        analysis_start_sample=0,
        analysis_stop_sample=n_sample,
        phase_delay_samples_modulo_period=0.0,
        rms_delta_db=0.0,
        correlation_after_phase_alignment=1.0,
        residual_rms_db_re_input_rms=-300.0,
        reference_signal=reference,
        phase_aligned_output=reference.copy(),
    )
    summary_row = ScenarioSummaryRow(
        scenario="reporting_boundary_test",
        method=method_id,
        evaluation_pattern="fixed_beam_multi_source",
        target_frequency_hz=8.0,
        target_azimuth_deg=45.0,
        target_peak_azimuth_deg=45.0,
        target_peak_error_deg=0.0,
        target_level_db_re_input_rms=0.0,
        sidelobe_peak_db_re_mainlobe_peak=-15.0,
        output_snr_db=20.0,
        interferer_level_at_target_beam_db_re_input_rms=-20.0,
        minimum_fir_energy_containment=0.99,
        target_waveform_rms_delta_db=0.0,
        target_waveform_correlation_after_phase_alignment=1.0,
        target_waveform_residual_rms_db_re_input_rms=-300.0,
        target_phase_delay_samples_modulo_period=0.0,
        streaming_one_block_max_abs_error=0.0,
        streaming_boundary_max_abs_error=0.0,
        streaming_valid_mask_matches_one_block=True,
        ebae_signal_count_at_target=-1,
        ebae_music_peak_azimuth_deg_at_target=float("nan"),
        ebae_fallback_at_target=False,
        runtime_factor=0.1,
        finite=True,
    )
    review_data = T2aReviewData(
        frequency_hz=np.asarray(frequency_hz, dtype=np.float64),
        beam_azimuth_deg=beam_azimuth_deg,
        fraz_by_component=fraz_by_component,
        valid_sample_counts={f"mixed_{method_id}": n_sample},
        streamed_waveforms={
            "target": {method_id: (beam_output, valid)},
            "mixed": {method_id: (beam_output, valid)},
        },
        one_block_mixed={method_id: (beam_output.copy(), valid.copy())},
        waveform_integrity_by_method={method_id: integrity},
        streaming_overall_error_by_method={method_id: 0.0},
        streaming_boundary_error_by_method={method_id: 0.0},
        streaming_valid_match_by_method={method_id: True},
        diagnostic_zoom_by_method={method_id: (0, 0, 16)},
        source_frequency_bl_by_method={method_id: np.max(fraz_level[:, 1:3], axis=1)},
        summary_rows=(summary_row,),
        runtime_s=0.01,
        runtime_factor=0.1,
        target_frequency_index=1,
        interferer_frequency_index=2,
        target_beam_index=0,
        reference_channel_index=0,
    )
    context = T2aReviewContext(
        scenario=config,
        scenario_metadata=dict(config.__dict__),
        selected_method_ids=(method_id,),
        review_title="reporting boundary test",
        positions_path=Path("COE_POS"),
        shading_path=Path("COE_CBFSHADING"),
        shading_frequency_step_hz=8.0,
        n_channel=1,
        predicted_aliases_deg={"target": (), "interferer": ()},
        rendered_mixed=reference[np.newaxis, :],
        active_channel_count=np.ones(n_frequency, dtype=np.float64),
        causal_delays_samples=np.zeros((n_beam, 1), dtype=np.int64),
        ebae_signal_count=np.zeros((n_frequency, n_beam), dtype=np.int64),
        ebae_music_peak_azimuth_deg=np.full((n_frequency, n_beam), np.nan),
        ebae_fallback_mask=np.zeros((n_frequency, n_beam), dtype=np.bool_),
        covariance_snapshot_count_by_beam=np.full(n_beam, 3, dtype=np.int64),
    )

    write_t2a_review_pack(tmp_path, context, review_data)

    expected_artifacts = {
        "scenario_summary.csv",
        "worst_cases.csv",
        "plot_arrays.npz",
        "metadata.json",
        "review_index.md",
        "rendered_input_spectrum.png",
        "input_waveform_diagnostics.png",
        f"output_waveform_diagnostics_{method_id}.png",
        f"target_waveform_integrity_{method_id}.png",
        "bl_fraz_fl.png",
        "source_frequency_bl_overlay.png",
    }
    assert expected_artifacts <= {path.name for path in tmp_path.iterdir()}
    with np.load(tmp_path / "plot_arrays.npz", allow_pickle=False) as plot_arrays:
        np.testing.assert_array_equal(plot_arrays["azimuth_deg"], beam_azimuth_deg)
        np.testing.assert_array_equal(
            plot_arrays["covariance_snapshot_count_by_beam"], np.full(n_beam, 3)
        )
        np.testing.assert_array_equal(
            plot_arrays[f"mixed_{method_id}_fraz_db_re_input_rms"], fraz_level
        )
        np.testing.assert_array_equal(
            plot_arrays[f"{method_id}_source_frequency_bl_db_re_input_rms"],
            review_data.source_frequency_bl_by_method[method_id],
        )
