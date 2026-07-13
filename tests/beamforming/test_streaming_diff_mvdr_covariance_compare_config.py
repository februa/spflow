"""streaming 差分 MVDR 評価スクリプトの外部 config 入力を確認する。"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from evaluations.beamforming import (
    evaluate_streaming_diff_mvdr_covariance_compare as streaming_eval,
)


def _small_config_payload(output_dir: Path) -> dict[str, object]:
    """CI で短時間に実行できる評価 config を返す。

    64 point rFFT、0.125 秒、4 channel、7 beam に縮小するが、通常共分散と
    beam 方向合算共分散の両方を通すため、外部 JSON 受付と出力生成の回帰確認に使える。
    """

    return {
        "fs_hz": 4096.0,
        "sound_speed_m_s": 1500.0,
        "n_ch": 4,
        "spacing_m": 0.05,
        "fft_size": 64,
        "duration_sec": 0.125,
        "frame_size": 256,
        "source_rms": 1.0,
        "diagonal_loading_ratio": 1.0e-2,
        "covariance_time_constant_sec": 1.0e6,
        "n_beam": 7,
        "azimuth_min_deg": 0.0,
        "azimuth_max_deg": 180.0,
        "output_dir": str(output_dir),
        "scenarios": [
            {
                "scenario_id": "small_tone_az090",
                "title": "Small tone source for config smoke test",
                "sources": [
                    {
                        "name": "tone_512",
                        "azimuth_deg": 90.0,
                        "kind": "tone",
                        "frequency_hz": 512.0,
                        "rms": 1.0,
                    }
                ],
            }
        ],
    }


def test_streaming_diff_mvdr_evaluation_accepts_json_config(tmp_path: Path) -> None:
    """外部 JSON config から軽量評価を実行し、RMS 正規化と成果物を確認する。"""

    default_config = streaming_eval.load_evaluation_config(None)
    config_path = tmp_path / "streaming_eval_config.json"
    output_dir = tmp_path / "streaming_eval_output"
    config_path.write_text(
        json.dumps(_small_config_payload(output_dir), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    try:
        config = streaming_eval.load_evaluation_config(config_path)
        zip_path = streaming_eval.build_report(config)
    finally:
        # build_report は既存 script との互換性のため module-level 定数へ反映する。
        # 他テストへ縮小設定を漏らさないよう、既定値へ戻しておく。
        streaming_eval._apply_config(default_config)

    assert zip_path.exists()
    assert (output_dir / "figures" / "small_tone_az090_beam_response.png").exists()
    assert (output_dir / "figures" / "small_tone_az090_spectrum.png").exists()
    assert (output_dir / "data" / "math_check.md").exists()

    with (output_dir / "data" / "summary_metrics.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    # one-sided rFFT bin power を帯域内で加算した RMS が、入力 source RMS と一致する。
    assert rows
    assert float(rows[0]["input_total_rms_from_bins"]) == pytest.approx(1.0, abs=1.0e-12)
    assert float(rows[0]["input_total_rms_expected"]) == pytest.approx(1.0, abs=1.0e-12)
