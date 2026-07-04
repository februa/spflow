"""整数遅延固定整相の BL/FRAZ/BTR を保存するサンプル。"""

# 非均一帯域フィルタバンク検証で代表条件になった `1536 Hz / 20 deg` を、
# 時間領域固定整相の段階でも同じアレイ条件で再確認できるようにする。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming.time_delay_diagnostics import (
    TimeDelayDiagnosticConfig,
    run_integer_delay_diagnostics,
)


def build_parser() -> argparse.ArgumentParser:
    """CLI 引数定義を返す。"""
    parser = argparse.ArgumentParser(description="整数遅延固定整相の BL/FRAZ/BTR を保存する。")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "beamforming" / "time_delay_integer_diagnostics",
        help="PNG と summary.json の保存先。",
    )
    parser.add_argument("--fs-hz", type=float, default=32768.0)
    parser.add_argument("--duration-s", type=float, default=1.0)
    parser.add_argument("--sound-speed-m-s", type=float, default=1500.0)
    parser.add_argument("--source-frequency-hz", type=float, default=1536.0)
    parser.add_argument("--source-level-db20", type=float, default=0.0)
    parser.add_argument("--source-azimuth-deg", type=float, default=20.0)
    parser.add_argument("--source-elevation-deg", type=float, default=0.0)
    parser.add_argument("--source-phase-deg", type=float, default=0.0)
    parser.add_argument("--noise-level-db20", type=float, default=-40.0)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--array-n-ch", type=int, default=160)
    parser.add_argument("--array-sensor-spacing-m", type=float, default=0.05)
    parser.add_argument("--az-min-deg", type=float, default=0.0)
    parser.add_argument("--az-max-deg", type=float, default=180.0)
    parser.add_argument("--n-beam-az-real", type=int, default=241)
    parser.add_argument("--n-beam-az-virtual", type=int, default=0)
    parser.add_argument("--display-elevation-deg", type=float, default=0.0)
    parser.add_argument("--btr-block-size", type=int, default=1024)
    return parser


def main() -> None:
    """診断を実行し、画像と summary を保存する。"""
    args = build_parser().parse_args()
    summary = run_integer_delay_diagnostics(
        TimeDelayDiagnosticConfig(
            output_dir=Path(args.output_dir),
            fs_hz=float(args.fs_hz),
            duration_s=float(args.duration_s),
            sound_speed_m_s=float(args.sound_speed_m_s),
            source_frequency_hz=float(args.source_frequency_hz),
            source_level_db20=float(args.source_level_db20),
            source_azimuth_deg=float(args.source_azimuth_deg),
            source_elevation_deg=float(args.source_elevation_deg),
            source_phase_deg=float(args.source_phase_deg),
            noise_level_db20=float(args.noise_level_db20),
            random_seed=int(args.random_seed),
            array_n_ch=int(args.array_n_ch),
            array_sensor_spacing_m=float(args.array_sensor_spacing_m),
            az_min_deg=float(args.az_min_deg),
            az_max_deg=float(args.az_max_deg),
            n_beam_az_real=int(args.n_beam_az_real),
            n_beam_az_virtual=int(args.n_beam_az_virtual),
            display_elevation_deg=float(args.display_elevation_deg),
            btr_block_size=int(args.btr_block_size),
        )
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
