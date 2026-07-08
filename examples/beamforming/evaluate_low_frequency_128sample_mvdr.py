"""低周波・128 sample 共分散での MVDR 安定性レポートを作る。"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scene_renderer_mvdr_stability_sweep import evaluate_frequency  # noqa: E402
from spflow.beamforming.diagnostic_plotting import require_matplotlib  # noqa: E402

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 実行環境に依存する。
    plt = None


FloatArray: TypeAlias = NDArray[np.float64]

FS_HZ = 32768.0
SOUND_SPEED_M_S = 1500.0
N_CH = 32
SPACING_M = 0.05
ACTIVE_APERTURE_M = SPACING_M * float(N_CH - 1)
TARGET_AZIMUTH_DEG = 20.0
INTERFERER_AZIMUTH_DEG = -30.0
SIGNAL_LEVEL_DB20 = 0.0
INTERFERER_LEVEL_DB20 = 0.0
FFT_SIZE = 128
N_SAMPLE = 128
INTEGRATION_TIME_S = float(N_SAMPLE) / FS_HZ
DIAGONAL_LOADING_RATIO = 1.0e-3
FREQUENCIES_HZ = (256.0, 512.0, 1024.0, 2048.0, 4096.0, 8960.0)
OUTPUT_DIR = ROOT / "artifacts" / "beamforming" / "fixed_delay_diff_mvdr" / "low_frequency_128sample_mvdr"
FIGURE_DIR = OUTPUT_DIR / "figures"
DATA_DIR = OUTPUT_DIR / "data"
LEVEL_UNIT_LABEL = "dB re input RMS"


@dataclass(frozen=True)
class FrequencyRow:
    """1 周波数・1 共分散条件の評価行を保持する。

    このクラスは、scene_renderer で生成した target + interferer 入力に対し、
    128 sample 共分散から設計した MVDR の応答指標を CSV/PNG へ渡す中間表現である。

    信号生成、MVDR 重み設計、図の描画は責務に含めない。
    信号処理上は、短時間共分散で低周波 MVDR が分離不能になるかを読むための
    scenario-by-method 行である。
    """

    covariance_source: str
    frequency_hz: float
    wavelength_m: float
    aperture_wavelength: float
    adjacent_phase_deg: float
    aperture_phase_deg: float
    cbf_interferer_db: float
    mvdr_interferer_db: float
    interferer_reduction_db: float
    cbf_rms_err: float
    mvdr_rms_err: float
    mvdr_improves_target_err: bool
    active_alias_limit_hz: float


def _plt():
    """matplotlib.pyplot を遅延取得する。"""

    if plt is None:
        raise RuntimeError("matplotlib is required to plot figures.")
    return plt


def _evaluate_rows() -> list[FrequencyRow]:
    """低周波 sweep を実行して評価行を返す。

    Returns:
        評価行の list。各行は 1 周波数・1 共分散条件を表す。

    Notes:
        `mixture` は target と interferer を含む 128 sample だけから共分散を作るため、
        実運用で target が統計に混入する条件を表す。`interferer-only` は理想参照であり、
        低周波でも MVDR が動ける上限性能として併記する。
    """

    rows: list[FrequencyRow] = []
    for covariance_source in ("mixture", "interferer-only"):
        for frequency_hz in FREQUENCIES_HZ:
            raw_row = evaluate_frequency(
                fs=FS_HZ,
                fft_size=FFT_SIZE,
                freq=float(frequency_hz),
                n_samples=N_SAMPLE,
                n_ch=N_CH,
                spacing_m=SPACING_M,
                sound_speed=SOUND_SPEED_M_S,
                target_deg=TARGET_AZIMUTH_DEG,
                signal_level_db20=SIGNAL_LEVEL_DB20,
                integration_time=INTEGRATION_TIME_S,
                diag_load=DIAGONAL_LOADING_RATIO,
                interferer_deg=INTERFERER_AZIMUTH_DEG,
                interferer_level_db20=INTERFERER_LEVEL_DB20,
                covariance_source=covariance_source,
                selector_mode="full",
                aperture_wavelengths=4.0,
                min_active_ch=4,
                dense_spacing_m=None,
                n_dense_ch=None,
            )
            if raw_row["status"] != "ok":
                continue

            wavelength_m = SOUND_SPEED_M_S / float(frequency_hz)
            # ULA 隣接 CH の位相差は 2π f d cos(theta) / c。
            # 低周波ではこの値が小さく、CH 間 steering が似るため空間分離が難しくなる。
            adjacent_phase_deg = float(
                np.rad2deg(
                    2.0
                    * np.pi
                    * float(frequency_hz)
                    * SPACING_M
                    * np.cos(np.deg2rad(TARGET_AZIMUTH_DEG))
                    / SOUND_SPEED_M_S
                )
            )
            aperture_phase_deg = adjacent_phase_deg * float(N_CH - 1)
            cbf_interferer_db = float(raw_row["cbf_interferer_db"])
            mvdr_interferer_db = float(raw_row["mvdr_interferer_db"])
            rows.append(
                FrequencyRow(
                    covariance_source=covariance_source,
                    frequency_hz=float(frequency_hz),
                    wavelength_m=float(wavelength_m),
                    aperture_wavelength=float(ACTIVE_APERTURE_M / wavelength_m),
                    adjacent_phase_deg=adjacent_phase_deg,
                    aperture_phase_deg=aperture_phase_deg,
                    cbf_interferer_db=cbf_interferer_db,
                    mvdr_interferer_db=mvdr_interferer_db,
                    interferer_reduction_db=float(cbf_interferer_db - mvdr_interferer_db),
                    cbf_rms_err=float(raw_row["cbf_rms_err"]),
                    mvdr_rms_err=float(raw_row["mvdr_rms_err"]),
                    mvdr_improves_target_err=bool(raw_row["mvdr_improves_target_err"]),
                    active_alias_limit_hz=float(raw_row["active_alias_limit_hz"]),
                )
            )
    return rows


def _rows_for_source(rows: list[FrequencyRow], covariance_source: str) -> list[FrequencyRow]:
    """指定した共分散条件の行だけを周波数昇順で返す。"""

    return sorted(
        [row for row in rows if row.covariance_source == covariance_source],
        key=lambda row: row.frequency_hz,
    )


def _row_array(rows: list[FrequencyRow], field_name: str) -> FloatArray:
    """FrequencyRow の float field を NumPy 配列へ変換する。"""

    return np.asarray([float(getattr(row, field_name)) for row in rows], dtype=np.float64)


def _plot_interferer_response(rows: list[FrequencyRow], output_path: Path) -> None:
    """干渉方向応答の周波数依存を保存する。"""

    mixture_rows = _rows_for_source(rows, "mixture")
    oracle_rows = _rows_for_source(rows, "interferer-only")
    frequency_hz = _row_array(mixture_rows, "frequency_hz")

    fig, axis = _plt().subplots(figsize=(10.8, 5.3))
    axis.semilogx(frequency_hz, _row_array(mixture_rows, "cbf_interferer_db"), marker="o", color="black", label="fixed_baseline")
    axis.semilogx(frequency_hz, _row_array(mixture_rows, "mvdr_interferer_db"), marker="o", color="tab:orange", label="MVDR from mixture covariance")
    axis.semilogx(frequency_hz, _row_array(oracle_rows, "mvdr_interferer_db"), marker="o", color="tab:blue", label="MVDR from interferer-only covariance")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel(f"Interferer response [{LEVEL_UNIT_LABEL}]")
    axis.set_title("128-sample covariance: low-frequency MVDR interferer response")
    axis.text(
        0.02,
        0.05,
        "mixture covariance uses only 128 samples containing target + interferer.\n"
        "interferer-only is an oracle reference, not the operational condition.",
        transform=axis.transAxes,
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.92},
    )
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _plot_physical_scale(rows: list[FrequencyRow], output_path: Path) -> None:
    """波長と開口位相差を可視化する。"""

    mixture_rows = _rows_for_source(rows, "mixture")
    frequency_hz = _row_array(mixture_rows, "frequency_hz")

    fig, axis_left = _plt().subplots(figsize=(10.8, 5.3))
    axis_right = axis_left.twinx()
    line_left = axis_left.semilogx(frequency_hz, _row_array(mixture_rows, "aperture_wavelength"), marker="o", color="tab:green", label="aperture / wavelength")
    line_right = axis_right.semilogx(frequency_hz, _row_array(mixture_rows, "aperture_phase_deg"), marker="o", color="tab:red", label="phase span across aperture")
    axis_left.axhline(1.0, color="0.4", linestyle=":", linewidth=1.0)
    axis_left.set_xlabel("Frequency [Hz]")
    axis_left.set_ylabel("Active aperture / wavelength [ratio]")
    axis_right.set_ylabel("Target steering phase span [deg]")
    axis_left.set_title("Low-frequency spatial aperture with 32ch, 0.05 m spacing")
    lines = line_left + line_right
    legend_labels: list[str] = [str(line.get_label()) for line in lines]
    axis_left.legend(lines, legend_labels, loc="best")
    axis_left.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _plot_error_response(rows: list[FrequencyRow], output_path: Path) -> None:
    """target 波形誤差の周波数依存を保存する。"""

    mixture_rows = _rows_for_source(rows, "mixture")
    oracle_rows = _rows_for_source(rows, "interferer-only")
    frequency_hz = _row_array(mixture_rows, "frequency_hz")

    fig, axis = _plt().subplots(figsize=(10.8, 5.3))
    axis.loglog(frequency_hz, _row_array(mixture_rows, "cbf_rms_err"), marker="o", color="black", label="fixed_baseline")
    axis.loglog(frequency_hz, _row_array(mixture_rows, "mvdr_rms_err"), marker="o", color="tab:orange", label="MVDR from mixture covariance")
    axis.loglog(frequency_hz, _row_array(oracle_rows, "mvdr_rms_err"), marker="o", color="tab:blue", label="MVDR from interferer-only covariance")
    axis.set_xlabel("Frequency [Hz]")
    axis.set_ylabel("RMS error to target-only reference [linear]")
    axis.set_title("128-sample covariance: target waveform error")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    _plt().close(fig)


def _write_csv(rows: list[FrequencyRow], output_path: Path) -> None:
    """scenario_summary.csv を保存する。"""

    fieldnames = list(FrequencyRow.__dataclass_fields__.keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field_name: getattr(row, field_name) for field_name in fieldnames})


def _write_npz(rows: list[FrequencyRow], output_path: Path) -> None:
    """PNG 作成元配列を NPZ に保存する。"""

    mixture_rows = _rows_for_source(rows, "mixture")
    oracle_rows = _rows_for_source(rows, "interferer-only")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        frequency_hz=_row_array(mixture_rows, "frequency_hz"),
        mixture_cbf_interferer_db=_row_array(mixture_rows, "cbf_interferer_db"),
        mixture_mvdr_interferer_db=_row_array(mixture_rows, "mvdr_interferer_db"),
        oracle_mvdr_interferer_db=_row_array(oracle_rows, "mvdr_interferer_db"),
        mixture_cbf_rms_err=_row_array(mixture_rows, "cbf_rms_err"),
        mixture_mvdr_rms_err=_row_array(mixture_rows, "mvdr_rms_err"),
        oracle_mvdr_rms_err=_row_array(oracle_rows, "mvdr_rms_err"),
        aperture_wavelength=_row_array(mixture_rows, "aperture_wavelength"),
        adjacent_phase_deg=_row_array(mixture_rows, "adjacent_phase_deg"),
        aperture_phase_deg=_row_array(mixture_rows, "aperture_phase_deg"),
    )


def _write_metadata(output_dir: Path) -> None:
    """評価条件と配列 shape を metadata.json に保存する。"""

    metadata = {
        "scenario_id": "low_frequency_128sample_mvdr",
        "evaluation_pattern": "fixed_beam_multi_source",
        "fs_hz": FS_HZ,
        "sound_speed_m_s": SOUND_SPEED_M_S,
        "n_ch": N_CH,
        "spacing_m": SPACING_M,
        "active_aperture_m": ACTIVE_APERTURE_M,
        "target_azimuth_deg": TARGET_AZIMUTH_DEG,
        "interferer_azimuth_deg": INTERFERER_AZIMUTH_DEG,
        "fft_size": FFT_SIZE,
        "n_sample": N_SAMPLE,
        "integration_time_s": INTEGRATION_TIME_S,
        "diagonal_loading_ratio": DIAGONAL_LOADING_RATIO,
        "frequencies_hz": list(FREQUENCIES_HZ),
        "level_reference": LEVEL_UNIT_LABEL,
        "array_shapes": {
            "frequency_hz": "[n_freq]",
            "*_interferer_db": "[n_freq]",
            "*_rms_err": "[n_freq]",
            "aperture_wavelength": "[n_freq]",
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_review_index(rows: list[FrequencyRow], output_dir: Path) -> None:
    """AI 向け review index を保存する。"""

    mixture_rows = _rows_for_source(rows, "mixture")
    low_row = mixture_rows[0]
    lines = [
        "# 低周波・128 sample 共分散 MVDR 評価",
        "",
        "## Scenario",
        "",
        "- target: 20 deg, 0 dB re input RMS",
        "- interferer: -30 deg, 0 dB re input RMS, target と同一周波数",
        f"- covariance block: `{N_SAMPLE}` sample = `{INTEGRATION_TIME_S:.9f}` s",
        f"- array: {N_CH} ch ULA, spacing {SPACING_M:.3f} m, aperture {ACTIVE_APERTURE_M:.3f} m",
        "- covariance_source `mixture`: target + interferer を含む実運用寄り条件。",
        "- covariance_source `interferer-only`: 理想参照。実運用の採否判断には直接使わない。",
        "",
        "## Artifacts",
        "",
        "- `figures/interferer_response_vs_frequency.png`: 干渉方向応答。mixture 共分散で抑圧が出ないことを見る主図。",
        "- `figures/physical_scale_vs_frequency.png`: 波長に対する開口長と target steering 位相幅。",
        "- `figures/target_error_vs_frequency.png`: target-only 参照に対する出力 RMS error。",
        "- `data/low_frequency_128sample_mvdr_arrays.npz`: 図作成元配列。",
        "- `scenario_summary.csv`: 周波数・共分散条件別 metric。",
        "- `metadata.json`: 評価条件、単位、shape。",
        "",
        "## Interpretation Notes",
        "",
        f"- {low_row.frequency_hz:.0f} Hz では波長 {low_row.wavelength_m:.3f} m に対して開口は {low_row.aperture_wavelength:.3f} λ、隣接 CH 位相差は {low_row.adjacent_phase_deg:.3f} deg。",
        "- `mixture` 共分散では target も統計に含まれるため、128 sample だけでは干渉方向だけを安定に学習できない。",
        "- `interferer-only` が大きく抑圧できる場合でも、それは理想参照がある条件であり、運用時に同じ性能を保証しない。",
    ]
    (output_dir / "review_index.md").write_text("\n".join(lines), encoding="utf-8")


def _zip_package(output_dir: Path) -> Path:
    """出力ディレクトリを zip 化する。"""

    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as package:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                package.write(path, path.relative_to(output_dir.parent))
    return zip_path


def build_report_package() -> Path:
    """評価を実行し、人間向け PNG と AI 向け report package を保存する。"""

    require_matplotlib()
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = _evaluate_rows()
    _plot_interferer_response(rows, FIGURE_DIR / "interferer_response_vs_frequency.png")
    _plot_physical_scale(rows, FIGURE_DIR / "physical_scale_vs_frequency.png")
    _plot_error_response(rows, FIGURE_DIR / "target_error_vs_frequency.png")
    _write_npz(rows, DATA_DIR / "low_frequency_128sample_mvdr_arrays.npz")
    _write_csv(rows, OUTPUT_DIR / "scenario_summary.csv")
    _write_metadata(OUTPUT_DIR)
    _write_review_index(rows, OUTPUT_DIR)
    return _zip_package(OUTPUT_DIR)


def main() -> None:
    """CLI entrypoint。"""

    zip_path = build_report_package()
    print(json.dumps({"output_dir": str(OUTPUT_DIR), "zip_path": str(zip_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

