"""低周波endfireで3整相方式のBL・FRAZ・スペクトルを生成する。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from spflow.beamforming import (  # noqa: E402
    DelayTable,
    design_fixed_delay_fractional_weights_from_delay_table,
    design_standard_fractional_delay_filter_bank,
)
from spflow.beamforming.diagnostic_plotting import require_matplotlib  # noqa: E402

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "beamforming"
    / "direction_cut_mvdr_method_comparison"
    / "review_pack"
)
SCENARIO_ID = "low_broadband_endfire_0deg_interferer_60deg"
FS_HZ = 32768.0
SOUND_SPEED_M_S = 1500.0
N_CHANNEL = 64
SPACING_M = 6.25
ANALYSIS_WIDTH_HZ = 16.0
TARGET_AZIMUTH_DEG = 0.0
INTERFERER_AZIMUTH_DEG = 60.0
TARGET_BAND_HZ = (40.0, 120.0)
NOISE_POWER_PER_BIN_RE_SOURCE_RMS = 1.0e-3
DIAGONAL_LOADING_RATIO = 1.0e-3
AZIMUTH_DEG = np.linspace(0.0, 180.0, 37, dtype=np.float64)
FREQUENCY_HZ = np.arange(16.0, 256.0 + 16.0, 16.0, dtype=np.float64)
METHOD_IDS = (
    "fixed_integer_fractional",
    "S1",
    "S2a",
    "T1",
    "T2a",
)


def _positions_m() -> NDArray[np.float64]:
    """ULA位置`[n_ch,3]`をm単位で返す。"""

    aperture_m = SPACING_M * (N_CHANNEL - 1)
    x_m = np.linspace(-aperture_m / 2.0, aperture_m / 2.0, N_CHANNEL)
    return np.stack((x_m, np.zeros_like(x_m), np.zeros_like(x_m)), axis=1)


def _directions(azimuth_deg: NDArray[np.float64]) -> NDArray[np.float64]:
    """方位`[n_direction]`を方向余弦`[n_direction,3]`へ変換する。"""

    rad = np.deg2rad(azimuth_deg)
    return np.stack((np.cos(rad), np.sin(rad), np.zeros_like(rad)), axis=1)


def _arrival_delay_s(
    positions_m: NDArray[np.float64], azimuth_deg: NDArray[np.float64]
) -> NDArray[np.float64]:
    """物理到来遅延`[n_direction,n_ch]`をs単位で返す。"""

    # DelayTable.from_geometryと同じく、到来遅延は-(r·u)/cとする。
    # この符号を反転すると、固定整相baselineのbeamが鏡像方位へ向く。
    return np.asarray(-_directions(azimuth_deg) @ positions_m.T / SOUND_SPEED_M_S)


def _steering(
    delays_s: NDArray[np.float64], frequencies_hz: NDArray[np.float64]
) -> NDArray[np.complex128]:
    """steering`[n_direction,n_frequency,n_ch]`を返す。"""

    # a(theta,f)=exp(-j2πfτ)。axis=0は方位、axis=1は周波数、axis=2はchannel。
    return np.asarray(
        np.exp(
            -1j
            * 2.0
            * np.pi
            * frequencies_hz[np.newaxis, :, np.newaxis]
            * delays_s[:, np.newaxis, :]
        ),
        dtype=np.complex128,
    )


def _mvdr_weight(
    covariance: NDArray[np.complex128], constraint: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """対角loading付きMVDR重み`[n_ch]`を返す。"""

    hermitian = 0.5 * (covariance + covariance.conj().T)
    loading = DIAGONAL_LOADING_RATIO * float(np.real(np.trace(hermitian))) / N_CHANNEL
    # source数が少ない条件でも共分散を可逆に保ち、weight発散を防ぐ。
    loaded = hermitian + loading * np.eye(N_CHANNEL, dtype=np.complex128)
    solved = np.linalg.solve(loaded, constraint)
    return np.asarray(solved / np.vdot(constraint, solved), dtype=np.complex128)


def _direction_cut_covariance(
    candidate_delay_s: NDArray[np.float64],
    source_delays_s: tuple[NDArray[np.float64], ...],
    source_steerings: tuple[NDArray[np.complex128], ...],
    source_powers: tuple[float, ...],
) -> NDArray[np.complex128]:
    """候補方位切り出し後の帯域内共分散`[n_ch,n_ch]`を返す。"""

    quantized_candidate_s = np.rint(candidate_delay_s * FS_HZ) / FS_HZ
    covariance = NOISE_POWER_PER_BIN_RE_SOURCE_RMS * np.eye(N_CHANNEL, dtype=np.complex128)
    for source_delay_s, source_steering, source_power in zip(
        source_delays_s, source_steerings, source_powers, strict=True
    ):
        residual_s = source_delay_s - quantized_candidate_s
        # 粗い1 bin内の周波数積分は、pair間の残留遅延に
        # sinc(ΔfΔτ)のcoherence低下を与える。
        pair_residual_s = residual_s[:, np.newaxis] - residual_s[np.newaxis, :]
        coherence = np.sinc(ANALYSIS_WIDTH_HZ * pair_residual_s)
        outer = source_steering[:, np.newaxis] * source_steering.conj()[np.newaxis, :]
        covariance += float(source_power) * coherence * outer
    return np.asarray(covariance, dtype=np.complex128)


def _coarse_same_time_covariance(
    source_delays_s: tuple[NDArray[np.float64], ...],
    source_steerings: tuple[NDArray[np.complex128], ...],
    source_powers: tuple[float, ...],
) -> NDArray[np.complex128]:
    """同一時間blockの粗い分析幅共分散`[n_ch,n_ch]`を返す。"""

    covariance = NOISE_POWER_PER_BIN_RE_SOURCE_RMS * np.eye(N_CHANNEL, dtype=np.complex128)
    for source_delay_s, source_steering, source_power in zip(
        source_delays_s, source_steerings, source_powers, strict=True
    ):
        # 整数遅延も方位別切り出しも行わないため、開口全体の
        # 物理遅延が1 bin内のcoherence低下にそのまま寄与する。
        pair_delay_s = source_delay_s[:, np.newaxis] - source_delay_s[np.newaxis, :]
        coherence = np.sinc(ANALYSIS_WIDTH_HZ * pair_delay_s)
        outer = source_steering[:, np.newaxis] * source_steering.conj()[np.newaxis, :]
        covariance += float(source_power) * coherence * outer
    return np.asarray(covariance, dtype=np.complex128)


def calculate_review_arrays() -> dict[str, NDArray[Any]]:
    """3方式のBL、FRAZ、周波数スペクトル配列を返す。"""

    positions_m = _positions_m()
    beam_delay_s = _arrival_delay_s(positions_m, AZIMUTH_DEG)
    source_delay_s = _arrival_delay_s(
        positions_m,
        np.asarray([TARGET_AZIMUTH_DEG, INTERFERER_AZIMUTH_DEG], dtype=np.float64),
    )
    beam_steering = _steering(beam_delay_s, FREQUENCY_HZ)
    source_steering = _steering(source_delay_s, FREQUENCY_HZ)
    target_mask = (FREQUENCY_HZ >= TARGET_BAND_HZ[0]) & (FREQUENCY_HZ <= TARGET_BAND_HZ[1])
    n_source_bin = int(np.count_nonzero(target_mask))
    # targetとinterfererはそれぞれ帯域積分RMS=1とし、各binに等powerを割り当てる。
    source_power_per_bin = np.zeros(FREQUENCY_HZ.shape, dtype=np.float64)
    source_power_per_bin[target_mask] = 1.0 / float(n_source_bin)

    filter_bank = design_standard_fractional_delay_filter_bank()
    delay_table = DelayTable.from_geometry(
        positions_m,
        _directions(AZIMUTH_DEG),
        FS_HZ,
        SOUND_SPEED_M_S,
        filter_bank,
    )
    fixed_weights = design_fixed_delay_fractional_weights_from_delay_table(
        delay_table,
        filter_bank,
        FREQUENCY_HZ,
        fs_hz=FS_HZ,
    )
    # weight shapeは`[n_frequency,n_beam,n_ch]`。MVDRも同じaxis順へ揃える。
    coarse_direct_weights = np.empty_like(fixed_weights)
    integer_then_weights = np.empty_like(fixed_weights)
    direction_direct_weights = np.empty_like(fixed_weights)
    direction_integer_weights = np.empty_like(fixed_weights)
    for frequency_index, frequency_hz in enumerate(FREQUENCY_HZ.tolist()):
        for beam_index in range(AZIMUTH_DEG.size):
            source_steerings_at_frequency = (
                source_steering[0, frequency_index],
                source_steering[1, frequency_index],
            )
            source_powers_at_frequency = (
                float(source_power_per_bin[frequency_index]),
                float(source_power_per_bin[frequency_index]),
            )
            coarse_covariance = _coarse_same_time_covariance(
                (source_delay_s[0], source_delay_s[1]),
                source_steerings_at_frequency,
                source_powers_at_frequency,
            )
            direction_covariance = _direction_cut_covariance(
                beam_delay_s[beam_index],
                (source_delay_s[0], source_delay_s[1]),
                source_steerings_at_frequency,
                source_powers_at_frequency,
            )
            constraint = beam_steering[beam_index, frequency_index]
            coarse_direct_weights[frequency_index, beam_index] = _mvdr_weight(
                coarse_covariance, constraint
            )
            direction_direct_weights[frequency_index, beam_index] = _mvdr_weight(
                direction_covariance, constraint
            )
            # 整数遅延前段に合わせ、共分散・steering・入力の
            # channel位相を同じ係数で回転する。
            integer_phase = np.exp(
                -1j
                * 2.0
                * np.pi
                * float(frequency_hz)
                * delay_table.delay_int[:, beam_index]
                / FS_HZ
            )
            # S2aはS1と同じS共分散を整数遅延後座標へunitary変換する。
            # ここへ候補方位別T共分散を入れると、実現座標と共分散構成の2軸が混ざる。
            integer_aligned_coarse_covariance = coarse_covariance
            rotated_coarse_covariance = (
                integer_phase[:, np.newaxis]
                * integer_aligned_coarse_covariance
                * integer_phase.conj()[np.newaxis, :]
            )
            rotated_direction_covariance = (
                integer_phase[:, np.newaxis]
                * direction_covariance
                * integer_phase.conj()[np.newaxis, :]
            )
            delayed_coarse_weight = _mvdr_weight(
                np.asarray(rotated_coarse_covariance, dtype=np.complex128),
                np.asarray(integer_phase * constraint, dtype=np.complex128),
            )
            delayed_direction_weight = _mvdr_weight(
                np.asarray(rotated_direction_covariance, dtype=np.complex128),
                np.asarray(integer_phase * constraint, dtype=np.complex128),
            )
            # 実出力はw_d^H D xである。元入力xへ対する等価weightは
            # D^H w_dなので、BL/FRAZ計算用に元入力位相基準へ戻す。
            integer_then_weights[frequency_index, beam_index] = np.asarray(
                integer_phase.conj() * delayed_coarse_weight,
                dtype=np.complex128,
            )
            direction_integer_weights[frequency_index, beam_index] = np.asarray(
                integer_phase.conj() * delayed_direction_weight,
                dtype=np.complex128,
            )

    weights_by_method = {
        METHOD_IDS[0]: fixed_weights,
        METHOD_IDS[1]: coarse_direct_weights,
        METHOD_IDS[2]: integer_then_weights,
        METHOD_IDS[3]: direction_direct_weights,
        METHOD_IDS[4]: direction_integer_weights,
    }
    arrays: dict[str, NDArray[Any]] = {
        "azimuth_deg": AZIMUTH_DEG,
        "frequency_hz": FREQUENCY_HZ,
        "source_frequency_indices": np.flatnonzero(target_mask).astype(np.int32),
        "input_spectrum_db_re_input_rms": 10.0
        * np.log10(
            np.maximum(
                2.0 * source_power_per_bin + NOISE_POWER_PER_BIN_RE_SOURCE_RMS,
                1.0e-12,
            )
        ),
    }
    target_beam_index = int(np.argmin(np.abs(AZIMUTH_DEG - TARGET_AZIMUTH_DEG)))
    for method_id, weights in weights_by_method.items():
        fraz_power = np.empty((AZIMUTH_DEG.size, FREQUENCY_HZ.size), dtype=np.float64)
        target_power = np.empty_like(fraz_power)
        interferer_power = np.empty_like(fraz_power)
        for frequency_index in range(FREQUENCY_HZ.size):
            # response shapeは`[n_beam]`。beamごとのw^H aをchannel axis=1で縮約する。
            target_response = np.einsum(
                "bc,c->b",
                weights[frequency_index].conj(),
                source_steering[0, frequency_index],
                optimize=True,
            )
            interferer_response = np.einsum(
                "bc,c->b",
                weights[frequency_index].conj(),
                source_steering[1, frequency_index],
                optimize=True,
            )
            target_power[:, frequency_index] = (
                source_power_per_bin[frequency_index] * np.abs(target_response) ** 2
            )
            interferer_power[:, frequency_index] = (
                source_power_per_bin[frequency_index] * np.abs(interferer_response) ** 2
            )
            noise_power = NOISE_POWER_PER_BIN_RE_SOURCE_RMS * np.sum(
                np.abs(weights[frequency_index]) ** 2, axis=1
            )
            fraz_power[:, frequency_index] = (
                target_power[:, frequency_index]
                + interferer_power[:, frequency_index]
                + noise_power
            )
        arrays[f"{method_id}_fraz_level_db"] = 10.0 * np.log10(
            np.maximum(fraz_power, 1.0e-12)
        )
        arrays[f"{method_id}_target_bl_level_db"] = 10.0 * np.log10(
            np.maximum(np.sum(target_power[:, target_mask], axis=1), 1.0e-12)
        )
        arrays[f"{method_id}_mixed_bl_level_db"] = 10.0 * np.log10(
            np.maximum(np.sum(fraz_power[:, target_mask], axis=1), 1.0e-12)
        )
        arrays[f"{method_id}_output_spectrum_db"] = arrays[
            f"{method_id}_fraz_level_db"
        ][target_beam_index]
    return arrays


def _plot_review(arrays: dict[str, NDArray[Any]]) -> None:
    """review packのBL、FRAZ、スペクトル図を保存する。"""

    plt = require_matplotlib()
    figure_dir = OUTPUT_DIR / "figures" / SCENARIO_ID
    data_dir = OUTPUT_DIR / "data"
    figure_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    # np.savezの可変長keyword型stubが配列をallow_pickleと誤推論しないよう、
    # 成果物境界の辞書だけをAny値としてkeyを保持する。
    plot_arrays: dict[str, Any] = dict(arrays)
    np.savez(data_dir / f"{SCENARIO_ID}.npz", **plot_arrays)

    figure, axis = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    for method_id in METHOD_IDS:
        axis.plot(AZIMUTH_DEG, arrays[f"{method_id}_mixed_bl_level_db"], label=method_id)
    axis.axvline(TARGET_AZIMUTH_DEG, color="tab:green", linestyle="--", label="target")
    axis.axvline(INTERFERER_AZIMUTH_DEG, color="tab:red", linestyle=":", label="interferer")
    axis.set(
        title="Low-frequency endfire mixed BL",
        xlabel="Waiting-beam azimuth [deg]",
        ylabel="Band-integrated RMS level [dB re input source RMS]",
        xlim=(0.0, 180.0),
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(figure_dir / "bl_overlay.png", dpi=160)
    figure.savefig(figure_dir / "source_frequency_bl_overlay.png", dpi=160)
    plt.close(figure)

    finite_values = np.concatenate(
        [np.asarray(arrays[f"{method_id}_fraz_level_db"]).reshape(-1) for method_id in METHOD_IDS]
    )
    vmax = float(np.max(finite_values))
    vmin = vmax - 80.0
    figure, axes = plt.subplots(1, 5, figsize=(24.0, 5.0), constrained_layout=True, sharey=True)
    image: Any = None
    for axis, method_id in zip(axes, METHOD_IDS, strict=True):
        image = axis.pcolormesh(
            AZIMUTH_DEG,
            FREQUENCY_HZ,
            np.asarray(arrays[f"{method_id}_fraz_level_db"]).T,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        axis.set(title=method_id, xlabel="Waiting-beam azimuth [deg]")
    axes[0].set_ylabel("Frequency [Hz]")
    figure.colorbar(image, ax=axes, label="Per-bin RMS level [dB re input RMS]")
    figure.savefig(figure_dir / "fraz_panel.png", dpi=160)
    plt.close(figure)

    fixed_fraz = np.asarray(arrays[f"{METHOD_IDS[0]}_fraz_level_db"])
    for method_id in METHOD_IDS[1:]:
        delta = np.asarray(arrays[f"{method_id}_fraz_level_db"]) - fixed_fraz
        figure, axis = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
        image = axis.pcolormesh(
            AZIMUTH_DEG,
            FREQUENCY_HZ,
            delta.T,
            shading="auto",
            cmap="coolwarm",
            vmin=-30.0,
            vmax=30.0,
        )
        axis.set(
            title=f"{method_id} - fixed FRAZ",
            xlabel="Waiting-beam azimuth [deg]",
            ylabel="Frequency [Hz]",
        )
        figure.colorbar(image, ax=axis, label="Level difference [dB re fixed FRAZ level]")
        figure.savefig(figure_dir / f"fraz_delta_{method_id}_vs_fixed.png", dpi=160)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    axis.plot(FREQUENCY_HZ, arrays["input_spectrum_db_re_input_rms"], color="black")
    axis.set(
        title="Pre-beamforming signal+noise spectrum",
        xlabel="Frequency [Hz]",
        ylabel="Per-bin RMS level [dB re input RMS]",
    )
    axis.grid(True, alpha=0.25)
    figure.savefig(figure_dir / "input_spectrum.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    for method_id in METHOD_IDS:
        axis.plot(FREQUENCY_HZ, arrays[f"{method_id}_output_spectrum_db"], label=method_id)
    axis.set(
        title="Post-beamforming target-beam spectrum",
        xlabel="Frequency [Hz]",
        ylabel="Per-bin RMS level [dB re input RMS]",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(figure_dir / "output_spectrum_overlay.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    fixed_spectrum = np.asarray(arrays[f"{METHOD_IDS[0]}_output_spectrum_db"])
    for method_id in METHOD_IDS[1:]:
        axis.plot(
            FREQUENCY_HZ,
            np.asarray(arrays[f"{method_id}_output_spectrum_db"]) - fixed_spectrum,
            label=f"{method_id} - fixed",
        )
    axis.set(
        title="Post-beamforming spectrum difference",
        xlabel="Frequency [Hz]",
        ylabel="Level difference [dB re fixed output spectrum]",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(figure_dir / "output_spectrum_delta.png", dpi=160)
    plt.close(figure)


def main() -> None:
    """代表scenarioの3方式review pack画像と配列を生成する。"""

    arrays = calculate_review_arrays()
    _plot_review(arrays)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    target_beam_index = int(np.argmin(np.abs(AZIMUTH_DEG - TARGET_AZIMUTH_DEG)))
    interferer_beam_index = int(np.argmin(np.abs(AZIMUTH_DEG - INTERFERER_AZIMUTH_DEG)))
    for method_id in METHOD_IDS:
        mixed_bl = np.asarray(arrays[f"{method_id}_mixed_bl_level_db"])
        peak_index = int(np.argmax(mixed_bl))
        rows.append(
            {
                "scenario": SCENARIO_ID,
                "method": method_id,
                "evaluation_pattern": "fixed_beam_multi_source",
                "peak_azimuth_deg": float(AZIMUTH_DEG[peak_index]),
                "target_beam_level_db_re_input_source_rms": float(mixed_bl[target_beam_index]),
                "interferer_beam_level_db_re_input_source_rms": float(
                    mixed_bl[interferer_beam_index]
                ),
                "source_count_expected": 2,
                "btr_status": "NOT_EVALUATED_STATIC_FREQUENCY_MODEL",
            }
        )
    with (OUTPUT_DIR / "scenario_summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (OUTPUT_DIR / "review_index.md").write_text(
        "\n".join(
            (
                "# 固定整相・粗い共分散MVDR・方位別共分散MVDR 5方式レビュー",
                "",
                f"- scenario: `{SCENARIO_ID}`",
                "- BLは待受beam軸、FRAZは`[beam,frequency]`、スペクトルはper-bin RMS level。",
                "- 固定整相は標準51相×128 tap FIR bankの実周波数応答を使用。",
                "- 本scenarioは静的周波数モデルであり、BTRとstreaming境界は未評価。",
                "",
            )
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
