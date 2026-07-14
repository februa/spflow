"""scene_renderer信号へMATLAB運用係数を使うT2a逐次整相を適用する。

一つの実行で、シナリオ生成、MATLAB生成rawアレイ係数読込、候補方位別T共分散、
T2a-MVDR残差FIR、Flowによるblock逐次処理、BL/FRAZ/FL評価を再現する。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "scene_renderer"))

from scene_renderer import (  # noqa: E402
    AcousticSource,
    AmbientField,
    BandLimitedNoiseSpectrum,
    ConstantEnvelope,
    FreeField,
    Receiver,
    Scene,
    SceneRenderer,
    StaticPose,
    tone_component_from_rms_level_db,
)
from scene_renderer.receiver import ArrayGeometry  # noqa: E402

from evaluations.beamforming.external_fixed_delay_diff_mvdr_inputs import (  # noqa: E402
    load_complex_shading_matlab_raw,
    load_positions_matlab_raw,
)
from spflow import Flow  # noqa: E402
from spflow.beamforming.ebae import EbaeConfig, design_ebae_weights_band  # noqa: E402
from spflow.beamforming_evaluation.diagnostic_plotting import centers_to_edges  # noqa: E402
from spflow.simulation import SignalBlock, StatefulIntegerDelay, VersionedCausalFIR  # noqa: E402

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]

SUPPORTED_METHOD_IDS = ("fixed_baseline", "t2a_mvdr", "t2a_ebae")


def _validate_method_ids(method_ids: tuple[str, ...]) -> tuple[str, ...]:
    """実行対象方式を重複のない非空tupleとして検証する。

    Args:
        method_ids: `SUPPORTED_METHOD_IDS`から選んだ方式識別子。

    Returns:
        入力順を維持した方式識別子。

    Raises:
        ValueError: 空、重複、または未対応方式を含む場合。
    """
    if len(method_ids) == 0:
        raise ValueError("method_ids must contain at least one method.")
    if len(set(method_ids)) != len(method_ids):
        raise ValueError("method_ids must not contain duplicates.")
    unsupported = [method_id for method_id in method_ids if method_id not in SUPPORTED_METHOD_IDS]
    if unsupported:
        raise ValueError(f"unsupported method_ids: {unsupported}")
    return method_ids


@dataclass(frozen=True)
class T2aScenarioConfig:
    """T2a逐次評価の物理条件と処理条件を保持する。

    入力はsampling、音速、音源、雑音、STFT、FIR、block条件であり、出力は
    `run_evaluation`が生成する信号と評価成果物である。アレイ係数そのものや
    MATLAB変数名は責務に含めない。信号処理上はsceneと処理周期の正本に位置づく。
    """

    fs_hz: float = 4096.0
    sound_speed_m_s: float = 1500.0
    duration_s: float = 6.0
    training_duration_s: float = 4.5
    target_azimuth_deg: float = 55.0
    target_frequency_hz: float = 512.0
    target_level_db_re_input_rms: float = 0.0
    interferer_azimuth_deg: float = 125.0
    interferer_frequency_hz: float = 768.0
    interferer_level_db_re_input_rms: float = 6.0
    noise_asd_level_db_re_input_rms_per_sqrt_hz: float = -42.0
    noise_band_hz: tuple[float, float] = (128.0, 1400.0)
    beam_azimuth_step_deg: float = 5.0
    analysis_fft_size: int = 256
    analysis_hop_size: int = 256
    residual_fir_tap_count: int = 128
    runtime_block_size: int = 173
    diagonal_loading_ratio: float = 1.0e-2
    random_seed: int = 20260715

    def __post_init__(self) -> None:
        """単位付きscalarとFFT/FIR境界を早期検証する。

        Raises:
            ValueError: sampling、時間、帯域、FFT、FIRまたはblock条件が不正な場合。
        """
        if self.fs_hz <= 0.0 or self.sound_speed_m_s <= 0.0:
            raise ValueError("fs_hz and sound_speed_m_s must be positive.")
        if not 0.0 < self.training_duration_s < self.duration_s:
            raise ValueError("training_duration_s must lie inside duration_s.")
        if not 0.0 < self.noise_band_hz[0] < self.noise_band_hz[1] < self.fs_hz / 2.0:
            raise ValueError("noise_band_hz must lie inside the positive Nyquist band.")
        if self.analysis_fft_size < 8 or self.analysis_hop_size <= 0:
            raise ValueError("analysis FFT and hop sizes must be positive and usable.")
        if not 1 <= self.residual_fir_tap_count <= self.analysis_fft_size:
            raise ValueError("residual_fir_tap_count must lie in [1, analysis_fft_size].")
        if self.runtime_block_size <= 0 or self.beam_azimuth_step_deg <= 0.0:
            raise ValueError("runtime block size and beam step must be positive.")


@dataclass(frozen=True)
class MatlabArrayCoefficients:
    """MATLABから読んだ周波数別アレイ係数を保持する。

    `positions_m`はshape `[n_ch,3]`、`frequency_hz`は`[n_band]`、
    `shading`と`active_channel_mask`は`[n_ch,n_band]`である。axis=0は物理channel、
    axis=1は係数周波数で、位置はm、周波数はHz、shadingは線形振幅である。

    本クラスはファイル境界の検証だけを担い、重み設計、補間、信号処理を担わない。
    """

    positions_m: FloatArray
    frequency_hz: FloatArray
    shading: ComplexArray
    active_channel_mask: BoolArray

    def __post_init__(self) -> None:
        """配列shape、有限性、周波数単調性、active条件を検証する。

        Raises:
            ValueError: MATLAB配列の契約を満たさない場合。
        """
        positions = np.asarray(self.positions_m, dtype=np.float64)
        frequency = np.asarray(self.frequency_hz, dtype=np.float64)
        shading = np.asarray(self.shading, dtype=np.complex128)
        active = np.asarray(self.active_channel_mask, dtype=np.bool_)
        if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
            raise ValueError("positions_m must have shape [n_ch, 3].")
        if frequency.ndim != 1 or frequency.size == 0 or not bool(np.all(np.diff(frequency) > 0.0)):
            raise ValueError("frequency_hz must be a non-empty strictly increasing 1-D array.")
        expected_shape = (positions.shape[0], frequency.size)
        if shading.shape != expected_shape or active.shape != expected_shape:
            raise ValueError("shading and active_channel_mask must have shape [n_ch, n_band].")
        if not bool(np.all(np.isfinite(positions))) or not bool(np.all(np.isfinite(shading))):
            raise ValueError("positions_m and shading must contain finite values.")
        if bool(np.any(np.count_nonzero(active, axis=0) == 0)):
            raise ValueError("every coefficient frequency must activate at least one channel.")
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "frequency_hz", frequency)
        object.__setattr__(self, "shading", shading)
        object.__setattr__(self, "active_channel_mask", active)

    def table_at(self, frequency_hz: float) -> tuple[ComplexArray, BoolArray]:
        """指定周波数以上で最も近い係数行をchannel軸で返す。

        Args:
            frequency_hz: 処理周波数。単位はHz。

        Returns:
            `(shading, active_mask)`。双方shapeは`[n_ch]`。

        Raises:
            ValueError: 周波数が負の場合。

        境界条件:
            表の中間では高周波側のactive setを選ぶ。疎配置のalias上限を超えて
            低周波側の広い素子間隔を使わないための安全側規約である。
        """
        if frequency_hz < 0.0:
            raise ValueError("frequency_hz must be non-negative.")
        index = int(np.searchsorted(self.frequency_hz, frequency_hz, side="left"))
        index = min(index, int(self.frequency_hz.size - 1))
        return self.shading[:, index].copy(), self.active_channel_mask[:, index].copy()


@dataclass(frozen=True)
class RenderedComponents:
    """scene_rendererが生成した成分分離信号を保持する。

    `target`、`interferer`、`noise`、`mixed`はshape `[n_ch,n_sample]`、
    axis=0がchannel、axis=1が時刻、振幅単位はinput RMS基準である。
    """

    target: FloatArray
    interferer: FloatArray
    noise: FloatArray
    mixed: FloatArray


@dataclass(frozen=True)
class FrequencyWeightDesign:
    """fixed、T2a-MVDR、T2a-EBAEの完成周波数重みと診断量を保持する。

    `weights`の各値はshape `[n_frequency,n_beam,n_ch]`、`causal_delays_samples`は
    `[n_beam,n_ch]`、`active_channel_count`は`[n_frequency]`である。EBAEの
    `signal_count`、`music_peak_azimuth_deg`、`fallback_mask`は`[n_frequency,n_beam]`。

    本結果型は完成値を運ぶだけで、FIR化、逐次処理、図表作成を担わない。
    """

    weights: dict[str, ComplexArray]
    causal_delays_samples: IntArray
    active_channel_count: FloatArray
    ebae_signal_count: IntArray
    ebae_music_peak_azimuth_deg: FloatArray
    ebae_fallback_mask: BoolArray


@dataclass(frozen=True)
class WaveformIntegrityResult:
    """target-only入力と整相出力を位相整列して得た波形完全性を保持する。

    `reference_signal`と`phase_aligned_output`は共通評価区間のshape `[n_sample]`、
    振幅単位はinput RMS基準である。`phase_delay_samples_modulo_period`はtarget toneの
    1周期を法とする位相差のsample換算であり、絶対伝搬遅延ではない。

    本結果型は完成した観測値と描画配列だけを保持し、EBAE重み設計、streaming処理、
    合否判定は責務に含めない。信号処理上はtarget-only無歪性の診断段に位置づく。
    """

    analysis_start_sample: int
    analysis_stop_sample: int
    phase_delay_samples_modulo_period: float
    rms_delta_db: float
    correlation_after_phase_alignment: float
    residual_rms_db_re_input_rms: float
    reference_signal: FloatArray
    phase_aligned_output: FloatArray


@dataclass(frozen=True)
class RuntimeBlock:
    """Flowへ渡す入力blockを保持する。

    `data`はshape `[n_ch,n_block_sample]`、`start_sample`は元系列上の先頭sampleである。
    """

    start_sample: int
    data: FloatArray


@dataclass(frozen=True)
class BeamBlock:
    """Flowから回収する完成状態付きbeam blockを保持する。

    `data`と`valid_mask`はshape `[n_beam,n_block_sample]`で、axis=0は待受方位、
    axis=1は時刻である。`method_id`は`fixed_baseline`、`t2a_mvdr`、`t2a_ebae`である。
    """

    method_id: str
    start_sample: int
    data: ComplexArray
    valid_mask: BoolArray


class MatlabArrayGeometry(ArrayGeometry):
    """MATLAB係数の任意3次元位置をscene_rendererへ公開する。

    入出力はshape `[n_ch,3]`、単位mである。座標変換やアレイ設計は責務に含めない。
    """

    def __init__(self, positions_m: FloatArray) -> None:
        """検証済みセンサ位置を保持する。

        Args:
            positions_m: ArrayFrame `[Bow, Starboard, Up]`位置。shape `[n_ch,3]`、単位m。
        """
        self._positions_m = np.asarray(positions_m, dtype=np.float64).copy()

    def positions(self) -> NDArray[Any]:
        """センサ位置のコピーをshape `[n_ch,3]`、単位mで返す。"""
        return self._positions_m.copy()


class StreamingBeamBranch:
    """一方式の整数遅延・残差FIR・channel和を逐次適用する。

    入力は`RuntimeBlock`、出力は`BeamBlock`である。整数遅延とFIR履歴を保持するが、
    係数設計、scene生成、評価図作成は責務に含めない。信号処理上はT2a実行段に位置づく。
    """

    def __init__(
        self,
        method_id: str,
        causal_delays_samples: IntArray,
        coefficients: ComplexArray,
    ) -> None:
        """beam×channel系列へ状態付き遅延器とFIRを構築する。

        Args:
            method_id: 方式識別子。
            causal_delays_samples: shape `[n_beam,n_ch]`、単位sample。
            coefficients: shape `[n_beam,n_ch,n_tap]`の因果FIR。

        Raises:
            ValueError: beam/channel shapeが一致しない場合。
        """
        delays = np.asarray(causal_delays_samples, dtype=np.int64)
        taps = np.asarray(coefficients, dtype=np.complex128)
        if delays.ndim != 2 or taps.ndim != 3 or taps.shape[:2] != delays.shape:
            raise ValueError("delays and coefficients must share [n_beam, n_ch].")
        self.method_id = method_id
        self._n_beam, self._n_ch = delays.shape
        # Stateful部品はseries軸だけを認識するため、[beam,ch]を一つのseries軸へ平坦化する。
        self._delay = StatefulIntegerDelay(delays.reshape(-1))
        self._fir = VersionedCausalFIR(taps.reshape(self._n_beam * self._n_ch, taps.shape[2]))

    @property
    def n_beam(self) -> int:
        """このbranchが生成する待受beam数を返す。"""
        return self._n_beam

    def process(self, block: RuntimeBlock) -> BeamBlock:
        """一つの入力blockを処理し、完成sampleだけをmaskで明示する。

        Args:
            block: channel入力。shape `[n_ch,n_block_sample]`、単位input RMS。

        Returns:
            beam出力。shape `[n_beam,n_block_sample]`。

        Raises:
            ValueError: channel数が係数と一致しない場合。
        """
        if block.data.ndim != 2 or block.data.shape[0] != self._n_ch:
            raise ValueError("runtime block must have shape [n_ch, n_block_sample].")
        # 全beamが同じ物理入力を受けるため、beam軸を追加して[beam,ch,time]を作る。
        expanded = np.broadcast_to(
            block.data[np.newaxis, :, :],
            (self._n_beam, self._n_ch, block.data.shape[1]),
        ).reshape(self._n_beam * self._n_ch, block.data.shape[1])
        delayed = self._delay.process(np.asarray(expanded, dtype=np.float64))
        filtered = self._fir.process(SignalBlock(delayed.data, delayed.valid_mask))
        filtered_data = filtered.data.reshape(self._n_beam, self._n_ch, block.data.shape[1])
        filtered_valid = filtered.valid_mask.reshape(self._n_beam, self._n_ch, block.data.shape[1])
        # 完成weightの各channel寄与を加算し、全channel完成時だけbeam sampleを公開する。
        return BeamBlock(
            self.method_id,
            block.start_sample,
            np.asarray(np.sum(filtered_data, axis=1), dtype=np.complex128),
            np.asarray(np.all(filtered_valid, axis=1), dtype=np.bool_),
        )


def load_matlab_array_coefficients(
    positions_path: Path,
    shading_path: Path,
    shading_frequency_step_hz: float,
) -> MatlabArrayCoefficients:
    """MATLAB生成rawからアレイ位置、周波数別active channel、shadingを読む。

    Args:
        positions_path: `COE_POS`相当のlittle-endian float32 raw。
        shading_path: `COE_CBFSHADING`相当のlittle-endian float32 raw。
        shading_frequency_step_hz: shading列の周波数間隔。単位はHz。

    Returns:
        検証済み周波数別アレイ係数。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
        ValueError: raw要素数、shape、周波数間隔またはactive条件が不正な場合。

    ファイル契約:
        `COE_POS`はMATLABの`reshape(pos,3,[]).T`で`[n_ch,3]`へ復元する。
        `COE_CBFSHADING`は`reshape(shading,n_ch,[])`の前半列をreal、後半列をimagとし、
        `[n_ch,n_frequency]`へ復元する。active channelは`abs(shading)>0`から導出する。
    """
    if shading_frequency_step_hz <= 0.0:
        raise ValueError("shading_frequency_step_hz must be positive.")
    positions = load_positions_matlab_raw(positions_path)
    shading = load_complex_shading_matlab_raw(shading_path, positions.shape[0])
    frequency = np.arange(shading.shape[1], dtype=np.float64) * float(shading_frequency_step_hz)
    # 独自rawにactive maskを重複保存せず、周波数別shading=0を非使用channelの正本とする。
    active = np.asarray(np.abs(shading) > 0.0, dtype=np.bool_)
    return MatlabArrayCoefficients(positions, frequency, shading, active)


def write_example_matlab_coefficients(
    positions_path: Path,
    shading_path: Path,
    config: T2aScenarioConfig,
) -> None:
    """スクリプト疎通確認用のMATLAB互換raw係数ファイルを生成する。

    Args:
        positions_path: 保存先`COE_POS`相当raw。
        shading_path: 保存先`COE_CBFSHADING`相当raw。
        config: 音速と評価帯域を与える条件。

    Returns:
        なし。

    Notes:
        実運用係数の代替ではない。低周波ほど広い開口、高周波ほど密な中央subsetを使う
        周波数切替契約の確認だけを目的とする。
    """
    n_ch = 16
    positions_x_m = (np.arange(n_ch, dtype=np.float64) - 0.5 * (n_ch - 1)) * 0.18
    positions = np.column_stack((positions_x_m, np.zeros(n_ch), np.zeros(n_ch)))
    frequency_step_hz = 512.0
    n_frequency = int(round((config.fs_hz / 2.0) / frequency_step_hz)) + 1
    active = np.zeros((n_ch, n_frequency), dtype=np.uint8)
    active[:, 0] = 1
    active[1::2, 1] = 1
    active[4:12, 2:4] = 1
    active[6:10, 4:] = 1
    shading = np.asarray(active, dtype=np.float64)
    for band_index in range(n_frequency):
        indices = np.flatnonzero(active[:, band_index])
        # Hann端点が0になるとactive maskと矛盾するため、0.25+0.75*Hannで正値を保つ。
        shading[indices, band_index] *= 0.25 + 0.75 * np.hanning(indices.size)
    positions_path.parent.mkdir(parents=True, exist_ok=True)
    shading_path.parent.mkdir(parents=True, exist_ok=True)
    # COE_POSはMATLABの[3,n_ch]列優先列を一次元rawへ書く契約である。
    np.asarray(positions.T, dtype="<f4").reshape(-1, order="F").tofile(positions_path)
    # COE_CBFSHADINGは[real列,imag列]を[n_ch,2*n_frequency]で列優先保存する。
    complex_shading = np.asarray(shading, dtype=np.complex128)
    raw_table = np.concatenate((complex_shading.real, complex_shading.imag), axis=1)
    np.asarray(raw_table, dtype="<f4").reshape(-1, order="F").tofile(shading_path)


def _direction_vectors(azimuth_deg: FloatArray) -> FloatArray:
    """相対方位をArrayFrame `[Bow,Starboard,Up]`方向余弦へ変換する。"""
    radians = np.deg2rad(np.asarray(azimuth_deg, dtype=np.float64))
    return np.column_stack((np.cos(radians), np.sin(radians), np.zeros_like(radians)))


def _arrival_delays_s(
    positions_m: FloatArray, azimuth_deg: FloatArray, sound_speed_m_s: float
) -> FloatArray:
    """基準点に対する到達遅延をshape `[n_beam,n_ch]`、単位sで返す。"""
    directions = _direction_vectors(azimuth_deg)
    # tau[beam,ch]=-r_ch・u_beam/c。scene_rendererのSourceProjectorと同じ到達遅延符号を使う。
    return np.asarray(-directions @ positions_m.T / sound_speed_m_s, dtype=np.float64)


def predict_uniform_subset_grating_azimuths(
    coefficients: MatlabArrayCoefficients,
    frequency_hz: float,
    steering_azimuth_deg: float,
    sound_speed_m_s: float,
) -> tuple[float, ...]:
    """周波数別active subsetが等間隔ULAなら理論グレーティング方位を返す。

    Args:
        coefficients: 物理位置と周波数別active mask。
        frequency_hz: 評価周波数。単位はHz。
        steering_azimuth_deg: 待受方位。単位はdeg。
        sound_speed_m_s: 音速。単位はm/s。

    Returns:
        `d(cos(theta_g)-cos(theta_0))=m lambda`を満たす0--180 degの方位。
        active位置がx軸等間隔ULAでない場合は空tuple。

    Raises:
        ValueError: 周波数または音速が正でない場合。
    """
    if frequency_hz <= 0.0 or sound_speed_m_s <= 0.0:
        raise ValueError("frequency_hz and sound_speed_m_s must be positive.")
    _, active = coefficients.table_at(frequency_hz)
    positions = coefficients.positions_m[active]
    if positions.shape[0] < 2:
        return ()
    # ULA式を任意非一様・3次元配置へ誤適用しないため、y/z一定とx等間隔を先に確認する。
    if not np.allclose(positions[:, 1:], positions[0, 1:], atol=1.0e-9):
        return ()
    sorted_x = np.sort(positions[:, 0])
    spacing = np.diff(sorted_x)
    spacing_m = float(np.median(spacing))
    if spacing_m <= 0.0 or not np.allclose(spacing, spacing_m, rtol=1.0e-6, atol=1.0e-9):
        return ()
    wavelength_m = sound_speed_m_s / frequency_hz
    steering_cosine = float(np.cos(np.deg2rad(steering_azimuth_deg)))
    maximum_order = int(np.ceil(2.0 * spacing_m / wavelength_m)) + 1
    aliases: list[float] = []
    for order in range(-maximum_order, maximum_order + 1):
        if order == 0:
            continue
        candidate_cosine = steering_cosine + order * wavelength_m / spacing_m
        if -1.0 <= candidate_cosine <= 1.0:
            aliases.append(float(np.rad2deg(np.arccos(candidate_cosine))))
    return tuple(sorted(aliases))


def render_scenario(
    coefficients: MatlabArrayCoefficients, config: T2aScenarioConfig
) -> RenderedComponents:
    """scene_rendererでtarget、interferer、海洋雑音を成分分離して生成する。

    Args:
        coefficients: sceneへ渡す物理センサ位置。
        config: 音源、雑音、環境、sampling条件。

    Returns:
        target-only、interferer-only、noise-only、mixed信号。各shape `[n_ch,n_sample]`。
    """
    receiver = Receiver(
        StaticPose([0.0, 0.0, 0.0], heading_deg=0.0),
        MatlabArrayGeometry(coefficients.positions_m),
    )
    target = AcousticSource.from_relative_bearing(
        config.target_azimuth_deg,
        1000.0,
        receiver.trajectory.pose(0.0),
        [
            tone_component_from_rms_level_db(
                config.target_frequency_hz,
                config.target_level_db_re_input_rms,
                ConstantEnvelope(),
            )
        ],
        identifier="target",
        role="target",
    )
    interferer = AcousticSource.from_relative_bearing(
        config.interferer_azimuth_deg,
        1000.0,
        receiver.trajectory.pose(0.0),
        [
            tone_component_from_rms_level_db(
                config.interferer_frequency_hz,
                config.interferer_level_db_re_input_rms,
                ConstantEnvelope(),
            )
        ],
        identifier="interferer",
        role="interferer",
    )
    ambient = AmbientField.from_asd_level_db(
        BandLimitedNoiseSpectrum(*config.noise_band_hz),
        config.noise_asd_level_db_re_input_rms_per_sqrt_hz,
        covariance=np.eye(coefficients.positions_m.shape[0], dtype=np.float64),
        noise_seed=config.random_seed,
        noise_filter_length=257,
        identifier="ambient",
        role="noise",
    )
    sample_count = int(round(config.duration_s * config.fs_hz))
    time_s = np.arange(sample_count, dtype=np.float64) / config.fs_hz
    rendered = SceneRenderer().render_components(
        Scene([target, interferer], [ambient], FreeField(config.sound_speed_m_s)),
        receiver,
        time_s,
    )
    return RenderedComponents(
        np.asarray(np.real(rendered.sum_by_role("target")), dtype=np.float64),
        np.asarray(np.real(rendered.sum_by_role("interferer")), dtype=np.float64),
        np.asarray(np.real(rendered.sum_by_role("noise")), dtype=np.float64),
        np.asarray(np.real(rendered.mixed), dtype=np.float64),
    )


def _candidate_snapshots(
    signal: FloatArray,
    causal_delays_samples: IntArray,
    config: T2aScenarioConfig,
) -> ComplexArray:
    """候補方位へ整数整相したtraining STFT snapshotを返す。

    出力shapeは`[n_frame,n_frequency,n_ch]`。axis=0は時間snapshot、axis=1は
    rFFT周波数、axis=2はchannelである。窓は振幅校正を変えない矩形窓とする。
    """
    training_samples = min(int(round(config.training_duration_s * config.fs_hz)), signal.shape[1])
    maximum_delay = int(np.max(causal_delays_samples))
    starts = np.arange(
        maximum_delay,
        training_samples - config.analysis_fft_size + 1,
        config.analysis_hop_size,
        dtype=np.int64,
    )
    if starts.size < 2:
        raise ValueError("training interval is too short after T2a integer-delay alignment.")
    frames = np.empty((starts.size, signal.shape[0], config.analysis_fft_size), dtype=np.float64)
    for frame_index, start in enumerate(starts):
        for channel_index, delay in enumerate(causal_delays_samples):
            begin = int(start) - int(delay)
            frames[frame_index, channel_index] = signal[
                channel_index, begin : begin + config.analysis_fft_size
            ]
    # FFT axis=2は各channelの時間軸。moveaxis後は[n_frame,n_frequency,n_ch]となる。
    return np.asarray(np.moveaxis(np.fft.rfft(frames, axis=2), 2, 1), dtype=np.complex128)


def _loaded_mvdr_weight(
    covariance: ComplexArray, constraint: ComplexArray, ratio: float
) -> ComplexArray:
    """trace比例対角loading付きMVDR重みを返す。"""
    hermitian = 0.5 * (covariance + covariance.conj().T)
    average_power = float(np.real(np.trace(hermitian))) / float(constraint.size)
    # snapshot不足や強相関時の特異化を避けるが、物理power尺度を変えないtrace比例量を使う。
    loaded = hermitian + ratio * max(average_power, np.finfo(float).tiny) * np.eye(
        constraint.size, dtype=np.complex128
    )
    solved = np.linalg.solve(loaded, constraint)
    # w=R^-1 a/(a^H R^-1 a)により残差座標でw^H a=1を保証する。
    return np.asarray(solved / np.vdot(constraint, solved), dtype=np.complex128)


def design_frequency_weights(
    training_signal: FloatArray,
    coefficients: MatlabArrayCoefficients,
    beam_azimuth_deg: FloatArray,
    config: T2aScenarioConfig,
    method_ids: tuple[str, ...] = SUPPORTED_METHOD_IDS,
) -> FrequencyWeightDesign:
    """候補方位別T共分散からfixed、T2a-MVDR、T2a-EBAE重みを設計する。

    Args:
        training_signal: 共分散を推定する入力。shape `[n_ch,n_sample]`、単位input RMS。
        coefficients: 物理位置、周波数別active channel、複素shading。
        beam_azimuth_deg: 待受方位。shape `[n_beam]`、単位deg。
        config: sampling、training、FFT、EBAE/MVDR条件。
        method_ids: 設計する方式。既定は3方式すべて。

    Returns:
        選択方式の重み、因果整数遅延、active数、EBAE診断量。重みは`w^H x`表現であり、
        FIR化時に共役を取る。EBAE未選択時のEBAE診断配列は未使用既定値を保持する。

    Raises:
        ValueError: 方式、入力shape、またはEBAEの`M^2` snapshot条件が不正な場合。
    """
    selected_method_ids = _validate_method_ids(method_ids)
    selected_method_set = set(selected_method_ids)
    arrival_delays_s = _arrival_delays_s(
        coefficients.positions_m, beam_azimuth_deg, config.sound_speed_m_s
    )
    integer_offsets = np.rint(arrival_delays_s * config.fs_hz).astype(np.int64)
    # T共分散のx[n+tau]と因果bufferのx[n-d]を対応させるためd=max(tau)-tauとする。
    causal_delays = np.max(integer_offsets, axis=1, keepdims=True) - integer_offsets
    frequency_hz = np.fft.rfftfreq(config.analysis_fft_size, d=1.0 / config.fs_hz)
    n_frequency = frequency_hz.size
    n_beam = beam_azimuth_deg.size
    n_ch = coefficients.positions_m.shape[0]
    weights = {
        method_id: np.zeros((n_frequency, n_beam, n_ch), dtype=np.complex128)
        for method_id in selected_method_ids
    }
    active_count = np.empty(n_frequency, dtype=np.float64)
    ebae_signal_count = np.zeros((n_frequency, n_beam), dtype=np.int64)
    ebae_music_peak_deg = np.full((n_frequency, n_beam), np.nan, dtype=np.float64)
    ebae_fallback = np.zeros((n_frequency, n_beam), dtype=np.bool_)
    # T2aの候補方位別時間切り出しは周波数に依存しないため、各beamで一度だけFFTする。
    # 同じsnapshotを周波数loop内で再生成すると、方式上不要な計算量がn_frequency倍になる。
    needs_covariance = bool({"t2a_mvdr", "t2a_ebae"} & selected_method_set)
    snapshots_by_beam = (
        [
            _candidate_snapshots(training_signal, causal_delays[beam_index], config)
            for beam_index in range(n_beam)
        ]
        if needs_covariance
        else []
    )
    for frequency_index, frequency in enumerate(frequency_hz):
        shading, active = coefficients.table_at(float(frequency))
        active_indices = np.flatnonzero(active)
        active_count[frequency_index] = float(active_indices.size)
        for beam_index in range(n_beam):
            is_real_spectrum_boundary = frequency_index in (0, n_frequency - 1)
            physical_tau = arrival_delays_s[beam_index, active_indices]
            integer_tau = integer_offsets[beam_index, active_indices] / config.fs_hz
            # 整数遅延後の残差steeringはD a=exp(-j2πf(tau-q/fs))。
            residual_constraint = np.exp(
                -1j * 2.0 * np.pi * frequency * (physical_tau - integer_tau)
            )
            channel_shading = shading[active_indices]
            fixed_active = residual_constraint / np.vdot(residual_constraint, residual_constraint)
            unshaded_by_method: dict[str, ComplexArray] = {}
            if "fixed_baseline" in selected_method_set:
                unshaded_by_method["fixed_baseline"] = np.asarray(fixed_active, dtype=np.complex128)
            if is_real_spectrum_boundary:
                # DC/Nyquistは実FIRのHermitian境界であり、複素適応位相を持たせない。
                if "t2a_mvdr" in selected_method_set:
                    unshaded_by_method["t2a_mvdr"] = np.asarray(fixed_active, dtype=np.complex128)
                if "t2a_ebae" in selected_method_set:
                    unshaded_by_method["t2a_ebae"] = np.asarray(fixed_active, dtype=np.complex128)
                    ebae_music_peak_deg[frequency_index, beam_index] = beam_azimuth_deg[beam_index]
            elif needs_covariance:
                snapshots = snapshots_by_beam[beam_index][:, frequency_index, :]
                ebae_snapshot_count = int(active_indices.size * active_indices.size)
                if "t2a_ebae" in selected_method_set and snapshots.shape[0] < ebae_snapshot_count:
                    raise ValueError(
                        "training interval must provide at least M**2 non-overlap snapshots "
                        f"for EBAE: frequency={frequency:g} Hz, M={active_indices.size}, "
                        f"required={ebae_snapshot_count}, observed={snapshots.shape[0]}."
                    )
                # EBAE選択時は宣言L=M^2と物理平均数を一致させ、比較するMVDRも同じ共分散を使う。
                selected_snapshot_count = (
                    ebae_snapshot_count if "t2a_ebae" in selected_method_set else snapshots.shape[0]
                )
                active_snapshot = snapshots[:selected_snapshot_count, active_indices]
                # R=E[xx^H]。snapshot axis=0を平均し、channel×channel共分散を得る。
                covariance = np.einsum(
                    "fc,fd->cd", active_snapshot, active_snapshot.conj(), optimize=True
                ) / float(active_snapshot.shape[0])
                if "t2a_mvdr" in selected_method_set:
                    unshaded_by_method["t2a_mvdr"] = _loaded_mvdr_weight(
                        covariance, residual_constraint, config.diagonal_loading_ratio
                    )
                if "t2a_ebae" in selected_method_set:
                    # D_b a(phi)により、候補bの整数遅延後座標へ全scan steeringを移す。
                    residual_scan = np.exp(
                        -1j
                        * 2.0
                        * np.pi
                        * frequency
                        * (
                            arrival_delays_s[:, active_indices].T
                            - integer_offsets[beam_index, active_indices, np.newaxis] / config.fs_hz
                        )
                    )
                    ebae_result = design_ebae_weights_band(
                        covariance,
                        residual_scan,
                        snapshot_count=ebae_snapshot_count,
                        config=EbaeConfig(
                            snapshot_rate_hz=config.fs_hz / config.analysis_hop_size,
                            integration_time_sec=(
                                ebae_snapshot_count * config.analysis_hop_size / config.fs_hz
                            ),
                            sigmoid_slope=10.0,
                            sigmoid_midpoint=0.5,
                            diagonal_loading=1.0,
                        ),
                    )
                    unshaded_by_method["t2a_ebae"] = np.asarray(
                        ebae_result.weights[:, beam_index], dtype=np.complex128
                    )
                    ebae_signal_count[frequency_index, beam_index] = ebae_result.signal_count
                    ebae_music_peak_deg[frequency_index, beam_index] = float(
                        beam_azimuth_deg[int(np.argmax(ebae_result.music_spectrum))]
                    )
                    ebae_fallback[frequency_index, beam_index] = ebae_result.used_fallback
            for method_id, unshaded in unshaded_by_method.items():
                # 実信号経路でgを掛ける意味をw^H x規約の重みへ移すため、重み側はconj(g)倍する。
                shaded = channel_shading.conj() * unshaded
                denominator = np.vdot(shaded, residual_constraint)
                if abs(denominator) <= np.finfo(np.float64).eps:
                    # shading後に無歪正規化できない場合は、不完全な適応値でなくCBFを採用する。
                    shaded = fixed_active
                    denominator = np.vdot(shaded, residual_constraint)
                weights[method_id][frequency_index, beam_index, active_indices] = (
                    shaded / denominator.conjugate()
                )
    return FrequencyWeightDesign(
        weights=weights,
        causal_delays_samples=np.asarray(causal_delays, dtype=np.int64),
        active_channel_count=active_count,
        ebae_signal_count=ebae_signal_count,
        ebae_music_peak_azimuth_deg=ebae_music_peak_deg,
        ebae_fallback_mask=ebae_fallback,
    )


def realize_residual_fir(weights: ComplexArray, tap_count: int) -> tuple[ComplexArray, FloatArray]:
    """残差周波数重みを共通tap窓の因果FIRへ変換する。

    Args:
        weights: `w^H x`重み。shape `[n_frequency,n_beam,n_ch]`。
        tap_count: 残差FIR長。単位sample。

    Returns:
        `(coefficients, energy_containment)`。係数shapeは`[n_beam,n_ch,n_tap]`、
        energy比shapeは`[n_beam]`。
    """
    n_fft = 2 * (weights.shape[0] - 1)
    n_beam, n_ch = weights.shape[1:]
    coefficients = np.empty((n_beam, n_ch, tap_count), dtype=np.complex128)
    energy_ratio = np.empty(n_beam, dtype=np.float64)
    for beam_index in range(n_beam):
        # FIR適用応答Hはy=sum_ch H_ch X_ch=sum_ch conj(w_ch)X_chなのでH=conj(w)。
        impulse = np.fft.irfft(weights[:, beam_index, :].conj(), n=n_fft, axis=0)
        energy = np.sum(impulse**2, axis=1)
        extended = np.concatenate((energy, energy[: tap_count - 1]))
        window_energy = np.convolve(extended, np.ones(tap_count), mode="valid")[:n_fft]
        start = int(np.argmax(window_energy))
        indices = (start + np.arange(tap_count)) % n_fft
        coefficients[beam_index] = np.asarray(impulse[indices].T, dtype=np.complex128)
        energy_ratio[beam_index] = float(
            window_energy[start] / max(float(np.sum(energy)), np.finfo(float).tiny)
        )
    return coefficients, energy_ratio


def run_streaming_flow(
    signal: FloatArray,
    branches: list[StreamingBeamBranch],
    block_size: int,
) -> dict[str, tuple[ComplexArray, BoolArray]]:
    """Flowで全方式のblock逐次処理を読みやすく接続する。

    `Flow.many(branches) -> process(block) -> collect`という通常Pythonの反復を使い、
    Flowへrate、DAG、状態管理を持たせない。各branchの状態は独立クラスが保持する。
    """
    n_beam = branches[0].n_beam
    output = {
        branch.method_id: np.empty((n_beam, signal.shape[1]), dtype=np.complex128)
        for branch in branches
    }
    valid = {
        branch.method_id: np.empty((n_beam, signal.shape[1]), dtype=np.bool_) for branch in branches
    }

    def collect(block: BeamBlock) -> None:
        """完成blockを元時刻位置へ格納し、次段へ値を流さない。"""
        stop = block.start_sample + block.data.shape[1]
        output[block.method_id][:, block.start_sample : stop] = block.data
        valid[block.method_id][:, block.start_sample : stop] = block.valid_mask

    for start in range(0, signal.shape[1], block_size):
        stop = min(start + block_size, signal.shape[1])
        runtime_block = RuntimeBlock(start, signal[:, start:stop])
        Flow.many(branches).map(lambda branch: branch.process(runtime_block)).map(collect)
    return {method: (output[method], valid[method]) for method in output}


def _fraz_level_db(
    output: ComplexArray, valid: BoolArray, config: T2aScenarioConfig
) -> tuple[FloatArray, FloatArray]:
    """完成beam波形からFRAZ levelを`dB re input RMS`で返す。"""
    common_valid = np.all(valid, axis=0)
    indices = np.flatnonzero(common_valid)
    if indices.size < config.analysis_fft_size:
        raise ValueError("streaming output has too few common valid samples for FRAZ.")
    first = int(indices[0])
    last = int(indices[-1]) + 1
    starts = np.arange(
        first,
        last - config.analysis_fft_size + 1,
        config.analysis_hop_size,
        dtype=np.int64,
    )
    spectra = np.stack(
        [
            np.fft.rfft(np.real(output[:, start : start + config.analysis_fft_size]), axis=1)
            for start in starts
        ],
        axis=0,
    )
    # 非DC/Nyquistの片側bin RMS powerは2|X/N|^2。端点だけ2倍しない。
    power = np.abs(spectra / float(config.analysis_fft_size)) ** 2
    if power.shape[2] > 2:
        power[:, :, 1:-1] *= 2.0
    mean_power = np.mean(power, axis=0)
    level = 10.0 * np.log10(np.maximum(mean_power, np.finfo(np.float64).tiny))
    frequency = np.fft.rfftfreq(config.analysis_fft_size, d=1.0 / config.fs_hz)
    return np.asarray(frequency, dtype=np.float64), np.asarray(level, dtype=np.float64)


def _make_branches(
    weights: dict[str, ComplexArray], delays: IntArray, tap_count: int
) -> tuple[list[StreamingBeamBranch], dict[str, FloatArray]]:
    """完成重みから履歴を共有しない方式別runtime branchを作る。"""
    branches: list[StreamingBeamBranch] = []
    energy: dict[str, FloatArray] = {}
    for method_id, method_weights in weights.items():
        coefficients, energy_ratio = realize_residual_fir(method_weights, tap_count)
        branches.append(StreamingBeamBranch(method_id, delays, coefficients))
        energy[method_id] = energy_ratio
    return branches, energy


def _one_sided_rms_spectrum(signal: FloatArray, fs_hz: float) -> tuple[FloatArray, FloatArray]:
    """実時間信号の片側per-bin RMS levelを計算する。

    Args:
        signal: 実信号。shape `[n_sample]`、振幅単位はinput RMS基準。
        fs_hz: 標本化周波数。単位はHz。

    Returns:
        `(frequency_hz, level_db)`。双方shape `[n_frequency]`。levelは
        `dB re input RMS`のper-bin RMSである。

    Raises:
        ValueError: 信号が1次元でない、2 sample未満、有限でない、またはfsが正でない場合。
    """
    values = np.asarray(signal, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("signal must be a one-dimensional array with at least two samples.")
    if fs_hz <= 0.0 or not bool(np.all(np.isfinite(values))):
        raise ValueError("fs_hz must be positive and signal must be finite.")
    spectrum = np.fft.rfft(values)
    # 実信号の片側per-bin RMS powerは内側binだけ2|X/N|^2とする。
    power = np.abs(spectrum / float(values.size)) ** 2
    if values.size % 2 == 0 and power.size > 2:
        # 偶数長では末尾がNyquistなので2倍しない。
        power[1:-1] *= 2.0
    elif values.size % 2 == 1 and power.size > 1:
        # 奇数長rFFTにはNyquist binがないため、DC以外をすべて2倍する。
        power[1:] *= 2.0
    frequency_hz = np.fft.rfftfreq(values.size, d=1.0 / fs_hz)
    level_db = 10.0 * np.log10(np.maximum(power, np.finfo(np.float64).tiny))
    return (
        np.asarray(frequency_hz, dtype=np.float64),
        np.asarray(level_db, dtype=np.float64),
    )


def calculate_target_waveform_integrity(
    reference_signal: FloatArray,
    beam_output: ComplexArray,
    valid_mask: BoolArray,
    config: T2aScenarioConfig,
) -> WaveformIntegrityResult:
    """target-only出力を入力へ位相整列し、波形完全性を計算する。

    Args:
        reference_signal: 基準channelのtarget-only入力。shape `[n_sample]`、input RMS基準。
        beam_output: target待受beamのtarget-only出力。shape `[n_sample]`、input RMS基準。
        valid_mask: beam出力の完成sample。shape `[n_sample]`、Trueだけを評価する。
        config: fs、target周波数、training区間を与えるscenario条件。

    Returns:
        位相差、RMS差、位相整列後相関、残差level、および共通評価区間の波形。

    Raises:
        ValueError: shape不一致、評価区間不足、非有限値、target周波数が無効、または
            基準信号のtarget成分が数値床以下の場合。

    境界条件:
        重み設計へ使ったtraining区間と未完成FIR履歴は除外する。単一toneでは絶対遅延を
        1周期ごとに一意化できないため、位相遅延はtarget周期を法とするsample数で返す。
    """
    reference = np.asarray(reference_signal, dtype=np.float64)
    output = np.asarray(beam_output, dtype=np.complex128)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if reference.ndim != 1 or output.ndim != 1 or valid.ndim != 1:
        raise ValueError("waveform integrity inputs must be one-dimensional.")
    if reference.shape != output.shape or reference.shape != valid.shape:
        raise ValueError("reference_signal, beam_output, and valid_mask must share shape.")
    if not 0.0 < config.target_frequency_hz < config.fs_hz / 2.0:
        raise ValueError("target_frequency_hz must lie inside the positive Nyquist band.")
    evaluation_valid = valid.copy()
    training_sample_count = int(round(config.training_duration_s * config.fs_hz))
    evaluation_valid[:training_sample_count] = False
    indices = np.flatnonzero(evaluation_valid)
    if indices.size < config.analysis_fft_size:
        raise ValueError("too few completed post-training samples for waveform integrity.")
    start = int(indices[0])
    stop = int(indices[-1]) + 1
    if not bool(np.all(evaluation_valid[start:stop])):
        # 内部欠損を一つの連続系列としてFFTすると、不連続を方式の歪みと混同するため拒否する。
        raise ValueError("waveform integrity requires one contiguous completed interval.")
    reference_segment = reference[start:stop]
    output_segment = np.real(output[start:stop])
    if not bool(np.all(np.isfinite(reference_segment))) or not bool(
        np.all(np.isfinite(output_segment))
    ):
        raise ValueError("waveform integrity interval must contain finite values.")

    sample_index = np.arange(start, stop, dtype=np.float64)
    # exact target周波数への複素射影で、FFT bin丸めに依存せず入力・出力の位相差を求める。
    carrier = np.exp(-1j * 2.0 * np.pi * config.target_frequency_hz * sample_index / config.fs_hz)
    reference_phasor = 2.0 * np.mean(reference_segment * carrier)
    output_phasor = 2.0 * np.mean(output_segment * carrier)
    if abs(reference_phasor) <= np.finfo(np.float64).eps:
        raise ValueError("reference target component is below the numerical floor.")
    phase_delta_rad = float(np.angle(output_phasor / reference_phasor))
    phase_delay_samples = (
        phase_delta_rad * config.fs_hz / (2.0 * np.pi * config.target_frequency_hz)
    )

    # 出力へ線形位相exp(-j2πfD/fs)を与え、target toneで観測した位相遅延Dを除去する。
    # 振幅応答は変えないため、整列後残差は位相遅延以外の波形変化を表す。
    output_spectrum = np.fft.rfft(output_segment)
    frequency_hz = np.fft.rfftfreq(output_segment.size, d=1.0 / config.fs_hz)
    phase_correction = np.exp(-1j * 2.0 * np.pi * frequency_hz * phase_delay_samples / config.fs_hz)
    phase_aligned = np.fft.irfft(output_spectrum * phase_correction, n=output_segment.size)
    input_rms = float(np.sqrt(np.mean(reference_segment**2)))
    output_rms = float(np.sqrt(np.mean(output_segment**2)))
    if input_rms <= np.finfo(np.float64).eps or output_rms <= np.finfo(np.float64).eps:
        raise ValueError("waveform integrity RMS must exceed the numerical floor.")
    correlation = float(np.corrcoef(reference_segment, phase_aligned)[0, 1])
    residual_rms = float(np.sqrt(np.mean((phase_aligned - reference_segment) ** 2)))
    return WaveformIntegrityResult(
        analysis_start_sample=start,
        analysis_stop_sample=stop,
        phase_delay_samples_modulo_period=phase_delay_samples,
        rms_delta_db=20.0 * np.log10(output_rms / input_rms),
        correlation_after_phase_alignment=correlation,
        residual_rms_db_re_input_rms=20.0 * np.log10(max(residual_rms, np.finfo(np.float64).tiny)),
        reference_signal=np.asarray(reference_segment, dtype=np.float64),
        phase_aligned_output=np.asarray(phase_aligned, dtype=np.float64),
    )


def _streaming_reference_errors(
    streamed_output: ComplexArray,
    one_block_output: ComplexArray,
    valid_mask: BoolArray,
    block_size: int,
) -> tuple[float, float]:
    """分割streamingと一括blockの全体誤差・block境界近傍誤差を返す。

    入力配列はshape `[n_sample]`、振幅単位はinput RMS基準である。戻り値は
    `(全完成区間の最大絶対誤差, 各block境界前後1 sampleの最大絶対誤差)`である。
    """
    streamed = np.asarray(streamed_output, dtype=np.complex128)
    one_block = np.asarray(one_block_output, dtype=np.complex128)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if streamed.ndim != 1 or streamed.shape != one_block.shape or streamed.shape != valid.shape:
        raise ValueError("streamed, one-block, and valid arrays must share one-dimensional shape.")
    if block_size <= 0 or not bool(np.any(valid)):
        raise ValueError(
            "block_size must be positive and valid_mask must contain completed samples."
        )
    difference = np.abs(streamed - one_block)
    overall_error = float(np.max(difference[valid]))
    boundary_mask = np.zeros(valid.shape, dtype=np.bool_)
    for boundary in range(block_size, streamed.size, block_size):
        # 境界直前・直後の2 sampleは、履歴更新漏れによる段差が最初に現れる位置である。
        boundary_mask[max(0, boundary - 1) : min(streamed.size, boundary + 1)] = True
    completed_boundary = boundary_mask & valid
    boundary_error = (
        float(np.max(difference[completed_boundary])) if bool(np.any(completed_boundary)) else 0.0
    )
    return overall_error, boundary_error


def _write_input_spectrum(output_path: Path, mixed: FloatArray, config: T2aScenarioConfig) -> None:
    """整相前rendered target+interferer+noiseの片側RMS spectrumを保存する。"""
    spectrum = np.fft.rfft(mixed, axis=1)
    power = np.abs(spectrum / float(mixed.shape[1])) ** 2
    power[:, 1:-1] *= 2.0
    level = 10.0 * np.log10(np.maximum(np.mean(power, axis=0), np.finfo(float).tiny))
    frequency = np.fft.rfftfreq(mixed.shape[1], d=1.0 / config.fs_hz)
    figure, axis = plt.subplots(figsize=(10.0, 4.0))
    axis.plot(frequency, level)
    axis.set(
        title="Pre-beamforming rendered target + interferer + noise",
        xlabel="Frequency [Hz]",
        ylabel="Per-bin RMS Level [dB re input RMS]",
        xlim=(0.0, config.fs_hz / 2.0),
    )
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _diagnostic_zoom_bounds(
    valid_mask: BoolArray,
    block_size: int,
    minimum_sample: int,
) -> tuple[int, int, int]:
    """完成区間内のblock境界と拡大表示範囲を選ぶ。

    Args:
        valid_mask: 出力完成sample。shape `[n_sample]`。
        block_size: streaming block長。単位sample。
        minimum_sample: training等を除外する最小sample index。

    Returns:
        `(boundary_sample, zoom_start, zoom_stop)`。すべてsample index。

    Raises:
        ValueError: 完成したblock境界を含む表示範囲を確保できない場合。
    """
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if valid.ndim != 1 or block_size <= 0 or minimum_sample < 0:
        raise ValueError("valid_mask, block_size, and minimum_sample are invalid.")
    half_width = min(max(block_size // 2, 32), 256)
    for boundary in range(block_size, valid.size, block_size):
        start = boundary - half_width
        stop = boundary + half_width
        if start >= minimum_sample and stop <= valid.size and bool(np.all(valid[start:stop])):
            return boundary, start, stop
    raise ValueError("no completed streaming boundary is available for diagnostic zoom.")


def _draw_block_boundaries(
    axis: Any,
    start_sample: int,
    stop_sample: int,
    block_size: int,
    fs_hz: float,
) -> None:
    """時間波形axisへruntime block境界をsample時刻で描く。"""
    first_boundary = ((start_sample + block_size - 1) // block_size) * block_size
    for boundary in range(first_boundary, stop_sample, block_size):
        axis.axvline(boundary / fs_hz, color="tab:red", linestyle=":", alpha=0.75)


def _write_input_waveform_diagnostics(
    output_path: Path,
    input_signal: FloatArray,
    reference_channel_index: int,
    zoom_start: int,
    zoom_stop: int,
    config: T2aScenarioConfig,
) -> None:
    """整相前mixed入力の全体波形、境界拡大波形、spectrumを保存する。

    Args:
        output_path: PNG保存先。
        input_signal: beamformer入力。shape `[n_ch,n_sample]`、input RMS基準。
        reference_channel_index: 表示する物理channel index。
        zoom_start: 拡大区間先頭。単位sample。
        zoom_stop: 拡大区間終端。単位sample、終端は含まない。
        config: fs、training時間、block長を与えるscenario条件。

    Raises:
        ValueError: channelまたは拡大範囲が入力shape外の場合。
    """
    values = np.asarray(input_signal, dtype=np.float64)
    if values.ndim != 2 or not 0 <= reference_channel_index < values.shape[0]:
        raise ValueError("input_signal or reference_channel_index is invalid.")
    if not 0 <= zoom_start < zoom_stop <= values.shape[1]:
        raise ValueError("input waveform zoom range lies outside the signal.")
    waveform = values[reference_channel_index]
    time_s = np.arange(waveform.size, dtype=np.float64) / config.fs_hz
    spectrum_start = int(round(config.training_duration_s * config.fs_hz))
    frequency_hz, level_db = _one_sided_rms_spectrum(waveform[spectrum_start:], config.fs_hz)
    upper_db = float(np.max(level_db)) + 3.0
    lower_db = upper_db - 120.0

    figure, axes = plt.subplots(3, 1, figsize=(12.0, 10.0))
    axes[0].plot(time_s, waveform, linewidth=0.7)
    axes[0].set(
        title=f"Pre-beamforming mixed input: channel {reference_channel_index}",
        xlabel="Time [s]",
        ylabel="Amplitude [re input RMS]",
    )
    zoom_time_s = time_s[zoom_start:zoom_stop]
    axes[1].plot(zoom_time_s, waveform[zoom_start:zoom_stop], linewidth=1.0)
    _draw_block_boundaries(axes[1], zoom_start, zoom_stop, config.runtime_block_size, config.fs_hz)
    axes[1].set(
        title="Input waveform zoom; red dotted lines are runtime block boundaries",
        xlabel="Time [s]",
        ylabel="Amplitude [re input RMS]",
    )
    axes[2].plot(frequency_hz, np.maximum(level_db, lower_db))
    axes[2].set(
        title="Pre-beamforming mixed input spectrum after training interval",
        xlabel="Frequency [Hz]",
        ylabel="Per-bin RMS Level [dB re input RMS]",
        xlim=(0.0, config.fs_hz / 2.0),
        ylim=(lower_db, upper_db),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_output_waveform_diagnostics(
    output_path: Path,
    method_id: str,
    streamed_output: ComplexArray,
    one_block_output: ComplexArray,
    valid_mask: BoolArray,
    zoom_start: int,
    zoom_stop: int,
    config: T2aScenarioConfig,
) -> None:
    """target待受beamのmixed出力波形、境界、一括誤差、spectrumを保存する。

    入力3配列はshape `[n_sample]`、振幅単位はinput RMS基準である。分割streamingと
    一括blockは同じ完成係数を使い、その差はblock境界が加えた実装誤差を表す。
    """
    streamed = np.asarray(streamed_output, dtype=np.complex128)
    one_block = np.asarray(one_block_output, dtype=np.complex128)
    valid = np.asarray(valid_mask, dtype=np.bool_)
    if streamed.ndim != 1 or streamed.shape != one_block.shape or streamed.shape != valid.shape:
        raise ValueError("output diagnostic arrays must share one-dimensional shape.")
    if not 0 <= zoom_start < zoom_stop <= streamed.size:
        raise ValueError("output waveform zoom range lies outside the signal.")
    real_output = np.real(streamed)
    plot_output = np.where(valid, real_output, np.nan)
    time_s = np.arange(streamed.size, dtype=np.float64) / config.fs_hz
    spectrum_start = max(zoom_start, int(round(config.training_duration_s * config.fs_hz)))
    completed = real_output[spectrum_start:]
    if not bool(np.all(valid[spectrum_start:])):
        raise ValueError("output spectrum interval contains incomplete samples.")
    frequency_hz, level_db = _one_sided_rms_spectrum(completed, config.fs_hz)
    upper_db = float(np.max(level_db)) + 3.0
    lower_db = upper_db - 120.0
    difference = np.real(streamed - one_block)

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    axes[0, 0].plot(time_s, plot_output, linewidth=0.7)
    axes[0, 0].set(
        title=f"Post-beamforming mixed output: {method_id}, target beam",
        xlabel="Time [s]",
        ylabel="Amplitude [re input RMS]",
    )
    axes[0, 1].plot(
        time_s[zoom_start:zoom_stop], real_output[zoom_start:zoom_stop], label="streaming"
    )
    axes[0, 1].plot(
        time_s[zoom_start:zoom_stop],
        np.real(one_block[zoom_start:zoom_stop]),
        linestyle="--",
        label="one block",
    )
    _draw_block_boundaries(
        axes[0, 1], zoom_start, zoom_stop, config.runtime_block_size, config.fs_hz
    )
    axes[0, 1].set(
        title="Output zoom at runtime block boundary",
        xlabel="Time [s]",
        ylabel="Amplitude [re input RMS]",
    )
    axes[0, 1].legend()
    axes[1, 0].plot(time_s[zoom_start:zoom_stop], difference[zoom_start:zoom_stop], color="tab:red")
    _draw_block_boundaries(
        axes[1, 0], zoom_start, zoom_stop, config.runtime_block_size, config.fs_hz
    )
    maximum_error = float(np.max(np.abs(difference[valid])))
    axes[1, 0].set(
        title=f"Streaming minus one-block reference; max |error|={maximum_error:.3g}",
        xlabel="Time [s]",
        ylabel="Error [re input RMS]",
    )
    axes[1, 1].plot(frequency_hz, np.maximum(level_db, lower_db))
    axes[1, 1].set(
        title="Post-beamforming mixed output spectrum after training interval",
        xlabel="Frequency [Hz]",
        ylabel="Per-bin RMS Level [dB re input RMS]",
        xlim=(0.0, config.fs_hz / 2.0),
        ylim=(lower_db, upper_db),
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_target_waveform_integrity(
    output_path: Path,
    method_id: str,
    integrity: WaveformIntegrityResult,
    config: T2aScenarioConfig,
) -> None:
    """target-only入力と位相整列後出力の波形・spectrum・残差を保存する。

    `integrity`内の波形はshape `[n_sample]`、振幅単位はinput RMS基準である。
    spectrumはper-bin RMSの`dB re input RMS`で同一軸表示する。
    """
    reference = integrity.reference_signal
    aligned = integrity.phase_aligned_output
    sample_count = reference.size
    time_s = (
        integrity.analysis_start_sample + np.arange(sample_count, dtype=np.float64)
    ) / config.fs_hz
    # target toneを少なくとも8周期表示し、過密な全区間overlayで局所歪みを隠さない。
    period_samples = max(1, int(round(config.fs_hz / config.target_frequency_hz)))
    zoom_count = min(sample_count, max(8 * period_samples, 64))
    frequency_hz, input_level_db = _one_sided_rms_spectrum(reference, config.fs_hz)
    _, output_level_db = _one_sided_rms_spectrum(aligned, config.fs_hz)
    upper_db = max(float(np.max(input_level_db)), float(np.max(output_level_db))) + 3.0
    lower_db = upper_db - 120.0
    residual = aligned - reference

    figure, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    axes[0, 0].plot(time_s[:zoom_count], reference[:zoom_count], label="input target-only")
    axes[0, 0].plot(
        time_s[:zoom_count], aligned[:zoom_count], linestyle="--", label="phase-aligned output"
    )
    axes[0, 0].set(
        title=f"Target-only waveform integrity: {method_id}",
        xlabel="Time [s]",
        ylabel="Amplitude [re input RMS]",
    )
    axes[0, 0].legend()
    axes[0, 1].plot(time_s[:zoom_count], residual[:zoom_count], color="tab:red")
    axes[0, 1].set(
        title="Phase-aligned output minus input",
        xlabel="Time [s]",
        ylabel="Residual [re input RMS]",
    )
    axes[1, 0].plot(frequency_hz, np.maximum(input_level_db, lower_db), label="input")
    axes[1, 0].plot(
        frequency_hz,
        np.maximum(output_level_db, lower_db),
        linestyle="--",
        label="phase-aligned output",
    )
    axes[1, 0].set(
        title="Target-only input/output spectrum",
        xlabel="Frequency [Hz]",
        ylabel="Per-bin RMS Level [dB re input RMS]",
        xlim=(0.0, config.fs_hz / 2.0),
        ylim=(lower_db, upper_db),
    )
    axes[1, 0].legend()
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.95,
        "\n".join(
            (
                "phase delay modulo period: "
                f"{integrity.phase_delay_samples_modulo_period:.6g} sample",
                f"output/input RMS delta: {integrity.rms_delta_db:.6g} dB",
                "correlation after phase alignment: "
                f"{integrity.correlation_after_phase_alignment:.9f}",
                f"residual RMS: {integrity.residual_rms_db_re_input_rms:.6g} dB re input RMS",
            )
        ),
        va="top",
        family="monospace",
    )
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_bl_fraz_fl(
    output_path: Path,
    beam_azimuth_deg: FloatArray,
    frequency_hz: FloatArray,
    fraz_by_method: dict[str, FloatArray],
    config: T2aScenarioConfig,
    predicted_target_aliases_deg: tuple[float, ...],
) -> None:
    """同一軸・同一level基準でBL、FL、FRAZを保存する。"""
    if len(fraz_by_method) == 1:
        # 単独方式は2×2へ詰め、比較方式が存在しない空panelを成果物へ残さない。
        figure, axes = plt.subplots(2, 2, figsize=(12.0, 9.0))
        bl_axis = axes[0, 0]
        fl_axis = axes[0, 1]
        source_bl_axis = axes[1, 0]
        fraz_axes = (axes[1, 1],)
    else:
        figure, axes = plt.subplots(2, 3, figsize=(18.0, 9.0))
        bl_axis = axes[0, 0]
        fl_axis = axes[0, 1]
        source_bl_axis = axes[0, 2]
        fraz_axes = tuple(axes[1])
    target_bin = int(np.argmin(np.abs(frequency_hz - config.target_frequency_hz)))
    target_beam = int(np.argmin(np.abs(beam_azimuth_deg - config.target_azimuth_deg)))
    for method_id, fraz in fraz_by_method.items():
        bl_axis.plot(beam_azimuth_deg, fraz[:, target_bin], label=method_id)
        fl_axis.plot(frequency_hz, fraz[target_beam], label=method_id)
        source_frequency_bl = np.maximum(
            fraz[:, target_bin],
            fraz[:, int(np.argmin(np.abs(frequency_hz - config.interferer_frequency_hz)))],
        )
        source_bl_axis.plot(beam_azimuth_deg, source_frequency_bl, label=method_id)
    bl_axis.axvline(config.target_azimuth_deg, color="black", linestyle="--")
    for alias_deg in predicted_target_aliases_deg:
        # 理論aliasは計算BLを見てから付ける説明ではなく、宣言幾何からの事前予測として描く。
        bl_axis.axvline(alias_deg, color="tab:orange", linestyle=":", alpha=0.8)
    bl_axis.set(
        title=f"BL at {frequency_hz[target_bin]:g} Hz",
        xlabel="Waiting-beam azimuth [deg]",
        ylabel="RMS Level [dB re input RMS]",
    )
    fl_axis.set(
        title=f"FL at {beam_azimuth_deg[target_beam]:g} deg",
        xlabel="Frequency [Hz]",
        ylabel="RMS Level [dB re input RMS]",
    )
    source_bl_axis.set(
        title="Source-frequency BL",
        xlabel="Waiting-beam azimuth [deg]",
        ylabel="RMS Level [dB re input RMS]",
    )
    azimuth_edges = centers_to_edges(beam_azimuth_deg)
    frequency_edges = centers_to_edges(frequency_hz)
    finite = np.concatenate([values[np.isfinite(values)] for values in fraz_by_method.values()])
    upper = float(np.max(finite))
    lower = upper - 80.0
    for axis, (method_id, fraz) in zip(fraz_axes, fraz_by_method.items(), strict=False):
        image = axis.pcolormesh(
            azimuth_edges,
            frequency_edges,
            fraz.T,
            shading="auto",
            vmin=lower,
            vmax=upper,
        )
        axis.set(
            title=f"FRAZ: {method_id}",
            xlabel="Waiting-beam azimuth [deg]",
            ylabel="Frequency [Hz]",
        )
        figure.colorbar(image, ax=axis, label="RMS Level [dB re input RMS]")
    for axis in fraz_axes[len(fraz_by_method) :]:
        # 選択方式より多い空panelを表示せず、存在しない方式の結果と誤認させない。
        axis.set_visible(False)
    for axis in (bl_axis, fl_axis, source_bl_axis):
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def run_evaluation(
    positions_path: Path,
    shading_path: Path,
    shading_frequency_step_hz: float,
    output_dir: Path,
    config: T2aScenarioConfig | None = None,
    method_ids: tuple[str, ...] = SUPPORTED_METHOD_IDS,
    review_title: str = "T2a scene_renderer streaming review pack",
) -> None:
    """MATLAB係数読込からscene生成、T2a逐次処理、評価保存まで実行する。

    Args:
        positions_path: `COE_POS`相当raw。
        shading_path: `COE_CBFSHADING`相当raw。
        shading_frequency_step_hz: shading周波数bin間隔。単位はHz。
        output_dir: review pack保存先。
        config: 省略時は再現可能な既定scenario。
        method_ids: 設計、Flow処理、評価、表示を行う方式識別子。
        review_title: `review_index.md`先頭に記録する評価名。

    Returns:
        なし。

    Raises:
        ValueError: 方式、係数、scene、設計、逐次処理の契約が不正な場合。
    """
    scenario = T2aScenarioConfig() if config is None else config
    selected_method_ids = _validate_method_ids(method_ids)
    coefficients = load_matlab_array_coefficients(
        positions_path, shading_path, shading_frequency_step_hz
    )
    if coefficients.frequency_hz[-1] < scenario.fs_hz / 2.0:
        raise ValueError("MATLAB coefficient frequencies must cover Nyquist.")
    output_dir.mkdir(parents=True, exist_ok=True)
    predicted_aliases = {
        "target": predict_uniform_subset_grating_azimuths(
            coefficients,
            scenario.target_frequency_hz,
            scenario.target_azimuth_deg,
            scenario.sound_speed_m_s,
        ),
        "interferer": predict_uniform_subset_grating_azimuths(
            coefficients,
            scenario.interferer_frequency_hz,
            scenario.interferer_azimuth_deg,
            scenario.sound_speed_m_s,
        ),
    }
    rendered = render_scenario(coefficients, scenario)
    beam_azimuth_deg = np.arange(
        0.0, 180.0 + 0.5 * scenario.beam_azimuth_step_deg, scenario.beam_azimuth_step_deg
    )
    weight_design = design_frequency_weights(
        rendered.mixed,
        coefficients,
        beam_azimuth_deg,
        scenario,
        method_ids=selected_method_ids,
    )
    component_signals = {
        "target": rendered.target,
        "interferer": rendered.interferer,
        "noise": rendered.noise,
        "mixed": rendered.mixed,
    }
    fraz: dict[str, dict[str, FloatArray]] = {name: {} for name in component_signals}
    valid_counts: dict[str, int] = {}
    frequency_hz = np.asarray(
        np.fft.rfftfreq(scenario.analysis_fft_size, d=1.0 / scenario.fs_hz),
        dtype=np.float64,
    )
    runtime_start = time.perf_counter()
    energy: dict[str, FloatArray] = {}
    # 波形完全性はmixed実入力とtarget-only無歪性を別目的で確認するため、両成分だけ保持する。
    streamed_waveforms: dict[str, dict[str, tuple[ComplexArray, BoolArray]]] = {}
    for component_id, signal in component_signals.items():
        branches, energy = _make_branches(
            weight_design.weights,
            weight_design.causal_delays_samples,
            scenario.residual_fir_tap_count,
        )
        streamed = run_streaming_flow(signal, branches, scenario.runtime_block_size)
        if component_id in ("target", "mixed"):
            streamed_waveforms[component_id] = streamed
        for method_id, (method_output, method_valid) in streamed.items():
            evaluation_valid = method_valid.copy()
            training_sample_count = int(round(scenario.training_duration_s * scenario.fs_hz))
            # 重み設計に使ったtraining区間を性能評価へ再利用せず、完成係数の後半区間だけを見る。
            evaluation_valid[:, :training_sample_count] = False
            observed_frequency_hz, level = _fraz_level_db(method_output, evaluation_valid, scenario)
            # 全component/methodは同じSTFT条件を使うため、周波数軸のずれは評価契約違反である。
            if not np.array_equal(observed_frequency_hz, frequency_hz):
                raise RuntimeError("FRAZ frequency axis changed between streaming branches.")
            fraz[component_id][method_id] = level
            valid_counts[f"{component_id}_{method_id}"] = int(
                np.count_nonzero(np.all(evaluation_valid, axis=0))
            )
    runtime_s = time.perf_counter() - runtime_start
    runtime_factor = runtime_s / (scenario.duration_s * len(component_signals))
    target_bin = int(np.argmin(np.abs(frequency_hz - scenario.target_frequency_hz)))
    interferer_bin = int(np.argmin(np.abs(frequency_hz - scenario.interferer_frequency_hz)))
    target_beam = int(np.argmin(np.abs(beam_azimuth_deg - scenario.target_azimuth_deg)))
    reference_channel = int(np.argmin(np.linalg.norm(coefficients.positions_m, axis=1)))

    # 同じ完成係数を一つのblockで適用し、分割境界が出力へ加えた差だけを直接観測する。
    one_block_branches, _ = _make_branches(
        weight_design.weights,
        weight_design.causal_delays_samples,
        scenario.residual_fir_tap_count,
    )
    one_block_mixed = run_streaming_flow(
        rendered.mixed, one_block_branches, block_size=rendered.mixed.shape[1]
    )
    waveform_integrity: dict[str, WaveformIntegrityResult] = {}
    streaming_overall_error: dict[str, float] = {}
    streaming_boundary_error: dict[str, float] = {}
    streaming_valid_match: dict[str, bool] = {}
    diagnostic_zoom: dict[str, tuple[int, int, int]] = {}
    for method_id in weight_design.weights:
        target_output, target_valid = streamed_waveforms["target"][method_id]
        mixed_output, mixed_valid = streamed_waveforms["mixed"][method_id]
        one_block_output, one_block_valid = one_block_mixed[method_id]
        waveform_integrity[method_id] = calculate_target_waveform_integrity(
            rendered.target[reference_channel],
            target_output[target_beam],
            target_valid[target_beam],
            scenario,
        )
        overall_error, boundary_error = _streaming_reference_errors(
            mixed_output[target_beam],
            one_block_output[target_beam],
            mixed_valid[target_beam],
            scenario.runtime_block_size,
        )
        streaming_overall_error[method_id] = overall_error
        streaming_boundary_error[method_id] = boundary_error
        streaming_valid_match[method_id] = bool(
            np.array_equal(mixed_valid[target_beam], one_block_valid[target_beam])
        )
        diagnostic_zoom[method_id] = _diagnostic_zoom_bounds(
            mixed_valid[target_beam],
            scenario.runtime_block_size,
            int(round(scenario.training_duration_s * scenario.fs_hz)),
        )
    guard_deg = max(10.0, 2.0 * scenario.beam_azimuth_step_deg)
    non_source = np.abs(beam_azimuth_deg - scenario.target_azimuth_deg) > guard_deg
    rows: list[dict[str, Any]] = []
    for method_id in weight_design.weights:
        target_bl = fraz["target"][method_id][:, target_bin]
        peak_index = int(np.argmax(target_bl))
        target_level = float(target_bl[target_beam])
        sidelobe_peak = float(np.max(target_bl[non_source]))
        target_power = 10.0 ** (fraz["target"][method_id][target_beam, target_bin] / 10.0)
        noise_power = 10.0 ** (fraz["noise"][method_id][target_beam, target_bin] / 10.0)
        rows.append(
            {
                "scenario": "sparse_frequency_switched_two_tone",
                "method": method_id,
                "evaluation_pattern": "sparse_array_design+fixed_beam_multi_source",
                "target_frequency_hz": frequency_hz[target_bin],
                "target_azimuth_deg": scenario.target_azimuth_deg,
                "target_peak_azimuth_deg": beam_azimuth_deg[peak_index],
                "target_peak_error_deg": abs(
                    beam_azimuth_deg[peak_index] - scenario.target_azimuth_deg
                ),
                "target_level_db_re_input_rms": target_level,
                "sidelobe_peak_db_re_mainlobe_peak": sidelobe_peak - float(np.max(target_bl)),
                "output_snr_db": 10.0
                * np.log10(
                    max(target_power, np.finfo(float).tiny) / max(noise_power, np.finfo(float).tiny)
                ),
                "interferer_level_at_target_beam_db_re_input_rms": float(
                    fraz["interferer"][method_id][target_beam, interferer_bin]
                ),
                "minimum_fir_energy_containment": float(np.min(energy[method_id])),
                "target_waveform_rms_delta_db": waveform_integrity[method_id].rms_delta_db,
                "target_waveform_correlation_after_phase_alignment": waveform_integrity[
                    method_id
                ].correlation_after_phase_alignment,
                "target_waveform_residual_rms_db_re_input_rms": waveform_integrity[
                    method_id
                ].residual_rms_db_re_input_rms,
                "target_phase_delay_samples_modulo_period": waveform_integrity[
                    method_id
                ].phase_delay_samples_modulo_period,
                "streaming_one_block_max_abs_error": streaming_overall_error[method_id],
                "streaming_boundary_max_abs_error": streaming_boundary_error[method_id],
                "streaming_valid_mask_matches_one_block": streaming_valid_match[method_id],
                "ebae_signal_count_at_target": (
                    int(weight_design.ebae_signal_count[target_bin, target_beam])
                    if method_id == "t2a_ebae"
                    else -1
                ),
                "ebae_music_peak_azimuth_deg_at_target": (
                    float(weight_design.ebae_music_peak_azimuth_deg[target_bin, target_beam])
                    if method_id == "t2a_ebae"
                    else float("nan")
                ),
                "ebae_fallback_at_target": (
                    bool(weight_design.ebae_fallback_mask[target_bin, target_beam])
                    if method_id == "t2a_ebae"
                    else False
                ),
                "runtime_factor": runtime_factor,
                "finite": bool(np.all(np.isfinite(fraz["mixed"][method_id]))),
            }
        )
    with (output_dir / "scenario_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # worst_casesは採否の自動判定ではなく、peak誤差とFIR包含率が悪い方式を先に見る索引である。
    worst_rows = sorted(
        rows,
        key=lambda row: (
            -float(row["target_peak_error_deg"]),
            float(row["minimum_fir_energy_containment"]),
        ),
    )
    with (output_dir / "worst_cases.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(worst_rows)
    npz_arrays: dict[str, Any] = {
        "azimuth_deg": beam_azimuth_deg,
        "frequency_hz": frequency_hz,
        "active_channel_count": weight_design.active_channel_count,
        "causal_integer_delays_samples": weight_design.causal_delays_samples,
        "t2a_ebae_signal_count": weight_design.ebae_signal_count,
        "t2a_ebae_music_peak_azimuth_deg": weight_design.ebae_music_peak_azimuth_deg,
        "t2a_ebae_fallback_mask": weight_design.ebae_fallback_mask,
        "diagnostic_time_s": np.arange(rendered.mixed.shape[1], dtype=np.float64) / scenario.fs_hz,
        "diagnostic_reference_channel_index": np.asarray(reference_channel, dtype=np.int64),
        "diagnostic_input_mixed_reference_channel": rendered.mixed[reference_channel],
    }
    for component_id, method_levels in fraz.items():
        for method_id, levels in method_levels.items():
            npz_arrays[f"{component_id}_{method_id}_fraz_db_re_input_rms"] = levels
    _write_input_spectrum(output_dir / "rendered_input_spectrum.png", rendered.mixed, scenario)
    first_method_id = next(iter(weight_design.weights))
    _, input_zoom_start, input_zoom_stop = diagnostic_zoom[first_method_id]
    _write_input_waveform_diagnostics(
        output_dir / "input_waveform_diagnostics.png",
        rendered.mixed,
        reference_channel,
        input_zoom_start,
        input_zoom_stop,
        scenario,
    )
    for method_id in weight_design.weights:
        mixed_output, mixed_valid = streamed_waveforms["mixed"][method_id]
        one_block_output, _ = one_block_mixed[method_id]
        _, zoom_start, zoom_stop = diagnostic_zoom[method_id]
        _write_output_waveform_diagnostics(
            output_dir / f"output_waveform_diagnostics_{method_id}.png",
            method_id,
            mixed_output[target_beam],
            one_block_output[target_beam],
            mixed_valid[target_beam],
            zoom_start,
            zoom_stop,
            scenario,
        )
        _write_target_waveform_integrity(
            output_dir / f"target_waveform_integrity_{method_id}.png",
            method_id,
            waveform_integrity[method_id],
            scenario,
        )
        integrity = waveform_integrity[method_id]
        npz_arrays[f"{method_id}_target_beam_mixed_output_real"] = np.real(
            mixed_output[target_beam]
        )
        npz_arrays[f"{method_id}_target_beam_mixed_valid_mask"] = mixed_valid[target_beam]
        npz_arrays[f"{method_id}_target_beam_mixed_one_block_real"] = np.real(
            one_block_output[target_beam]
        )
        npz_arrays[f"{method_id}_target_integrity_input"] = integrity.reference_signal
        npz_arrays[f"{method_id}_target_integrity_phase_aligned_output"] = (
            integrity.phase_aligned_output
        )
    _write_bl_fraz_fl(
        output_dir / "bl_fraz_fl.png",
        beam_azimuth_deg,
        frequency_hz,
        fraz["mixed"],
        scenario,
        predicted_aliases["target"],
    )
    source_frequency_bl = {
        method_id: np.maximum(
            fraz["mixed"][method_id][:, target_bin],
            fraz["mixed"][method_id][:, interferer_bin],
        )
        for method_id in weight_design.weights
    }
    figure, axis = plt.subplots(figsize=(10.0, 4.5))
    for method_id, levels in source_frequency_bl.items():
        axis.plot(beam_azimuth_deg, levels, label=method_id)
        npz_arrays[f"{method_id}_source_frequency_bl_db_re_input_rms"] = levels
    axis.axvline(scenario.target_azimuth_deg, color="black", linestyle="--", label="target")
    axis.axvline(scenario.interferer_azimuth_deg, color="gray", linestyle=":", label="interferer")
    axis.set(
        title="Source-frequency BL overlay",
        xlabel="Waiting-beam azimuth [deg]",
        ylabel="RMS Level [dB re input RMS]",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "source_frequency_bl_overlay.png", dpi=160)
    plt.close(figure)
    # FRAZ、波形、境界参照、source-frequency BLが全方式分完成してから一つのNPZを公開する。
    np.savez_compressed(output_dir / "plot_arrays.npz", **npz_arrays)
    metadata = {
        "scenario": scenario.__dict__,
        "positions_path": str(positions_path),
        "shading_path": str(shading_path),
        "shading_frequency_step_hz": shading_frequency_step_hz,
        "n_channel": coefficients.positions_m.shape[0],
        "active_channel_count_by_frequency": weight_design.active_channel_count.tolist(),
        "t2a_ebae_fallback_count": int(np.count_nonzero(weight_design.ebae_fallback_mask)),
        "valid_sample_counts": valid_counts,
        "runtime_s": runtime_s,
        "runtime_factor": runtime_factor,
        "level_reference": "BL/FRAZ/FL: dB re input RMS",
        "evaluation_patterns": ["sparse_array_design", "fixed_beam_multi_source"],
        "selected_method_ids": list(selected_method_ids),
        "waveform_diagnostics": {
            "input_reference_channel_index": reference_channel,
            "output_beam_azimuth_deg": float(beam_azimuth_deg[target_beam]),
            "spectrum_reference": "per-bin RMS level, dB re input RMS",
            "phase_delay_definition": "sample delay modulo one target-tone period",
            "method_metrics": {
                method_id: {
                    "target_waveform_rms_delta_db": waveform_integrity[method_id].rms_delta_db,
                    "target_waveform_correlation_after_phase_alignment": waveform_integrity[
                        method_id
                    ].correlation_after_phase_alignment,
                    "target_waveform_residual_rms_db_re_input_rms": waveform_integrity[
                        method_id
                    ].residual_rms_db_re_input_rms,
                    "target_phase_delay_samples_modulo_period": waveform_integrity[
                        method_id
                    ].phase_delay_samples_modulo_period,
                    "streaming_one_block_max_abs_error": streaming_overall_error[method_id],
                    "streaming_boundary_max_abs_error": streaming_boundary_error[method_id],
                    "streaming_valid_mask_matches_one_block": streaming_valid_match[method_id],
                }
                for method_id in weight_design.weights
            },
        },
        "predicted_uniform_subset_grating_azimuths_deg": predicted_aliases,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if selected_method_ids == ("t2a_ebae",):
        method_description = (
            "MATLAB係数の周波数別active channelとshadingを適用し、候補方位別T共分散から"
            "T2a-EBAE残差重みだけを設計、Flow処理、評価、表示した。EBAE内部の成立条件を"
            "満たさない場合に使う固定整相fallbackは安全契約として残すが、`fixed_baseline`と"
            "`t2a_mvdr`の独立branchは生成しない。比較baselineを含まないため、本pack単独で"
            "方式間の採否は判断しない。"
        )
    else:
        method_description = (
            "MATLAB係数の周波数別active channelとshadingを適用し、選択方式 "
            f"{', '.join(selected_method_ids)} を同じFlow、完成区間、表示軸で評価した。"
        )
    (output_dir / "review_index.md").write_text(
        f"# {review_title}\n\n"
        f"{method_description}\n\n"
        "- `rendered_input_spectrum.png`: 整相前target+interferer+noiseのper-bin RMS。\n"
        "- `input_waveform_diagnostics.png`: 基準channel入力の全体・block境界拡大波形とspectrum。\n"
        "- `output_waveform_diagnostics_<method>.png`: target待受beam出力、境界拡大、"
        "一括block差、spectrum。\n"
        "- `target_waveform_integrity_<method>.png`: target-only入力と位相整列後出力の"
        "波形、残差、spectrum。\n"
        "- `bl_fraz_fl.png`: 整相後mixed信号のBL、FL、FRAZ。\n"
        "- `source_frequency_bl_overlay.png`: 全source真値周波数の最大BL。\n"
        "- `scenario_summary.csv`: peak、sidelobe、SNR、FIR、波形完全性、境界、runtime観測値。\n"
        "- `worst_cases.csv`: レビュー優先順で並べた同じ観測値。自動採否には使わない。\n"
        "- `plot_arrays.npz`: 描画前配列。BL/FRAZ/FLはdB re input RMS。\n\n"
        "波形完全性はtarget-only入力の原点最近傍channelとtarget待受beam出力を比較する。"
        "単一toneの位相遅延は1周期ごとに同値なため、絶対伝搬遅延ではなく1周期を法とする。"
        "分割streamingと同じ係数を一括blockへ適用した差によりblock境界由来の不連続を確認する。\n\n"
        "本scenarioは自由音場、水平固定音源、channel独立帯域雑音である。海面・海底反射、"
        "音速プロファイル、係数更新過渡は扱わず、それらの成立性を本結果から判断しない。\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    """CLI引数を解析する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--positions-raw", type=Path, required=True, help="COE_POS相当raw")
    parser.add_argument("--shading-raw", type=Path, required=True, help="COE_CBFSHADING相当raw")
    parser.add_argument(
        "--shading-frequency-step-hz",
        type=float,
        required=True,
        help="shading列の周波数間隔[Hz]",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/beamforming/t2a_scene_renderer_streaming/review_pack"),
    )
    parser.add_argument(
        "--write-example-coefficients",
        action="store_true",
        help="実行前に疎通確認用rawを指定2ファイルへ生成する",
    )
    return parser.parse_args()


def main() -> None:
    """CLIから統合T2a評価を実行する。"""
    args = _parse_args()
    config = T2aScenarioConfig()
    if bool(args.write_example_coefficients):
        write_example_matlab_coefficients(args.positions_raw, args.shading_raw, config)
    run_evaluation(
        args.positions_raw,
        args.shading_raw,
        args.shading_frequency_step_hz,
        args.output_dir,
        config,
    )


if __name__ == "__main__":
    main()
