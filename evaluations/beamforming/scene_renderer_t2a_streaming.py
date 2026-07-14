"""scene_renderer信号へMATLAB運用係数を使うT2a逐次整相を適用する。

一つの実行で、シナリオ生成、MATLAB生成rawアレイ係数読込、候補方位別T共分散、
T2a-MVDR残差FIR、通常Pythonによるblock逐次処理、BL/FRAZ/FL評価を再現する。
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from evaluations.beamforming.scene_renderer_t2a_review_reporting import (  # noqa: E402
    ScenarioSummaryRow,
    T2aReviewContext,
    T2aReviewData,
    write_t2a_review_pack,
)
from evaluations.beamforming.scene_renderer_t2a_waveform_reporting import (  # noqa: E402
    WaveformIntegrityResult,
    calculate_streaming_reference_errors,
    calculate_target_waveform_integrity,
    select_diagnostic_zoom_bounds,
)
from spflow import StepScheduler  # noqa: E402
from spflow.beamforming.ebae import EbaeConfig, design_ebae_weights_band  # noqa: E402
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
    adaptive_weight_update_interval_s: float = 1.0
    adaptive_weight_design_items_per_cycle: int | None = None
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
        if self.adaptive_weight_update_interval_s <= 0.0:
            raise ValueError("adaptive_weight_update_interval_s must be positive.")
        if int(round(self.adaptive_weight_update_interval_s * self.fs_hz)) < 1:
            raise ValueError("adaptive weight update interval must cover at least one sample.")
        if (
            self.adaptive_weight_design_items_per_cycle is not None
            and self.adaptive_weight_design_items_per_cycle <= 0
        ):
            raise ValueError("adaptive_weight_design_items_per_cycle must be positive or None.")
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
class RuntimeBlock:
    """方式branchへ渡す入力blockを保持する。

    `data`はshape `[n_ch,n_block_sample]`、`start_sample`は元系列上の先頭sampleである。
    """

    start_sample: int
    data: FloatArray


@dataclass(frozen=True)
class BeamBlock:
    """方式branchから回収する完成状態付きbeam blockを保持する。

    `data`と`valid_mask`はshape `[n_beam,n_block_sample]`で、axis=0は待受方位、
    axis=1は時刻である。`method_id`は`fixed_baseline`、`t2a_mvdr`、`t2a_ebae`である。
    """

    method_id: str
    start_sample: int
    data: ComplexArray
    valid_mask: BoolArray


@dataclass(frozen=True)
class CompletedWeightUpdate:
    """保持した同一入力周期へ適用できる完成係数更新を保持する。

    `effective_start_sample`と`cycle_stop_sample`は新係数で処理する入力周期
    `[start,stop)`、`source_snapshot_stop_sample`は設計入力の観測終端、`version`は単調増加する
    完成版番号である。`coefficients`の各値はshape `[n_beam,n_ch,n_tap]`、
    `energy_containment`の各値はshape `[n_beam]`である。

    本結果型は完成係数と設計診断を運ぶだけで、共分散計算、FIR状態、信号適用を担わない。
    """

    effective_start_sample: int
    cycle_stop_sample: int
    source_snapshot_stop_sample: int
    version: int
    coefficients: dict[str, ComplexArray]
    energy_containment: dict[str, FloatArray]
    weight_design: FrequencyWeightDesign


@dataclass(frozen=True)
class WeightUpdateCycleResult:
    """一つの適応処理周期の区間と完成係数の有無を表す。

    `[cycle_start_sample,cycle_stop_sample)`は今回信号経路へ渡す入力周期である。
    `completed_update`は`StepScheduler`が今回全itemを完了した場合だけ存在し、未完成時は
    `None`として前回完成FIRを維持する。本結果型は周期境界を運ぶだけで信号処理を担わない。
    """

    cycle_start_sample: int
    cycle_stop_sample: int
    completed_update: CompletedWeightUpdate | None


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
        # delays[beam,ch]は候補方位ごとに各channelへ与える因果整数遅延[sample]、
        # taps[beam,ch,tap]はその整数遅延後に適用する複素残差FIR係数である。
        delays = np.asarray(causal_delays_samples, dtype=np.int64)
        taps = np.asarray(coefficients, dtype=np.complex128)
        if delays.ndim != 2 or taps.ndim != 3 or taps.shape[:2] != delays.shape:
            raise ValueError("delays and coefficients must share [n_beam, n_ch].")

        # method_idはfixed/MVDR/EBAEの出力収集先を識別する。各branchは方式ごとに
        # 独立した遅延・FIR履歴を持ち、別方式の途中状態を共有しない。
        self.method_id = method_id
        self._n_beam, self._n_ch = delays.shape

        # 状態部品は先頭軸を独立seriesとして扱うため、[beam,ch]をbeam-major順の
        # [beam*ch]へ平坦化する。delaysとtapsへ同じ変換を行い、series indexごとの
        # 「整数遅延後に対応する残差FIRを適用する」という組合せを維持する。
        self._delay = StatefulIntegerDelay(delays.reshape(-1))
        self._fir = VersionedCausalFIR(taps.reshape(self._n_beam * self._n_ch, taps.shape[2]))

    @property
    def n_beam(self) -> int:
        """このbranchが生成する待受beam数を返す。"""
        return self._n_beam

    @property
    def active_coefficient_version(self) -> int:
        """現在信号へ適用している完成FIR係数の版番号を返す。"""
        return self._fir.active_version

    def request_coefficient_update(
        self,
        coefficients: ComplexArray,
        *,
        version: int,
    ) -> None:
        """完成した全beam・channel係数を次のblock先頭での置換対象にする。

        Args:
            coefficients: 新しい因果FIR。shape `[n_beam,n_ch,n_tap]`。
            version: 現在版と予約済み版より大きい完成版番号。

        Returns:
            なし。

        Raises:
            ValueError: beam/channel/tap shape、dtype、有限性、版番号が既存FIRと異なる場合。

        境界条件:
            呼び出し時点ではactive係数を変更しない。次の`process()`先頭で全系列を
            同時に切り替え、更新中またはblock途中の係数を外部へ公開しない。
        """
        taps = np.asarray(coefficients, dtype=np.complex128)
        if taps.ndim != 3 or taps.shape[:2] != (self._n_beam, self._n_ch):
            raise ValueError("updated coefficients must have shape [n_beam, n_ch, n_tap].")
        # VersionedCausalFIRのseries軸はbeam-majorの[beam*ch]である。初期構築時と同じ
        # flatten規約を使い、更新後も各beam・channelとtap列の対応を維持する。
        self._fir.request_update(
            taps.reshape(self._n_beam * self._n_ch, taps.shape[2]),
            version=version,
        )

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
        # block.data shape: [ch,time]。同じセンサ入力を全待受beamへ分岐し、
        # beamごとに異なる整相遅延を適用できる[beam,ch,time]を作る。
        beam_channel_input = np.broadcast_to(
            block.data[np.newaxis, :, :],
            (self._n_beam, self._n_ch, block.data.shape[1]),
        ).reshape(self._n_beam * self._n_ch, block.data.shape[1])

        # delayed.data shape: [beam*ch,time]。各系列へ候補方位別の因果整数遅延を与え、
        # 到来時刻をbeam内で揃える。valid_mask=Falseは初回の遅延履歴不足sampleを表す。
        delayed = self._delay.process(np.asarray(beam_channel_input, dtype=np.float64))

        # filtered.data shape: [beam*ch,time]。整数遅延後の各channelへ、周波数重み
        # w[f,beam,ch]に対応する残差FIR応答H=conj(w)を適用する。入力valid_maskを
        # FIR履歴の完成条件と合わせ、未完成sampleが後段へ完成値として出ないようにする。
        filtered = self._fir.process(SignalBlock(delayed.data, delayed.valid_mask))

        # 平坦化していたseries軸を[beam,ch]へ戻す。filtered_dataは各channelの複素寄与、
        # filtered_validはその寄与が整数遅延とFIRの両方で完成済みかを示す。
        filtered_data = filtered.data.reshape(self._n_beam, self._n_ch, block.data.shape[1])
        filtered_valid = filtered.valid_mask.reshape(self._n_beam, self._n_ch, block.data.shape[1])

        # y[beam,time]=sum_ch H[beam,ch]*x[ch]としてchannel寄与を加算する。
        # 一つでも履歴不足のchannelがあれば、そのbeam sampleは未完成として公開しない。
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
    # 出力方式をここで確定し、不要な共分散推定や適応重み設計を実行しない。
    # selected_method_idsの順序は、戻り値weightsの方式順としても維持する。
    selected_method_ids = _validate_method_ids(method_ids)
    selected_method_set = set(selected_method_ids)

    # arrival_delays_s shape: [beam,ch]、単位s。アレイ基準点に対する物理到来時間差を、
    # 各待受方位beamと各センサchannelの組合せについて計算する。
    arrival_delays_s = _arrival_delays_s(
        coefficients.positions_m, beam_azimuth_deg, config.sound_speed_m_s
    )

    # integer_offsets shape: [beam,ch]、単位sample。物理遅延を最近傍sampleへ量子化し、
    # 整数遅延器が担当する成分q=round(fs*tau)と、残差FIRが担当する小数成分へ分ける。
    integer_offsets = np.rint(arrival_delays_s * config.fs_hz).astype(np.int64)

    # causal_delays[beam,ch] shape: [beam,ch]、単位sample。T共分散のx[n+q]と
    # 因果bufferのx[n-d]を対応させるため、beam内の共通遅延を加えてd=max_ch(q)-qとする。
    # この共通遅延はbeam出力時刻だけを後方へ移し、channel間の相対遅延を変えない。
    causal_delays = np.max(integer_offsets, axis=1, keepdims=True) - integer_offsets

    # frequency_hzはrFFTの非負周波数軸[n_frequency]。weightsのaxis順は
    # [frequency,beam,ch]に固定し、周波数ごとのactive channel表と直接対応させる。
    frequency_hz = np.fft.rfftfreq(config.analysis_fft_size, d=1.0 / config.fs_hz)
    n_frequency = frequency_hz.size
    n_beam = beam_azimuth_deg.size
    n_ch = coefficients.positions_m.shape[0]

    # inactive channelは重み0のまま残す。これにより周波数ごとにactive数が変化しても、
    # 全方式の出力shapeを[n_frequency,n_beam,n_ch]へ固定できる。
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
    # snapshots_by_beam[beam] shape: [frame,frequency,ch]。
    needs_covariance = bool({"t2a_mvdr", "t2a_ebae"} & selected_method_set)
    snapshots_by_beam = (
        [
            _candidate_snapshots(training_signal, causal_delays[beam_index], config)
            for beam_index in range(n_beam)
        ]
        if needs_covariance
        else []
    )

    # 外側をfrequency、内側をbeamとし、同じ周波数では全beamへ同一のactive channelと
    # shading表を使う。共分散と制約ベクトルは各frequency・beamの組合せごとに設計する。
    for frequency_index, frequency in enumerate(frequency_hz):
        shading, active = coefficients.table_at(float(frequency))
        active_indices = np.flatnonzero(active)
        active_count[frequency_index] = float(active_indices.size)
        for beam_index in range(n_beam):
            is_real_spectrum_boundary = frequency_index in (0, n_frequency - 1)

            # physical_tauとinteger_tauはactive channelだけのshape [n_active_ch]。
            # 両者の差tau-q/fsが整数整相後に残る小数遅延であり、残差FIRが補償する。
            physical_tau = arrival_delays_s[beam_index, active_indices]
            integer_tau = integer_offsets[beam_index, active_indices] / config.fs_hz

            # residual_constraint a_res shape: [n_active_ch]。
            # 整数遅延後の残差steeringはD a=exp(-j2πf(tau-q/fs))であり、
            # 最終重みは無歪条件w^H a_res=1を満たすように正規化する。
            residual_constraint = np.exp(
                -1j * 2.0 * np.pi * frequency * (physical_tau - integer_tau)
            )
            channel_shading = shading[active_indices]

            # fixed_active=a_res/(a_res^H a_res)は残差座標上の固定整相重み。
            # 適応設計が成立しない場合にも公開できる、安全側の完成重みとして使用する。
            fixed_active = residual_constraint / np.vdot(residual_constraint, residual_constraint)

            # このfrequency・beamで成立した方式別の未shading重みだけを一時保持する。
            # weights本体への格納は、全方式共通のshadingと無歪正規化を適用した後に行う。
            unshaded_by_method: dict[str, ComplexArray] = {}
            if "fixed_baseline" in selected_method_set:
                unshaded_by_method["fixed_baseline"] = np.asarray(fixed_active, dtype=np.complex128)
            if is_real_spectrum_boundary:
                # DC/Nyquistは実FIRを作るrFFTのHermitian境界であり、独立な負周波数対を
                # 持たない。この境界では適応解を作らず、完成済みfixed重みを採用する。
                if "t2a_mvdr" in selected_method_set:
                    unshaded_by_method["t2a_mvdr"] = np.asarray(fixed_active, dtype=np.complex128)
                if "t2a_ebae" in selected_method_set:
                    unshaded_by_method["t2a_ebae"] = np.asarray(fixed_active, dtype=np.complex128)
                    ebae_music_peak_deg[frequency_index, beam_index] = beam_azimuth_deg[beam_index]
            elif needs_covariance:
                # snapshots shape: [frame,ch]。候補beamへ整数整相済みの同一周波数binを取り出す。
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

                # active_snapshot shape: [selected_frame,n_active_ch]。EBAEを比較に含む場合は
                # 宣言snapshot数L=M^2と共分散の実平均数を一致させ、方式間の条件差を作らない。
                active_snapshot = snapshots[:selected_snapshot_count, active_indices]

                # covariance shape: [n_active_ch,n_active_ch]。
                # R=E[x x^H]としてframe axisを平均し、整数整相後のchannel間共分散を得る。
                covariance = np.einsum(
                    "fc,fd->cd", active_snapshot, active_snapshot.conj(), optimize=True
                ) / float(active_snapshot.shape[0])
                if "t2a_mvdr" in selected_method_set:
                    unshaded_by_method["t2a_mvdr"] = _loaded_mvdr_weight(
                        covariance, residual_constraint, config.diagonal_loading_ratio
                    )
                if "t2a_ebae" in selected_method_set:
                    # residual_scan shape: [n_active_ch,n_beam]。D_b a(phi)により、現在の
                    # 候補beam bの整数遅延後座標へ全scan方位phiのsteeringを移す。
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

            # 方式固有の設計後に、MATLABのchannel shading gを実信号経路へ反映する。
            # y=w^H x規約では信号へgを掛ける操作を重み側のconj(g)倍として表す。
            for method_id, unshaded in unshaded_by_method.items():
                shaded = channel_shading.conj() * unshaded
                denominator = np.vdot(shaded, residual_constraint)
                if abs(denominator) <= np.finfo(np.float64).eps:
                    # shading後に無歪正規化できない場合は、不完全な適応値でなくCBFを採用する。
                    shaded = fixed_active
                    denominator = np.vdot(shaded, residual_constraint)

                # d=shaded^H a_resに対してw=shaded/conj(d)とすると、
                # w^H a_res=d/d=1となる。active位置だけへ格納し、inactive位置は0を保つ。
                weights[method_id][frequency_index, beam_index, active_indices] = (
                    shaded / denominator.conjugate()
                )

    # ここで返すweightsは全frequency・beamの設計が完了した不変snapshotであり、
    # streaming実行中に部分更新しない。EBAE診断量も同じ軸で対応付けて返す。
    return FrequencyWeightDesign(
        weights=weights,
        causal_delays_samples=np.asarray(causal_delays, dtype=np.int64),
        active_channel_count=active_count,
        ebae_signal_count=ebae_signal_count,
        ebae_music_peak_azimuth_deg=ebae_music_peak_deg,
        ebae_fallback_mask=ebae_fallback,
    )


def design_initial_fixed_weights(
    reference_signal: FloatArray,
    coefficients: MatlabArrayCoefficients,
    beam_azimuth_deg: FloatArray,
    config: T2aScenarioConfig,
    method_ids: tuple[str, ...] = SUPPORTED_METHOD_IDS,
) -> FrequencyWeightDesign:
    """適応係数が未完成の場合に使う固定整相重みを方式別に作る。

    Args:
        reference_signal: channel数を確定する入力。shape `[n_ch,n_sample]`、単位input RMS。
        coefficients: 物理位置、周波数別active channel、複素shading。
        beam_azimuth_deg: 待受方位。shape `[n_beam]`、単位deg。
        config: sampling、FFT、遅延、shading条件。
        method_ids: runtimeへ作る方式識別子。

    Returns:
        全方式IDへ同じ完成fixed重みを割り当てた`FrequencyWeightDesign`。

    Raises:
        ValueError: 方式、入力shape、係数または方位条件が不正な場合。

    境界条件:
        完成更新を作れない短い終端入力や設計異常では、MVDR/EBAEの部分値を公開せず、
        無歪条件を満たす固定整相重みを安全側の完成値として使えるようにする。
    """
    selected_method_ids = _validate_method_ids(method_ids)
    fixed_design = design_frequency_weights(
        reference_signal,
        coefficients,
        beam_azimuth_deg,
        config,
        method_ids=("fixed_baseline",),
    )
    fixed_weights = fixed_design.weights["fixed_baseline"]
    return FrequencyWeightDesign(
        weights={method_id: fixed_weights.copy() for method_id in selected_method_ids},
        causal_delays_samples=fixed_design.causal_delays_samples.copy(),
        active_channel_count=fixed_design.active_channel_count.copy(),
        ebae_signal_count=fixed_design.ebae_signal_count.copy(),
        ebae_music_peak_azimuth_deg=fixed_design.ebae_music_peak_azimuth_deg.copy(),
        ebae_fallback_mask=fixed_design.ebae_fallback_mask.copy(),
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
    # rFFT bin数Kは偶数長実FFTのN/2+1に対応するため、元の時間長はN=2(K-1)。
    # weightsのaxis順は[frequency,beam,ch]であり、各beamの全channelを同じN点系でFIR化する。
    n_fft = 2 * (weights.shape[0] - 1)
    n_beam, n_ch = weights.shape[1:]

    # coefficients[beam,ch,tap]はstreaming FIRへ渡す因果tap、energy_ratio[beam]は
    # 元のN点周期インパルス応答energyのうち、選択した共通tap区間に残る割合である。
    coefficients = np.empty((n_beam, n_ch, tap_count), dtype=np.complex128)
    energy_ratio = np.empty(n_beam, dtype=np.float64)
    for beam_index in range(n_beam):
        # weights[:,beam,:] shape: [frequency,ch]。設計規約y=w^H xに対し、実際に
        # 信号へ畳み込む伝達関数はH=conj(w)なので、共役後にirFFTしてh[n,ch]を得る。
        # impulse shape: [n_fft,ch]。axis=0が周期時間index、axis=1がchannelである。
        impulse = np.fft.irfft(weights[:, beam_index, :].conj(), n=n_fft, axis=0)

        # 全channelで同じtap開始位置を選ばないと、channelごとに異なる時間shiftが加わり、
        # 設計済みの相対位相が崩れる。そこで各時間indexのchannel合計energyを評価する。
        tap_energy_by_time = np.sum(impulse**2, axis=1)

        # irFFT応答はN点周期列なので、末尾から先頭へ跨ぐtap窓も候補に含める。
        # 先頭tap_count-1点を連結し、長さtap_countの全N個の循環区間energyを計算する。
        extended_energy = np.concatenate((tap_energy_by_time, tap_energy_by_time[: tap_count - 1]))
        window_energy = np.convolve(extended_energy, np.ones(tap_count), mode="valid")[:n_fft]

        # 最大energy区間を全channel共通で選び、循環順序を保ったままtap index 0..tap_count-1
        # へ並べ直す。この並べ替えによりVersionedCausalFIRへ渡せる因果係数となる。
        start = int(np.argmax(window_energy))
        indices = (start + np.arange(tap_count)) % n_fft
        coefficients[beam_index] = np.asarray(impulse[indices].T, dtype=np.complex128)

        # energy_ratioはFIR打切りによる応答欠落をbeam単位で診断する値。
        # 無信号重みでも0除算しないよう、分母にfloat最小正規化値を適用する。
        energy_ratio[beam_index] = float(
            window_energy[start] / max(float(np.sum(tap_energy_by_time)), np.finfo(float).tiny)
        )
    return coefficients, energy_ratio


@dataclass(frozen=True)
class _WeightDesignSnapshot:
    """時間分割中に変更されないrolling信号snapshotと世代を保持する。"""

    training_signal: FloatArray
    generation: int
    observed_through_sample: int

    def __post_init__(self) -> None:
        """shapeと有限性を検証し、所有する読み取り専用copyへ固定する。"""
        signal = np.asarray(self.training_signal, dtype=np.float64)
        if signal.ndim != 2 or signal.shape[0] == 0 or signal.shape[1] == 0:
            raise ValueError("training_signal must have shape [n_ch, n_sample].")
        if not bool(np.all(np.isfinite(signal))):
            raise ValueError("training_signal must contain only finite values.")
        if self.generation < 0:
            raise ValueError("generation must be non-negative.")
        if self.observed_through_sample <= 0:
            raise ValueError("observed_through_sample must be positive.")
        owned_signal = signal.copy()
        owned_signal.flags.writeable = False
        object.__setattr__(self, "training_signal", owned_signal)


@dataclass(frozen=True)
class _ScheduledWeightDesign:
    """StepSchedulerが全item完了後だけ公開する重み・FIR完成値を保持する。"""

    weight_design: FrequencyWeightDesign
    coefficients: dict[str, ComplexArray]
    energy_containment: dict[str, FloatArray]
    source_generation: int
    source_snapshot_stop_sample: int


class _OnlineWeightDesignCallback:
    """重み設計と方式別FIR化をStepSchedulerのitemへ分けるcallbackである。

    最初のitemで一つのrolling snapshotから全方式の周波数重みを設計し、後続itemで
    方式ごとの残差FIRを生成する。全item完了時だけ`_ScheduledWeightDesign`を公開し、
    途中の周波数重みや一部方式だけのFIRを信号経路へ渡さない。
    """

    def __init__(
        self,
        coefficients: MatlabArrayCoefficients,
        beam_azimuth_deg: FloatArray,
        config: T2aScenarioConfig,
        method_ids: tuple[str, ...],
    ) -> None:
        """固定設計条件と空の作業領域を保持する。"""
        self._coefficients = coefficients
        self._beam_azimuth_deg = np.asarray(beam_azimuth_deg, dtype=np.float64).copy()
        self._config = config
        self._method_ids = method_ids
        self._work_design: FrequencyWeightDesign | None = None
        self._work_coefficients: dict[str, ComplexArray] = {}
        self._work_energy: dict[str, FloatArray] = {}
        self._previous: _ScheduledWeightDesign | None = None

    def signature(self, inputs: _WeightDesignSnapshot) -> int:
        """異なるrolling snapshotの部分結果を混ぜないためgenerationを返す。"""
        return inputs.generation

    def on_start(self, inputs: _WeightDesignSnapshot) -> tuple[str | None, ...]:
        """重み設計itemと方式別FIR化itemを順序付きで生成する。"""
        del inputs
        self.reset_cycle()
        # Noneは全方式共通の周波数重み設計、後続文字列は方式別FIR化を表す。
        return (None, *self._method_ids)

    def on_step(self, item: str | None, inputs: _WeightDesignSnapshot) -> None:
        """一つの設計itemを処理し、未完成作業領域へだけ保存する。"""
        if item is None:
            self._work_design = design_frequency_weights(
                inputs.training_signal,
                self._coefficients,
                self._beam_azimuth_deg,
                self._config,
                method_ids=self._method_ids,
            )
            return
        design = self._work_design
        if design is None:
            raise RuntimeError("frequency weights must complete before FIR realization.")
        fir_coefficients, energy_ratio = realize_residual_fir(
            design.weights[item],
            self._config.residual_fir_tap_count,
        )
        self._work_coefficients[item] = fir_coefficients
        self._work_energy[item] = energy_ratio

    def on_finish(
        self,
        inputs: _WeightDesignSnapshot,
        done: bool,
    ) -> _ScheduledWeightDesign | None:
        """全item完了時だけ一つの完成値へ昇格し、未完成時は前回完成値を返す。"""
        if done:
            design = self._work_design
            if design is None or set(self._work_coefficients) != set(self._method_ids):
                raise RuntimeError("all weight-design items must complete before publication.")
            # 方式別FIRがすべて揃った後に参照を一度で置換し、部分世代を公開しない。
            self._previous = _ScheduledWeightDesign(
                weight_design=design,
                coefficients=dict(self._work_coefficients),
                energy_containment=dict(self._work_energy),
                source_generation=inputs.generation,
                source_snapshot_stop_sample=inputs.observed_through_sample,
            )
        return self._previous

    def reset_cycle(self) -> None:
        """進行中snapshotの作業値だけを破棄し、前回完成値を保持する。"""
        self._work_design = None
        self._work_coefficients = {}
        self._work_energy = {}


class OnlineT2aWeightUpdater:
    """mixed入力の移動training窓から運用中の完成T2a係数を生成する。

    入力は時系列順の`RuntimeBlock`、出力は更新時刻だけ生成される
    `CompletedWeightUpdate | None`である。最新`training_duration_s`秒を保持し、最初の
    warm-up完了後は`adaptive_weight_update_interval_s`秒ごとに重みと残差FIRを再設計する。

    本クラスは共分散窓、更新時刻、`StepScheduler`による計算時間分割、完成版の生成を担う。
    信号へのFIR適用と方式branchの履歴は責務に含めない。設計が複数周期へまたがる間は
    前回完成FIRを維持し、全item完成後だけ`VersionedCausalFIR`へ渡す。
    """

    def __init__(
        self,
        coefficients: MatlabArrayCoefficients,
        beam_azimuth_deg: FloatArray,
        config: T2aScenarioConfig,
        method_ids: tuple[str, ...],
    ) -> None:
        """更新周期と空のrolling training窓を初期化する。

        Args:
            coefficients: 物理位置、周波数別active channel、複素shading。
            beam_azimuth_deg: 待受方位。shape `[n_beam]`、単位deg。
            config: sampling、training窓、更新間隔、FFT、FIR条件。
            method_ids: 設計する方式識別子。fixed単独では更新対象がない。

        Raises:
            ValueError: 方式が不正、または適応方式を一つも含まない場合。
        """
        selected_method_ids = _validate_method_ids(method_ids)
        adaptive_method_ids = tuple(
            method_id for method_id in selected_method_ids if method_id != "fixed_baseline"
        )
        if len(adaptive_method_ids) == 0:
            raise ValueError("online updater requires at least one adaptive method.")
        self._beam_azimuth_deg = np.asarray(beam_azimuth_deg, dtype=np.float64).copy()
        self._adaptive_method_ids = adaptive_method_ids
        self._training_sample_count = int(round(config.training_duration_s * config.fs_hz))
        self._update_interval_samples = int(
            round(config.adaptive_weight_update_interval_s * config.fs_hz)
        )
        self._next_update_sample = self._training_sample_count
        # 初回training窓のうち、更新間隔より前の区間には完成適応係数が存在しない。
        # 最初の完成版はtraining窓末尾の1更新周期だけへ適用し、それ以前は固定整相を使う。
        self._current_cycle_start_sample = max(
            0,
            self._training_sample_count - self._update_interval_samples,
        )
        self._expected_start_sample = 0
        self._rolling_signal = np.empty((coefficients.positions_m.shape[0], 0), dtype=np.float64)
        self._next_version = 1
        self._completed_updates: list[CompletedWeightUpdate] = []
        self._latest_weight_design: FrequencyWeightDesign | None = None
        self._weight_scheduler = StepScheduler(
            _OnlineWeightDesignCallback(
                coefficients,
                self._beam_azimuth_deg,
                config,
                selected_method_ids,
            ),
            items_per_cycle=config.adaptive_weight_design_items_per_cycle,
        )
        self._designing_snapshot: _WeightDesignSnapshot | None = None
        self._waiting_snapshot: _WeightDesignSnapshot | None = None
        self._next_snapshot_generation = 0

    @property
    def next_update_sample(self) -> int:
        """次にtraining窓を確定する元系列上のsample位置を返す。"""
        return self._next_update_sample

    @property
    def completed_updates(self) -> tuple[CompletedWeightUpdate, ...]:
        """時系列順に完成した係数更新の不変tupleを返す。"""
        return tuple(self._completed_updates)

    @property
    def latest_weight_design(self) -> FrequencyWeightDesign | None:
        """最後に全周波数・全beamが完成した設計を返す。未更新時は`None`。"""
        return self._latest_weight_design

    def process(self, block: RuntimeBlock) -> WeightUpdateCycleResult | None:
        """mixed入力blockをrolling窓へ加え、更新境界で設計を一段進める。

        Args:
            block: mixed channel入力。shape `[n_ch,n_block_sample]`、単位input RMS。

        Returns:
            block終端が更新境界なら周期区間と完成更新の有無を持つ固定型結果、
            境界以外は`None`。

        Raises:
            ValueError: channel shape、時系列連続性、更新境界を跨ぐblockが不正な場合。

        境界条件:
            更新境界を跨ぐblockは受け付けない。呼び出し側は更新周期の入力を保持し、
            境界で完成した係数を保持済みの同一周期へ適用する。
        """
        values = np.asarray(block.data, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] != self._rolling_signal.shape[0]:
            raise ValueError("online update block must have shape [n_ch, n_block_sample].")
        if block.start_sample != self._expected_start_sample:
            raise ValueError("online update blocks must be contiguous and chronological.")
        stop_sample = block.start_sample + values.shape[1]
        if stop_sample > self._next_update_sample:
            raise ValueError("online update block must not cross the next update boundary.")

        # rolling_signal shape: [ch,time]。運用memoryをtraining窓長へ制限し、更新ごとに
        # 最新training_duration_s秒だけから共分散snapshotを再構成する。
        self._rolling_signal = np.concatenate((self._rolling_signal, values), axis=1)
        if self._rolling_signal.shape[1] > self._training_sample_count:
            self._rolling_signal = self._rolling_signal[:, -self._training_sample_count :].copy()
        self._expected_start_sample = stop_sample
        if stop_sample != self._next_update_sample:
            return None
        if self._rolling_signal.shape[1] != self._training_sample_count:
            raise RuntimeError("complete update boundary must contain one full training window.")

        # snapshotは所有copyを持つため、次のrolling更新で時間分割中の入力が変化しない。
        newest_snapshot = _WeightDesignSnapshot(
            self._rolling_signal,
            self._next_snapshot_generation,
            stop_sample,
        )
        self._next_snapshot_generation += 1
        if self._designing_snapshot is None:
            self._designing_snapshot = newest_snapshot
        else:
            # 設計中世代は完了まで固定し、新着snapshotは最新1件だけ待機させる。
            self._waiting_snapshot = newest_snapshot

        designing_snapshot = self._designing_snapshot
        if designing_snapshot is None:
            raise RuntimeError("designing snapshot must exist at an update boundary.")
        scheduler_result = self._weight_scheduler.process_result(designing_snapshot)
        completed: CompletedWeightUpdate | None = None
        if scheduler_result.updated:
            scheduled_design = scheduler_result.value
            if scheduled_design is None:
                raise RuntimeError("completed scheduler cycle must publish a weight design.")
            completed_coefficients = {
                method_id: scheduled_design.coefficients[method_id]
                for method_id in self._adaptive_method_ids
            }
            completed = CompletedWeightUpdate(
                effective_start_sample=self._current_cycle_start_sample,
                cycle_stop_sample=stop_sample,
                source_snapshot_stop_sample=scheduled_design.source_snapshot_stop_sample,
                version=self._next_version,
                coefficients=completed_coefficients,
                energy_containment=scheduled_design.energy_containment,
                weight_design=scheduled_design.weight_design,
            )
            self._completed_updates.append(completed)
            self._latest_weight_design = scheduled_design.weight_design
            self._next_version += 1
            self._designing_snapshot = self._waiting_snapshot
            self._waiting_snapshot = None

        cycle_result = WeightUpdateCycleResult(
            cycle_start_sample=self._current_cycle_start_sample,
            cycle_stop_sample=stop_sample,
            completed_update=completed,
        )
        self._current_cycle_start_sample = stop_sample
        self._next_update_sample += self._update_interval_samples
        return cycle_result


def run_streaming_beam_branches(
    signal: FloatArray,
    branches: list[StreamingBeamBranch],
    block_size: int,
    *,
    coefficient_updater: OnlineT2aWeightUpdater | None = None,
    coefficient_updates: tuple[CompletedWeightUpdate, ...] = (),
) -> dict[str, tuple[ComplexArray, BoolArray]]:
    """入力をblock分割し、同じblockを方式別beam branchへ適用する。

    Args:
        signal: channel入力。shape `[n_ch,n_sample]`、input RMS基準。
        branches: 方式別の状態付き整数delay・FIR処理branch。
        block_size: 入力分割長。単位sample。
        coefficient_updater: mixed入力から更新をオンライン生成するruntime。省略時は生成しない。
        coefficient_updates: 別実行で完成済みの更新列。同じ時刻で成分分離信号へ再適用する。

    Returns:
        方式ごとの`(output, valid_mask)`。双方shape `[n_beam,n_sample]`。

    Raises:
        ValueError: branchが空、方式IDが重複する、beam数が一致しない、block長、入力shape、
            更新時刻、または更新runtimeの組合せが不正な場合。

    信号処理上の位置づけ:
        fixed、T2a-MVDR、T2a-EBAEを並列な方式経路として適用する。
        mixed実行では更新周期の入力を保持し、係数経路を先に完了してから同一周期を信号経路へ
        渡す。StepSchedulerが未完成の周期は前回完成係数を使う。成分分離実行ではmixedから
        得た更新列を同じsample区間へ再適用する。
        本処理では各branchが常に一つの完成`BeamBlock`を返し、処理レート差もないため、
        `Flow`による0/1/many伝播は使わない。block分割、方式分岐、時刻位置への収集を
        通常のPython制御構文で明示する。
    """
    if len(branches) == 0:
        raise ValueError("branches must contain at least one streaming branch.")
    if signal.ndim != 2 or block_size <= 0:
        raise ValueError("signal must have shape [n_ch, n_sample] and block_size must be positive.")
    if coefficient_updater is not None and len(coefficient_updates) > 0:
        raise ValueError("coefficient_updater and coefficient_updates are mutually exclusive.")
    method_ids = [branch.method_id for branch in branches]
    if len(set(method_ids)) != len(method_ids):
        raise ValueError("streaming branch method_id values must be unique.")
    n_beam = branches[0].n_beam
    if any(branch.n_beam != n_beam for branch in branches[1:]):
        raise ValueError("all streaming branches must have the same n_beam.")
    update_samples = [update.effective_start_sample for update in coefficient_updates]
    if update_samples != sorted(set(update_samples)):
        raise ValueError("coefficient update samples must be unique and strictly increasing.")
    if any(sample < 0 or sample >= signal.shape[1] for sample in update_samples):
        raise ValueError("coefficient updates must lie inside the processed signal interval.")
    if any(
        update.cycle_stop_sample <= update.effective_start_sample
        or update.cycle_stop_sample > signal.shape[1]
        for update in coefficient_updates
    ):
        raise ValueError("coefficient update cycles must be non-empty and lie inside the signal.")
    if any(
        current.cycle_stop_sample > following.effective_start_sample
        for current, following in zip(coefficient_updates, coefficient_updates[1:], strict=False)
    ):
        raise ValueError("coefficient update cycles must not overlap and must be chronological.")
    branch_by_method = {branch.method_id: branch for branch in branches}
    if any(
        method_id not in branch_by_method
        for update in coefficient_updates
        for method_id in update.coefficients
    ):
        raise ValueError("coefficient update method must have a matching streaming branch.")
    output = {
        branch.method_id: np.empty((n_beam, signal.shape[1]), dtype=np.complex128)
        for branch in branches
    }
    valid = {
        branch.method_id: np.empty((n_beam, signal.shape[1]), dtype=np.bool_) for branch in branches
    }

    start = 0
    update_input_start = 0
    replay_update_index = 0
    pending_online_cycle: WeightUpdateCycleResult | None = None
    while start < signal.shape[1]:
        segment_stop = signal.shape[1]
        if coefficient_updater is not None:
            # 係数経路は更新周期のmixed入力を先に保持する。周期終端で重みとFIRが完成してから、
            # 同じ[start,stop)入力周期を下の信号経路へ渡すため、出力遅延は最大1更新周期となる。
            while update_input_start < signal.shape[1] and pending_online_cycle is None:
                update_input_stop = min(
                    update_input_start + block_size,
                    signal.shape[1],
                    coefficient_updater.next_update_sample,
                )
                pending_online_cycle = coefficient_updater.process(
                    RuntimeBlock(
                        update_input_start,
                        signal[:, update_input_start:update_input_stop],
                    )
                )
                update_input_start = update_input_stop
            if pending_online_cycle is not None:
                if pending_online_cycle.cycle_start_sample < start:
                    raise RuntimeError(
                        "weight-update cycle must not precede the pending signal cycle."
                    )
                if pending_online_cycle.cycle_start_sample > start:
                    # 初回training窓の先頭側にはまだ適応完成版がないため、固定整相で公開する。
                    segment_stop = pending_online_cycle.cycle_start_sample
                else:
                    completed_update = pending_online_cycle.completed_update
                    if completed_update is not None:
                        for method_id, updated_coefficients in (
                            completed_update.coefficients.items()
                        ):
                            branch_by_method[method_id].request_coefficient_update(
                                updated_coefficients,
                                version=completed_update.version,
                            )
                    # scheduler未完成なら係数予約を行わず、前回完成FIRで同じ周期を処理する。
                    segment_stop = pending_online_cycle.cycle_stop_sample
                    pending_online_cycle = None
        elif replay_update_index < len(coefficient_updates):
            replay_update = coefficient_updates[replay_update_index]
            if replay_update.effective_start_sample == start:
                # mixedで完成した係数を同一入力周期へ再適用する。成分別に係数を再設計せず、
                # target+interferer+noiseの線形分解条件と版時刻を維持する。
                for method_id, updated_coefficients in replay_update.coefficients.items():
                    branch_by_method[method_id].request_coefficient_update(
                        updated_coefficients,
                        version=replay_update.version,
                    )
                segment_stop = replay_update.cycle_stop_sample
                replay_update_index += 1
            elif replay_update.effective_start_sample > start:
                segment_stop = replay_update.effective_start_sample

        # 完成FIRを予約した後に、対応する同一入力周期を通常のruntime blockへ分けて処理する。
        # 各branchのdelay/FIR履歴はbranch自身が保持し、方式間では共有しない。
        for block_start in range(start, segment_stop, block_size):
            block_stop = min(block_start + block_size, segment_stop)
            runtime_block = RuntimeBlock(block_start, signal[:, block_start:block_stop])
            for branch in branches:
                completed_block = branch.process(runtime_block)

                # branch出力を元系列上の同じ時刻位置へ戻す。初回履歴不足sampleの完成状態は
                # valid_maskを保存し、後段のBL/FRAZ/FL評価から除外できるようにする。
                completed_stop = completed_block.start_sample + completed_block.data.shape[1]
                output[completed_block.method_id][
                    :, completed_block.start_sample : completed_stop
                ] = completed_block.data
                valid[completed_block.method_id][
                    :, completed_block.start_sample : completed_stop
                ] = completed_block.valid_mask
        start = segment_stop
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
        method_ids: 設計、逐次処理、評価、表示を行う方式識別子。
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
    # 適応係数が完成しない短い終端や異常時にも公開可能な固定整相重みを初期値にする。
    # 通常はtraining窓を保持し、mixed入力から完成した係数を同じ入力周期へ適用する。
    initial_weight_design = design_initial_fixed_weights(
        rendered.mixed,
        coefficients,
        beam_azimuth_deg,
        scenario,
        method_ids=selected_method_ids,
    )
    review_data, latest_weight_design = _evaluate_streaming_scenario(
        scenario=scenario,
        coefficients=coefficients,
        rendered=rendered,
        beam_azimuth_deg=np.asarray(beam_azimuth_deg, dtype=np.float64),
        weight_design=initial_weight_design,
    )
    # 評価計算が全方式分完成してからreporting境界へ渡し、途中結果を成果物へ公開しない。
    review_context = T2aReviewContext(
        scenario=scenario,
        scenario_metadata=dict(scenario.__dict__),
        selected_method_ids=selected_method_ids,
        review_title=review_title,
        positions_path=positions_path,
        shading_path=shading_path,
        shading_frequency_step_hz=shading_frequency_step_hz,
        n_channel=coefficients.positions_m.shape[0],
        predicted_aliases_deg=predicted_aliases,
        rendered_mixed=rendered.mixed,
        active_channel_count=latest_weight_design.active_channel_count,
        causal_delays_samples=latest_weight_design.causal_delays_samples,
        ebae_signal_count=latest_weight_design.ebae_signal_count,
        ebae_music_peak_azimuth_deg=latest_weight_design.ebae_music_peak_azimuth_deg,
        ebae_fallback_mask=latest_weight_design.ebae_fallback_mask,
    )
    write_t2a_review_pack(output_dir, review_context, review_data)


def _evaluate_streaming_scenario(
    *,
    scenario: T2aScenarioConfig,
    coefficients: MatlabArrayCoefficients,
    rendered: RenderedComponents,
    beam_azimuth_deg: FloatArray,
    weight_design: FrequencyWeightDesign,
) -> tuple[T2aReviewData, FrequencyWeightDesign]:
    """完成したsceneと重みを逐次処理し、固定型の評価結果を返す。

    Args:
        scenario: sampling、source、FFT、FIR、block条件。
        coefficients: 検証済み位置・周波数別active channel・shading。
        rendered: target、interferer、noise、mixed。各shapeは[n_ch,n_sample]。
        beam_azimuth_deg: 待受方位。shape [n_beam]、単位deg。
        weight_design: 完成周波数重みとEBAE診断量。

    Returns:
        `(review_data, latest_weight_design)`。前者は全componentのFRAZ、波形完全性、
        block境界、方式別指標、後者は最後に完成したオンライン設計診断である。
        本関数はファイルを保存せず、reporting側が直列化する。

    Raises:
        ValueError: 完成sampleまたは波形評価の契約が不正な場合。
        RuntimeError: component間でFRAZ周波数軸が変化した場合。
    """
    # mixedを先に実行して完成係数の更新時刻列を確定し、各分離成分へ同じ版を再適用する。
    # target/noiseごとに適応設計をやり直すと、線形な成分分解評価が成立しない。
    component_signals = {
        "mixed": rendered.mixed,
        "target": rendered.target,
        "interferer": rendered.interferer,
        "noise": rendered.noise,
    }
    fraz: dict[str, dict[str, FloatArray]] = {name: {} for name in component_signals}
    valid_counts: dict[str, int] = {}
    frequency_hz = np.asarray(
        np.fft.rfftfreq(scenario.analysis_fft_size, d=1.0 / scenario.fs_hz),
        dtype=np.float64,
    )
    runtime_start = time.perf_counter()
    energy: dict[str, FloatArray] = {}
    report_energy: dict[str, FloatArray] = {}
    coefficient_updates: tuple[CompletedWeightUpdate, ...] = ()
    latest_weight_design = weight_design
    adaptive_method_ids = tuple(
        method_id for method_id in weight_design.weights if method_id != "fixed_baseline"
    )
    # 波形完全性はmixed実入力とtarget-only無歪性を別目的で確認するため、両成分だけ保持する。
    streamed_waveforms: dict[str, dict[str, tuple[ComplexArray, BoolArray]]] = {}
    for component_id, signal in component_signals.items():
        branches, energy = _make_branches(
            weight_design.weights,
            weight_design.causal_delays_samples,
            scenario.residual_fir_tap_count,
        )
        if len(report_energy) == 0:
            report_energy = energy
        coefficient_updater = (
            OnlineT2aWeightUpdater(
                coefficients,
                beam_azimuth_deg,
                scenario,
                tuple(weight_design.weights),
            )
            if component_id == "mixed" and len(adaptive_method_ids) > 0
            else None
        )
        streamed = run_streaming_beam_branches(
            signal,
            branches,
            scenario.runtime_block_size,
            coefficient_updater=coefficient_updater,
            coefficient_updates=coefficient_updates if component_id != "mixed" else (),
        )
        if coefficient_updater is not None:
            coefficient_updates = coefficient_updater.completed_updates
            completed_design = coefficient_updater.latest_weight_design
            if completed_design is not None:
                latest_weight_design = completed_design
            if len(coefficient_updates) > 0:
                # 表へ載せるFIR energyは最後に完成した版とする。各版の係数は
                # component再適用用のCompletedWeightUpdate内に保持されている。
                report_energy = coefficient_updates[-1].energy_containment
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

    # 通常block境界を除き、係数更新境界だけで分割した参照を作る。更新境界まで一つの
    # blockに含めると、block先頭latchというVersionedCausalFIRの契約を変えてしまう。
    one_block_branches, _ = _make_branches(
        weight_design.weights,
        weight_design.causal_delays_samples,
        scenario.residual_fir_tap_count,
    )
    one_block_mixed = run_streaming_beam_branches(
        rendered.mixed,
        one_block_branches,
        block_size=rendered.mixed.shape[1],
        coefficient_updates=coefficient_updates,
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
        overall_error, boundary_error = calculate_streaming_reference_errors(
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
        diagnostic_zoom[method_id] = select_diagnostic_zoom_bounds(
            mixed_valid[target_beam],
            scenario.runtime_block_size,
            int(round(scenario.training_duration_s * scenario.fs_hz)),
        )
    guard_deg = max(10.0, 2.0 * scenario.beam_azimuth_step_deg)
    non_source = np.abs(beam_azimuth_deg - scenario.target_azimuth_deg) > guard_deg
    rows: list[ScenarioSummaryRow] = []
    for method_id in weight_design.weights:
        target_bl = fraz["target"][method_id][:, target_bin]
        peak_index = int(np.argmax(target_bl))
        target_level = float(target_bl[target_beam])
        sidelobe_peak = float(np.max(target_bl[non_source]))
        target_power = 10.0 ** (fraz["target"][method_id][target_beam, target_bin] / 10.0)
        noise_power = 10.0 ** (fraz["noise"][method_id][target_beam, target_bin] / 10.0)
        rows.append(
            ScenarioSummaryRow(
                scenario="sparse_frequency_switched_two_tone",
                method=method_id,
                evaluation_pattern="sparse_array_design+fixed_beam_multi_source",
                target_frequency_hz=float(frequency_hz[target_bin]),
                target_azimuth_deg=scenario.target_azimuth_deg,
                target_peak_azimuth_deg=float(beam_azimuth_deg[peak_index]),
                target_peak_error_deg=float(
                    abs(beam_azimuth_deg[peak_index] - scenario.target_azimuth_deg)
                ),
                target_level_db_re_input_rms=target_level,
                sidelobe_peak_db_re_mainlobe_peak=(sidelobe_peak - float(np.max(target_bl))),
                output_snr_db=float(
                    10.0
                    * np.log10(
                        max(target_power, np.finfo(float).tiny)
                        / max(noise_power, np.finfo(float).tiny)
                    )
                ),
                interferer_level_at_target_beam_db_re_input_rms=float(
                    fraz["interferer"][method_id][target_beam, interferer_bin]
                ),
                minimum_fir_energy_containment=float(np.min(report_energy[method_id])),
                target_waveform_rms_delta_db=waveform_integrity[method_id].rms_delta_db,
                target_waveform_correlation_after_phase_alignment=(
                    waveform_integrity[method_id].correlation_after_phase_alignment
                ),
                target_waveform_residual_rms_db_re_input_rms=(
                    waveform_integrity[method_id].residual_rms_db_re_input_rms
                ),
                target_phase_delay_samples_modulo_period=(
                    waveform_integrity[method_id].phase_delay_samples_modulo_period
                ),
                streaming_one_block_max_abs_error=streaming_overall_error[method_id],
                streaming_boundary_max_abs_error=streaming_boundary_error[method_id],
                streaming_valid_mask_matches_one_block=streaming_valid_match[method_id],
                ebae_signal_count_at_target=(
                    int(latest_weight_design.ebae_signal_count[target_bin, target_beam])
                    if method_id == "t2a_ebae"
                    else -1
                ),
                ebae_music_peak_azimuth_deg_at_target=(
                    float(
                        latest_weight_design.ebae_music_peak_azimuth_deg[
                            target_bin, target_beam
                        ]
                    )
                    if method_id == "t2a_ebae"
                    else float("nan")
                ),
                ebae_fallback_at_target=(
                    bool(latest_weight_design.ebae_fallback_mask[target_bin, target_beam])
                    if method_id == "t2a_ebae"
                    else False
                ),
                runtime_factor=runtime_factor,
                finite=bool(np.all(np.isfinite(fraz["mixed"][method_id]))),
            )
        )
    # source-frequency BLはtarget/interferer各真値周波数の最大levelを方位ごとに保持する。
    # 異周波数sourceをtarget周波数だけのBLで不可視と誤判定しないための評価配列である。
    source_frequency_bl = {
        method_id: np.maximum(
            fraz["mixed"][method_id][:, target_bin],
            fraz["mixed"][method_id][:, interferer_bin],
        )
        for method_id in weight_design.weights
    }

    # 全component、全方式、分割境界参照、評価行が完成してから不変の結果型へまとめる。
    # reporting側は本結果を直列化するだけで、信号処理式や指標計算を再実装しない。
    return T2aReviewData(
        frequency_hz=frequency_hz,
        beam_azimuth_deg=beam_azimuth_deg,
        fraz_by_component=fraz,
        valid_sample_counts=valid_counts,
        streamed_waveforms=streamed_waveforms,
        one_block_mixed=one_block_mixed,
        waveform_integrity_by_method=waveform_integrity,
        streaming_overall_error_by_method=streaming_overall_error,
        streaming_boundary_error_by_method=streaming_boundary_error,
        streaming_valid_match_by_method=streaming_valid_match,
        diagnostic_zoom_by_method=diagnostic_zoom,
        source_frequency_bl_by_method=source_frequency_bl,
        summary_rows=tuple(rows),
        runtime_s=runtime_s,
        runtime_factor=runtime_factor,
        target_frequency_index=target_bin,
        interferer_frequency_index=interferer_bin,
        target_beam_index=target_beam,
        reference_channel_index=reference_channel,
    ), latest_weight_design


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
        "--adaptive-weight-update-interval-s",
        type=float,
        default=1.0,
        help="適応重みの完成更新間隔[s]。既定値1.0",
    )
    parser.add_argument(
        "--adaptive-weight-design-items-per-cycle",
        type=int,
        default=None,
        help="1更新周期にStepSchedulerで処理する設計item数。省略時は全item",
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
    config = T2aScenarioConfig(
        adaptive_weight_update_interval_s=float(args.adaptive_weight_update_interval_s),
        adaptive_weight_design_items_per_cycle=args.adaptive_weight_design_items_per_cycle,
    )
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
