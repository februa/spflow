"""平面波toneとchannel非相関noiseから決定論的なアレイsceneを生成する。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow._validation import (
    require,
    require_non_negative_float,
    require_positive_float,
)
from spflow.level_conversion import LevelConverter, level_20log10_rms
from spflow.simulation.numerics import SimulationPrecision

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class ToneSceneSource:
    """アレイsceneへ配置する一つの平面波toneを表す。

    方位角、俯仰角、周波数、入力RMS level、初期位相、任意の振幅変調を入力として、
    `synthesize_tone_scene` が各channelへ到来遅延を与えるためのsource条件を保持する。

    アレイ形状、noise、beamforming処理、評価metricは責務に含めない。
    信号処理上は複数sourceを線形重ね合わせする前の一つの理想平面波に位置づく。
    """

    azimuth_deg: float
    frequency_hz: float
    level_db20: float = 0.0
    elevation_deg: float = 0.0
    phase_deg: float = 0.0
    amplitude_modulation_hz: float = 0.0
    amplitude_modulation_depth: float = 0.0
    amplitude_modulation_phase_deg: float = 0.0
    label: str | None = None

    def __post_init__(self) -> None:
        """角度、周波数、変調率、ラベルの境界条件を検証する。"""

        require(np.isfinite(float(self.azimuth_deg)), "azimuth_deg must be finite.")
        require(np.isfinite(float(self.elevation_deg)), "elevation_deg must be finite.")
        require_positive_float("frequency_hz", float(self.frequency_hz))
        require(np.isfinite(float(self.level_db20)), "level_db20 must be finite.")
        require(np.isfinite(float(self.phase_deg)), "phase_deg must be finite.")
        require_non_negative_float("amplitude_modulation_hz", float(self.amplitude_modulation_hz))
        require(
            0.0 <= float(self.amplitude_modulation_depth) <= 0.99,
            "amplitude_modulation_depth must lie in [0.0, 0.99].",
        )
        require(
            np.isfinite(float(self.amplitude_modulation_phase_deg)),
            "amplitude_modulation_phase_deg must be finite.",
        )
        if self.label is not None:
            require(len(str(self.label)) > 0, "label must not be empty when provided.")


@dataclass(frozen=True)
class ToneScene:
    """生成済みのmulti-channel時間波形と時間軸を保持する。

    `signal`はshape `[n_ch, n_sample]`、`time_axis_s`はshape `[n_sample]`であり、
    axis=0はchannel、axis=1は時間sample、時間軸の単位は秒である。

    beamforming、FFT、level評価、ファイル保存は責務に含めない。
    信号処理上はシミュレーションsource生成と方式処理の境界に位置づく固定shape結果型である。
    """

    signal: FloatArray
    time_axis_s: FloatArray

    def __post_init__(self) -> None:
        """信号と時間軸の固定shape、dtype、有限性を検証する。"""

        require(self.signal.ndim == 2, "signal must have shape (n_ch, n_sample).")
        require(self.signal.shape[0] > 0, "signal must contain at least one channel.")
        require(self.signal.shape[1] > 0, "signal must contain at least one sample.")
        require(self.time_axis_s.ndim == 1, "time_axis_s must have shape (n_sample,).")
        require(
            self.signal.shape[1] == self.time_axis_s.shape[0],
            "signal sample axis and time_axis_s length must match.",
        )
        require(
            np.issubdtype(self.signal.dtype, np.floating),
            "signal must have a real floating dtype.",
        )
        require(
            np.issubdtype(self.time_axis_s.dtype, np.floating),
            "time_axis_s must have a real floating dtype.",
        )
        require(
            bool(np.all(np.isfinite(self.signal))),
            "signal must contain only finite values.",
        )
        require(
            bool(np.all(np.isfinite(self.time_axis_s))),
            "time_axis_s must contain only finite values.",
        )


def direction_from_azimuth_elevation(
    azimuth_deg: float,
    elevation_deg: float,
) -> NDArray[np.float64]:
    """方位角と俯仰角から右手系の方向余弦ベクトルを計算する。

    Args:
        azimuth_deg: 方位角。単位はdeg。
        elevation_deg: 俯仰角。単位はdeg。

    Returns:
        方向余弦ベクトル。shapeは`[3]`、axis=0は`x, y, z`で無次元。

    Raises:
        ValueError: 角度が有限値でない場合。

    境界条件:
        方位角は周期量なので範囲を制限しない。俯仰角も座標変換自体は任意の有限値を受け付ける。
    """

    require(np.isfinite(float(azimuth_deg)), "azimuth_deg must be finite.")
    require(np.isfinite(float(elevation_deg)), "elevation_deg must be finite.")
    azimuth_rad = np.deg2rad(float(azimuth_deg))
    elevation_rad = np.deg2rad(float(elevation_deg))
    cos_elevation = np.cos(elevation_rad)
    return np.array(
        [
            np.cos(azimuth_rad) * cos_elevation,
            np.sin(azimuth_rad) * cos_elevation,
            np.sin(elevation_rad),
        ],
        dtype=np.float64,
    )


def synthesize_tone_scene(
    *,
    array_positions_m: FloatArray,
    sources: Sequence[ToneSceneSource],
    fs_hz: float,
    duration_s: float,
    sound_speed_m_s: float,
    noise_level_db20: float,
    random_seed: int,
    precision: SimulationPrecision = SimulationPrecision.SINGLE,
    level_converter: LevelConverter | None = None,
) -> ToneScene:
    """平面波tone群とchannel非相関noiseを加算したアレイsceneを生成する。

    Args:
        array_positions_m: センサ位置。shapeは`[n_ch, 3]`、axis=0はchannel、
            axis=1は`x, y, z`座標、単位はm。
        sources: 一つ以上の平面波tone条件。各levelは`dB re input RMS`。
        fs_hz: サンプリング周波数。単位はHz。
        duration_s: 生成時間。単位は秒。
        sound_speed_m_s: 伝搬速度。単位はm/s。
        noise_level_db20: 各channelの時間領域noise RMS level。
            単位は`dB re input RMS`であり、ASDではない。
        random_seed: channel非相関noiseを再現する乱数seed。
        precision: 出力`signal`と`time_axis_s`の実数dtypeを一括選択する精度。
        level_converter: sourceとnoiseの入力dBを線形RMSへ変換する契約。`None`の場合は
            `0 dB re input RMS = 1 RMS`のdefinitionを使う。出力評価にも同じconverterを
            保持して使うことで、入力地点と評価地点のreferenceを一致させられる。

    Returns:
        生成scene。`signal`のshapeは`[n_ch, n_sample]`、`time_axis_s`のshapeは
        `[n_sample]`。axis=1が時間sample、時間軸の単位は秒。

    Raises:
        ValueError: 配列shape、有限性、source数、周波数、時間、音速が不正な場合。

    境界条件:
        sample数は`round(duration_s * fs_hz)`で決め、1未満になる条件は拒否する。
        source位相とnoise系列の再現性を保つため、同じ入力とseedから同じ結果を返す。
    """

    positions_m = np.asarray(array_positions_m, dtype=np.float64)
    require(
        positions_m.ndim == 2 and positions_m.shape[1] == 3,
        "array_positions_m must have shape (n_ch, 3).",
    )
    require(positions_m.shape[0] > 0, "array_positions_m must not be empty.")
    require(
        bool(np.all(np.isfinite(positions_m))),
        "array_positions_m must contain only finite values.",
    )
    require(len(sources) > 0, "sources must not be empty.")
    require_positive_float("fs_hz", float(fs_hz))
    require_positive_float("duration_s", float(duration_s))
    require_positive_float("sound_speed_m_s", float(sound_speed_m_s))
    require(np.isfinite(float(noise_level_db20)), "noise_level_db20 must be finite.")

    if level_converter is None:
        # 既存APIの0 dB=1 RMSを維持しつつ、手書きの10**(L/20)を残さない。
        default_definition = level_20log10_rms(
            reference_rms=1.0,
            reference_label="input RMS",
        )
        effective_level_converter = LevelConverter.for_definition(default_definition)
    else:
        effective_level_converter = level_converter

    n_sample = int(round(float(duration_s) * float(fs_hz)))
    require(n_sample > 0, "duration_s * fs_hz must produce at least one sample.")
    time_axis_s = np.arange(n_sample, dtype=np.float64) / float(fs_hz)
    signal = np.zeros((positions_m.shape[0], n_sample), dtype=np.float64)

    for source in sources:
        source_direction = direction_from_azimuth_elevation(
            azimuth_deg=float(source.azimuth_deg),
            elevation_deg=float(source.elevation_deg),
        )
        # source levelはRMS定義であり、実cosineのpeak=sqrt(2)*RMSへの変換を
        # 入力地点と出力評価地点で共有するLevelConverterへ委譲する。
        peak_amplitude = effective_level_converter.input_to_real_cosine_peak(
            float(source.level_db20)
        )
        phase_rad = np.deg2rad(float(source.phase_deg))
        modulation_phase_rad = np.deg2rad(float(source.amplitude_modulation_phase_deg))

        # tau[ch] = -(r_ch^T u) / c は基準点に対する各channelの到来時刻差で、単位は秒。
        # sourceごとに同じ符号規約で生成し、線形重ね合わせでmulti-source sceneを作る。
        arrival_delay_s = -(positions_m @ source_direction) / float(sound_speed_m_s)

        # envelope[n] = 1 + depth*cos(2*pi*f_mod*t + phi_mod)。
        # block共分散の時間変化を再現しつつ、depth<1により包絡の符号反転を防ぐ。
        amplitude_envelope = 1.0 + float(source.amplitude_modulation_depth) * np.cos(
            2.0 * np.pi * float(source.amplitude_modulation_hz) * time_axis_s + modulation_phase_rad
        )

        # signalのbroadcasting後shapeは[n_ch, n_sample]。
        # axis=0へchannel別遅延、axis=1へ時間sampleを配置する。
        signal += (
            peak_amplitude
            * amplitude_envelope[np.newaxis, :]
            * np.cos(
                2.0
                * np.pi
                * float(source.frequency_hz)
                * (time_axis_s[np.newaxis, :] - arrival_delay_s[:, np.newaxis])
                + phase_rad
            )
        )

    # noise_level_db20は時間領域sample RMS基準であり、ASDからの帯域積分値ではない。
    noise_std = effective_level_converter.input_to_rms(float(noise_level_db20))
    random_generator = np.random.default_rng(int(random_seed))
    signal += noise_std * random_generator.standard_normal(signal.shape)

    return ToneScene(
        signal=np.asarray(signal, dtype=precision.real_dtype),
        time_axis_s=np.asarray(time_axis_s, dtype=precision.real_dtype),
    )


__all__ = [
    "ToneScene",
    "ToneSceneSource",
    "direction_from_azimuth_elevation",
    "synthesize_tone_scene",
]
