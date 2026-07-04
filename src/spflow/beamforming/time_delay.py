"""spflow.beamforming.time_delay を実装するモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .._validation import require, require_positive_float, require_positive_int


def _normalize_array_positions(array_pos_m: np.ndarray) -> np.ndarray:
    """アレイ位置ベクトルを shape `[n_ch, 3]` の `float64` 配列へ正規化する。"""
    positions = np.asarray(array_pos_m, dtype=np.float64)
    require(positions.ndim == 2 and positions.shape[1] == 3, "array_pos_m must have shape (n_ch, 3).")
    require(positions.shape[0] > 0, "array_pos_m must not be empty.")
    require(np.all(np.isfinite(positions)), "array_pos_m must contain only finite values.")
    return positions


def _normalize_direction_cosines(dir_cos: np.ndarray) -> np.ndarray:
    """方向余弦テーブルを shape `[n_beam, 3]` の `float64` 配列へ正規化する。"""
    directions = np.asarray(dir_cos, dtype=np.float64)
    require(directions.ndim == 2, "dir_cos must be a 2-D array.")
    require(np.all(np.isfinite(directions)), "dir_cos must contain only finite values.")

    if directions.shape[1] == 3:
        return directions
    if directions.shape[0] == 3 and directions.shape[1] != 3:
        # 既存の `make_directions()` は [3, n_beam] を返すため、
        # 時間領域固定整相では [n_beam, 3] へ転置してから幾何式へ渡す。
        return directions.T
    raise ValueError("dir_cos must have shape (n_beam, 3) or (3, n_beam).")


@dataclass(frozen=True)
class FractionalDelayFilterBank:
    """小数遅延 FIR フィルタ群を保存・読込可能な形で保持する。

    このクラスは、小数遅延量の離散グリッド `frac_grid` と、それに対応する
    FIR フィルタ群 `frac_filters` を一体で保持する。

    入力は小数遅延候補 `[n_frac_filter]` と FIR 係数表 `[n_frac_filter, n_tap]` であり、
    出力は nearest-neighbor 選択用の filter index や `.npz` ファイルへの保存結果である。

    実際の固定整相処理で各チャネルへ FIR 畳み込みを適用する責務は持たない。
    信号処理上は、時間領域固定ビームフォーマの小数サンプル遅延補償部に対応する。
    """

    frac_grid: np.ndarray
    frac_filters: np.ndarray

    def __post_init__(self) -> None:
        """保存形式と nearest-neighbor 選択に必要な shape 条件を検証する。"""
        frac_grid = np.asarray(self.frac_grid, dtype=np.float64)
        frac_filters = np.asarray(self.frac_filters, dtype=np.float64)

        require(frac_grid.ndim == 1, "frac_grid must have shape (n_frac_filter,).")
        require(frac_grid.size > 0, "frac_grid must not be empty.")
        require(np.all(np.isfinite(frac_grid)), "frac_grid must contain only finite values.")
        require(np.all(np.diff(frac_grid) >= 0.0), "frac_grid must be sorted in ascending order.")

        require(frac_filters.ndim == 2, "frac_filters must have shape (n_frac_filter, n_tap).")
        require(frac_filters.shape[0] == frac_grid.size, "frac_filters and frac_grid must agree on n_frac_filter.")
        require(frac_filters.shape[1] > 0, "frac_filters must contain at least one tap.")
        require(np.all(np.isfinite(frac_filters)), "frac_filters must contain only finite values.")

        object.__setattr__(self, "frac_grid", frac_grid)
        object.__setattr__(self, "frac_filters", frac_filters)

    @property
    def n_frac_filter(self) -> int:
        """小数遅延候補数を返す。"""
        return int(self.frac_grid.size)

    @property
    def n_tap(self) -> int:
        """各 FIR フィルタのタップ長を返す。"""
        return int(self.frac_filters.shape[1])

    def select_indices(self, delay_frac: np.ndarray) -> np.ndarray:
        """各小数遅延量に最も近い FIR フィルタ番号を返す。

        Args:
            delay_frac: 小数遅延量。shape は `[n_ch, n_beam]`、単位は sample。
                `DelayTable.delay_frac` をそのまま渡すことを想定する。

        Returns:
            最近傍フィルタ番号。shape は `[n_ch, n_beam]`。

        Raises:
            ValueError: `delay_frac` が 2 次元でない場合。
        """
        delay_frac_array = np.asarray(delay_frac, dtype=np.float64)
        require(delay_frac_array.ndim == 2, "delay_frac must have shape (n_ch, n_beam).")

        # delay_frac[..., None] shape: [n_ch, n_beam, 1]
        # frac_grid[None, None, :] shape: [1, 1, n_frac_filter]
        # broadcasting により、各チャネル・各ビームの小数遅延と候補グリッドとの差を一括計算する。
        return np.argmin(
            np.abs(delay_frac_array[..., np.newaxis] - self.frac_grid[np.newaxis, np.newaxis, :]),
            axis=-1,
        ).astype(np.int64)

    def save_npz(self, path: str | Path) -> None:
        """小数遅延 FIR バンクを `.npz` 形式で保存する。"""
        np.savez(
            Path(path),
            frac_grid=self.frac_grid,
            frac_filters=self.frac_filters,
        )

    @classmethod
    def load_npz(cls, path: str | Path) -> "FractionalDelayFilterBank":
        """保存済み `.npz` から小数遅延 FIR バンクを読み込む。"""
        with np.load(Path(path), allow_pickle=False) as saved:
            return cls(
                frac_grid=saved["frac_grid"],
                frac_filters=saved["frac_filters"],
            )


def design_windowed_sinc_fractional_delay_filter(mu: float, n_tap: int) -> np.ndarray:
    """窓付き sinc により単一の小数遅延 FIR を設計する。

    Args:
        mu: 小数遅延量。単位は sample。想定範囲は `-0.5 <= mu <= 0.5`。
        n_tap: FIR タップ長。単位は tap。

    Returns:
        FIR 係数。shape は `[n_tap]`。

    Raises:
        ValueError: `mu` や `n_tap` が不正な場合。
    """
    require(-0.5 <= float(mu) <= 0.5, "mu must lie in [-0.5, 0.5].")
    require_positive_int("n_tap", n_tap)

    n = np.arange(n_tap, dtype=np.float64)
    center = 0.5 * (n_tap - 1)

    # 理想小数遅延 FIR は sinc(n - center - mu) で与える。
    # center は全フィルタ共通の群遅延であり、チャネル間の相対整相に効くのは mu だけである。
    taps = np.sinc(n - center - float(mu))
    taps *= np.hamming(n_tap)

    dc_gain = np.sum(taps)
    require(abs(dc_gain) > 0.0, "fractional delay filter normalization failed.")

    # DC 利得を 1 に正規化しておくことで、固定整相後の振幅が
    # フィルタ選択だけで系統的に増減しないようにする。
    taps /= dc_gain
    return taps


def design_fractional_delay_filter_bank(n_frac_filter: int, n_tap: int) -> FractionalDelayFilterBank:
    """小数遅延候補を等間隔に並べた FIR バンクを設計する。

    Args:
        n_frac_filter: 小数遅延候補数。単位は本数。
        n_tap: 各 FIR のタップ長。単位は tap。

    Returns:
        保存可能な小数遅延 FIR バンク。

    Raises:
        ValueError: `n_frac_filter` または `n_tap` が不正な場合。
    """
    require_positive_int("n_frac_filter", n_frac_filter)
    require_positive_int("n_tap", n_tap)

    frac_grid = np.linspace(-0.5, 0.5, n_frac_filter, dtype=np.float64)
    frac_filters = np.stack(
        [design_windowed_sinc_fractional_delay_filter(float(mu), n_tap) for mu in frac_grid],
        axis=0,
    )
    return FractionalDelayFilterBank(frac_grid=frac_grid, frac_filters=frac_filters)


@dataclass(frozen=True)
class DelayTable:
    """時間領域固定整相で使う到達遅延と補償遅延の分解結果を保持する。

    このクラスは、アレイ位置、方向余弦、音速、サンプリング周波数から求めた
    到達時刻差 `arrival_delay_sec` と、固定整相のために必要な非負補償遅延
    `steering_delay_sample` を保持する。

    出力は整数遅延 `delay_int`、小数遅延 `delay_frac`、必要に応じて
    小数遅延 FIR の選択番号 `frac_filter_index` である。

    各チャネル信号への実際の遅延適用や FIR 畳み込みは責務に含めない。
    信号処理上は、時間領域 Delay-and-Sum 固定ビームフォーマの設計テーブルに対応する。
    """

    arrival_delay_sec: np.ndarray
    steering_delay_sample: np.ndarray
    delay_int: np.ndarray
    delay_frac: np.ndarray
    frac_filter_index: np.ndarray | None = None

    def __post_init__(self) -> None:
        """shape 整合とサンプル遅延分解の基本条件を検証する。"""
        arrival_delay_sec = np.asarray(self.arrival_delay_sec, dtype=np.float64)
        steering_delay_sample = np.asarray(self.steering_delay_sample, dtype=np.float64)
        delay_int = np.asarray(self.delay_int, dtype=np.int64)
        delay_frac = np.asarray(self.delay_frac, dtype=np.float64)

        require(arrival_delay_sec.ndim == 2, "arrival_delay_sec must have shape (n_ch, n_beam).")
        require(
            steering_delay_sample.shape == arrival_delay_sec.shape,
            "steering_delay_sample and arrival_delay_sec must agree on shape.",
        )
        require(delay_int.shape == arrival_delay_sec.shape, "delay_int and arrival_delay_sec must agree on shape.")
        require(
            delay_frac.shape == arrival_delay_sec.shape,
            "delay_frac and arrival_delay_sec must agree on shape.",
        )
        require(np.all(delay_int >= 0), "delay_int must be non-negative after causal offsetting.")

        if self.frac_filter_index is not None:
            frac_filter_index = np.asarray(self.frac_filter_index, dtype=np.int64)
            require(
                frac_filter_index.shape == arrival_delay_sec.shape,
                "frac_filter_index and arrival_delay_sec must agree on shape.",
            )
            object.__setattr__(self, "frac_filter_index", frac_filter_index)

        object.__setattr__(self, "arrival_delay_sec", arrival_delay_sec)
        object.__setattr__(self, "steering_delay_sample", steering_delay_sample)
        object.__setattr__(self, "delay_int", delay_int)
        object.__setattr__(self, "delay_frac", delay_frac)

    @property
    def n_ch(self) -> int:
        """チャネル数を返す。"""
        return int(self.delay_int.shape[0])

    @property
    def n_beam(self) -> int:
        """ビーム数を返す。"""
        return int(self.delay_int.shape[1])

    @property
    def max_delay_int(self) -> int:
        """整数遅延の最大値を返す。"""
        return int(np.max(self.delay_int))

    @classmethod
    def from_geometry(
        cls,
        array_pos_m: np.ndarray,
        dir_cos: np.ndarray,
        fs_hz: float,
        sound_speed_m_s: float,
        fractional_filter_bank: FractionalDelayFilterBank | None = None,
    ) -> "DelayTable":
        """アレイ幾何と方向余弦から固定整相用の遅延表を設計する。

        Args:
            array_pos_m: センサ位置。shape は `[n_ch, 3]`、単位は m。
            dir_cos: 方向余弦。shape は `[n_beam, 3]` または `[3, n_beam]`。
            fs_hz: サンプリング周波数。単位は Hz。
            sound_speed_m_s: 音速。単位は m/s。
            fractional_filter_bank: 小数遅延 FIR バンク。与えた場合は
                `delay_frac` に最近傍な filter index も算出する。

        Returns:
            時間領域固定整相用の遅延表。

        Raises:
            ValueError: 入力 shape や物理パラメータが不正な場合。
        """
        positions = _normalize_array_positions(array_pos_m)
        directions = _normalize_direction_cosines(dir_cos)
        require_positive_float("fs_hz", float(fs_hz))
        require_positive_float("sound_speed_m_s", float(sound_speed_m_s))

        # arrival_delay_sec[ch, beam] = -(r_ch^T u_beam) / c。
        # 音響中心より早着なら負、遅着なら正という設計書の符号規約に対応する。
        arrival_delay_sec = -(positions @ directions.T) / float(sound_speed_m_s)

        # 固定整相では早着チャネルを遅らせるため、補償遅延は到達遅延の逆符号とする。
        steering_delay_sample = -arrival_delay_sec * float(fs_hz)

        # 時間領域の逐次処理では未来サンプル参照ができないため、
        # ビームごとに最小遅延を引いて全チャネルの補償遅延を非負化する。
        steering_delay_sample = steering_delay_sample - np.min(steering_delay_sample, axis=0, keepdims=True)

        # round 分解により、補償遅延を整数部と ±0.5 sample 程度の小数部へ分ける。
        # 後段の小数遅延 FIR は delay_frac のみを担当し、delay_int は単純なサンプルシフトで処理する。
        delay_int = np.rint(steering_delay_sample).astype(np.int64)
        delay_frac = steering_delay_sample - delay_int

        frac_filter_index = None
        if fractional_filter_bank is not None:
            frac_filter_index = fractional_filter_bank.select_indices(delay_frac)

        return cls(
            arrival_delay_sec=arrival_delay_sec,
            steering_delay_sample=steering_delay_sample,
            delay_int=delay_int,
            delay_frac=delay_frac,
            frac_filter_index=frac_filter_index,
        )


class IntegerDelayAndSumBeamformer:
    """整数サンプル遅延だけで時間領域固定整相を行うビームフォーマ。

    このクラスは、`DelayTable.delay_int` に基づいて各チャネルを整数サンプル分だけ
    遅延させ、チャネル平均により固定ビーム出力を生成する。

    入力はチャネル時系列 `[n_ch, n_sample]`、出力は固定整相後の
    ビーム時系列 `[n_beam, n_sample]` である。

    小数遅延 FIR の畳み込み、SLC の適応キャンセル、方位センサ連動の状態制御は
    このクラスの責務に含めない。
    信号処理上は、SLC 前段の時間領域固定 Delay-and-Sum ビームフォーマに位置づく。
    """

    def __init__(self, delay_table: DelayTable, average_channels: bool = True) -> None:
        self.delay_table = delay_table
        self.average_channels = bool(average_channels)

    @classmethod
    def from_geometry(
        cls,
        array_pos_m: np.ndarray,
        dir_cos: np.ndarray,
        fs_hz: float,
        sound_speed_m_s: float,
        average_channels: bool = True,
        fractional_filter_bank: FractionalDelayFilterBank | None = None,
    ) -> "IntegerDelayAndSumBeamformer":
        """アレイ幾何から整数遅延固定整相ビームフォーマを構成する。"""
        return cls(
            delay_table=DelayTable.from_geometry(
                array_pos_m=array_pos_m,
                dir_cos=dir_cos,
                fs_hz=fs_hz,
                sound_speed_m_s=sound_speed_m_s,
                fractional_filter_bank=fractional_filter_bank,
            ),
            average_channels=average_channels,
        )

    def process(
        self,
        x: np.ndarray,
        return_steered_channels: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """整数遅延整相により固定ビーム出力を生成する。

        Args:
            x: 入力チャネル信号。shape は `[n_ch, n_sample]`。
                axis=0 は受波器チャネル、axis=1 は時間サンプルである。
            return_steered_channels: `True` の場合はチャネル平均前の
                整相信号 `[n_beam, n_ch, n_sample]` も返す。

        Returns:
            `return_steered_channels=False` の場合は固定整相出力
            `[n_beam, n_sample]` を返す。
            `True` の場合は `(beam_output, steered_channel_output)` を返す。

        Raises:
            ValueError: 入力 shape が想定と異なる場合。
        """
        input_signal = np.asarray(x)
        require(input_signal.ndim == 2, "x must have shape (n_ch, n_sample).")
        require(
            input_signal.shape[0] == self.delay_table.n_ch,
            "x and delay_table must agree on n_ch.",
        )

        # 整数遅延後に平均を取るため、整数入力でも丸め落ちしないよう少なくとも float32 に上げる。
        working_dtype = np.result_type(input_signal.dtype, np.float32)
        channel_signal = np.asarray(input_signal, dtype=working_dtype)

        n_ch, n_sample = channel_signal.shape
        n_beam = self.delay_table.n_beam

        # steered_channel_output shape: [n_beam, n_ch, n_sample]
        # axis=0 はビーム、axis=1 はチャネル、axis=2 は時間サンプルである。
        steered_channel_output = np.zeros((n_beam, n_ch, n_sample), dtype=working_dtype)

        for beam_idx in range(n_beam):
            for ch_idx in range(n_ch):
                delay_sample = int(self.delay_table.delay_int[ch_idx, beam_idx])
                if delay_sample >= n_sample:
                    # ブロック長より大きい遅延では有効サンプルが残らないため、
                    # そのチャネル寄与はゼロのままとし、異常な wrap-around を防ぐ。
                    continue

                # y[n] = x[n - d] に対応する整数遅延を、先頭ゼロ詰め・末尾切り捨てで実装する。
                # 未来サンプルを参照しない因果実装のため、ビームごとの共通遅延だけ出力全体が後ろへずれる。
                steered_channel_output[beam_idx, ch_idx, delay_sample:] = channel_signal[
                    ch_idx,
                    : n_sample - delay_sample,
                ]

        if self.average_channels:
            # 固定 Delay-and-Sum ではチャネル平均でビーム出力を得る。
            # 平均化により、同相整列した target 成分は保持しつつチャネル数依存の利得増加を避ける。
            beam_output = np.mean(steered_channel_output, axis=1, dtype=working_dtype)
        else:
            beam_output = np.sum(steered_channel_output, axis=1, dtype=working_dtype)

        if return_steered_channels:
            return beam_output, steered_channel_output
        return beam_output
