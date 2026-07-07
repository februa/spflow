"""外部アレイ係数を fixed-delay diff-MVDR 評価用 ndarray へ変換する補助関数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from spflow.beamforming.time_delay import FractionalDelayFilterBank

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]


@dataclass(frozen=True)
class ExternalBeamformingInput:
    """外部アレイ定義を ndarray として保持する。

    このクラスは、実アレイ位置、周波数別 channel shading、小数遅延 FIR バンクを
    fixed-delay diff-MVDR 評価へ渡すための入力束である。

    入力は外部ファイルや MATLAB から渡された ndarray であり、出力は評価 API が参照する
    正規化済み ndarray である。raw file の I/O、scene 合成、MVDR 設計は責務に含めない。
    信号処理上は、実アレイ・実 shading・実小数遅延 FIR を評価系へ注入する境界に位置づく。
    """

    array_positions_m: FloatArray
    shading_by_channel_bin: ComplexArray
    shading_frequency_step_hz: float
    fractional_delay_filter_bank: FractionalDelayFilterBank


def load_float32_le(path: str | Path) -> NDArray[np.float32]:
    """little-endian float32 raw file を 1 次元 ndarray として読み込む。

    Args:
        path: raw binary file の path。MATLAB の `fread(..., 'float32', 'ieee-le')` に対応する。

    Returns:
        読み込んだ値。shape は `[n_value]`、dtype は `float32`。

    Raises:
        ValueError: ファイルが空の場合。
    """
    values = np.fromfile(Path(path), dtype="<f4")
    if values.size == 0:
        raise ValueError(f"empty float32 raw file: {path}")
    return np.asarray(values, dtype=np.float32)


def positions_from_matlab_raw(raw_values: NDArray[Any]) -> FloatArray:
    """MATLAB 互換の `reshape(pos, 3, [])` 結果を `[n_ch, 3]` へ変換する。

    Args:
        raw_values: COE_POS から読んだ 1 次元配列。shape は `[3 * n_ch]`、単位は m。

    Returns:
        センサ位置。shape は `[n_ch, 3]`、axis=0 は channel、axis=1 は x/y/z [m]。

    Raises:
        ValueError: 要素数が 3 の倍数でない場合、または非有限値を含む場合。
    """
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim != 1 or values.size % 3 != 0:
        raise ValueError("COE_POS raw values must have shape [3 * n_ch].")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("COE_POS contains non-finite values.")

    # MATLAB の reshape(pos, 3, []) は column-major で [3, n_ch] を作る。
    # NumPy では order='F' を指定して同じ並びにし、評価 API の [n_ch, 3] へ転置する。
    return np.asarray(values.reshape((3, values.size // 3), order="F").T, dtype=np.float64)


def load_positions_matlab_raw(path: str | Path) -> FloatArray:
    """COE_POS raw file を `[n_ch, 3]` の位置配列として読み込む。"""
    return positions_from_matlab_raw(load_float32_le(path))


def complex_shading_from_matlab_raw(raw_values: NDArray[Any], n_ch: int) -> ComplexArray:
    """MATLAB 互換の COE_CBFSHADING 配列を complex shading に変換する。

    Args:
        raw_values: COE_CBFSHADING から読んだ 1 次元配列。
        n_ch: channel 数。COE_POS から得た値と一致させる。

    Returns:
        複素 shading。shape は `[n_ch, n_bin]`。
        axis=0 は channel、axis=1 は shading 周波数 bin である。

    Raises:
        ValueError: `reshape(shading, nCh, [])` 後の列数が偶数でない場合。
    """
    if n_ch <= 0:
        raise ValueError("n_ch must be positive.")
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim != 1 or values.size % n_ch != 0:
        raise ValueError("COE_CBFSHADING raw values must have shape [n_ch * n_column].")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("COE_CBFSHADING contains non-finite values.")

    # MATLAB の reshape(shading, nCh, []) と同じ column-major 配列にする。
    # 前半列が real、後半列が imag なので、列数は 2 * n_bin でなければならない。
    table = values.reshape((int(n_ch), values.size // int(n_ch)), order="F")
    if table.shape[1] % 2 != 0:
        raise ValueError("COE_CBFSHADING column count must be even: real half + imag half.")
    half = table.shape[1] // 2
    return np.asarray(table[:, :half] + 1j * table[:, half:], dtype=np.complex128)


def load_complex_shading_matlab_raw(path: str | Path, n_ch: int) -> ComplexArray:
    """COE_CBFSHADING raw file を `[n_ch, n_bin]` の complex shading として読み込む。"""
    return complex_shading_from_matlab_raw(load_float32_le(path), n_ch=int(n_ch))


def fractional_delay_filter_bank_from_ndarrays(
    frac_grid: NDArray[Any], frac_filters: NDArray[Any]
) -> FractionalDelayFilterBank:
    """小数遅延 FIR バンクを ndarray から作る。

    Args:
        frac_grid: 小数遅延候補。shape は `[n_frac_filter]`、単位は sample。
        frac_filters: FIR 係数。shape は `[n_frac_filter, n_tap]`。

    Returns:
        `FractionalDelayFilterBank`。
    """
    return FractionalDelayFilterBank(
        frac_grid=np.asarray(frac_grid, dtype=np.float64),
        frac_filters=np.asarray(frac_filters, dtype=np.float64),
    )


def load_fractional_delay_filter_bank_npz(path: str | Path) -> FractionalDelayFilterBank:
    """`frac_grid` と `frac_filters` を含む npz から小数遅延 FIR バンクを読み込む。"""
    return FractionalDelayFilterBank.load_npz(path)


def fractional_delay_filter_bank_from_matlab_raw(
    raw_values: NDArray[Any],
    *,
    n_tap: int,
    frac_min: float = -0.5,
    frac_max: float = 0.5,
) -> FractionalDelayFilterBank:
    """MATLAB 互換の COE_DLYFILT 配列から小数遅延 FIR バンクを作る。

    Args:
        raw_values: COE_DLYFILT から読んだ 1 次元配列。
        n_tap: MATLAB の `reshape(delayfilter, n_tap, [])` に使う tap 数。
            単位は sample。例: `COE_DLYFILT_128` では 128。
        frac_min: 小数遅延 grid の最小値。単位は sample。
        frac_max: 小数遅延 grid の最大値。単位は sample。

    Returns:
        小数遅延 FIR バンク。`frac_filters` shape は `[n_frac_filter, n_tap]`。

    Raises:
        ValueError: raw 要素数が `n_tap` の倍数でない場合。

    境界条件:
        raw file には filter 係数だけが入り、frac_grid は含まれない想定である。
        そのため従来方式と同じく `frac_min` から `frac_max` までを等間隔 grid とみなす。
    """
    if int(n_tap) <= 0:
        raise ValueError("n_tap must be positive.")
    values = np.asarray(raw_values, dtype=np.float64)
    if values.ndim != 1 or values.size % int(n_tap) != 0:
        raise ValueError("COE_DLYFILT raw values must have shape [n_tap * n_frac_filter].")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("COE_DLYFILT contains non-finite values.")
    n_frac_filter = values.size // int(n_tap)
    if n_frac_filter <= 0:
        raise ValueError("COE_DLYFILT must contain at least one filter.")

    # MATLAB の reshape(delayfilter, n_tap, []) と同じ column-major 配列を作る。
    # 各 column が 1 つの小数遅延 FIR なので、spflow の [n_frac_filter, n_tap] へ転置する。
    filters_by_tap_filter = values.reshape((int(n_tap), n_frac_filter), order="F")
    frac_filters = np.asarray(filters_by_tap_filter.T, dtype=np.float64)
    frac_grid = np.linspace(float(frac_min), float(frac_max), n_frac_filter, dtype=np.float64)
    return FractionalDelayFilterBank(frac_grid=frac_grid, frac_filters=frac_filters)


def load_fractional_delay_filter_bank_matlab_raw(
    path: str | Path,
    *,
    n_tap: int,
    frac_min: float = -0.5,
    frac_max: float = 0.5,
) -> FractionalDelayFilterBank:
    """COE_DLYFILT raw file を小数遅延 FIR バンクとして読み込む。"""
    return fractional_delay_filter_bank_from_matlab_raw(
        load_float32_le(path),
        n_tap=int(n_tap),
        frac_min=float(frac_min),
        frac_max=float(frac_max),
    )


def make_external_beamforming_input(
    *,
    array_positions_m: NDArray[Any],
    shading_by_channel_bin: NDArray[Any],
    shading_frequency_step_hz: float,
    fractional_delay_frac_grid: NDArray[Any],
    fractional_delay_filters: NDArray[Any],
) -> ExternalBeamformingInput:
    """公開 API 用に、外部 ndarray を評価入力へ正規化する。

    Args:
        array_positions_m: センサ位置。shape は `[n_ch, 3]`、単位は m。
        shading_by_channel_bin: 複素 shading。shape は `[n_ch, n_bin]`。
        shading_frequency_step_hz: shading bin 間隔。単位は Hz。
        fractional_delay_frac_grid: 小数遅延候補。shape は `[n_frac_filter]`、単位は sample。
        fractional_delay_filters: 小数遅延 FIR 係数。shape は `[n_frac_filter, n_tap]`。

    Returns:
        正規化済み外部入力。

    Raises:
        ValueError: shape、単位、有限性が不正な場合。
    """
    positions = np.asarray(array_positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError("array_positions_m must have shape [n_ch, 3].")
    if not bool(np.all(np.isfinite(positions))):
        raise ValueError("array_positions_m contains non-finite values.")

    shading = np.asarray(shading_by_channel_bin, dtype=np.complex128)
    if shading.ndim != 2 or shading.shape[0] != positions.shape[0] or shading.shape[1] == 0:
        raise ValueError("shading_by_channel_bin must have shape [n_ch, n_bin].")
    if not bool(np.all(np.isfinite(shading))):
        raise ValueError("shading_by_channel_bin contains non-finite values.")
    if float(shading_frequency_step_hz) <= 0.0:
        raise ValueError("shading_frequency_step_hz must be positive.")

    return ExternalBeamformingInput(
        array_positions_m=positions,
        shading_by_channel_bin=shading,
        shading_frequency_step_hz=float(shading_frequency_step_hz),
        fractional_delay_filter_bank=fractional_delay_filter_bank_from_ndarrays(
            fractional_delay_frac_grid,
            fractional_delay_filters,
        ),
    )


def select_shading_for_frequencies(
    shading_by_channel_bin: NDArray[Any],
    shading_frequency_step_hz: float,
    frequencies_hz: NDArray[Any],
) -> ComplexArray:
    """評価周波数に対応する shading を nearest bin で取り出す。

    Args:
        shading_by_channel_bin: 複素 shading。shape は `[n_ch, n_shading_bin]`。
        shading_frequency_step_hz: shading bin 間隔。単位は Hz。
        frequencies_hz: 評価周波数。shape は `[n_freq]`、単位は Hz。

    Returns:
        評価周波数別 shading。shape は `[n_freq, n_ch]`。

    Raises:
        ValueError: 評価周波数が shading table の範囲外に出る場合。
    """
    shading = np.asarray(shading_by_channel_bin, dtype=np.complex128)
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if shading.ndim != 2:
        raise ValueError("shading_by_channel_bin must have shape [n_ch, n_shading_bin].")
    if frequencies.ndim != 1:
        raise ValueError("frequencies_hz must have shape [n_freq].")
    if not bool(np.all(np.isfinite(frequencies))):
        raise ValueError("frequencies_hz contains non-finite values.")
    step_hz = float(shading_frequency_step_hz)
    if step_hz <= 0.0:
        raise ValueError("shading_frequency_step_hz must be positive.")

    # MATLAB 側の dF_shad と同じく、bin 周波数を k * dF とみなす。
    # 実機 shading は離散 table なので、評価周波数には最近傍 bin を対応させる。
    indices = np.rint(frequencies / step_hz).astype(np.int64)
    if not bool(np.all((0 <= indices) & (indices < shading.shape[1]))):
        raise ValueError("frequencies_hz exceeds the shading frequency table range.")
    return np.asarray(shading[:, indices].T, dtype=np.complex128)


def apply_frequency_shading_to_weights(
    fixed_weights: NDArray[Any], shading_by_frequency_channel: NDArray[Any]
) -> ComplexArray:
    """固定整相の周波数重みに channel shading を反映する。

    Args:
        fixed_weights: 固定整相重み。shape は `[n_freq, n_beam, n_ch]`。
            `y = w^H X` の規約で使う重みである。
        shading_by_frequency_channel: shading。shape は `[n_freq, n_ch]`。

    Returns:
        shading 適用後の重み。shape は `[n_freq, n_beam, n_ch]`。

    境界条件:
        実信号経路で channel 応答を `g[ch, f]` 倍してから和を取る場合、
        `w^H X` 規約の重み側には `conj(g)` を掛ける必要がある。
    """
    weights = np.asarray(fixed_weights, dtype=np.complex128)
    shading = np.asarray(shading_by_frequency_channel, dtype=np.complex128)
    if weights.ndim != 3:
        raise ValueError("fixed_weights must have shape [n_freq, n_beam, n_ch].")
    if shading.shape != (weights.shape[0], weights.shape[2]):
        raise ValueError("shading_by_frequency_channel must have shape [n_freq, n_ch].")
    return np.asarray(weights * np.conj(shading[:, np.newaxis, :]), dtype=np.complex128)


def effective_aperture_by_shading(
    array_positions_m: NDArray[Any], shading_by_channel_bin: NDArray[Any]
) -> FloatArray:
    """MATLAB 例と同じ定義で、shading 有効 channel の開口長を返す。

    Args:
        array_positions_m: センサ位置。shape は `[n_ch, 3]`、単位は m。
        shading_by_channel_bin: 複素 shading。shape は `[n_ch, n_bin]`。

    Returns:
        周波数 bin ごとの実効開口長。shape は `[n_bin]`、単位は m。
    """
    positions = np.asarray(array_positions_m, dtype=np.float64)
    shading = np.asarray(shading_by_channel_bin, dtype=np.complex128)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("array_positions_m must have shape [n_ch, 3].")
    if shading.ndim != 2 or shading.shape[0] != positions.shape[0]:
        raise ValueError("shading_by_channel_bin must have shape [n_ch, n_bin].")

    aperture = np.zeros(shading.shape[1], dtype=np.float64)
    for bin_index in range(shading.shape[1]):
        used_indices = np.flatnonzero(shading[:, bin_index] != 0.0)
        if used_indices.size <= 1:
            # 有効 channel が 0 または 1 個では開口を作れないため、0 m と定義する。
            aperture[bin_index] = 0.0
            continue
        max_index = used_indices[int(np.argmax(positions[used_indices, 0]))]
        min_index = used_indices[int(np.argmin(positions[used_indices, 0]))]
        aperture[bin_index] = float(np.linalg.norm(positions[max_index] - positions[min_index]))
    return aperture
