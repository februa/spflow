"""非均一 example 用のアレイ入力生成 helper をまとめる。"""

# 非均一木構造では分割仕様と streaming 状態の組み合わせで挙動が大きく変わるため、
# 実運用に近い入出力条件を一式そろえて可視化・書き出しできる例として管理する。

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from spflow.beamforming.array_design import BandwiseArrayDesign
from spflow.filterbank.formal_nonuniform_tree import FormalNonuniformTreeFilterBank


DEFAULT_ACTIVE_COUNTS = (32, 32, 24, 20, 16, 12, 8, 4)


def band_specs_for_fs(fs_hz: float):
    """指定サンプリング周波数に対応する leaf 帯域仕様を返す。"""
    return FormalNonuniformTreeFilterBank.default_for_fs(fs_hz).band_specs


def build_nested_sparse_positions(
    *,
    n_dense_ch: int,
    dense_spacing_m: float,
    n_outer_pairs: int,
    outer_spacing_m: float,
) -> np.ndarray:
    """中央密・端疎の 1 次元受波器位置を生成する。"""
    if n_dense_ch <= 0:
        raise ValueError('n_dense_ch must be positive.')
    if dense_spacing_m <= 0.0:
        raise ValueError('dense_spacing_m must be positive.')
    if n_outer_pairs < 0:
        raise ValueError('n_outer_pairs must be non-negative.')
    if outer_spacing_m < dense_spacing_m:
        raise ValueError('outer_spacing_m must be at least dense_spacing_m.')

    inner = (np.arange(n_dense_ch, dtype=np.float32) - 0.5 * (n_dense_ch - 1)) * dense_spacing_m
    if n_outer_pairs == 0:
        return inner.astype(np.float32)

    inner_edge = 0.5 * (n_dense_ch - 1) * dense_spacing_m
    outer = inner_edge + outer_spacing_m * np.arange(1, n_outer_pairs + 1, dtype=np.float32)
    return np.concatenate([-outer[::-1], inner, outer]).astype(np.float32)


def _dominant_linear_coordinate(channel_positions_m: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """線形配置を代表する 1 次元座標へ射影し、並び替え情報も返す。"""
    positions = np.asarray(channel_positions_m, dtype=np.float32)
    if positions.ndim == 1:
        scalar = positions.copy()
        order = np.argsort(scalar)
        return scalar, order, scalar[order]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError('channel_positions_m must have shape (n_ch,) or (n_ch, 3).')

    spans = np.ptp(positions, axis=0)
    dominant_axis = int(np.argmax(spans))
    scalar = positions[:, dominant_axis].astype(np.float32, copy=True)
    order = np.argsort(scalar)
    return scalar, order, scalar[order]


def build_progressive_shading_table(
    *,
    channel_positions_m: np.ndarray,
    fs_hz: float,
    sound_speed: float,
    aperture_wavelengths: float = 4.0,
    min_active_ch: int = 4,
    max_active_counts: tuple[int, ...] | None = None,
) -> np.ndarray:
    """leaf 帯域仕様に合わせて帯域別シェーディング表を組み立てる。"""
    if fs_hz <= 0.0:
        raise ValueError('fs_hz must be positive.')
    if sound_speed <= 0.0:
        raise ValueError('sound_speed must be positive.')
    if aperture_wavelengths <= 0.0:
        raise ValueError('aperture_wavelengths must be positive.')
    if min_active_ch <= 0:
        raise ValueError('min_active_ch must be positive.')

    specs = band_specs_for_fs(fs_hz)
    positions = np.asarray(channel_positions_m, dtype=np.float32)
    n_ch = int(positions.shape[0])
    scalar, order, ordered = _dominant_linear_coordinate(positions)
    shading_table = np.zeros((n_ch, len(specs)), dtype=np.float32)

    if max_active_counts is None:
        default_counts = np.array(DEFAULT_ACTIVE_COUNTS, dtype=np.int64)
        if default_counts.size != len(specs):
            default_counts = np.full(len(specs), n_ch, dtype=np.int64)
        scaled = np.clip(default_counts, 1, n_ch)
    else:
        scaled = np.asarray(max_active_counts, dtype=np.int64)
        if scaled.shape != (len(specs),):
            raise ValueError('max_active_counts must have shape (n_band,).')
        scaled = np.clip(scaled, 1, n_ch)

    for band_index, spec in enumerate(specs):
        desired_aperture_m = aperture_wavelengths * (sound_speed / max(spec.center_frequency_hz, 1.0))
        center = 0.5 * float(ordered[0] + ordered[-1])
        selected_mask = np.abs(ordered - center) <= 0.5 * desired_aperture_m + 1e-12
        selected_sorted = order[selected_mask]

        if selected_sorted.size < min_active_ch:
            target_count = min(n_ch, max(min_active_ch, int(scaled[band_index])))
            start = max(0, (n_ch - target_count) // 2)
            stop = start + target_count
            selected_sorted = order[start:stop]
        else:
            max_count = int(scaled[band_index])
            if selected_sorted.size > max_count:
                trim = selected_sorted.size - max_count
                left_trim = trim // 2
                right_trim = trim - left_trim
                selected_sorted = selected_sorted[left_trim:selected_sorted.size - right_trim]

        shading_table[selected_sorted, band_index] = 1.0

    return shading_table


def build_default_array_design(
    *,
    fs_hz: float,
    sound_speed: float,
    n_dense_ch: int,
    dense_spacing_m: float,
    n_outer_pairs: int,
    outer_spacing_m: float,
    aperture_wavelengths: float,
    min_active_ch: int,
) -> BandwiseArrayDesign:
    """example 既定の中央密・端疎アレイ設計を生成する。"""
    positions = build_nested_sparse_positions(
        n_dense_ch=n_dense_ch,
        dense_spacing_m=dense_spacing_m,
        n_outer_pairs=n_outer_pairs,
        outer_spacing_m=outer_spacing_m,
    )
    shading_table = build_progressive_shading_table(
        channel_positions_m=positions,
        fs_hz=fs_hz,
        sound_speed=sound_speed,
        aperture_wavelengths=aperture_wavelengths,
        min_active_ch=min_active_ch,
    )
    return BandwiseArrayDesign.from_ndarrays(
        channel_positions_m=positions,
        shading_table=shading_table,
    )


def load_array_design(
    *,
    channel_positions_path: str | None,
    shading_table_path: str | None,
    fs_hz: float,
    sound_speed: float,
    n_dense_ch: int,
    dense_spacing_m: float,
    n_outer_pairs: int,
    outer_spacing_m: float,
    aperture_wavelengths: float,
    min_active_ch: int,
) -> BandwiseArrayDesign:
    """外部 ndarray または既定生成条件から array design を解決する。"""
    if channel_positions_path is None and shading_table_path is None:
        return build_default_array_design(
            fs_hz=fs_hz,
            sound_speed=sound_speed,
            n_dense_ch=n_dense_ch,
            dense_spacing_m=dense_spacing_m,
            n_outer_pairs=n_outer_pairs,
            outer_spacing_m=outer_spacing_m,
            aperture_wavelengths=aperture_wavelengths,
            min_active_ch=min_active_ch,
        )

    if channel_positions_path is None:
        raise ValueError('channel_positions_path must be provided when shading_table_path is set.')

    channel_positions = np.load(channel_positions_path)
    if shading_table_path is None:
        shading_table = build_progressive_shading_table(
            channel_positions_m=channel_positions,
            fs_hz=fs_hz,
            sound_speed=sound_speed,
            aperture_wavelengths=aperture_wavelengths,
            min_active_ch=min_active_ch,
        )
    else:
        shading_table = np.load(shading_table_path)

    expected_shape = (int(np.asarray(channel_positions).shape[0]), len(band_specs_for_fs(fs_hz)))
    if np.asarray(shading_table).shape != expected_shape:
        raise ValueError(
            'shading_table shape mismatch. '
            f'Expected {expected_shape}. '
            'Use generate_nonuniform_array_inputs.py to regenerate matching ndarray files.'
        )

    return BandwiseArrayDesign.from_ndarrays(
        channel_positions_m=channel_positions,
        shading_table=shading_table,
    )


def save_array_inputs(output_dir: str | Path, design: BandwiseArrayDesign, *, fs_hz: float, sound_speed: float) -> None:
    """array design を example 再利用向けの ndarray 群として保存する。"""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    positions_path = out_dir / 'channel_positions.npy'
    shading_path = out_dir / 'shading_table.npy'
    summary_path = out_dir / 'array_summary.json'

    np.save(positions_path, np.asarray(design.channel_positions_m, dtype=np.float32))
    np.save(shading_path, np.asarray(design.shading_table, dtype=np.float32))

    summary = {
        'fs_hz': float(fs_hz),
        'sound_speed': float(sound_speed),
        'n_ch': int(design.n_ch),
        'n_band': int(design.n_band),
        'active_channel_counts_per_band': design.active_channel_counts_per_band().tolist(),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
