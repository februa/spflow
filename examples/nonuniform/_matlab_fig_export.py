"""MATLAB 表示可能な .fig を一括生成するための補助関数。"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

_STAGED_EXPORTS: list[dict] = []


def _ensure_2d(values: np.ndarray) -> np.ndarray:
    """CSV 保存しやすい 2 次元配列へ整形する。"""
    array = np.asarray(values)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array


def _write_csv(path: Path, values: np.ndarray) -> str:
    """配列を CSV として保存し、MATLAB 用の絶対パス文字列を返す。"""
    array = _ensure_2d(np.asarray(values, dtype=np.float32))
    np.savetxt(path, array, delimiter=',')
    return path.resolve().as_posix()


def _materialize_axis_spec(axis_spec: dict, axis_dir: Path) -> dict:
    """Python 内の numpy 配列を CSV へ退避した MATLAB 向け spec を返す。"""
    materialized = {
        'kind': axis_spec['kind'],
        'index': int(axis_spec['index']),
        'xlabel': str(axis_spec.get('xlabel', '')),
        'ylabel': str(axis_spec.get('ylabel', '')),
        'title': str(axis_spec.get('title', '')),
        'grid': bool(axis_spec.get('grid', True)),
        'legend_location': str(axis_spec.get('legend_location', 'northeast')),
    }

    if axis_spec['kind'] == 'line':
        materialized['lines'] = []
        for line_index, line_spec in enumerate(axis_spec['lines']):
            x_path = axis_dir / f'line_{line_index:02d}_x.csv'
            y_path = axis_dir / f'line_{line_index:02d}_y.csv'
            materialized['lines'].append({
                'x_csv': _write_csv(x_path, line_spec['x']),
                'y_csv': _write_csv(y_path, line_spec['y']),
                'label': str(line_spec.get('label', '')),
                'line_width': float(line_spec.get('line_width', 1.0)),
                'line_style': str(line_spec.get('line_style', '-')),
            })
        return materialized

    if axis_spec['kind'] == 'image':
        image_path = axis_dir / 'image.csv'
        materialized['image_csv'] = _write_csv(image_path, axis_spec['image'])
        materialized['x_lim'] = [float(axis_spec['x_lim'][0]), float(axis_spec['x_lim'][1])]
        materialized['y_lim'] = [float(axis_spec['y_lim'][0]), float(axis_spec['y_lim'][1])]
        materialized['colorbar_label'] = str(axis_spec.get('colorbar_label', ''))
        materialized['colormap'] = str(axis_spec.get('colormap', 'parula'))
        materialized['vlines'] = []
        for vline_spec in axis_spec.get('vlines', []):
            materialized['vlines'].append({
                'x': float(vline_spec['x']),
                'label': str(vline_spec.get('label', '')),
                'line_width': float(vline_spec.get('line_width', 1.0)),
                'line_style': str(vline_spec.get('line_style', '--')),
                'color_rgb': [float(v) for v in vline_spec.get('color_rgb', [1.0, 1.0, 1.0])],
            })
        materialized['polylines'] = []
        for line_index, line_spec in enumerate(axis_spec.get('polylines', [])):
            x_path = axis_dir / f'polyline_{line_index:02d}_x.csv'
            y_path = axis_dir / f'polyline_{line_index:02d}_y.csv'
            materialized['polylines'].append({
                'x_csv': _write_csv(x_path, line_spec['x']),
                'y_csv': _write_csv(y_path, line_spec['y']),
                'label': str(line_spec.get('label', '')),
                'line_width': float(line_spec.get('line_width', 1.0)),
                'line_style': str(line_spec.get('line_style', '--')),
                'color_rgb': [float(v) for v in line_spec.get('color_rgb', [1.0, 1.0, 1.0])],
            })
        return materialized

    raise ValueError(f"unsupported axis kind: {axis_spec['kind']}")


def _escape_matlab_string(text: str) -> str:
    """MATLAB 文字列リテラル用にクォートをエスケープする。"""
    return text.replace("'", "''")


def stage_matlab_figure(base_path: Path, spec: dict) -> None:
    """MATLAB .fig 再構成用の spec と CSV 群を出力キューへ積む。"""
    spec_dir = base_path.parent / f'{base_path.name}_matlab'
    spec_dir.mkdir(parents=True, exist_ok=True)

    materialized_axes = []
    for axis_number, axis_spec in enumerate(spec['axes']):
        axis_dir = spec_dir / f'axis_{axis_number:02d}'
        axis_dir.mkdir(parents=True, exist_ok=True)
        materialized_axes.append(_materialize_axis_spec(axis_spec, axis_dir))

    materialized_spec = {
        'nrows': int(spec['nrows']),
        'ncols': int(spec['ncols']),
        'suptitle': str(spec.get('suptitle', '')),
        'caption': str(spec.get('caption', '')),
        'axes': materialized_axes,
    }
    spec_path = spec_dir / 'figure_spec.json'
    spec_path.write_text(json.dumps(materialized_spec, indent=2, ensure_ascii=False), encoding='utf-8')
    _STAGED_EXPORTS.append({
        'spec_path': spec_path.resolve().as_posix(),
        'output_fig_path': base_path.with_suffix('.fig').resolve().as_posix(),
    })


def flush_matlab_figure_exports(strict: bool = False) -> bool:
    """溜め込んだ figure spec を MATLAB 1 回起動でまとめて .fig 化する。"""
    if not _STAGED_EXPORTS:
        return True

    matlab_path = shutil.which('matlab')
    if matlab_path is None:
        if strict:
            raise RuntimeError('MATLAB is not available, so .fig could not be generated.')
        return False

    manifest_path = Path(_STAGED_EXPORTS[0]['spec_path']).parent.parent / 'matlab_figure_manifest.json'
    manifest_path.write_text(json.dumps({'figures': _STAGED_EXPORTS}, indent=2, ensure_ascii=False), encoding='utf-8')
    renderer_dir = Path(__file__).resolve().parent
    batch_command = (
        f"addpath('{_escape_matlab_string(renderer_dir.as_posix())}'); "
        f"spflow_render_figures_batch('{_escape_matlab_string(manifest_path.resolve().as_posix())}');"
    )
    completed = subprocess.run(
        [matlab_path, '-batch', batch_command],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode == 0:
        _STAGED_EXPORTS.clear()
        return True

    message = completed.stderr.strip() or completed.stdout.strip() or 'unknown MATLAB error'
    if strict:
        raise RuntimeError(f'MATLAB .fig export failed: {message}')
    return False
