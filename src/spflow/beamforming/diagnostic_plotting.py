"""BL/FRAZ/BTR 診断図の共通描画部品をまとめるモジュール。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .._validation import require

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - 実行環境依存
    plt = None


@dataclass(frozen=True)
class BeamDiagnosticPlotUsageNotes:
    """BL/FRAZ/BTR の解釈上の注意事項を保持する。

    このクラスは、ビーム応答図 BL、周波数-方位図 FRAZ、時間-方位図 BTR を
    どの前提で読めばよいかを、用途別の注意事項として保持する。

    入力は common, bl, fraz, btr の各カテゴリに属する注意文であり、
    出力は Markdown 文字列や辞書形式である。

    図そのものの数値計算やファイル保存は責務に含めない。
    信号処理上は、診断図を誤読しないための表示契約を定義する補助オブジェクトである。
    """

    common: tuple[str, ...]
    bl: tuple[str, ...]
    fraz: tuple[str, ...]
    btr: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        """注意事項を JSON 化しやすい辞書へ変換する。

        Returns:
            カテゴリ名をキーとし、各カテゴリの注意文リストを値とする辞書。
        """
        return {
            "common": list(self.common),
            "bl": list(self.bl),
            "fraz": list(self.fraz),
            "btr": list(self.btr),
        }

    def as_markdown(self) -> str:
        """注意事項を Markdown 文章へ整形する。

        Returns:
            見出し付きの Markdown 文字列。改行終端を含む。
        """
        sections = [
            ("共通", self.common),
            ("BL", self.bl),
            ("FRAZ", self.fraz),
            ("BTR", self.btr),
        ]
        lines = ["# BL/FRAZ/BTR 使用上の注意", ""]
        for title, notes in sections:
            lines.append(f"## {title}")
            lines.extend([f"- {note}" for note in notes])
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"


def build_beam_diagnostic_plot_usage_notes() -> BeamDiagnosticPlotUsageNotes:
    """BL/FRAZ/BTR の再利用時に参照すべき注意事項を返す。

    Returns:
        表示軸の解釈、正規化の意味、適用範囲をカテゴリ別にまとめた注意事項。
    """
    return BeamDiagnosticPlotUsageNotes(
        common=(
            "方位軸を等 cos 空間で設計した場合、degree 空間ではビーム間隔が非一様になる。",
            "そのため FRAZ/BTR の画像描画では、ビーム中心値を線形補間せずセル境界を明示する必要がある。",
            "BL/FRAZ/BTR のピーク方位は連続角ではなく nearest beam center へ量子化される。",
        ),
        bl=(
            "BL は指定周波数 1 点でのビーム応答であり、帯域平均応答ではない。",
            "複数音源条件では、BL を音源周波数ごとに個別評価し、他周波数音源の影響と混同しない。",
            "target 方位とピーク方位の差は、主にビームグリッド量子化と整数遅延近似の影響で生じる。",
        ),
        fraz=(
            "FRAZ は one-sided RMS スペクトルを表示するため、負周波数側は描かない。",
            "方位軸が非一様な場合に imshow(extent=...) を使うと見かけ上のピーク位置がずれるため、非一様セル描画を使う。",
        ),
        btr=(
            "BTR は各時刻で最大ビームを 0 dB に正規化した相対表示であり、時刻間の絶対レベル比較には使わない。",
            "複数同時音源条件では『最大ビームのトラック』が任意の target 方位と一致しないことがある。",
            "target 方位追従の確認に BTR を使うときは、まず単一音源条件で検証する。",
        ),
    )


def write_beam_diagnostic_plot_usage_notes(path: str | Path, notes: BeamDiagnosticPlotUsageNotes) -> None:
    """使用上の注意事項を Markdown ファイルとして保存する。

    Args:
        path: 保存先パス。単位は filesystem path。
        notes: 保存する注意事項オブジェクト。

    Returns:
        なし。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(notes.as_markdown(), encoding="utf-8")


def configure_matplotlib_japanese() -> None:
    """matplotlib で日本語キャプションが崩れにくい設定を入れる。

    Returns:
        なし。

    Raises:
        AssertionError: matplotlib が import 済みでない場合。
    """
    assert plt is not None
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Yu Gothic",
        "Yu Gothic UI",
        "Meiryo",
        "MS Gothic",
        "IPAexGothic",
        "Noto Sans CJK JP",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def require_matplotlib() -> None:
    """画像保存に必要な matplotlib が使えることを確認する。

    Returns:
        なし。

    Raises:
        RuntimeError: matplotlib が利用できず PNG 保存契約を満たせない場合。
    """
    if plt is None:
        raise RuntimeError("matplotlib is required to save BL/FRAZ/BTR figures.")
    configure_matplotlib_japanese()


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    """非一様な中心座標列から pcolormesh 用のセル境界を作る。

    Args:
        centers: 単調増加する中心座標列。shape は `[n_center]`。
            方位なら単位は deg、周波数なら Hz、時間なら s を想定する。

    Returns:
        セル境界列。shape は `[n_center + 1]`。
        `pcolormesh` の x/y edge 軸へそのまま渡せる。

    Raises:
        ValueError: 入力が 1 次元でない、空、または単調増加でない場合。
    """
    axis_centers = np.asarray(centers, dtype=np.float64)
    require(axis_centers.ndim == 1, "centers must be a 1-D array.")
    require(axis_centers.size > 0, "centers must not be empty.")

    if axis_centers.size == 1:
        return np.array([axis_centers[0] - 0.5, axis_centers[0] + 0.5], dtype=np.float64)

    require(np.all(np.diff(axis_centers) > 0.0), "centers must be strictly increasing.")
    edges = np.empty(axis_centers.size + 1, dtype=np.float64)

    # 内部境界は隣接 2 点の中点とし、非一様サンプリング間隔を画像セル幅へ保存する。
    # 等 cos 走査の方位軸を degree 空間で正しく見せるため、この edge 化が必須になる。
    edges[1:-1] = 0.5 * (axis_centers[:-1] + axis_centers[1:])
    edges[0] = axis_centers[0] - 0.5 * (axis_centers[1] - axis_centers[0])
    edges[-1] = axis_centers[-1] + 0.5 * (axis_centers[-1] - axis_centers[-2])
    return edges


def add_caption(fig, caption: str) -> None:
    """図下部へ短い説明文を付ける。

    Args:
        fig: matplotlib figure。
        caption: 図の前提条件やピーク位置を要約する文章。

    Returns:
        なし。
    """
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=9)


def save_figure(fig, path: str | Path) -> None:
    """PNG 図を保存して figure を閉じる。

    Args:
        fig: 保存対象の matplotlib figure。
        path: 保存先 PNG パス。

    Returns:
        なし。
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _normalize_marker_points(
    target_azimuth_deg: float | None,
    target_frequency_hz: float | None,
    target_points: Sequence[tuple[float, float, str]] | None,
    *,
    default_label: str,
) -> list[tuple[float, float, str]]:
    """単一点指定と複数点指定を共通の点列へ正規化する。"""
    if target_points is not None:
        return [(float(azimuth_deg), float(frequency_hz), str(label)) for azimuth_deg, frequency_hz, label in target_points]

    if target_azimuth_deg is None or target_frequency_hz is None:
        return []
    return [(float(target_azimuth_deg), float(target_frequency_hz), default_label)]


def plot_bl_response(
    axis_az_deg: np.ndarray,
    beam_levels_db20: np.ndarray,
    *,
    target_azimuth_deg: float,
    peak_azimuth_deg: float,
    title: str,
    caption: str,
    output_path: str | Path,
    response_label: str = "Beam response",
) -> None:
    """BL, すなわち指定周波数でのビーム応答を保存する。

    Args:
        axis_az_deg: 方位中心列。shape は `[n_beam]`、単位は deg。
        beam_levels_db20: 各ビームの RMS レベル。shape は `[n_beam]`、単位は dB20。
        target_azimuth_deg: 真値として重ね描きする target 方位。単位は deg。
        peak_azimuth_deg: 最大応答ビーム中心。単位は deg。
        title: 図タイトル。
        caption: 図下部に表示する使用条件要約。
        output_path: 保存先 PNG パス。
        response_label: 凡例へ表示する応答系列名。

    Returns:
        なし。
    """
    fig, axis = plt.subplots(figsize=(10, 4.5))
    axis.plot(axis_az_deg, beam_levels_db20, linewidth=1.5, label=response_label)
    axis.axvline(float(target_azimuth_deg), color="black", linestyle=":", linewidth=1.0, label="Target azimuth")
    axis.axvline(float(peak_azimuth_deg), color="tab:red", linestyle="--", linewidth=1.0, label="Peak azimuth")
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel("RMS Level [dB20]")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    axis.legend(loc="best")
    add_caption(fig, caption)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.96))
    save_figure(fig, output_path)


def plot_fraz_heatmap(
    axis_az_deg: np.ndarray,
    freqs_hz: np.ndarray,
    fraz_levels_db20: np.ndarray,
    *,
    target_azimuth_deg: float | None = None,
    target_frequency_hz: float | None = None,
    peak_azimuth_deg: float | None = None,
    peak_frequency_hz: float | None = None,
    target_points: Sequence[tuple[float, float, str]] | None = None,
    peak_points: Sequence[tuple[float, float, str]] | None = None,
    title: str,
    caption: str,
    output_path: str | Path,
    colorbar_label: str = "Level [dB20 RMS]",
) -> None:
    """FRAZ, すなわち周波数-方位レベル分布を保存する。

    Args:
        axis_az_deg: 方位中心列。shape は `[n_beam]`、単位は deg。
        freqs_hz: 周波数中心列。shape は `[n_freq]`、単位は Hz。
        fraz_levels_db20: 周波数-方位レベル。shape は `[n_beam, n_freq]`、単位は dB20 RMS。
            axis=0 がビーム、axis=1 が周波数ビンである。
        target_azimuth_deg: 単一点の真値方位。単位は deg。
        target_frequency_hz: 単一点の真値周波数。単位は Hz。
        peak_azimuth_deg: 単一点のピーク方位。単位は deg。
        peak_frequency_hz: 単一点のピーク周波数。単位は Hz。
        target_points: 複数 target を重ね描きする場合の `(azimuth_deg, frequency_hz, label)` 列。
        peak_points: 複数 peak を重ね描きする場合の `(azimuth_deg, frequency_hz, label)` 列。
        title: 図タイトル。
        caption: 図下部に表示する使用条件要約。
        output_path: 保存先 PNG パス。
        colorbar_label: カラーバーのラベル。

    Returns:
        なし。
    """
    fig, axis = plt.subplots(figsize=(10, 5.5))
    azimuth_edges_deg = centers_to_edges(axis_az_deg)
    frequency_edges_hz = centers_to_edges(freqs_hz)

    # fraz_levels_db20.T shape: [n_freq, n_beam]
    # axis=0 を y 軸の周波数、axis=1 を x 軸の方位へ対応させるため転置する。
    # 非一様な方位中心を pcolormesh のセル境界へ展開し、等 cos 走査の歪みを避ける。
    image = axis.pcolormesh(
        azimuth_edges_deg,
        frequency_edges_hz,
        fraz_levels_db20.T,
        shading="flat",
        cmap="viridis",
    )

    normalized_target_points = _normalize_marker_points(
        target_azimuth_deg,
        target_frequency_hz,
        target_points,
        default_label="Target",
    )
    normalized_peak_points = _normalize_marker_points(
        peak_azimuth_deg,
        peak_frequency_hz,
        peak_points,
        default_label="Peak",
    )

    for marker_index, (azimuth_deg, frequency_hz, label) in enumerate(normalized_target_points):
        axis.axvline(
            azimuth_deg,
            color="white",
            linestyle=":" if marker_index == 0 else "-.",
            linewidth=1.0,
            label=label,
        )
        axis.axhline(
            frequency_hz,
            color="white",
            linestyle="--" if marker_index == 0 else (0, (3, 3)),
            linewidth=1.0,
        )

    for marker_index, (azimuth_deg, frequency_hz, label) in enumerate(normalized_peak_points):
        axis.plot(
            [azimuth_deg],
            [frequency_hz],
            marker="o",
            color="tab:red" if marker_index == 0 else "tab:orange",
            markersize=5.0,
            label=label,
        )

    axis.set_xlim(float(azimuth_edges_deg[0]), float(azimuth_edges_deg[-1]))
    axis.set_ylim(float(frequency_edges_hz[0]), float(frequency_edges_hz[-1]))
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel("Frequency [Hz]")
    axis.set_title(title)
    axis.legend(loc="upper right")
    fig.colorbar(image, ax=axis, label=colorbar_label)
    add_caption(fig, caption)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.96))
    save_figure(fig, output_path)


def plot_btr_heatmap(
    axis_az_deg: np.ndarray,
    times_s: np.ndarray,
    btr_relative_levels_db: np.ndarray,
    *,
    btr_peak_azimuths_deg: np.ndarray | None = None,
    target_azimuth_deg: float | None = None,
    target_azimuths_deg: np.ndarray | Sequence[float] | None = None,
    title: str,
    caption: str,
    output_path: str | Path,
    colorbar_label: str = "Relative Level [dB]",
) -> None:
    """BTR, すなわち time-azimuth のビームトラックを保存する。

    Args:
        axis_az_deg: 方位中心列。shape は `[n_beam]`、単位は deg。
        times_s: 時刻中心列。shape は `[n_time]`、単位は s。
        btr_relative_levels_db: 時間-方位相対レベル。shape は `[n_time, n_beam]`、単位は dB。
            axis=0 が時間ブロック、axis=1 がビームである。
        btr_peak_azimuths_deg: 各時間ブロックの最大ビーム方位。shape は `[n_time]`、単位は deg。
            複数同時音源で peak track が代表値にならない場合は `None` を渡す。
        target_azimuth_deg: 単一点の真値方位。単位は deg。
        target_azimuths_deg: 複数 target 方位列。shape は `[n_target]`、単位は deg。
        title: 図タイトル。
        caption: 図下部に表示する使用条件要約。
        output_path: 保存先 PNG パス。
        colorbar_label: カラーバーのラベル。

    Returns:
        なし。
    """
    fig, axis = plt.subplots(figsize=(10, 5.5))
    azimuth_edges_deg = centers_to_edges(axis_az_deg)
    time_edges_s = centers_to_edges(times_s)

    # btr_relative_levels_db shape: [n_time, n_beam]
    # BTR も FRAZ と同様に非一様な方位中心を edge 化しないと、
    # ピークトラックが左へ圧縮されて見かけの方位がずれる。
    image = axis.pcolormesh(
        azimuth_edges_deg,
        time_edges_s,
        btr_relative_levels_db,
        shading="flat",
        cmap="viridis",
        vmin=-12.0,
        vmax=0.0,
    )

    if btr_peak_azimuths_deg is not None:
        axis.plot(btr_peak_azimuths_deg, times_s, color="white", linestyle="--", linewidth=1.0, label="Peak track")

    if target_azimuths_deg is None and target_azimuth_deg is not None:
        normalized_target_azimuths_deg = np.array([float(target_azimuth_deg)], dtype=np.float64)
    elif target_azimuths_deg is None:
        normalized_target_azimuths_deg = np.empty(0, dtype=np.float64)
    else:
        normalized_target_azimuths_deg = np.asarray(target_azimuths_deg, dtype=np.float64)

    for target_index, target_azimuth in enumerate(normalized_target_azimuths_deg):
        axis.axvline(
            float(target_azimuth),
            color="black" if target_index == 0 else "tab:red",
            linestyle=":" if target_index == 0 else "-.",
            linewidth=1.0,
            label="Target azimuth" if target_index == 0 else f"Target azimuth {target_index + 1}",
        )

    axis.set_xlim(float(azimuth_edges_deg[0]), float(azimuth_edges_deg[-1]))
    axis.set_ylim(float(time_edges_s[0]), float(time_edges_s[-1]))
    axis.set_xlabel("Azimuth [deg]")
    axis.set_ylabel("Time [s]")
    axis.set_title(title)
    axis.legend(loc="upper right")
    fig.colorbar(image, ax=axis, label=colorbar_label)
    add_caption(fig, caption)
    fig.tight_layout(rect=(0.03, 0.04, 1.0, 0.96))
    save_figure(fig, output_path)


__all__ = [
    "BeamDiagnosticPlotUsageNotes",
    "build_beam_diagnostic_plot_usage_notes",
    "write_beam_diagnostic_plot_usage_notes",
    "configure_matplotlib_japanese",
    "require_matplotlib",
    "centers_to_edges",
    "add_caption",
    "save_figure",
    "plot_bl_response",
    "plot_fraz_heatmap",
    "plot_btr_heatmap",
]
