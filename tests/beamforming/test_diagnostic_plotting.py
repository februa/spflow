"""BL/FRAZ/BTR 描画部品に関する回帰試験。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from spflow.beamforming.diagnostic_plotting import (
    build_beam_diagnostic_plot_usage_notes,
    centers_to_edges,
    write_beam_diagnostic_plot_usage_notes,
)


def test_centers_to_edges_preserve_nonuniform_equal_cos_spacing():
    """等 cos 走査由来の非一様方位中心から pcolormesh 用 edge 軸を作れることを確認する。"""
    # 以前に BTR/FRAZ の表示位置がずれた原因は、非一様な中心列を
    # 線形等間隔とみなして描画した点にあったため、中点境界が正しく出ることを固定する。
    centers_deg = np.array([0.0, 10.0, 30.0, 60.0], dtype=np.float64)

    edges_deg = centers_to_edges(centers_deg)

    assert np.allclose(edges_deg, np.array([-5.0, 5.0, 20.0, 45.0, 75.0], dtype=np.float64))


def test_usage_notes_markdown_include_axis_and_btr_cautions():
    """使用上の注意事項を書き出した Markdown に軸解釈と BTR 正規化の注意が含まれることを確認する。"""
    notes = build_beam_diagnostic_plot_usage_notes()
    output_path = Path.cwd() / "artifacts" / "beamforming" / "diagnostic_plotting_test" / "plot_usage_notes.md"

    write_beam_diagnostic_plot_usage_notes(output_path, notes)

    text = output_path.read_text(encoding="utf-8")
    assert "# BL/FRAZ/BTR 使用上の注意" in text
    assert "等 cos 空間" in text
    assert "imshow(extent=...)" in text
    assert "各時刻で最大ビームを 0 dB に正規化" in text
    assert "dB re" in text
