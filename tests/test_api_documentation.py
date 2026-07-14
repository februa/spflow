"""実装済み機能一覧とソースコードの同期を検証する。"""

import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_implemented_feature_catalog_is_current() -> None:
    """公開 API の変更がコミット済み機能一覧へ反映されていることを確認する。

    HTML 生成には optional dependency が必要になるため、このテストでは AST から生成する
    Markdown 一覧の一致だけを検査する。これにより通常の test 環境を重くせず、公開部品の
    追加・削除に伴う一覧更新漏れを検出する。
    """

    result = subprocess.run(
        [sys.executable, "tools/build_api_docs.py", "--check"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_implemented_feature_catalog_contains_each_responsibility_group() -> None:
    """主要な責務分類と公開部品が機能一覧から検索できることを確認する。

    基本部品、周波数処理、フィルタバンク、beamforming、array事前設計、評価、SLC、
    シミュレーション支援を代表するmoduleを一つずつ選ぶ。単に空の一覧と同期した状態を
    成功としないための境界条件である。
    """

    catalog = (REPOSITORY_ROOT / "doc" / "SpFlow" / "実装済み機能一覧.md").read_text(
        encoding="utf-8"
    )

    assert "### `spflow.flow`" in catalog
    assert "### `spflow.frequency.overlap_save`" in catalog
    assert "### `spflow.filterbank.polyphase`" in catalog
    assert "### `spflow.beamforming.cbf`" in catalog
    assert "### `spflow.beamforming.application`" in catalog
    assert "### `spflow.array_design.operational_array`" in catalog
    assert "### `spflow.beamforming_evaluation.evaluation_arrays`" in catalog
    assert "### `spflow.sidelobe_cancellation.beam_domain`" in catalog
    assert "### `spflow.simulation.numerics`" in catalog
    assert "`SimulationPrecision`" in catalog
