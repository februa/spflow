"""逐次処理の中核importへ評価・描画依存が逆流しないことを検証する。"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Any

import spflow
import spflow.beamforming as beamforming


def _loaded_modules_after_import(statement: str) -> dict[str, bool]:
    """独立Python processでimport後の代表module読込状態を取得する。

    Args:
        statement: 独立processで最初に実行するPython import文。

    Returns:
        module名を読込済み状態へ対応付けた辞書。

    Raises:
        subprocess.CalledProcessError: import文が正常終了しなかった場合。
        json.JSONDecodeError: 子processが検査結果以外を標準出力へ書いた場合。

    Notes:
        親processでは他testが既にbeamformingやMatplotlibをimportしている可能性がある。
        import境界そのものを検証するため、状態を共有しない独立processを使う。
    """
    code = (
        "import json, sys\n"
        f"{statement}\n"
        "names = ['spflow.beamforming', 'spflow.filterbank', "
        "'spflow.beamforming_evaluation.diagnostic_plotting', 'matplotlib']\n"
        "print(json.dumps({name: name in sys.modules for name in names}))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    raw_result: Any = json.loads(completed.stdout)
    if not isinstance(raw_result, dict):
        raise TypeError("import境界の子process結果はdictでなければなりません。")
    # JSON値がboolであることは生成側で固定しており、ここでは組み込みboolへ明示変換する。
    return {str(name): bool(loaded) for name, loaded in raw_result.items()}


def test_flow_and_frame_buffer_import_does_not_load_optional_responsibility_packages() -> None:
    """中核逐次処理部品だけの利用時にbeamforming・描画依存を読まないことを確認する。"""
    loaded = _loaded_modules_after_import("from spflow import Flow, FrameBuffer")

    assert loaded == {
        "spflow.beamforming": False,
        "spflow.filterbank": False,
        "spflow.beamforming_evaluation.diagnostic_plotting": False,
        "matplotlib": False,
    }


def test_cbf_compatibility_import_does_not_load_evaluation_plotting() -> None:
    """CBFの互換flat importがCBF責務moduleだけを遅延解決することを確認する。"""
    loaded = _loaded_modules_after_import("from spflow import CBFBeamformer")

    assert loaded["spflow.beamforming"] is True
    assert loaded["spflow.beamforming_evaluation.diagnostic_plotting"] is False
    assert loaded["matplotlib"] is False


def test_root_public_names_are_eager_or_registered_for_lazy_resolution() -> None:
    """ルート公開名が通常importまたは責務packageの遅延表に存在することを確認する。"""
    eager_names = set(vars(spflow))
    lazy_names = spflow._BEAMFORMING_EXPORTS | spflow._FILTERBANK_EXPORTS

    assert set(spflow.__all__) <= eager_names | lazy_names


def test_beamforming_public_names_are_registered_to_one_responsibility_module() -> None:
    """beamforming公開名に遅延解決先の責務moduleが一意に登録されることを確認する。"""
    assert set(beamforming.__all__) <= set(beamforming._EXPORT_MODULES)


def test_evaluation_names_are_not_reexported_from_core_flat_apis() -> None:
    """評価支援APIがspflow直下やbeamforming処理APIへ再混入しないことを確認する。"""
    evaluation_names = {
        "SourceSectorMask",
        "build_beam_level_display_arrays",
        "get_evaluation_criteria_for_pattern",
        "plot_bl_response",
    }

    # 評価部品は明示的にspflow.beamforming_evaluationから選ばせる。
    # core flat APIへ混ぜると、通常の重み設計・適用と評価policyの責務境界が見えなくなる。
    assert evaluation_names.isdisjoint(spflow.__all__)
    assert evaluation_names.isdisjoint(beamforming.__all__)
