"""examples、evaluations、toolsの配置契約を検証する。"""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_ROOT = REPOSITORY_ROOT / "examples"


def test_examples_do_not_contain_evaluation_or_artifact_cli_names() -> None:
    """公開API例へ評価・sweep・artifact生成CLIが再混入しないことを確認する。"""
    forbidden_name_parts = (
        "_eval",
        "_sweep",
        "_diagnostics",
        "_performance",
        "_comparison",
        "design_",
        "generate_",
        "optimize_",
    )
    misplaced = [
        path.relative_to(REPOSITORY_ROOT).as_posix()
        for path in EXAMPLES_ROOT.rglob("*.py")
        if path.name != "__init__.py"
        and any(part in path.stem for part in forbidden_name_parts)
    ]

    assert misplaced == []


def test_examples_only_keep_short_public_api_responsibility_groups() -> None:
    """現在のexampleがstreamingとbeamformingの公開API例だけであることを確認する。"""
    runnable_examples = {
        path.relative_to(EXAMPLES_ROOT).as_posix()
        for path in EXAMPLES_ROOT.rglob("*.py")
        if path.name != "__init__.py"
    }

    assert runnable_examples == {
        "beamforming/delay_and_sum.py",
        "beamforming/streaming_cbf.py",
        "beamforming/streaming_mvdr_weights.py",
        "streaming/basic_pipeline.py",
        "streaming/none_cycle.py",
        "streaming/step_scheduler_completion.py",
    }
