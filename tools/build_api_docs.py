"""spflow の実装済み機能一覧と HTML API リファレンスを生成する。"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
PACKAGE_ROOT = SOURCE_ROOT / "spflow"
DEFAULT_CATALOG_PATH = REPOSITORY_ROOT / "doc" / "SpFlow" / "実装済み機能一覧.md"
BUILD_ROOT = REPOSITORY_ROOT / "build"
DEFAULT_HTML_PATH = BUILD_ROOT / "api-docs"

CATEGORY_LABELS = {
    "beamforming": "ビームフォーミング",
    "beamforming_evaluation": "ビームフォーミング評価支援",
    "filterbank": "フィルタバンク",
    "frequency": "周波数領域処理",
    "simulation": "シミュレーション支援",
}


@dataclass(frozen=True)
class _ModuleFeature:
    """一つの公開 module から抽出した API 索引情報を保持する。

    module、source_path、summary、classes、functions、exports を入力として保持し、
    Markdown 生成へ固定形の情報を渡す。Python module の import や信号処理の実行は
    責務に含めない。
    """

    module: str
    source_path: str
    summary: str
    classes: tuple[str, ...]
    functions: tuple[str, ...]
    exports: tuple[str, ...]


def _module_name(path: Path) -> str:
    relative_path = path.relative_to(SOURCE_ROOT).with_suffix("")
    parts = list(relative_path.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _first_paragraph(docstring: str | None) -> str:
    if docstring is None:
        return "module docstring に責務の記載なし"

    paragraph_lines: list[str] = []
    for line in docstring.strip().splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            if paragraph_lines:
                break
            continue
        paragraph_lines.append(stripped_line)
    return " ".join(paragraph_lines) or "module docstring に責務の記載なし"


def _literal_exports(tree: ast.Module) -> tuple[str, ...]:
    for node in tree.body:
        value_node: ast.expr | None = None
        if isinstance(node, ast.Assign):
            if any(
                isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
            ):
                value_node = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == "__all__":
                value_node = node.value

        if value_node is None:
            continue
        try:
            exports = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            return ()
        if isinstance(exports, (list, tuple)) and all(isinstance(name, str) for name in exports):
            return tuple(exports)
        return ()
    return ()


def _read_module_feature(path: Path) -> _ModuleFeature:
    # UTF-8 BOM を含む既存 module も Python の import と同じ内容として解析する。
    # `utf-8-sig` は BOM がない通常の UTF-8 ファイルにも同じように適用できる。
    source = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(source, filename=str(path))

    # import せず AST だけを読むため、一覧生成が optional dependency や実行時副作用に
    # 影響されない。定義名は module 直下に限定し、内部 helper は掲載しない。
    classes = tuple(
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and not node.name.startswith("_")
    )
    functions = tuple(
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and not node.name.startswith("_")
    )
    return _ModuleFeature(
        module=_module_name(path),
        source_path=path.relative_to(REPOSITORY_ROOT).as_posix(),
        summary=_first_paragraph(ast.get_docstring(tree)),
        classes=classes,
        functions=functions,
        exports=_literal_exports(tree),
    )


def _collect_features() -> tuple[_ModuleFeature, ...]:
    features: list[_ModuleFeature] = []
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        # `_validation` のような private module は公開部品の索引には含めない。
        # `__init__` は package の公式な再公開入口を示すため対象に残す。
        if path.name != "__init__.py" and path.stem.startswith("_"):
            continue
        feature = _read_module_feature(path)
        if feature.classes or feature.functions or feature.exports:
            features.append(feature)
    return tuple(features)


def _category(feature: _ModuleFeature) -> tuple[int, str]:
    module_parts = feature.module.split(".")
    if len(module_parts) == 1 or module_parts[1] not in CATEGORY_LABELS:
        return (0, "基本部品")
    top_level_package = module_parts[1]
    label = CATEGORY_LABELS[top_level_package]
    order = list(CATEGORY_LABELS).index(top_level_package) + 1
    return (order, label)


def _markdown_names(names: tuple[str, ...]) -> str:
    if not names:
        return "なし"
    return ", ".join(f"`{name}`" for name in names)


def _escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|")


def _render_catalog(features: tuple[_ModuleFeature, ...]) -> str:
    categories: dict[tuple[int, str], list[_ModuleFeature]] = {}
    for feature in features:
        categories.setdefault(_category(feature), []).append(feature)

    lines = [
        "# spflow 実装済み機能一覧",
        "",
        "> このファイルは `python tools/build_api_docs.py` でソースコードから自動生成する。"
        "直接編集しない。",
        "",
        "公開 module の責務と import パスを俯瞰するための索引である。"
        "詳細な引数、戻り値、shape、単位は、",
        "同じツールが `build/api-docs/` に生成する HTML API リファレンスを参照する。",
        "",
        "## 分類別サマリ",
        "",
        "| 分類 | module 数 | class 数 | function 数 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, category_features in sorted(categories.items()):
        lines.append(
            f"| {category[1]} | {len(category_features)} | "
            f"{sum(len(feature.classes) for feature in category_features)} | "
            f"{sum(len(feature.functions) for feature in category_features)} |"
        )

    for category, category_features in sorted(categories.items()):
        lines.extend(["", f"## {category[1]}"])
        for feature in category_features:
            lines.extend(
                [
                    "",
                    f"### `{feature.module}`",
                    "",
                    f"- 責務: {_escape_table_cell(feature.summary)}",
                    f"- 実装: `{feature.source_path}`",
                    f"- 公開 class: {_markdown_names(feature.classes)}",
                    f"- 公開 function: {_markdown_names(feature.functions)}",
                ]
            )
            if feature.exports:
                lines.append(
                    f"- package からの再公開名 (`__all__`): {_markdown_names(feature.exports)}"
                )

    return "\n".join(lines) + "\n"


def _write_or_check_catalog(catalog_path: Path, *, check: bool) -> None:
    rendered_catalog = _render_catalog(_collect_features())
    if check:
        catalog_is_current = (
            catalog_path.is_file() and catalog_path.read_text(encoding="utf-8") == rendered_catalog
        )
        if not catalog_is_current:
            raise SystemExit(
                f"実装済み機能一覧が最新ではありません: {catalog_path}\n"
                "python tools/build_api_docs.py --skip-html を実行してください。"
            )
        print(f"実装済み機能一覧は最新です: {catalog_path}")
        return

    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(rendered_catalog, encoding="utf-8")
    print(f"実装済み機能一覧を生成しました: {catalog_path}")


def _build_html(output_path: Path) -> None:
    if importlib.util.find_spec("pdoc") is None:
        raise SystemExit('pdoc が見つかりません。pip install -e ".[docs]" を実行してください。')

    resolved_output_path = output_path.resolve()
    output_is_in_repository = resolved_output_path.is_relative_to(REPOSITORY_ROOT)
    output_is_in_build = resolved_output_path.is_relative_to(BUILD_ROOT)
    if output_is_in_repository and not output_is_in_build:
        raise ValueError("リポジトリ内の HTML 出力先は build/ 配下に限定します。")

    # 削除済み module の古い HTML を残さないため、出力先を生成ごとに作り直す。
    # 上の検証により、リポジトリ内のソースや文書を誤って削除することを防いでいる。
    if resolved_output_path.exists():
        shutil.rmtree(resolved_output_path)

    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{SOURCE_ROOT}{os.pathsep}{existing_python_path}"
        if existing_python_path
        else str(SOURCE_ROOT)
    )
    module_names = [feature.module for feature in _collect_features()]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pdoc",
            "-o",
            str(resolved_output_path),
            "--docformat",
            "google",
            *module_names,
        ],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
    )
    print(f"HTML API リファレンスを生成しました: {resolved_output_path}")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="実装済み機能一覧と pdoc HTML API リファレンスを生成する。"
    )
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_HTML_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="機能一覧が最新か検査する。HTML は生成しない。",
    )
    parser.add_argument(
        "--skip-html",
        action="store_true",
        help="機能一覧だけを更新し、HTML は生成しない。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """コマンドライン引数に従って API ドキュメントを生成する。

    Args:
        argv: コマンドライン引数。None の場合は実行プロセスの引数を使う。

    Returns:
        生成または検査に成功した場合は 0。

    Raises:
        SystemExit: 一覧が古い場合、または pdoc が未導入の場合。
        ValueError: HTML 出力先がソースを破壊し得る場所を指す場合。
        subprocess.CalledProcessError: pdoc による生成が失敗した場合。
    """

    args = _parse_args(argv)
    _write_or_check_catalog(args.catalog, check=bool(args.check))
    if not args.check and not args.skip_html:
        _build_html(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
