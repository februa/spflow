"""整相方式の設計結果と評価結果を統合した日本語Word文書を生成する。"""

# pyright: reportMissingImports=false
# python-docxは製品依存ではなくCodex文書runtimeから供給される成果物生成専用依存である。

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

from tools.render_formal_s2a_t2a_word import (
    CAUTION_FILL,
    HEADER_FILL,
    MUTED,
    TABLE_WIDTH_DXA,
    _add_body,
    _add_bullet,
    _add_heading,
    _configure_document,
    _set_cell_margins,
    _set_cell_shading,
    _set_run_font,
    _set_table_geometry,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_MARKDOWN = REPOSITORY_ROOT / "doc" / "SpFlow" / "整相方式設計結果.md"
OUTPUT_PATH = REPOSITORY_ROOT / "output" / "word" / "整相方式設計・評価結果.docx"
BLUE = RGBColor(46, 116, 181)
DARK_BLUE = RGBColor(31, 77, 120)


def _clean_inline_markdown(text: str) -> str:
    """Word本文へ不要なMarkdown記号だけを除去し、技術記号は保持する。"""
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = text.replace("**", "").replace("__", "")
    return text.replace("`", "")


def _add_title_block(document: Document) -> None:
    """設計と評価を統合した文書であることを冒頭に明示する。"""
    title = document.add_paragraph()
    title.paragraph_format.space_after = Pt(5)
    _set_run_font(title.add_run("整相方式 設計・評価結果"), size_pt=24, bold=True)
    subtitle = document.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(14)
    _set_run_font(
        subtitle.add_run("S/T共分散 × 直接/整数遅延分離 × a/b実現方式"),
        size_pt=13,
        color=MUTED,
    )
    _add_body(
        document,
        "本書は整相方式の数式、処理順、方式間の等価性、有限FIR実現、成立条件、"
        "および正式S2a/T2a評価結果を一体としてレビューするWord版である。"
        "EBAEとMVDRは重み設計部品、S/T・1/2・a/bは共分散とFIR実現座標の選択として分離して記述する。",
    )


def _mark_header_row(table: Any) -> None:
    """screen readerと改ページ後の再表示のため先頭行をheaderにする。"""
    properties = table.rows[0]._tr.get_or_add_trPr()
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    properties.append(marker)


def _add_method_flow(document: Document) -> None:
    """旧図1の代わりに方式軸と処理順を欠落なくWord-nativeで示す。"""
    _add_heading(document, "0. 方式全体の処理フロー", 1)
    _add_body(
        document,
        "方式名は、共分散構成（S/T）、FIR実現座標（1/2）、整数遅延後の実現構造（a/b）の3軸で読む。"
        "EBAEまたはMVDRは完成周波数重みを生成する交換可能な部品であり、整相方式名には含めない。",
    )

    headers = ("方式", "共分散", "重み設計前の座標変換", "時間領域実現", "意味")
    rows = (
        ("S1", "元入力の同一時刻S共分散", "なし", "元入力座標で完成重みを直接FIR化", "S共分散＋直接FIR"),
        ("S2a", "整数遅延前入力からS共分散", "整数遅延分の位相回転", "実整数遅延buffer＋残差完成重みFIR", "S共分散＋整数遅延分離"),
        ("S2b", "S2aと同じS共分散", "S2aと同じ位相回転", "固定主枝－差分補正枝", "S2aと同じ完成重みの別構造"),
        ("T1", "候補方位別時間切り出しT共分散", "切り出し時刻差の位相を元入力座標へ補正", "元入力座標で完成重みを直接FIR化", "T共分散＋直接FIR"),
        ("T2a", "候補方位別時間切り出しT共分散", "整数遅延後の残差座標へ回転", "実整数遅延buffer＋残差完成重みFIR", "T共分散＋整数遅延分離"),
        ("T2b", "T2aと同じT共分散", "T2aと同じ位相回転", "固定主枝－差分補正枝", "T2aと同じ完成重みの別構造"),
    )
    table = document.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for index, text in enumerate(headers):
        cell = table.rows[0].cells[index]
        _set_cell_shading(cell, HEADER_FILL)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_run_font(paragraph.add_run(text), size_pt=8.0, bold=True)
    for values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(values):
            paragraph = cells[index].paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER if index == 0 else WD_ALIGN_PARAGRAPH.LEFT
            _set_run_font(paragraph.add_run(value), size_pt=8.0)
    _set_table_geometry(table, (800, 2050, 2100, 2350, 2060))
    _mark_header_row(table)

    caption = document.add_paragraph()
    caption.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption.paragraph_format.space_before = Pt(4)
    caption.paragraph_format.space_after = Pt(8)
    _set_run_font(
        caption.add_run("図1相当　S/T・1/2・a/b方式の処理順と責務分離"),
        size_pt=9.0,
        color=MUTED,
    )

    for text in (
        "S2a/T2aのa方式: 完成残差重みを直接FIR化して整数遅延後信号へ適用する。",
        "S2b/T2bのb方式: 固定整相主枝から差分補正枝を減算し、a方式と同じ完成出力を得る。",
        "Ns=0ではEBAE補正項が存在せず、EBAE重みはCBFと一致する。",
        "S2a/T2aの共分散は整数遅延前入力から生成し、重み側だけを整数遅延後の残差座標へ回転する。",
    ):
        _add_bullet(document, text)


def _add_markdown_table(document: Document, lines: list[str]) -> None:
    """Markdown表を内容幅に応じた固定DXA表へ変換する。"""
    parsed = [[_clean_inline_markdown(cell.strip()) for cell in line.strip().strip("|").split("|")] for line in lines]
    if len(parsed) < 2:
        return
    headers = parsed[0]
    data_rows = parsed[2:] if all(set(cell) <= {"-", ":"} for cell in parsed[1]) else parsed[1:]
    column_count = len(headers)
    if column_count == 0:
        return
    # 短い識別列を狭くし、説明列へ残りを配分する。全cell幅の合計は9360 DXAに固定する。
    weights = [max(3, min(16, len(header))) for header in headers]
    if column_count >= 2:
        weights[-1] = max(weights[-1], 18)
    total_weight = sum(weights)
    widths = [max(600, int(TABLE_WIDTH_DXA * weight / total_weight)) for weight in weights]
    widths[-1] += TABLE_WIDTH_DXA - sum(widths)
    if widths[-1] < 600:
        deficit = 600 - widths[-1]
        widths[-1] = 600
        widths[0] -= deficit

    table = document.add_table(rows=1, cols=column_count)
    table.style = "Table Grid"
    for index, text in enumerate(headers):
        cell = table.rows[0].cells[index]
        _set_cell_shading(cell, HEADER_FILL)
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        _set_run_font(paragraph.add_run(text), size_pt=7.8, bold=True)
    for row in data_rows:
        cells = table.add_row().cells
        normalized = row + [""] * (column_count - len(row))
        for index, text in enumerate(normalized[:column_count]):
            cells[index].vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            paragraph = cells[index].paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == column_count - 1 else WD_ALIGN_PARAGRAPH.CENTER
            _set_run_font(paragraph.add_run(text), size_pt=7.8)
    _set_table_geometry(table, tuple(widths))
    _mark_header_row(table)
    spacer = document.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)


def _add_source_image(document: Document, markdown_path: str, caption: str, figure_number: int) -> None:
    """Markdown基準の相対画像を解決し、本文幅とalt textを設定する。"""
    path = (SOURCE_MARKDOWN.parent / markdown_path).resolve()
    if not path.exists():
        _add_body(document, f"[画像未生成: {path}]", bold_lead="注意: ")
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    inline_shape = paragraph.add_run().add_picture(str(path), width=Inches(6.35))
    alt_text = f"図{figure_number} {caption}"
    inline_shape._inline.docPr.set("descr", alt_text)
    inline_shape._inline.docPr.set("title", f"図{figure_number}")
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.space_after = Pt(8)
    _set_run_font(caption_paragraph.add_run(alt_text), size_pt=9.0, color=MUTED)


def _add_code_or_equation(document: Document, lines: list[str]) -> None:
    """数式とcodeを改行保持のmonospace blockとして表示する。"""
    table = document.add_table(rows=1, cols=1)
    table.style = "Table Grid"
    _set_cell_shading(table.cell(0, 0), "F7F8FA")
    paragraph = table.cell(0, 0).paragraphs[0]
    paragraph.paragraph_format.space_after = Pt(0)
    run = paragraph.add_run("\n".join(lines))
    run.font.name = "Menlo"
    run.font.size = Pt(8.5)
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Yu Gothic")
    _set_table_geometry(table, (TABLE_WIDTH_DXA,))
    # 1 cellの数式blockもアクセシビリティ監査上の先頭行を明示する。
    _mark_header_row(table)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def _render_markdown(document: Document) -> None:
    """既存設計書の全章をWordの見出し、表、図、本文へ変換する。"""
    lines = SOURCE_MARKDOWN.read_text(encoding="utf-8").splitlines()
    index = 0
    figure_number = 2
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if paragraph_buffer:
            _add_body(document, _clean_inline_markdown(" ".join(paragraph_buffer)))
            paragraph_buffer.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            index += 1
            continue
        if stripped.startswith("```"):
            flush_paragraph()
            block: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                block.append(lines[index])
                index += 1
            _add_code_or_equation(document, block)
            index += 1
            continue
        if stripped == "\\[":
            flush_paragraph()
            block = []
            index += 1
            while index < len(lines) and lines[index].strip() != "\\]":
                block.append(lines[index])
                index += 1
            _add_code_or_equation(document, block)
            index += 1
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph()
            level = min(3, len(heading_match.group(1)))
            text = _clean_inline_markdown(heading_match.group(2))
            if level == 1 and text == "整相方式 設計結果":
                index += 1
                continue
            _add_heading(document, text, level)
            index += 1
            continue
        image_match = re.match(r"^!\[([^]]*)\]\(([^)]+)\)$", stripped)
        if image_match:
            flush_paragraph()
            _add_source_image(document, image_match.group(2), image_match.group(1), figure_number)
            figure_number += 1
            index += 1
            continue
        if stripped.startswith("|"):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index])
                index += 1
            _add_markdown_table(document, table_lines)
            continue
        bullet_match = re.match(r"^[-*]\s+(.+)$", stripped)
        if bullet_match:
            flush_paragraph()
            _add_bullet(document, _clean_inline_markdown(bullet_match.group(1)))
            index += 1
            continue
        number_match = re.match(r"^(\d+)\.\s+(.+)$", stripped)
        if number_match:
            flush_paragraph()
            paragraph = document.add_paragraph(style="List Number")
            paragraph.paragraph_format.left_indent = Inches(0.5)
            paragraph.paragraph_format.first_line_indent = Inches(-0.25)
            paragraph.paragraph_format.space_after = Pt(4)
            _set_run_font(paragraph.add_run(_clean_inline_markdown(number_match.group(2))), size_pt=10.5)
            index += 1
            continue
        paragraph_buffer.append(stripped)
        index += 1
    flush_paragraph()


def build_document() -> Path:
    """設計書全章と評価成果物を統合した日本語DOCXを生成する。"""
    document = Document()
    _configure_document(document)
    document.sections[0].header.paragraphs[0].clear()
    header = document.sections[0].header.paragraphs[0]
    _set_run_font(header.add_run("SpFlow | 整相方式 設計・評価結果"), size_pt=9.0, color=MUTED)
    _add_title_block(document)
    _add_method_flow(document)
    document.add_page_break()
    _render_markdown(document)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    document.save(OUTPUT_PATH)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(build_document())
