"""整相方式設計書に対応する評価artifactをRelease添付用ZIPへまとめる。"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = REPOSITORY_ROOT / "artifacts" / "beamforming"
RELEASE_ID = "alignment-method-evaluation-v1"
OUTPUT_ROOT = REPOSITORY_ROOT / "artifacts" / "releases"


@dataclass(frozen=True)
class ArtifactGroup:
    """設計書の章と評価artifactディレクトリの対応を表す。

    source_directoryはリポジトリ作業領域内の生成物、release_directoryはZIP内の配置である。
    評価処理そのものやartifactの再計算は責務に含めない。
    """

    document_sections: tuple[str, ...]
    evaluation_purpose: str
    source_directory: str
    release_directory: str
    evaluation_script: str


GROUPS = (
    ArtifactGroup(("5",), "整数遅延MVDRの共分散破綻条件", "coarse_covariance_integer_delay_mvdr", "05_coarse_covariance_integer_delay_mvdr", "evaluations/beamforming/coarse_covariance_integer_delay_mvdr.py"),
    ArtifactGroup(("6",), "S/T・1/2方式の5方式比較", "direction_cut_mvdr_method_comparison", "06_direction_cut_mvdr_method_comparison", "evaluations/beamforming/direction_cut_mvdr_method_comparison.py"),
    ArtifactGroup(("11",), "EBAE/MVDRとFIR実現方式の比較", "ebae_mvdr_s0_s1_t1_t2_fir_sweep", "11_ebae_mvdr_s1_s2a_t1_t2a_fir_sweep", "evaluations/beamforming/ebae_mvdr_s1_s2a_t1_t2a_fir_sweep.py"),
    ArtifactGroup(("12",), "S/T方向性sanity確認", "ebae_mvdr_s_t_directionality_sanity", "12_ebae_mvdr_s_t_directionality_sanity", "evaluations/beamforming/ebae_mvdr_s_t_directionality_sanity.py"),
    ArtifactGroup(("13",), "T1/T2a必要tap数", "ebae_t1_t2_tap_requirement_sweep", "13_ebae_t1_t2a_tap_requirement_sweep", "evaluations/beamforming/ebae_t1_t2a_tap_requirement_sweep.py"),
    ArtifactGroup(("15",), "S2a/S2b・T2a/T2b等価性", "ebae_s2a_s2b_t2a_t2b_equivalence", "15_ebae_s2a_s2b_t2a_t2b_equivalence", "evaluations/beamforming/ebae_s2a_s2b_t2a_t2b_equivalence.py"),
    ArtifactGroup(("16.1--16.8",), "単一信号・tap・SNR成立性", "alignment_single_source_validation_matrix", "16_alignment_single_source_validation_matrix", "evaluations/beamforming/alignment_single_source_validation_matrix.py"),
    ArtifactGroup(("16.9--16.12",), "近傍強弱信号の可視性", "alignment_weak_source_visibility_sweep", "16_alignment_weak_source_visibility_sweep", "evaluations/beamforming/alignment_weak_source_visibility_sweep.py"),
    ArtifactGroup(("17",), "低周波狭帯域・広帯域tap sweep", "low_frequency_narrow_broad_tap_sweep", "17_low_frequency_narrow_broad_tap_sweep", "evaluations/beamforming/low_frequency_narrow_broad_tap_sweep.py"),
    ArtifactGroup(("18",), "広帯域endfireのS2a/T2a切り分け", "s2a_broad_endfire", "18_s2a_broad_endfire", "evaluations/beamforming/s2a_broad_endfire.py"),
    ArtifactGroup(("19",), "有限tap・実整数delay buffer正式評価", "formal_s2a_t2a_endfire", "19_formal_s2a_t2a_endfire", "evaluations/beamforming/formal_s2a_t2a_endfire.py"),
)


def _sha256(path: Path) -> str:
    """Release内ファイルの同一性確認用SHA-256を返す。"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_release() -> Path:
    """評価artifactを章別に配置し、manifestとSHA-256を含むZIPを生成する。

    Returns:
        Release添付用ZIPの絶対path。

    Raises:
        FileNotFoundError: 対応表に記載したartifactディレクトリが存在しない場合。
    """
    staging = OUTPUT_ROOT / RELEASE_ID
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    file_records: list[dict[str, str | int]] = []
    for group in GROUPS:
        source = ARTIFACT_ROOT / group.source_directory
        if not source.is_dir():
            raise FileNotFoundError(source)
        destination = staging / group.release_directory
        shutil.copytree(source, destination)
        for path in sorted(destination.rglob("*")):
            if path.is_file() and path.name != ".DS_Store":
                # pathはZIP rootからの相対位置で記録し、OS固有の絶対pathをReleaseへ持ち込まない。
                relative = path.relative_to(staging).as_posix()
                file_records.append({"path": relative, "size_bytes": path.stat().st_size, "sha256": _sha256(path)})

    manifest = {
        "release_id": RELEASE_ID,
        "document": "output/word/整相方式設計・評価結果.docx",
        "document_source": "doc/SpFlow/整相方式設計結果.md",
        "artifact_groups": [asdict(group) for group in GROUPS],
        "files": file_records,
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checksum_lines = [f"{record['sha256']}  {record['path']}" for record in file_records]
    (staging / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    archive = OUTPUT_ROOT / f"{RELEASE_ID}.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        for path in sorted(staging.rglob("*")):
            if path.is_file() and path.name != ".DS_Store":
                zip_file.write(path, arcname=f"{RELEASE_ID}/{path.relative_to(staging).as_posix()}")
    return archive


if __name__ == "__main__":
    print(build_release())
