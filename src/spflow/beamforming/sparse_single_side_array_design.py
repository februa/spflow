"""互換性のため片舷array設計APIを新しい責務packageから再公開する。"""

from ..array_design.sparse_single_side import (
    SparseSingleSideArrayDesignConfig,
    SparseSingleSideArrayDesignResult,
    build_sparse_single_side_array_design,
    run_sparse_single_side_array_design,
)

__all__ = [
    "SparseSingleSideArrayDesignConfig",
    "SparseSingleSideArrayDesignResult",
    "build_sparse_single_side_array_design",
    "run_sparse_single_side_array_design",
]
