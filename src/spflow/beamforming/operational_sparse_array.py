"""互換性のため運用array設計APIを新しい責務packageから再公開する。"""

from ..array_design.operational_array import (
    OperationalSparseArrayDefinition,
    OperationalSparseArrayDesignConfig,
    design_operational_sparse_array,
    load_operational_sparse_array,
    save_operational_sparse_array,
)

__all__ = [
    "OperationalSparseArrayDefinition",
    "OperationalSparseArrayDesignConfig",
    "design_operational_sparse_array",
    "load_operational_sparse_array",
    "save_operational_sparse_array",
]
