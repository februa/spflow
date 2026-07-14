"""アレイ幾何とchannel shadingの事前設計・評価部品を公開する。"""

from .bandwise import BandwiseArrayDesign
from .operational_array import (
    OperationalSparseArrayDefinition,
    OperationalSparseArrayDesignConfig,
    design_operational_sparse_array,
    load_operational_sparse_array,
    save_operational_sparse_array,
)
from .shading import (
    OperationalFixedBeamShadingDesignConfig,
    OperationalShadingDefinition,
    OperationalShadingDesignConfig,
    load_operational_shading,
    run_operational_fixed_beam_shading_design,
    run_operational_shading_design,
)
from .sparse_single_side import (
    SparseSingleSideArrayDesignConfig,
    SparseSingleSideArrayDesignResult,
    build_sparse_single_side_array_design,
    run_sparse_single_side_array_design,
)

__all__ = [
    "BandwiseArrayDesign",
    "OperationalFixedBeamShadingDesignConfig",
    "OperationalShadingDefinition",
    "OperationalShadingDesignConfig",
    "OperationalSparseArrayDefinition",
    "OperationalSparseArrayDesignConfig",
    "SparseSingleSideArrayDesignConfig",
    "SparseSingleSideArrayDesignResult",
    "build_sparse_single_side_array_design",
    "design_operational_sparse_array",
    "load_operational_shading",
    "load_operational_sparse_array",
    "run_operational_fixed_beam_shading_design",
    "run_operational_shading_design",
    "run_sparse_single_side_array_design",
    "save_operational_sparse_array",
]
