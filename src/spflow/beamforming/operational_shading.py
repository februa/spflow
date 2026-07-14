"""互換性のためchannel shading設計APIを新しい責務packageから再公開する。"""

from ..array_design.shading import (
    OperationalFixedBeamShadingDesignConfig,
    OperationalShadingDefinition,
    OperationalShadingDesignConfig,
    load_operational_shading,
    run_operational_fixed_beam_shading_design,
    run_operational_shading_design,
)
from ..array_design.shading import (
    _kaiser_bessel_channel_window as _kaiser_bessel_channel_window,
)
from ..array_design.shading import (
    _sidelobe_distribution_metrics as _sidelobe_distribution_metrics,
)

__all__ = [
    "OperationalFixedBeamShadingDesignConfig",
    "OperationalShadingDefinition",
    "OperationalShadingDesignConfig",
    "load_operational_shading",
    "run_operational_fixed_beam_shading_design",
    "run_operational_shading_design",
]
