"""整相シミュレーション部品の互換公開ファサード。

実装責務は設定、ULA伝搬、共分散、重み設計、FIR実現、source levelの各moduleへ
分離する。このmoduleは既存のimport経路を維持するための再exportだけを担う。
"""

from spflow.simulation.alignment_config import AlignmentSimulationConfig
from spflow.simulation.alignment_coordinates import (
    ALIGNMENT_METHOD_IDS,
    to_original_input_coordinates,
)
from spflow.simulation.alignment_covariance import calculate_alignment_source_covariance
from spflow.simulation.alignment_weight_design import (
    ALIGNMENT_ALGORITHM_IDS,
    AlignmentWeightDesign,
    design_alignment_weights,
)
from spflow.simulation.frequency_weight_fir import (
    FrequencyWeightFirApproximation,
    approximate_frequency_weights_with_fir,
)
from spflow.simulation.source_beam_level import calculate_source_beam_level_db
from spflow.simulation.ula_propagation import (
    calculate_frequency_steering,
    calculate_ula_arrival_delays_s,
)

__all__ = [
    "ALIGNMENT_ALGORITHM_IDS",
    "ALIGNMENT_METHOD_IDS",
    "AlignmentSimulationConfig",
    "AlignmentWeightDesign",
    "FrequencyWeightFirApproximation",
    "approximate_frequency_weights_with_fir",
    "calculate_alignment_source_covariance",
    "calculate_frequency_steering",
    "calculate_source_beam_level_db",
    "calculate_ula_arrival_delays_s",
    "design_alignment_weights",
    "to_original_input_coordinates",
]
