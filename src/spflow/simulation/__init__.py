"""信号処理方式の設計・検証に使う、再利用可能なシミュレーション支援部品。"""

from spflow.simulation.alignment import (
    ALIGNMENT_ALGORITHM_IDS,
    ALIGNMENT_METHOD_IDS,
    AlignmentSimulationConfig,
    AlignmentWeightDesign,
    FrequencyWeightFirApproximation,
    approximate_frequency_weights_with_fir,
    calculate_alignment_source_covariance,
    calculate_frequency_steering,
    calculate_source_beam_level_db,
    calculate_ula_arrival_delays_s,
    design_alignment_weights,
    to_original_input_coordinates,
)
from spflow.simulation.numerics import SimulationPrecision
from spflow.simulation.streaming import (
    SignalBlock,
    StatefulIntegerDelay,
    VersionedCausalFIR,
)
from spflow.simulation.tone_scene import (
    ToneScene,
    ToneSceneSource,
    direction_from_azimuth_elevation,
    synthesize_tone_scene,
)

__all__ = [
    "ALIGNMENT_ALGORITHM_IDS",
    "ALIGNMENT_METHOD_IDS",
    "AlignmentSimulationConfig",
    "AlignmentWeightDesign",
    "FrequencyWeightFirApproximation",
    "SignalBlock",
    "SimulationPrecision",
    "StatefulIntegerDelay",
    "ToneScene",
    "ToneSceneSource",
    "VersionedCausalFIR",
    "approximate_frequency_weights_with_fir",
    "calculate_alignment_source_covariance",
    "calculate_frequency_steering",
    "calculate_source_beam_level_db",
    "calculate_ula_arrival_delays_s",
    "design_alignment_weights",
    "direction_from_azimuth_elevation",
    "synthesize_tone_scene",
    "to_original_input_coordinates",
]
