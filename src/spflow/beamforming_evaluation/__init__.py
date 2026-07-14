"""beamforming方式に依存しない小さな評価支援部品を公開する。"""

from spflow.beamforming_evaluation.fractional_response import (
    calculate_fractional_beam_response_matrix,
    normalize_evaluation_channel_weights,
)
from spflow.beamforming_evaluation.level_metrics import (
    calculate_real_tone_response_rms_level_db20,
    calculate_rms_level_db20,
)
from spflow.beamforming_evaluation.scan_grid import BeamScanGrid, build_beam_scan_grid
from spflow.beamforming_evaluation.signal_levels import (
    calculate_block_rms_levels_db20,
    calculate_one_sided_rms_spectrum_db20,
    calculate_tone_projection_rms_level_db20,
)

from .abf_like_metrics import (
    AbfLikeMetricDecision,
    AbfLikeNonSourceMetrics,
    SourceSectorMask,
    build_source_sector_mask,
    build_source_sector_mask_from_azimuths,
    calculate_abf_like_non_source_metrics,
    detect_source_beam_indices_from_level_peaks,
    judge_abf_like_non_source_metrics,
)
from .bl_component_metrics import (
    BlComponentEvaluation,
    BlLocalPeak,
    MixedBlConsistency,
    NoiseOnlyBlMetrics,
    TargetOnlyBlMetrics,
    evaluate_mixed_bl_consistency,
    evaluate_noise_only_bl,
    evaluate_target_only_bl,
)
from .diagnostic_plotting import (
    BeamDiagnosticPlotUsageNotes,
    build_beam_diagnostic_plot_usage_notes,
    centers_to_edges,
    plot_bl_comparison,
    plot_bl_response,
    plot_btr_heatmap,
    plot_fraz_heatmap,
    write_beam_diagnostic_plot_usage_notes,
)
from .evaluation_arrays import (
    BeamLevelDisplayArrays,
    BlShapeFeatures,
    build_beam_level_display_arrays,
    calculate_bl_shape_features,
    calculate_btr_relative_level_db,
)
from .evaluation_criteria import (
    BeamformingEvaluationCriterion,
    BeamformingEvaluationPattern,
    get_evaluation_criteria_for_pattern,
    list_beamforming_evaluation_criteria,
    list_beamforming_evaluation_patterns,
    write_beamforming_evaluation_criteria_markdown,
)

__all__ = [
    "AbfLikeMetricDecision",
    "AbfLikeNonSourceMetrics",
    "BeamDiagnosticPlotUsageNotes",
    "BeamLevelDisplayArrays",
    "BeamScanGrid",
    "BeamformingEvaluationCriterion",
    "BeamformingEvaluationPattern",
    "BlComponentEvaluation",
    "BlLocalPeak",
    "BlShapeFeatures",
    "MixedBlConsistency",
    "NoiseOnlyBlMetrics",
    "SourceSectorMask",
    "TargetOnlyBlMetrics",
    "build_beam_diagnostic_plot_usage_notes",
    "build_beam_level_display_arrays",
    "build_beam_scan_grid",
    "build_source_sector_mask",
    "build_source_sector_mask_from_azimuths",
    "calculate_block_rms_levels_db20",
    "calculate_abf_like_non_source_metrics",
    "calculate_bl_shape_features",
    "calculate_btr_relative_level_db",
    "calculate_fractional_beam_response_matrix",
    "calculate_one_sided_rms_spectrum_db20",
    "calculate_real_tone_response_rms_level_db20",
    "calculate_rms_level_db20",
    "calculate_tone_projection_rms_level_db20",
    "centers_to_edges",
    "detect_source_beam_indices_from_level_peaks",
    "evaluate_mixed_bl_consistency",
    "evaluate_noise_only_bl",
    "evaluate_target_only_bl",
    "get_evaluation_criteria_for_pattern",
    "judge_abf_like_non_source_metrics",
    "list_beamforming_evaluation_criteria",
    "list_beamforming_evaluation_patterns",
    "normalize_evaluation_channel_weights",
    "plot_bl_comparison",
    "plot_bl_response",
    "plot_btr_heatmap",
    "plot_fraz_heatmap",
    "write_beam_diagnostic_plot_usage_notes",
    "write_beamforming_evaluation_criteria_markdown",
]
