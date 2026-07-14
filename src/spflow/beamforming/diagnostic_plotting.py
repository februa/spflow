"""互換性のためbeam診断描画をbeamforming_evaluationから再公開する。"""

from ..beamforming_evaluation.diagnostic_plotting import (
    BeamDiagnosticPlotUsageNotes,
    add_caption,
    build_beam_diagnostic_plot_usage_notes,
    centers_to_edges,
    configure_matplotlib_japanese,
    plot_bl_comparison,
    plot_bl_response,
    plot_btr_heatmap,
    plot_fraz_heatmap,
    require_matplotlib,
    save_figure,
    write_beam_diagnostic_plot_usage_notes,
)
from ..beamforming_evaluation.diagnostic_plotting import (
    plt as plt,
)

__all__ = [
    "BeamDiagnosticPlotUsageNotes",
    "add_caption",
    "build_beam_diagnostic_plot_usage_notes",
    "centers_to_edges",
    "configure_matplotlib_japanese",
    "plot_bl_comparison",
    "plot_bl_response",
    "plot_btr_heatmap",
    "plot_fraz_heatmap",
    "require_matplotlib",
    "save_figure",
    "write_beam_diagnostic_plot_usage_notes",
]
