function spflow_render_figures_batch(manifest_path)
%SPFLOW_RENDER_FIGURES_BATCH 評価manifestに含まれる複数図を.figへ変換する。

manifest = jsondecode(fileread(manifest_path));
for figure_index = 1:numel(manifest.figures)
    entry = manifest.figures(figure_index);
    spflow_render_figure(char(entry.spec_path), char(entry.output_fig_path));
end
end
