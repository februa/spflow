function spflow_render_figure(spec_path, output_fig_path)
%SPFLOW_RENDER_FIGURE 評価用JSON + CSV specからMATLAB .figを再構成する。

spec = jsondecode(fileread(spec_path));
figure_handle = figure('Visible', 'off', 'Color', 'w');
tile_layout = tiledlayout(spec.nrows, spec.ncols, 'Padding', 'compact', 'TileSpacing', 'compact');

for axis_index = 1:numel(spec.axes)
    axis_spec = spec.axes(axis_index);
    nexttile(tile_layout, axis_spec.index);
    hold on;

    switch axis_spec.kind
        case 'line'
            for line_index = 1:numel(axis_spec.lines)
                line_spec = axis_spec.lines(line_index);
                x_values = readmatrix(line_spec.x_csv);
                y_values = readmatrix(line_spec.y_csv);
                plot(x_values(:), y_values(:), ...
                    'LineWidth', line_spec.line_width, ...
                    'LineStyle', char(line_spec.line_style), ...
                    'DisplayName', char(line_spec.label));
            end

        case 'image'
            image_matrix = readmatrix(axis_spec.image_csv);
            imagesc(axis_spec.x_lim, axis_spec.y_lim, image_matrix);
            set(gca, 'YDir', 'normal');
            colormap(gca, char(axis_spec.colormap));
            colorbar_handle = colorbar;
            ylabel(colorbar_handle, char(axis_spec.colorbar_label));
            for line_index = 1:numel(axis_spec.vlines)
                line_spec = axis_spec.vlines(line_index);
                xline(line_spec.x, ...
                    'LineWidth', line_spec.line_width, ...
                    'LineStyle', char(line_spec.line_style), ...
                    'Color', line_spec.color_rgb, ...
                    'DisplayName', char(line_spec.label));
            end
            for line_index = 1:numel(axis_spec.polylines)
                line_spec = axis_spec.polylines(line_index);
                x_values = readmatrix(line_spec.x_csv);
                y_values = readmatrix(line_spec.y_csv);
                plot(x_values(:), y_values(:), ...
                    'LineWidth', line_spec.line_width, ...
                    'LineStyle', char(line_spec.line_style), ...
                    'Color', line_spec.color_rgb, ...
                    'DisplayName', char(line_spec.label));
            end

        otherwise
            error('Unsupported axis kind: %s', axis_spec.kind);
    end

    xlabel(char(axis_spec.xlabel));
    ylabel(char(axis_spec.ylabel));
    title(char(axis_spec.title));
    if axis_spec.grid
        grid on;
    else
        grid off;
    end

    if has_legend_entries(axis_spec)
        legend('Location', char(axis_spec.legend_location));
    end
    hold off;
end

sgtitle(char(spec.suptitle));
annotation(figure_handle, 'textbox', [0.05, 0.005, 0.90, 0.04], ...
    'String', char(spec.caption), ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', ...
    'EdgeColor', 'none', ...
    'Interpreter', 'none');
savefig(figure_handle, output_fig_path);
close(figure_handle);
end

function result = has_legend_entries(axis_spec)
%HAS_LEGEND_ENTRIES 凡例対象のラベルが 1 つでもあるかを返す。
result = false;
if isfield(axis_spec, 'lines')
    for line_index = 1:numel(axis_spec.lines)
        if strlength(string(axis_spec.lines(line_index).label)) > 0
            result = true;
            return;
        end
    end
end
if isfield(axis_spec, 'vlines')
    for line_index = 1:numel(axis_spec.vlines)
        if strlength(string(axis_spec.vlines(line_index).label)) > 0
            result = true;
            return;
        end
    end
end
if isfield(axis_spec, 'polylines')
    for line_index = 1:numel(axis_spec.polylines)
        if strlength(string(axis_spec.polylines(line_index).label)) > 0
            result = true;
            return;
        end
    end
end
end
