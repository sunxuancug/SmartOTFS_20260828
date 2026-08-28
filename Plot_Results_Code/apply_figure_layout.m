function apply_figure_layout(layout_file)
%APPLY_FIGURE_LAYOUT Restore positions of open MATLAB figures.
%
% Usage:
%   apply_figure_layout
%   apply_figure_layout('my_layout.mat')

    if nargin < 1 || isempty(layout_file)
        script_dir = fileparts(mfilename('fullpath'));
        layout_file = fullfile(script_dir, 'figure_layout.mat');
    end

    if ~exist(layout_file, 'file')
        warning('Figure layout file not found: %s', layout_file);
        return;
    end

    data = load(layout_file, 'layout');
    if ~isfield(data, 'layout')
        warning('Layout file has no variable named layout: %s', layout_file);
        return;
    end

    figs = findall(0, 'Type', 'figure');
    for i = 1:numel(data.layout)
        fig = find_figure_by_layout(figs, data.layout(i));
        if isempty(fig) || ~isvalid(fig)
            continue;
        end
        fig.Units = 'pixels';
        fig.Position = data.layout(i).Position;
    end

    fprintf('Applied figure layout: %s\n', layout_file);
end

function fig = find_figure_by_layout(figs, item)
    fig = [];
    if isfield(item, 'Name') && ~isempty(item.Name)
        for k = 1:numel(figs)
            if strcmp(figs(k).Name, item.Name)
                fig = figs(k);
                return;
            end
        end
    end

    if isfield(item, 'Number')
        for k = 1:numel(figs)
            if figs(k).Number == item.Number
                fig = figs(k);
                return;
            end
        end
    end
end
