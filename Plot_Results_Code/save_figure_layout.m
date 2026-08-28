function save_figure_layout(layout_file)
%SAVE_FIGURE_LAYOUT Save positions of all open MATLAB figures.
%
% Usage:
%   save_figure_layout
%   save_figure_layout('my_layout.mat')

    if nargin < 1 || isempty(layout_file)
        script_dir = fileparts(mfilename('fullpath'));
        layout_file = fullfile(script_dir, 'figure_layout.mat');
    end

    figs = findall(0, 'Type', 'figure');
    if isempty(figs)
        warning('No open figures found.');
        return;
    end

    layout = struct('Name', {}, 'Number', {}, 'Position', {});
    for i = 1:numel(figs)
        fig = figs(i);
        old_units = fig.Units;
        fig.Units = 'pixels';
        layout(i).Name = fig.Name;
        layout(i).Number = fig.Number;
        layout(i).Position = fig.Position;
        fig.Units = old_units;
    end

    save(layout_file, 'layout');
    fprintf('Saved figure layout: %s\n', layout_file);
    for i = 1:numel(layout)
        fprintf('  Figure %d | %s | [%g %g %g %g]\n', ...
            layout(i).Number, layout(i).Name, layout(i).Position);
    end
end
