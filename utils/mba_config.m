function cfg = mba_config()
%MBA_CONFIG  Central data/output locations for this repository.
%   This is the single source of truth for where input data is read from and
%   where figures/results are written. Edit the defaults below, or override
%   them without touching code via environment variables:
%
%       MBA_DATA_ROOT    folder containing the input datasets (.mat files)
%       MBA_FIGURE_ROOT  folder where figures and results are written
%
%   Defaults when the variables are unset:
%       data_root   = <repo>/example_data
%       figure_root = <repo>/results
%
%   Data files are resolved with MBA_P, e.g.  load(mba_p('RawData3.mat'))
%   resolves to  <data_root>/RawData3.mat.
%
%   See also MBA_P, SAVEMYFIG.

% Repository root = parent of the folder containing this file (utils/).
repoDir = fileparts(fileparts(mfilename('fullpath')));

dataRoot = getenv('MBA_DATA_ROOT');
if isempty(dataRoot)
    dataRoot = fullfile(repoDir, 'example_data');
end

figRoot = getenv('MBA_FIGURE_ROOT');
if isempty(figRoot)
    figRoot = fullfile(repoDir, 'results');
end

cfg = struct('data_root', dataRoot, 'figure_root', figRoot);
end
