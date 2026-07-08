function init()
%INIT  Add the repository's code folders to the MATLAB path.
%   Run this once at the start of a session:
%       >> init
%
%   It adds ising-model/ and utils/ (recursively) to the path so the model
%   code and its helper functions can be found. Data and output locations are
%   configured separately in MBA_CONFIG.

repoDir = fileparts(mfilename('fullpath'));
addpath(genpath(fullfile(repoDir, 'ising-model')));
addpath(genpath(fullfile(repoDir, 'utils')));
fprintf('added ising-model/ and utils/ to the path.\n');
cfg = mba_config();
fprintf('  data_root   = %s\n', cfg.data_root);
fprintf('  figure_root = %s\n', cfg.figure_root);
end
