function p = mba_p(varargin)
%MBA_P  Resolve a data file/path relative to the configured data root.
%   p = mba_p('RawData3.mat')        -> <MBA_DATA_ROOT>/RawData3.mat
%   p = mba_p('sub','file.mat')      -> <MBA_DATA_ROOT>/sub/file.mat
%
%   The data root is defined in MBA_CONFIG (default <repo>/example_data,
%   overridable via the MBA_DATA_ROOT environment variable).
%
%   See also MBA_CONFIG.
cfg = mba_config();
p = fullfile(cfg.data_root, varargin{:});
end
