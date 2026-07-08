function r = mba_repo_root()
%MBA_REPO_ROOT  Absolute path to the repository root.
%   Resolves relative to this file (utils/mba_repo_root.m), so it works
%   regardless of the current working directory.
r = fileparts(fileparts(mfilename('fullpath')));
end
