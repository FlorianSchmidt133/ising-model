function tf = isAbsolutePath(p)
%ISABSOLUTEPATH  True for absolute paths (Windows drive, UNC, or POSIX root).
%   Used to decide whether a figure save path should be resolved against the
%   configured figure_root (see MBA_CONFIG) or used as-is.
p = char(p);
tf = ~isempty(regexp(p, '^([A-Za-z]:[\\/]|[\\/]{2}|[\\/])', 'once'));
end
