function saveMyFigAsPDF(defaultName, savePath)
% saveMyFigAsPDF saves all open MATLAB figures into a single PDF file (appending).
%   saveMyFigAsPDF(defaultName, savePath)
%   - defaultName: (Optional) Base name for the PDF file. Default is 'Figure'.
%   - savePath: (Optional) Folder to save the PDF. If empty, saves to Desktop/<calling script>.
%
% The PDF will be named <date>_<defaultName>.pdf and will contain all open figures.

if nargin < 1 || isempty(defaultName)
    defaultName = 'Figure';
end
if nargin < 2
    savePath = [];
end

% Determine the Target Folder (same logic as saveMyFig). Output locations are
% configured centrally in MBA_CONFIG; relative paths resolve against figure_root.
cfg = mba_config();
if isempty(savePath)
    scriptName = getCallingScriptName();
    if isempty(scriptName)
        scriptName = 'saveMyFig';
        warning('Unable to determine the calling script''s name. Saving figures in the ''saveMyFig'' folder.');
    end
    targetFolder = fullfile(cfg.figure_root, scriptName);
elseif isAbsolutePath(savePath)
    targetFolder = char(savePath);
else
    targetFolder = fullfile(cfg.figure_root, char(savePath));
end
if ~exist(targetFolder, 'dir')
    mkdirStatus = mkdir(targetFolder);
    if ~mkdirStatus
        error('Failed to create folder: %s', targetFolder);
    end
end

% Get Current Date
currentDate = datetime('now');
datePrefix = datestr(currentDate, 'yyyymmdd');

% PDF filename
pdfFilename = sprintf('%s_%s.pdf', datePrefix, defaultName);
pdfFullPath = fullfile(targetFolder, makeValidFilename(pdfFilename));

% Get all open figures
figures = findall(0, 'Type', 'figure');
if isempty(figures)
    warning('No open figures to save.');
    return;
end

% Sort figures by number (for consistent order)
[~, idx] = sort([figures.Number]);
figures = figures(idx);

% Save all figures to a single PDF (append)
for i = 1:length(figures)
    fh = figures(i);
    if i == 1
        exportgraphics(fh, pdfFullPath, 'ContentType', 'vector', 'BackgroundColor', 'none');
    else
        exportgraphics(fh, pdfFullPath, 'ContentType', 'vector', 'BackgroundColor', 'none', 'Append', true);
    end
    fprintf('Figure (Handle: %d) appended to: %s\n', fh.Number, pdfFullPath);
end
end

function scriptName = getCallingScriptName()
    stack = dbstack('-completenames');
    scriptName = '';
    for i = 2:length(stack)
        [~, name, ext] = fileparts(stack(i).file);
        if strcmp(ext, '.m') && ~strcmp(name, 'saveMyFigAsPDF') && ~contains(name, '@')
            scriptName = name;
            break;
        end
    end
end

function validName = makeValidFilename(name)
    invalidChars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*'];
    replacement = '_';
    validName = name;
    for i = 1:length(invalidChars)
        validName(validName == invalidChars(i)) = replacement;
    end
    validName = strtrim(validName);
    if isempty(validName)
        validName = 'Figure';
    end
end 