function saveMyFig(defaultName, savePath, figHandle)
    % saveMyFig saves MATLAB figure(s) to a specified folder.
    % If a file path is provided via the second argument, the figures are saved there.
    % Otherwise, it saves the figures to a folder on the Desktop named after the calling script.
    %
    % If the calling script's name cannot be determined, it saves to a folder named 'saveMyFig'.
    % It saves each figure in .png, .svg, and .pdf formats with filenames prefixed by the current date.
    %
    % Additionally, if the figure's UserData property is not empty, the data is saved in a .mat file
    % using the same base filename as the figure.
    %
    % Moreover, if the figure has a title saved in the 'Name' property (and that title is not just the default),
    % that title is incorporated into the filename.
    %
    % After successfully saving all formats for all figures, all figures are automatically closed.
    %
    % Parameters:
    %   defaultName - (Optional) Default name for the figure if it doesn't have one.
    %                 Default is 'Figure'.
    %   savePath    - (Optional) Full file path (folder) where figures should be saved.
    %                 If empty or not provided, figures are saved to a folder on the Desktop.
    %   figHandle   - Handle to the figure to be saved. If 'All', saves all existing figures.
    %                 If empty, uses the current figure.
    %
    % Example:
    %   saveMyFig('MyPlot', 'MyFigures', gcf);         % Saves the current figure to 'MyFigures' with defaultName 'MyPlot'.
    %   saveMyFig('DefaultName', '', 'All');              % Saves all existing figures with their respective names to the Desktop folder.
    %   saveMyFig();                                      % Saves the current figure with defaultName 'Figure' to the Desktop folder.
    
    %% Input Validation and Defaults
    if nargin < 1 || isempty(defaultName)
        defaultName = 'Figure';
    end
    
    if nargin < 2
        savePath = [];
    end
    
    if nargin < 3 || isempty(figHandle)
        figHandle = gcf; % Use current figure if no handle is provided
    end

    %% Determine the Target Folder
    % Output locations are configured centrally in MBA_CONFIG. If no savePath
    % is given, figures go to <figure_root>/<calling script name>. If a
    % relative savePath is given, it is resolved against <figure_root>; an
    % absolute savePath is used as-is.
    cfg = mba_config();
    if isempty(savePath)
        % Determine Calling Script's Name
        scriptName = getCallingScriptName();
        if isempty(scriptName)
            % Default to 'saveMyFig' folder if calling script's name cannot be determined
            scriptName = 'saveMyFig';
            warning('Unable to determine the calling script''s name. Saving figures in the ''saveMyFig'' folder.');
        end
        targetFolder = fullfile(cfg.figure_root, scriptName);
    elseif isAbsolutePath(savePath)
        % Use the provided absolute savePath as the target folder
        targetFolder = char(savePath);
    else
        % Resolve a relative savePath against the configured figure root
        targetFolder = fullfile(cfg.figure_root, char(savePath));
    end

    %% Check if Folder Exists, If Not, Create It
    if ~exist(targetFolder, 'dir')
        mkdirStatus = mkdir(targetFolder);
        if ~mkdirStatus
            error('Failed to create folder: %s', targetFolder);
        end
    end

    %% Get Current Date
    currentDate = datetime('now');
    datePrefix = datestr(currentDate, 'yyyymmdd'); % Format: YYYYMMDD

    %% Determine Figures to Save
    if ischar(figHandle) && strcmpi(figHandle, 'All')
        % If figHandle is 'All', retrieve all figure handles
        figures = findall(0, 'Type', 'figure');
    else
        % Otherwise, ensure figHandle is a valid figure handle
        if isscalar(figHandle) && ishandle(figHandle) && strcmp(get(figHandle, 'Type'), 'figure')
            figures = figHandle;
        else
            error('Invalid figHandle provided. It must be a figure handle or the string ''All''.');
        end
    end

    %% Store all figure handles and numbers for later closing
    figureHandles = figures;
    figureNumbers = arrayfun(@(fh) fh.Number, figures);
    
    %% Iterate Through Each Figure and Save
    for i = 1:length(figures)
        fh = figures(i);
        %% Store figure number early (before any operations that might invalidate the handle)
        figNumber = figureNumbers(i);
        
        %% Get Figure Name and Incorporate Title if Available
        titleStr = get(fh, 'Name');
        if isempty(titleStr) || strcmp(titleStr, 'Figure') || strcmp(titleStr, defaultName)
            figName = defaultName;
        else
            figName = sprintf('%s_%s', defaultName, titleStr);
        end
        set(fh, 'Name', figName); % Update figure name in MATLAB

        %% Construct Base Filename with Date Prefix
        baseFilename = sprintf('%s_%s', datePrefix, figName);
        safeBaseFilename = makeValidFilename(baseFilename);

        %% Define File Extensions
        fileExtensions = {'png', 'svg', 'pdf'};

        %% Determine Unique Suffix
        % Initialize suffix as empty (no suffix)
        suffix = '';
        isUnique = false;

        while ~isUnique
            % Construct potential filenames with current suffix
            potentialFilenames = cellfun(@(ext) ...
                sprintf('%s%s.%s', safeBaseFilename, suffix, ext), ...
                fileExtensions, 'UniformOutput', false);

            % Check if any of the potential filenames exist
            filesExist = cellfun(@(fname) exist(fullfile(targetFolder, fname), 'file'), potentialFilenames);

            if ~any(filesExist)
                isUnique = true; % Found a unique suffix
            else
                % Increment suffix
                if isempty(suffix)
                    suffix = '_1';
                else
                    num = sscanf(suffix, '_%d');
                    if isempty(num)
                        suffix = '_1';
                    else
                        suffix = sprintf('_%d', num + 1);
                    end
                end
            end
        end

        %% Final Filenames with Unique Suffix
        finalFilenames = cellfun(@(ext) ...
            sprintf('%s%s.%s', safeBaseFilename, suffix, ext), ...
            fileExtensions, 'UniformOutput', false);

        %% Save the Figure in Both Formats
        for i = 1:length(fileExtensions)
            ext = lower(fileExtensions{i});
            filename = finalFilenames{i};
            fullPath = fullfile(targetFolder, filename);

            try
                % Check if figure contains a tiledlayout
                tiledLayoutObj = findall(fh, 'Type', 'tiledlayout');
                hasTiledLayout = ~isempty(tiledLayoutObj);
                
                if hasTiledLayout
                    % Store original figure properties
                    originalVisible = get(fh, 'Visible');
                    originalUnits = get(fh, 'Units');
                    originalColor = get(fh, 'Color');
                    originalRenderer = get(fh, 'Renderer');
                    
                    % Temporarily set figure properties for clean export
                    set(fh, 'Visible', 'on');
                    set(fh, 'Units', 'pixels');
                    set(fh, 'Color', 'white');
                    set(fh, 'Renderer', 'painters'); % Force vector renderer
                    drawnow; % Ensure figure is fully rendered
                end
                
                switch ext
                    case 'png'
                        % Save as high-resolution PNG using exportgraphics with maximum resolution
                        if hasTiledLayout
                            try
                                % Method 1: Export tiledlayout directly with transparent background
                                exportgraphics(tiledLayoutObj(1), fullPath, 'Resolution', 150, 'BackgroundColor', 'none');
                            catch
                                try
                                    % Method 2: Export figure with transparent background
                                    exportgraphics(fh, fullPath, 'Resolution', 150, 'BackgroundColor', 'none');
                                catch
                                    % Method 3: Use print as last resort
                                    print(fh, fullPath, '-dpng', '-r600');
                                end
                            end
                        else
                            exportgraphics(fh, fullPath, 'Resolution', 600, 'BackgroundColor', 'white');
                        end
                    case 'svg'
                        % Save as vector SVG using print command
                        if hasTiledLayout
                            try
                                % Method 1: Export tiledlayout as SVG using print
                                print(fh, fullPath, '-dsvg', '-painters');
                            catch
                                try
                                    % Method 2: Try without painters renderer
                                    print(fh, fullPath, '-dsvg');
                                catch
                                    % Method 3: Use saveas as last resort
                                    saveas(fh, fullPath, 'svg');
                                end
                            end
                        else
                            % For regular figures, use print for SVG
                            print(fh, fullPath, '-dsvg', '-painters');
                        end
                    case 'pdf'
                        % Save as PDF using exportgraphics or print command
                        if hasTiledLayout
                            try
                                % Method 1: Export tiledlayout directly as PDF
                                exportgraphics(tiledLayoutObj(1), fullPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
                            catch
                                try
                                    % Method 2: Export figure as PDF
                                    exportgraphics(fh, fullPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
                                catch
                                    % Method 3: Use print as last resort
                                    print(fh, fullPath, '-dpdf', '-fillpage');
                                end
                            end
                        else
                            try
                                % Try exportgraphics first for better quality
                                exportgraphics(fh, fullPath, 'ContentType', 'vector', 'BackgroundColor', 'white');
                            catch
                                % Fall back to print command
                                print(fh, fullPath, '-dpdf', '-fillpage');
                            end
                        end
                    otherwise
                        warning('Unsupported file extension: %s. Skipping save for this format.', ext);
                end
                
                if hasTiledLayout
                    % Restore original figure properties
                    set(fh, 'Visible', originalVisible);
                    set(fh, 'Units', originalUnits);
                    set(fh, 'Color', originalColor);
                    set(fh, 'Renderer', originalRenderer);
                end
                
                fprintf('Figure (Handle: %d) saved as: %s\n', figNumber, fullPath);
            catch ME
                warning('Failed to save figure (Handle: %d) as %s: %s', figNumber, ext, ME.message);
            end
        end

        %% Save associated UserData if present
        userData = get(fh, 'UserData');
        if ~isempty(userData)
            dataFilename = sprintf('%s%s.mat', safeBaseFilename, suffix);
            fullDataPath = fullfile(targetFolder, dataFilename);
            try
                save(fullDataPath, 'userData');
                fprintf('UserData (Handle: %d) saved as: %s\n', figNumber, fullDataPath);
            catch ME
                warning('Failed to save UserData for figure (Handle: %d): %s', figNumber, ME.message);
            end
        end

        %% Update the figure's name in MATLAB to reflect the saved filenames
        % Concatenate the saved file names separated by a comma.
        updatedName = strjoin(finalFilenames, ', ');
        set(fh, 'Name', updatedName);
    end
    
    % %% Close all figures after all saves are complete
    % for i = 1:length(figureHandles)
    %     fh = figureHandles(i);
    %     figNumber = figureNumbers(i);
    %     try
    %         % Check if figure handle is still valid before closing
    %         if isvalid(fh) && ishandle(fh)
    %             close(fh);
    %         end
    %     catch ME
    %         % Silently handle any errors during closing
    %     end
    % end
    % fprintf('\nAll open figures have been closed.\n');
end

function tf = isAbsolutePath(p)
    % Returns true for Windows drive paths (), UNC paths (\\server),
    % and POSIX absolute paths (/...).
    p = char(p);
    tf = ~isempty(regexp(p, '^([A-Za-z]:[\\/]|[\\/]{2}|[\\/])', 'once'));
end

function scriptName = getCallingScriptName()
    % getCallingScriptName retrieves the name of the script that called this function.
    %
    % Returns:
    %   scriptName - Name of the calling script without the .m extension.

    stack = dbstack('-completenames');
    scriptName = '';

    % Iterate through the stack to find the first script (not a function)
    for i = 2:length(stack) % Start at 2 to skip this function
        [~, name, ext] = fileparts(stack(i).file);
        if strcmp(ext, '.m') && ~strcmp(name, 'saveMyFig') && ~contains(name, '@')
            scriptName = name;
            break;
        end
    end
end

function validName = makeValidFilename(name)
    % makeValidFilename replaces or removes invalid characters for filenames.
    % This ensures that the filename is valid across different operating systems.

    invalidChars = ['<', '>', ':', '"', '/', '\', '|', '?', '*'];
    replacement = '_';
    validName = name;
    for i = 1:length(invalidChars)
        validName(validName == invalidChars(i)) = replacement;
    end
    validName = strtrim(validName); % Remove leading and trailing whitespace
    if isempty(validName)
        validName = 'Figure';
    end
end

