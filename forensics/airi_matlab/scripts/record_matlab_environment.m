% Record MATLAB environment and which -all for critical functions.
% This is the Phase A probe. It does not run ReDisCA.

if ~exist('FORENSIC_ROOT','var') || isempty(FORENSIC_ROOT)
    this_dir = fileparts(mfilename('fullpath'));
    FORENSIC_ROOT = fileparts(this_dir);
end
outdir = fullfile(FORENSIC_ROOT, 'environment');
if ~exist(outdir,'dir'); mkdir(outdir); end
addpath(fullfile(FORENSIC_ROOT, 'vendor', 'pinned_stock_SPoC', 'SPoC'));
addpath(fullfile(FORENSIC_ROOT, 'vendor', 'pinned_stock_SPoC', 'utils'));

fid = fopen(fullfile(outdir, 'matlab_which_all.txt'), 'w');
fprintf(fid, 'computer=%s\n', computer);
fprintf(fid, 'version=%s\n', version);
try
    v = ver;
    for k = 1:length(v)
        fprintf(fid, 'toolbox\t%s\t%s\n', v(k).Name, v(k).Version);
    end
catch
end
funcs = {'spoc','whiten_data','create_Cxxz','random_phase_surrogate', ...
         'butter','filtfilt','eig','cov','std','rng','ft_topoplotER'};
for i = 1:length(funcs)
    fprintf(fid, '\n==== which -all %s ====\n', funcs{i});
    w = which(funcs{i}, '-all');
    if isempty(w)
        fprintf(fid, 'NOT FOUND\n');
    elseif ischar(w)
        fprintf(fid, '%s\n', w);
    else
        for j = 1:length(w)
            fprintf(fid, '%s\n', w{j});
        end
    end
end
fclose(fid);
fprintf('Wrote %s\n', fullfile(outdir, 'matlab_which_all.txt'));
