% SPoC random-phase inference at seeds 1, 2, 3 (Phase F).
% Requires a prior literal run that saved airi_literal_after_spoc.mat
% so the expensive filter/average/pair construction is not repeated.
% If that file is absent, this script stops; it does not silently rebuild
% with a different code path.
%
% The SPoC null remains random_phase_surrogate (not z permutation).

if ~exist('FORENSIC_ROOT','var') || isempty(FORENSIC_ROOT)
    this_dir = fileparts(mfilename('fullpath'));
    FORENSIC_ROOT = fileparts(this_dir);
end
lit = fullfile(FORENSIC_ROOT, 'results', 'literal', 'airi_literal_after_spoc.mat');
if exist(lit,'file')~=2
    error(['Literal result not found: %s\n' ...
           'Run scripts/run_literal_sensor_space.m first.'], lit);
end
addpath(fullfile(FORENSIC_ROOT, 'vendor', 'pinned_stock_SPoC', 'SPoC'));
addpath(fullfile(FORENSIC_ROOT, 'vendor', 'pinned_stock_SPoC', 'utils'));
addpath(fullfile(FORENSIC_ROOT, 'instrumented'));

S = load(lit);
seeds = [1 2 3];
outdir = fullfile(FORENSIC_ROOT, 'results', 'seeds');
if ~exist(outdir,'dir'); mkdir(outdir); end

for si = 1:length(seeds)
    FORENSIC_SEED = seeds(si);
    FORENSIC_OUT_DIR = outdir;
    rng(FORENSIC_SEED, 'twister');
    [W1, A1, lambda_values1, p_values1, Cxx1, Cxxz1, Cxxe1, lambda_samples, r_samples] = ...
        spoc_save_surrogates(S.Xspoc, S.z, 'n_bootstrapping_iterations', 1000);
    save(fullfile(outdir, sprintf('seed_%d_spoc.mat', FORENSIC_SEED)), ...
        'W1','A1','lambda_values1','p_values1','Cxx1','Cxxz1','Cxxe1', ...
        'lambda_samples','r_samples','FORENSIC_SEED','-v7.3');
    fprintf('seed %d: p_values1(1:min(8,end)) = %s\n', FORENSIC_SEED, ...
        mat2str(p_values1(1:min(8,end)), 6));
end
