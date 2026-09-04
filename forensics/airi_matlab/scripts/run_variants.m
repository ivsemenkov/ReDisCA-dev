% Controlled one-at-a-time variants (Phase H).
% ONLY after a literal AIRI run has been saved.
% Each variant changes ONE construction choice, then calls the same SPoC.

if ~exist('FORENSIC_ROOT','var') || isempty(FORENSIC_ROOT)
    this_dir = fileparts(mfilename('fullpath'));
    FORENSIC_ROOT = fileparts(this_dir);
end
lit = fullfile(FORENSIC_ROOT, 'results', 'literal', 'airi_literal_after_spoc.mat');
if exist(lit,'file')~=2
    error(['Literal result not found: %s\n' ...
           'Complete the literal AIRI run before variants.'], lit);
end
addpath(fullfile(FORENSIC_ROOT, 'vendor', 'pinned_stock_SPoC', 'SPoC'));
addpath(fullfile(FORENSIC_ROOT, 'vendor', 'pinned_stock_SPoC', 'utils'));

S = load(lit);
outdir = fullfile(FORENSIC_ROOT, 'results', 'variants');
if ~exist(outdir,'dir'); mkdir(outdir); end
Nconds = S.Nconds;
mx = S.mx;
D = S.D;
trange = S.trange;
n_boot = 1000;

% ---------- Variant 1: UNIQUE PAIRS i < j ----------
[X_u, z_u, pi_u, pj_u] = local_build_pairs(mx, D, trange, 'unique');
rng(1,'twister');
[W,A,lam,p,Cxx,Cxxz,Cxxe] = spoc(X_u, z_u, 'n_bootstrapping_iterations', n_boot);
save(fullfile(outdir,'variant1_unique_pairs.mat'), ...
    'X_u','z_u','pi_u','pj_u','W','A','lam','p','Cxx','Cxxz','Cxxe','-v7.3');

% ---------- Variant 2: NO TEMPORAL DEMEANING (Gram / T) ----------
% Keep directed pairs. Replace MATLAB cov (centered, /(T-1)) with
% uncentered Gram X'*X / T corresponding to the printed paper-style
% unscaled correlation of the difference time series.
[X_d, z_d] = local_build_pairs(mx, D, trange, 'directed');
[nT, nCh, Ne] = size(X_d);
Cxxe_g = zeros(nCh, nCh, Ne);
Cxx_g = zeros(nCh, nCh);
for e = 1:Ne
    Xe = squeeze(X_d(:,:,e));           % T x channels
    G = (Xe' * Xe) / nT;                % uncentered Gram / T
    Cxxe_g(:,:,e) = G;
    Cxx_g = Cxx_g + G;
end
Cxx_g = Cxx_g / Ne;
Cxxe_g_mf = Cxxe_g;
for e = 1:Ne
    Cxxe_g_mf(:,:,e) = Cxxe_g(:,:,e) - Cxx_g;
end
z_std = (z_d - mean(z_d(:))) ./ std(z_d(:));
Cxxz_g = create_Cxxz(Cxxe_g_mf, z_std);
rng(1,'twister');
[W,A,lam,p,Cxx,Cxxz,Cxxe] = spoc([], z_d, ...
    'n_bootstrapping_iterations', n_boot, ...
    'Cxx', Cxx_g, 'Cxxz', Cxxz_g, 'Cxxe', Cxxe_g_mf);
save(fullfile(outdir,'variant2_uncentered_gram.mat'), ...
    'Cxx_g','Cxxz_g','Cxxe_g_mf','W','A','lam','p','-v7.3');

% ---------- Variant 3: SUM vs MEAN scaling of Cxx / Cxxz ----------
% Mean scaling is stock SPoC (divide by Ne). Sum scaling multiplies
% both Cxx and Cxxz by Ne. Generalized eigenvectors of (Cxxz, Cxx)
% are invariant to a common positive scale; eigenvalues scale.
Cxx_sum = S.Cxx1 * size(S.Xspoc,3);
Cxxz_sum = S.Cxxz1 * size(S.Xspoc,3);
Cxxe_sum = S.Cxxe1 * size(S.Xspoc,3);
rng(1,'twister');
[W,A,lam,p,Cxx,Cxxz,Cxxe] = spoc([], S.z, ...
    'n_bootstrapping_iterations', n_boot, ...
    'Cxx', Cxx_sum, 'Cxxz', Cxxz_sum, 'Cxxe', Cxxe_sum);
save(fullfile(outdir,'variant3_sum_scaling.mat'), ...
    'Cxx_sum','Cxxz_sum','W','A','lam','p','-v7.3');

% ---------- Variant 4: population SD (N) vs sample SD (N-1) ----------
z_raw = S.z_before_spoc_standardization;
z_pop = (z_raw - mean(z_raw(:))) ./ std(z_raw(:), 1);
% Feed already-standardized z, but spoc() will standardize again with
% sample SD. To isolate the convention we precompute Cxxz with pop-SD
% z and pass Cxx/Cxxe from the literal run, skipping spoc's z std by
% passing Cxxz directly.
Cxxz_pop = create_Cxxz(S.Cxxe1, z_pop);
rng(1,'twister');
[W,A,lam,p,Cxx,Cxxz,Cxxe] = spoc([], z_pop, ...
    'n_bootstrapping_iterations', n_boot, ...
    'Cxx', S.Cxx1, 'Cxxz', Cxxz_pop, 'Cxxe', S.Cxxe1);
save(fullfile(outdir,'variant4_population_sd.mat'), ...
    'z_pop','Cxxz_pop','W','A','lam','p','-v7.3');

fprintf('Variants written to %s\n', outdir);

function [Xspoc, z, pair_i, pair_j] = local_build_pairs(mx, D, trange, mode)
Nconds = length(mx);
Xspoc = [];
z = [];
pair_i = [];
pair_j = [];
e = 1;
for i_cnd = 1:Nconds
    for j_cnd = 1:Nconds
        if strcmp(mode,'unique')
            if j_cnd <= i_cnd, continue; end
        else
            if i_cnd == j_cnd, continue; end
        end
        Xi = mx{i_cnd}(:,trange);
        Xj = mx{j_cnd}(:,trange);
        Xspoc(:,:,e) = Xi'-Xj';
        z(e) = D(i_cnd,j_cnd);
        pair_i(e) = i_cnd;
        pair_j(e) = j_cnd;
        e = e + 1;
    end
end
end
