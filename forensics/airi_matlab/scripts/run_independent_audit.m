% Independent reconstruction of SPoC inputs/outputs from a saved literal .mat
% (Phase E). Does not modify the primary run. Requires MATLAB.
%
% Compares:
%   Cxx vs mean of cov(Xspoc(:,:,e))
%   Cxxz vs create_Cxxz(Cxxe, z_std)
%   rank of Cxx at SPoC threshold ev(1)*1e-6
%   w' Cxx w
%   A vs Cxx*W / (W'*Cxx*W)

if ~exist('FORENSIC_ROOT','var') || isempty(FORENSIC_ROOT)
    this_dir = fileparts(mfilename('fullpath'));
    FORENSIC_ROOT = fileparts(this_dir);
end
lit = fullfile(FORENSIC_ROOT, 'results', 'literal', 'airi_literal_after_spoc.mat');
if exist(lit,'file')~=2
    error('Literal result not found: %s', lit);
end
addpath(fullfile(FORENSIC_ROOT, 'vendor', 'pinned_stock_SPoC', 'utils'));
S = load(lit);
X = S.Xspoc;
z = S.z_before_spoc_standardization;
if isempty(z)
    z = S.z;
end
[nT, nCh, Ne] = size(X);

% 1-4 pair / z audit
pair_table = [S.pair_i(:), S.pair_j(:), z(:)];
z_mean = mean(z(:));
z_std_sample = std(z(:));
z_std_pop = std(z(:), 1);
z_std = (z - z_mean) ./ z_std_sample;

% 6 selected pair epochs: first, a reverse pair, and a high-z pair
sel = [1, 6, find(z==max(z),1,'first')];
pair_cov = struct();
for k = 1:length(sel)
    e = sel(k);
    Xe = squeeze(X(:,:,e));
    pair_cov(k).e = e;
    pair_cov(k).pair = [S.pair_i(e), S.pair_j(e)];
    pair_cov(k).temporal_mean = mean(Xe, 1);
    pair_cov(k).matlab_cov = cov(Xe);
end

% 7 Cxx from Cxxe / pair covariances
Cxx_from_cov = zeros(nCh, nCh);
Cxxe_re = zeros(nCh, nCh, Ne);
for e = 1:Ne
    Ctmp = cov(squeeze(X(:,:,e)));
    Cxx_from_cov = Cxx_from_cov + Ctmp;
    Cxxe_re(:,:,e) = Ctmp;
end
Cxx_from_cov = Cxx_from_cov / Ne;
Cxxe_meanfree = Cxxe_re;
for e = 1:Ne
    Cxxe_meanfree(:,:,e) = Cxxe_re(:,:,e) - Cxx_from_cov;
end

% 8 Cxxz from standardized z
Cxxz_re = create_Cxxz(Cxxe_meanfree, z_std);

% 9 rank at SPoC threshold
ev = sort(eig(S.Cxx1), 'descend');
tol = ev(1) * 1e-6;
rank_spoc = sum(ev > tol);

% 12-13 normalization / pattern
W = S.W1;
nW = size(W,2);
wnorm = zeros(1,nW);
for k = 1:nW
    wnorm(k) = W(:,k)' * S.Cxx1 * W(:,k);
end
A_re = S.Cxx1 * W / (W' * S.Cxx1 * W);

err = struct();
err.Cxx_maxabs = max(abs(S.Cxx1(:) - Cxx_from_cov(:)));
err.Cxx_maxrel = err.Cxx_maxabs / max(abs(S.Cxx1(:)));
err.Cxxe_maxabs = max(abs(S.Cxxe1(:) - Cxxe_meanfree(:)));
err.Cxxz_maxabs = max(abs(S.Cxxz1(:) - Cxxz_re(:)));
err.Cxxz_maxrel = err.Cxxz_maxabs / max(abs(S.Cxxz1(:)));
err.A_maxabs = max(abs(S.A1(:) - A_re(:)));
err.wnorm_minus_1_maxabs = max(abs(wnorm - 1));

outdir = fullfile(FORENSIC_ROOT, 'results', 'literal');
save(fullfile(outdir, 'independent_audit.mat'), ...
    'pair_table','z_mean','z_std_sample','z_std_pop','z_std', ...
    'pair_cov','Cxx_from_cov','Cxxz_re','ev','tol','rank_spoc', ...
    'wnorm','A_re','err','sel','-v7.3');

fid = fopen(fullfile(outdir, 'independent_audit.txt'), 'w');
fprintf(fid, 'n_pairs = %d\n', Ne);
fprintf(fid, 'both_directions = %d\n', ...
    isequal(sortrows([S.pair_i(:), S.pair_j(:)]), ...
            sortrows([[S.pair_j(:); S.pair_i(:)], [S.pair_i(:); S.pair_j(:)]])));
fprintf(fid, 'z_mean = %.16g\n', z_mean);
fprintf(fid, 'z_sample_std = %.16g\n', z_std_sample);
fprintf(fid, 'rank_spoc_threshold = %.16g\n', tol);
fprintf(fid, 'rank_spoc = %d\n', rank_spoc);
fprintf(fid, 'Cxx_maxabs = %.16g\n', err.Cxx_maxabs);
fprintf(fid, 'Cxx_maxrel = %.16g\n', err.Cxx_maxrel);
fprintf(fid, 'Cxxe_maxabs = %.16g\n', err.Cxxe_maxabs);
fprintf(fid, 'Cxxz_maxabs = %.16g\n', err.Cxxz_maxabs);
fprintf(fid, 'Cxxz_maxrel = %.16g\n', err.Cxxz_maxrel);
fprintf(fid, 'A_maxabs = %.16g\n', err.A_maxabs);
fprintf(fid, 'wnorm_minus_1_maxabs = %.16g\n', err.wnorm_minus_1_maxabs);
fprintf(fid, 'lambda_values1 =\n');
fprintf(fid, ' %.16g\n', S.lambda_values1(:));
fclose(fid);
fprintf('Wrote independent_audit.txt\n');
