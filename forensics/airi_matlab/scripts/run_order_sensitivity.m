% Pair-order sensitivity diagnostic (Phase I).
% Methodological diagnostic, NOT part of the literal paper reproduction.
% Keeps the same 30 directed pair observations; only permutes presentation
% order. Deterministic ReDisCA fit should be invariant; random-phase
% p-values may change because the surrogate treats z as an ordered sequence.

if ~exist('FORENSIC_ROOT','var') || isempty(FORENSIC_ROOT)
    this_dir = fileparts(mfilename('fullpath'));
    FORENSIC_ROOT = fileparts(this_dir);
end
lit = fullfile(FORENSIC_ROOT, 'results', 'literal', 'airi_literal_after_spoc.mat');
if exist(lit,'file')~=2
    error('Literal result not found: %s', lit);
end
addpath(fullfile(FORENSIC_ROOT, 'vendor', 'pinned_stock_SPoC', 'SPoC'));
addpath(fullfile(FORENSIC_ROOT, 'vendor', 'pinned_stock_SPoC', 'utils'));

S = load(lit);
outdir = fullfile(FORENSIC_ROOT, 'results', 'order_sensitivity');
if ~exist(outdir,'dir'); mkdir(outdir); end

X0 = S.Xspoc;
z0 = S.z;
Ne = size(X0, 3);
orders = struct();
orders(1).name = 'literal';
orders(1).perm = 1:Ne;
rng(42,'twister');
orders(2).name = 'randperm_42';
orders(2).perm = randperm(Ne);
rng(99,'twister');
orders(3).name = 'randperm_99';
orders(3).perm = randperm(Ne);
orders(4).name = 'reverse';
orders(4).perm = Ne:-1:1;

seed_grid = [1 2 3];
summary = [];
for oi = 1:length(orders)
    perm = orders(oi).perm;
    Xp = X0(:,:,perm);
    zp = z0(perm);
    % Deterministic fit: 0 bootstrap iterations
    [W,A,lam,~,Cxx,Cxxz,Cxxe] = spoc(Xp, zp, 'n_bootstrapping_iterations', 0);
    save(fullfile(outdir, sprintf('order_%s_fit.mat', orders(oi).name)), ...
        'W','A','lam','Cxx','Cxxz','Cxxe','perm','-v7.3');
    for si = 1:length(seed_grid)
        rng(seed_grid(si), 'twister');
        [~,~,lam_s,p] = spoc(Xp, zp, 'n_bootstrapping_iterations', 1000);
        save(fullfile(outdir, sprintf('order_%s_seed_%d.mat', ...
            orders(oi).name, seed_grid(si))), 'lam_s','p','perm','-v7.3');
        row = [oi, seed_grid(si), p(1:min(6,end))];
        summary = [summary; row]; %#ok<AGROW>
        fprintf('%s seed %d p(1:6)=%s\n', orders(oi).name, seed_grid(si), ...
            mat2str(p(1:min(6,end)),4));
    end
end
save(fullfile(outdir,'order_sensitivity_summary.mat'), 'summary','orders','-v7.3');
