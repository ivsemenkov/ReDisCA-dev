% FORENSIC INSTRUMENTED COPY of
%   AIRI-Institute/ReDisCA:Redisca_tools_faces_3_random_norm_correct.m
%   commit 15bc19cdc76989da202714b257f6de4d26a42c51
% Original remains byte-identical in ../original/AIRI-ReDisCA/
% See INSTRUMENTATION.md for the complete modification list.
%
% Scientific path through spoc(...) is unchanged.

close all
clear all

%% ===== FORENSIC: explicit paths (after clear all) =====
if ~exist('FORENSIC_ROOT','var') || isempty(FORENSIC_ROOT)
    this_file = mfilename('fullpath');
    FORENSIC_ROOT = fileparts(fileparts(this_file));
end
if ~exist('FORENSIC_DATA_DIR','var') || isempty(FORENSIC_DATA_DIR)
    FORENSIC_DATA_DIR = fullfile(FORENSIC_ROOT, 'data');
end
if ~exist('FORENSIC_SPOC_ROOT','var') || isempty(FORENSIC_SPOC_ROOT)
    FORENSIC_SPOC_ROOT = fullfile(FORENSIC_ROOT, 'vendor', 'pinned_stock_SPoC');
end
if ~exist('FORENSIC_OUT_DIR','var') || isempty(FORENSIC_OUT_DIR)
    FORENSIC_OUT_DIR = fullfile(FORENSIC_ROOT, 'results', 'literal');
end
if ~exist('FORENSIC_SEED','var') || isempty(FORENSIC_SEED)
    FORENSIC_SEED = 1;
end
if ~exist('FORENSIC_USE_INSTRUMENTED_SPOC','var') || isempty(FORENSIC_USE_INSTRUMENTED_SPOC)
    FORENSIC_USE_INSTRUMENTED_SPOC = false;
end
if ~exist(FORENSIC_OUT_DIR, 'dir')
    mkdir(FORENSIC_OUT_DIR);
end
addpath(fullfile(FORENSIC_SPOC_ROOT, 'SPoC'));
addpath(fullfile(FORENSIC_SPOC_ROOT, 'utils'));
if FORENSIC_USE_INSTRUMENTED_SPOC
    addpath(fullfile(FORENSIC_ROOT, 'instrumented'));
end
%% ===== END FORENSIC PATHS =====

bVerbose = true;
bEveryOther = true;
ThRDMArr = {'face','facevstool', 'tool','toolvsface','meaning','meaning1'};
RDM = ThRDMArr(2);
Nmc = 100;
bRandomizeLabels = false;
bExportGraphics = false;
lowCutOff   = 0.25;
highCutOff  = 20;

data = load(fullfile(FORENSIC_DATA_DIR, 'ibfctfprespm8_AD_run1_raw_tsss_mc.mat'));

data_meg =  load(fullfile(FORENSIC_DATA_DIR, 'MEG_AD_run1.mat'));


nTrials = length(data.D.trials);

for idx = 1:nTrials
    idTrial(idx) = str2num(data.D.trials(idx).label);
    sTrial{idx}  = data.D.trials(idx).label;
    tTrial(idx,1) =  str2num(data.D.trials(idx).label(1));
    tTrial(idx,2) =  str2num(data.D.trials(idx).label(2));
    tTrial(idx,3) =  str2num(data.D.trials(idx).label(3));
end;

bValid = (tTrial(:,1) == 1 | (tTrial(:,1) == 2 & tTrial(:,2) == 0));

bFaceSimple1  = bValid & (tTrial(:,2) == 5) & tTrial(:,3) == 1; % Type 3
bFaceSimple2  = bValid & (tTrial(:,2) == 6) & tTrial(:,3) == 1; % Type 3
bToolSimple1  = bValid & (tTrial(:,2) == 7) & tTrial(:,3) == 1; % Type 4
bToolSimple2  = bValid & (tTrial(:,2) == 8) & tTrial(:,3) == 1; % Type 4
bNonsense1    = bValid & (tTrial(:,2) == 0); % Type 5
bNonsense2    = bValid & (tTrial(:,2) == 9); % Type 5

idxTrial{1} = find(bFaceSimple1);
idxTrial{2} = find(bFaceSimple2);
idxTrial{3} = find(bToolSimple1);
idxTrial{4} = find(bToolSimple2);
idxTrial{5} = find(bNonsense1);
idxTrial{6} = find(bNonsense2);

if(bRandomizeLabels)
    idx = [];
    for i =1:length(idxTrial)
        idx = [idx ;idxTrial{i}(:)];
    end
    
    idx_shuffled = idx(randperm(length(idx)));
    rng = 1:length(idxTrial{1});
    for i =1:length(idxTrial)
        idxTrial{i} = idx_shuffled(rng);
        rng = rng + length(idxTrial{1});
    end
end;

%prefilter
[bf,af] = butter(3,[lowCutOff,highCutOff]/500);
for i = 1:size(data_meg.d,3)
    data_meg.d(:,:,i) = filtfilt(bf,af,data_meg.d(:,:,i)')';
end;

Nconds = length(idxTrial);

for idx = 1:length(idxTrial)
    mx{idx} = mean(data_meg.d(1:204,:,idxTrial{idx}),3);
end;


D = zeros(Nconds,Nconds);
if(strcmp(RDM,'face'))
    D(1,2) = 0.1; D(1,3) = 1; D(1,4) = 1;   D(1,5) = 1;   D(1,6) = 1;
                  D(2,3) = 1; D(2,4) = 1;   D(2,5) = 1;   D(2,6) = 1;
                              D(3,4) = 0.1; D(3,5) = 0.1; D(3,6) = 0.1;
                                            D(4,5) = 0.1; D(4,6) = 0.1;
                                            D(5,6) = 0.1;
elseif(strcmp(RDM,'facevstool'))

    D(1,2) = 0.1; D(1,3) = 1;   D(1,4) = 1;   D(1,5) = 0.5;   D(1,6) = 0.5;
                  D(2,3) = 1;   D(2,4) = 1;   D(2,5) = 0.5;   D(2,6) = 0.5;
                                D(3,4) = 0.1; D(3,5) = 0.5;   D(3,6) = 0.5;
                                              D(4,5) = 0.5;   D(4,6) =  0.5;
                                                            D(5,6) = 0.1;

elseif (strcmp(RDM,'tool'))
    D(1,2) = 0.1; D(1,3) = 1; D(1,4) = 1;   D(1,5) = 0.1;   D(1,6) = 0.1;
                  D(2,3) = 1; D(2,4) = 1;   D(2,5) = .1;    D(2,6) = .1;
                              D(3,4) = 0.1; D(3,5) = 1;     D(3,6) = 1;
                                            D(4,5) = 1;     D(4,6) = 1;
                                                            D(5,6) = 0.1;
elseif (strcmp(RDM,'tool1'))
    D(1,2) = 0.1; D(1,3) = 1; D(1,4) = 1;   D(1,5) = 0.5;   D(1,6) = 0.5;
                  D(2,3) = 1; D(2,4) = 1;   D(2,5) = .5;    D(2,6) = .5;
                              D(3,4) = 0.1; D(3,5) = .5;    D(3,6) = .5;
                                            D(4,5) = .5;     D(4,6) = .5;
                                                            D(5,6) = 0.1;

elseif (strcmp(RDM,'meaning'))

    D(1,2) = 0.1; D(1,3) = 0.1; D(1,4) = 0.1; D(1,5) = 1;   D(1,6) = 1;
                  D(2,3) = 0.1; D(2,4) = 0.1; D(2,5) = 1;   D(2,6) = 1;
                                D(3,4) = 0.1; D(3,5) = 1;   D(3,6) = 1;
                                              D(4,5) = 1;   D(4,6) =  1;
                                                            D(5,6) = 0.1;
elseif(strcmp(RDM,'meaning1'))

    D(1,2) = 0.1; D(1,3) = 1;   D(1,4) = 1;   D(1,5) = 1;   D(1,6) = 1;
                  D(2,3) = 1;   D(2,4) = 1;   D(2,5) = 1;   D(2,6) = 1;
                                D(3,4) = 0.1; D(3,5) = 1;   D(3,6) = 1;
                                              D(4,5) = 1;   D(4,6) =  1;
                                                            D(5,6) = 0.1;
end;

D = D+D';

clear Xspoc z
trange = 600:1500;
e = 1;
for i_cnd = 1:Nconds
    for j_cnd = 1:Nconds
        if(i_cnd==j_cnd) 
            continue; % avoid using nulls in the dissimilarity matrices
        end;
        %create epoched arrays
        Xi = mx{i_cnd}(:,trange);
        Xj = mx{j_cnd}(:,trange);
        Xspoc(:,:,e) = Xi'-Xj';
        % fetch the corresponding z-values
         z(e) = D(i_cnd,j_cnd);
        e = e+1;
    end;
end;

%% ===== FORENSIC: pair-table + RNG immediately before SPoC =====
pair_i = [];
pair_j = [];
ee = 1;
for i_cnd = 1:Nconds
    for j_cnd = 1:Nconds
        if(i_cnd==j_cnd)
            continue;
        end
        pair_i(ee) = i_cnd;
        pair_j(ee) = j_cnd;
        ee = ee + 1;
    end
end
z_before_spoc_standardization = z;
z_mean_before_std = mean(z(:));
z_std_sample = std(z(:));  % MATLAB sample SD, N-1
z_std_population = std(z(:), 1);
which_spoc = which('spoc', '-all');
which_whiten = which('whiten_data', '-all');
which_cxxz = which('create_Cxxz', '-all');
which_rps = which('random_phase_surrogate', '-all');
rng(FORENSIC_SEED, 'twister');
%% ===== END FORENSIC RNG =====

[W1, A1, lambda_values1, p_values1,Cxx1, Cxxz1, Cxxe1] = ...
spoc(Xspoc, z, 'n_bootstrapping_iterations',1000);

%% ===== FORENSIC: save immediately after SPoC, then stop =====
% Original continues into Nmc time-series permutation, FieldTrip plots,
% and exportgraphics. Those are after the sensor-space ReDisCA/SPoC
% computation. Source localization is a later forensic stage.
which_spoc_after = which('spoc', '-all');
save(fullfile(FORENSIC_OUT_DIR, 'airi_literal_after_spoc.mat'), ...
    'mx', 'D', 'Xspoc', 'z', 'W1', 'A1', 'lambda_values1', 'p_values1', ...
    'Cxx1', 'Cxxz1', 'Cxxe1', 'trange', 'idxTrial', ...
    'pair_i', 'pair_j', 'Nconds', 'RDM', 'lowCutOff', 'highCutOff', ...
    'z_before_spoc_standardization', 'z_mean_before_std', 'z_std_sample', ...
    'z_std_population', 'FORENSIC_SEED', 'bf', 'af', ...
    'which_spoc', 'which_whiten', 'which_cxxz', 'which_rps', ...
    'bValid', 'bFaceSimple1', 'bFaceSimple2', 'bToolSimple1', ...
    'bToolSimple2', 'bNonsense1', 'bNonsense2', ...
    '-v7.3');
fid = fopen(fullfile(FORENSIC_OUT_DIR, 'which_resolved.txt'), 'w');
fprintf(fid, 'seed=%g\n', FORENSIC_SEED);
fprintf(fid, 'spoc:\n%s\n', char(which_spoc));
fprintf(fid, 'whiten_data:\n%s\n', char(which_whiten));
fprintf(fid, 'create_Cxxz:\n%s\n', char(which_cxxz));
fprintf(fid, 'random_phase_surrogate:\n%s\n', char(which_rps));
fclose(fid);
fprintf('FORENSIC: saved %s and returning before plotting/source-loc.\n', ...
    fullfile(FORENSIC_OUT_DIR, 'airi_literal_after_spoc.mat'));
return;
%% ===== END FORENSIC SAVE (original plotting/source-loc not executed) =====
