% Run the instrumented AIRI sensor-space script (literal pair construction).
% Requires MATLAB + Signal Processing Toolbox (butter, filtfilt).
% Does not run source localization or FieldTrip plotting.
%
% Usage:
%   FORENSIC_ROOT = '/path/to/forensics/airi_matlab';
%   FORENSIC_SEED = 1;
%   run(fullfile(FORENSIC_ROOT, 'scripts', 'run_literal_sensor_space.m'))

if ~exist('FORENSIC_ROOT','var') || isempty(FORENSIC_ROOT)
    this_dir = fileparts(mfilename('fullpath'));
    FORENSIC_ROOT = fileparts(this_dir);
end
if ~exist('FORENSIC_SEED','var') || isempty(FORENSIC_SEED)
    FORENSIC_SEED = 1;
end
FORENSIC_DATA_DIR = fullfile(FORENSIC_ROOT, 'data');
FORENSIC_SPOC_ROOT = fullfile(FORENSIC_ROOT, 'vendor', 'pinned_stock_SPoC');
FORENSIC_OUT_DIR = fullfile(FORENSIC_ROOT, 'results', 'literal');
FORENSIC_USE_INSTRUMENTED_SPOC = false;

assert(exist(fullfile(FORENSIC_DATA_DIR,'MEG_AD_run1.mat'),'file')==2, ...
    'Missing MEG data. See README.md Phase B.');
assert(exist(fullfile(FORENSIC_DATA_DIR,'ibfctfprespm8_AD_run1_raw_tsss_mc.mat'),'file')==2, ...
    'Missing SPM header mat.');

diary(fullfile(FORENSIC_OUT_DIR, 'matlab_session.log'));
try
    ver
    ver('signal')
catch
end
which spoc -all
which butter -all
which filtfilt -all

run(fullfile(FORENSIC_ROOT, 'instrumented', ...
    'Redisca_tools_faces_3_random_norm_correct_instrumented.m'));
diary off
