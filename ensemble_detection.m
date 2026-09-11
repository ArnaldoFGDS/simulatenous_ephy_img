%Index	Value (ms)
% 1	908.718750
% 2	1258.452325
% 3	1583.109850
% 4	1922.109750
% 5	2259.129425
% 6	2600.821625
% 7	2938.389675
% 8	3281.966375


%% ===== 1. Load imaging =====
fname = 'D:\img\nad12\img\251028\img\r08\r08_from_tiffs_results_cnmf_sort';
data = load(fname);

S_all = data.est.S;
good_idx = data.proc.idx_components;
firing_rate_img = S_all(good_idx,:);

frame_rate = 30.95;
nFrames = size(firing_rate_img,2);

%% ===== 2. Load ephys and bin =====
ksDir = 'D:\output\nad12\251028\c01\filt\kilosort4';
fs = 40000;
sync_start = 3281.966375;
edges = sync_start + (0:nFrames)/frame_rate;

spike_times    = readNPY(fullfile(ksDir,'spike_times.npy'));
spike_clusters = readNPY(fullfile(ksDir,'spike_clusters.npy'));

spike_times_s = double(spike_times)/fs;

in_window = spike_times_s >= edges(1) & spike_times_s < edges(end);
spike_times_s  = spike_times_s(in_window);
spike_clusters = spike_clusters(in_window);

fid = fopen(fullfile(ksDir,'cluster_group.tsv'),'r');
C = textscan(fid,'%f%s','Delimiter','\t','HeaderLines',1);
fclose(fid);
good_clusters = C{1}(strcmpi(C{2},'good'));

is_good_spike  = ismember(spike_clusters, good_clusters);
spike_times_s  = spike_times_s(is_good_spike);
spike_clusters = spike_clusters(is_good_spike);

good_clusters_present = unique(spike_clusters,'stable');
nGood = numel(good_clusters_present);

firing_rate_eph = zeros(nGood, nFrames, 'single');
for i = 1:nGood
    cid = good_clusters_present(i);
    t_i = spike_times_s(spike_clusters==cid);
    firing_rate_eph(i,:) = histcounts(t_i, edges);
end

%% ===== 3. Merge imaging + ephys =====
firing_rate = [firing_rate_img; firing_rate_eph];

% -- Build unit_ids for ALL rows (img + eph) --------------------------------
% Imaging: assign negative IDs (-1, -2, ...) so they can be distinguished
% Ephys  : use the real kilosort cluster IDs
unit_ids_all = [(-1:-1:-size(firing_rate_img,1))'; ...
                double(good_clusters_present(:))];
% unit_ids_all(i) = identifier of row i of firing_rate (before any filtering)
% --------------------------------------------------------------------------

addpath([pwd '/functions/'])

%% ===== params estimation =====
estimate_params = 0;
include_shuff_version = 1;
est_params.ensamble_method     = 'svd';
est_params.normalize           = 'norm_mean_std';
est_params.smooth_SD           = 0;
est_params.num_comp            = 2:2:10;
est_params.shuffle_data_chunks = 0;
est_params.reps                = 1;
est_params.n_rep               = 1:est_params.reps;
est_params_list = f_build_param_list(est_params, {'smooth_SD', 'num_comp', 'n_rep'});
if include_shuff_version
    est_params_list_s = est_params_list;
end

%% ===== params NMF =====
ens_params.ensamble_method            = 'nmf';
ens_params.num_comp                   = 13;
ens_params.smooth_SD                  = 110;
ens_params.normalize                  = 'norm_mean_std';
ens_params.ensamble_extraction        = 'thresh';
ens_params.ensamble_extraction_thresh = 'shuff';
ens_params.signal_z_thresh            = 2.5;
ens_params.shuff_thresh_percent       = 95;
ens_params.hcluster_method            = 'average';
ens_params.hcluster_distance_metric   = 'cosine';
ens_params.corr_cell_thresh_percent   = 95;
ens_params.plot_stuff                 = 0;
ens_params.vol_period                 = 1/frame_rate*1000;

%% ===== 4. Remove inactive cells =====
active_cells = sum(firing_rate,2) > 0;
firing_rate  = firing_rate(active_cells,:);

% -- NEW: unit_ids after the active_cells filter ---------------------------
unit_ids_active = unit_ids_all(active_cells);
% unit_ids_active(i) = identifier of row i of firing_rate (active cells only)
% Negative IDs = imaging neurons, positive IDs = kilosort ephys clusters
% --------------------------------------------------------------------------

num_cells   = size(firing_rate,1);
perm        = randperm(num_cells);
firing_rate = firing_rate(perm,:);

% -- NEW: apply the same randperm to unit_ids ------------------------------
unit_ids_active_perm = unit_ids_active(perm);
% unit_ids_active_perm(i) = true identifier of row i of firing_rate
% and hence of firing_rate_sm (smoothing does not change row order)
% --------------------------------------------------------------------------

firing_rate_s      = f_shuffle_data(firing_rate);
firing_rate_norm   = f_normalize(firing_rate,   est_params.normalize);
firing_rate_norm_s = f_normalize(firing_rate_s, est_params.normalize);

%% ===== 5. Estimate params (optional) =====
if estimate_params
    est_params_list = f_ens_estimate_dim_params(firing_rate_norm, est_params_list, ens_params.vol_period);
    [~, min_ind] = min([est_params_list.test_err]);
    fprintf('Optimal smooth_SD = %d; num_comp = %d\n', est_params_list(min_ind).smooth_SD, est_params_list(min_ind).num_comp);

    if include_shuff_version
        fprintf('Estimating params shuff n/%d reps: ', numel(est_params_list_s));
        for n_par = 1:numel(est_params_list_s)
            params1 = est_params_list_s(n_par);
            params1.vol_period = ens_params.vol_period;
            accuracy = f_ens_estimate_corr_dim_cv(firing_rate_norm_s, params1);
            temp_fields = fields(accuracy);
            for n_fl = 1:numel(temp_fields)
                est_params_list_s(n_par).(temp_fields{n_fl}) = accuracy.(temp_fields{n_fl});
            end
            fprintf('--%d', n_par);
        end
        fprintf('\nDone\n');
        [~, min_ind] = min([est_params_list_s.test_err]);
        fprintf('Shuff optimal smooth_SD = %d; num_comp = %d\n', est_params_list_s(min_ind).smooth_SD, est_params_list(min_ind).num_comp);
    end
    f_plot_cv_error_3D(est_params_list, est_params_list_s, 'smooth_SD', 'num_comp', 'test_err');
    ax1 = gca;
    ax1.Title.String = sprintf('%s L2 error from raw, (%s)', est_params.ensamble_method, ax1.Title.String);
end

%% ===== 6. Smooth =====
firing_rate_sm = f_smooth_gauss(firing_rate, ens_params.smooth_SD*frame_rate/1000);

%% ===== 7. Extract ensembles =====
disp('Extracting ensembles...');
ens_out = f_ensemble_analysis_YS_raster(firing_rate_sm, ens_params);

%% ===== 8. Visualization =====
f_plot_raster_mean(firing_rate_sm(ens_out.ord_cell,:), 1);
title('raster cell sorted');

for n_comp = 1:numel(ens_out.cells.ens_list)
    cells1  = ens_out.cells.ens_list{n_comp};
    trials1 = ens_out.trials.ens_list{n_comp};
    scores1 = ens_out.cells.ens_scores(n_comp,:);
    coeffs1 = ens_out.coeffs(:, n_comp);
    coeffs1 = coeffs1(cells1);
    f_plot_ensamble_deets(firing_rate_sm, cells1, trials1, scores1, coeffs1);
    title([ens_params.ensamble_method ' ensemble ' num2str(n_comp)]);
end

%% ===== 9. SAVE =====
% -- NEW: save the row -> unit_id mapping ----------------------------------
% unit_ids_active_perm(i) = identifier of row i of firing_rate_sm
%   Negative IDs -> imaging neurons  (e.g. -1 = good_idx(1), -2 = good_idx(2)...)
%   Positive IDs -> kilosort ephys clusters (real cluster IDs)
% Use this vector to recover the correct neuron from an ens_list index.
%% ===== 9. SAVE =====
save_dir = 'D:\img\nad12\img\251028\img\r08\paper_detection';
if ~exist(save_dir, 'dir'), mkdir(save_dir); end

save(fullfile(save_dir, 'unit_ids_active_perm.mat'), 'unit_ids_active_perm', '-v7.3');
save(fullfile(save_dir, 'perm.mat'),                 'perm',                 '-v7.3');
save(fullfile(save_dir, 'active_cells.mat'),         'active_cells',         '-v7.3');
save(fullfile(save_dir, 'unit_ids_active.mat'),      'unit_ids_active',      '-v7.3');
save(fullfile(save_dir, 'ens_out.mat'),              'ens_out',              '-v7.3');
save(fullfile(save_dir, 'firing_rate_sm.mat'),       'firing_rate_sm',       '-v7.3');

fprintf('\nFiles saved in: %s\n', save_dir);
fprintf('    -> unit_ids_active_perm.mat\n');
fprintf('    -> perm.mat\n');
fprintf('    -> active_cells.mat\n');
fprintf('    -> unit_ids_active.mat\n');
fprintf('    -> ens_out.mat\n');
fprintf('    -> firing_rate_sm.mat\n');
disp('Done');
