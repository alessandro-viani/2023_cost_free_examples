clc;clear; close all;

%% Setup
n_data = 100;
true_noise_std = linspace(1, 10, n_data);

n_dip = 4;
create_all = false;
run_all = false;
run_err = false;

cfg.evol_exp = 200;
cfg.t_start = 40;
cfg.t_stop = 60;
cfg.n_samples = 200;
cfg.lambda = 1;
cfg.NDIP = 10;

%% Data Creation
create_all_data(n_dip, true_noise_std, n_data, create_all);

%% Run FB
eval_all_fb(n_dip, true_noise_std, run_all, run_err, cfg);

%% Run PM
eval_all_pm(n_dip, true_noise_std, run_all, run_err, cfg);

%% Number of likelihood evaluation
n_like = n_like_eval(n_dip, n_data, cfg, run_err);
boxplot(n_like, {'PropFB', 'FB'});

%% Comparison
comparison(n_dip, n_data);

%% FUNCTIONS
function n_like = n_like_eval(n_dip, n_data, cfg, run_err)

if run_err
    n_like_hy = zeros(n_data,1);
    n_like_pm = zeros(n_data,1);
    for j=1:n_data
        disp(j);
        load([strcat('sol/sol_hy_', num2str(n_dip),'_', num2str(j), '.mat')]);
        load([strcat('sol/sol_pm_', num2str(n_dip),'_', num2str(j), '.mat')]);

        for i=1:cfg.evol_exp
            num_particles = posterior_hy.mod_sel(:,i) * 200;
            num_pos_evol = linspace(0, numel(num_particles)-1, numel(num_particles))';
            n_like_pos = num_particles .* num_pos_evol; % per ogni dipolo nella particella lo muovo
            n_like_par = 3*num_particles; % per ogni particella evolvo i 3 paramterti dipmom_std, noise_std, num
            n_like_hy(j) = n_like_hy(j) + ceil(sum(n_like_pos + n_like_par));

            num_particles = posterior_prop.old_mod_sel(:,i) * 200;
            num_pos_evol = linspace(0, numel(num_particles)-1, numel(num_particles))';
            n_like_pos = num_particles .* num_pos_evol; % per ogni dipolo nella particella lo muovo
            n_like_par = 2*num_particles; % per ogni particella evolvo i 2 paramterti dipmom_std, num
            n_like_pm(j) = n_like_pm(j) + ceil(sum(n_like_pos + n_like_par));
        end
    end

    n_like = [n_like_pm, n_like_hy];
    save('sol/n_like.mat', 'n_like');
else
    load('sol/n_like.mat');
end
end

function create_all_data(n_dip, true_noise_std, n_data, create_all)

if create_all==true
    load('data/triangulation_auditory_EEG.mat');
    for i=1:n_data
        disp(i);
        noise_std = true_noise_std(i);
        name = ['new_', num2str(n_dip), '_', num2str(i)];
        [~, ~, ~, ~] = data_creation(n_dip, V, L, noise_std, name);
    end
end
end

function eval_all_fb(n_dip, true_noise_std, run_all, run_err, cfg)
load('data/triangulation_auditory_EEG.mat');
if run_all==true
    for j=1:numel(true_noise_std)
        load(['data/', strcat('data_new_', num2str(n_dip), '_', num2str(j),'.mat')]);
        cfg.noise_std = 0.5*min(true_noise_std);
        posterior_hy = inverse_SESAME_hyper_noise(data, L, V, cfg);
        save([strcat('sol/sol_hy_', num2str(n_dip),'_', num2str(j), '.mat')], 'posterior_hy');
    end
end
if run_err==true
    %ERROR
    est_num_hy = zeros(size(true_noise_std,2),1);
    OSPA_hy = zeros(size(true_noise_std,2),1);
    err_cm_hy = zeros(size(true_noise_std,2),1);
    err_map_hy = zeros(size(true_noise_std,2),1);
    ESS_hy = zeros(size(true_noise_std,2),1);
    cpu_hy = zeros(size(true_noise_std,2),1);
    max_num_hy = zeros(size(true_noise_std,2),1);
    gof_hy = zeros(size(true_noise_std,2),1);
    for ii=1:size(true_noise_std,2)
        disp(['iter: ',num2str(ii)]);
        load([strcat('data/data_new_', num2str(n_dip),'_', num2str(ii),'.mat')]);
        load([strcat('sol/sol_hy_', num2str(n_dip),'_', num2str(ii),'.mat')]);

        est_num_hy(ii) = numel(posterior_hy.estimated_dipoles);
        if est_num_hy(ii) > 0
            Compute_PERM;
            OSPA_hy(ii) = Compute_OSPA(V(posterior_hy.estimated_dipoles,:), V(pos,:), PERM);
        else
            OSPA_hy(ii) = nan;
        end

        err_cm_hy(ii) = (true_noise_std(ii) - posterior_hy.noise_cm_hy)/true_noise_std(ii);
        err_map_hy(ii) = (true_noise_std(ii) - posterior_hy.noise_map_hy)/true_noise_std(ii);
        ESS_hy(ii) = posterior_hy.ESS(end);
        cpu_hy(ii) = posterior_hy.cpu_time;
        [~, idx] = max(posterior_hy.mod_sel);
        max_num_hy(ii) = max(idx);
        gof_hy(ii) = posterior_hy.gof;
    end
    save(['sol/analytics_hy_', num2str(n_dip),'.mat'], 'est_num_hy', 'OSPA_hy', 'err_cm_hy', 'err_map_hy', 'ESS_hy', 'cpu_hy', 'max_num_hy', 'gof_hy');
end
end

function eval_all_pm(n_dip, true_noise_std, run_all, run_err, cfg)
load('data/triangulation_auditory_EEG.mat');
if run_all==true
    for j=1:numel(true_noise_std)
        load(['data/', strcat('data_new_', num2str(n_dip),'_', num2str(j),'.mat')]);
        cfg.noise_std = 0.5*min(true_noise_std);
        posterior_prop = inverse_SESAME_prop_method(data, L, V, cfg);
        save([strcat('sol/sol_pm_', num2str(n_dip),'_', num2str(j), '.mat')], 'posterior_prop');
    end
end
if run_err==true
    %ERROR
    est_num_prop = zeros(size(true_noise_std,2),1);
    OSPA_prop = zeros(size(true_noise_std,2),1);
    err_cm_prop = zeros(size(true_noise_std,2),1);
    err_map_prop = zeros(size(true_noise_std,2),1);
    ESS_prop = zeros(size(true_noise_std,2),1);
    cpu_prop = zeros(size(true_noise_std,2),1);
    max_num_prop = zeros(size(true_noise_std,2),1);
    gof_prop = zeros(size(true_noise_std,2),1);

    est_num_prop_eb = zeros(size(true_noise_std,2),1);
    OSPA_prop_eb = zeros(size(true_noise_std,2),1);

    for ii=1:size(true_noise_std,2)
        disp(['iter: ',num2str(ii)]);
        load([strcat('data/data_new_', num2str(n_dip),'_', num2str(ii),'.mat')]);
        load([strcat('sol/sol_pm_', num2str(n_dip),'_', num2str(ii),'.mat')]);

        est_num_prop(ii) = numel(posterior_prop.estimated_dipoles);
        if est_num_prop(ii) > 0
            Compute_PERM;
            OSPA_prop(ii) = Compute_OSPA(V(posterior_prop.estimated_dipoles,:), V(pos,:), PERM);
        else
            OSPA_prop(ii) = nan;
        end

        err_cm_prop(ii) = (true_noise_std(ii) - posterior_prop.noise_cm_prop)/true_noise_std(ii);
        err_map_prop(ii) = (true_noise_std(ii) - posterior_prop.noise_map_prop)/true_noise_std(ii);
        ESS_prop(ii) = posterior_prop.ESS(end);
        cpu_prop(ii) = posterior_prop.cpu_time;
        [~, idx] = max(posterior_prop.old_mod_sel);
        max_num_prop(ii) = max(idx);

        est_num_prop_eb(ii) = numel(posterior_prop.estimated_dipoles_eb);
        if est_num_prop_eb(ii) > 0
            Compute_PERM;
            OSPA_prop_eb(ii) = Compute_OSPA(V(posterior_prop.estimated_dipoles_eb,:), V(pos,:), PERM);
        else
            OSPA_prop_eb(ii) = nan;
        end

        gof_prop(ii) = posterior_prop.gof;
    end
    save(['sol/analytics_pm_', num2str(n_dip),'.mat'], 'est_num_prop', 'est_num_prop_eb', 'OSPA_prop', 'OSPA_prop_eb', 'err_cm_prop', 'err_map_prop', 'ESS_prop', 'cpu_prop', 'max_num_prop', 'gof_prop');
end
end

function comparison(n_dip, cutoff)
load(['sol/analytics_pm_', num2str(n_dip), '.mat']);
load(['sol/analytics_hy_', num2str(n_dip), '.mat']);

disp('---------');
disp(['Right number found HY: ', num2str(100*numel(find(est_num_hy==n_dip))/100), '%']);
disp(['Right number found PROP: ', num2str(100*numel(find(est_num_prop==n_dip))/100), '%']);

err_cm_hy = sort(err_cm_hy);
err_map_hy = sort(err_map_hy);
err_cm_prop = sort(err_cm_prop);
err_map_prop = sort(err_map_prop);

subplot(2,2,1);
error = [OSPA_hy(1:cutoff), OSPA_prop(1:cutoff), OSPA_prop_eb(1:cutoff)];
boxplot(error, 'Notch', 'off', 'Labels', {'OSPA_hy', 'OSPA_prop', 'OSPA_prop_eb'});

subplot(2,2,2);
error = [err_cm_hy(1:cutoff), err_map_hy(1:cutoff), err_cm_prop(1:cutoff), err_map_prop(1:cutoff)];
boxplot(error, 'Notch', 'off', 'Labels', {'err_cm_hy', 'err_map_hy', 'err_cm_prop', 'err_map_prop'});
title('fixed iter');

subplot(2,2,3);
error = [cpu_hy(1:cutoff), cpu_prop(1:cutoff)];
boxplot(error, 'Notch', 'off', 'Labels', {'cpu_hy', 'cpu_prop'});

subplot(2,2,4);
error = [gof_hy(1:cutoff), gof_prop(1:cutoff)];
boxplot(error, 'Notch', 'off', 'Labels', {'gof_hy', 'gof_prop'});

figure;
subplot(1,3,1);
hold on;
plot(est_num_hy(1:cutoff),'b');
plot(est_num_prop(1:cutoff),'r');
subplot(1,3,2);
histogram(max_num_hy(1:cutoff)-1);
subplot(1,3,3);
histogram(max_num_prop(1:cutoff)-1);
end
