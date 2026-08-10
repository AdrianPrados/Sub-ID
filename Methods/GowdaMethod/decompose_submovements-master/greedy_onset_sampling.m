function [t0_samples, D_samples] = greedy_onset_sampling(hand_vel, hand_speed, n_samples, DT, no_submovements, varargin)
    defaults = {'greed_factor', 3, 'plot_dist', 0};
    greed_factor = get_var('greed_factor', 'defaults', defaults, varargin{:});
    plot_dist = get_var('plot_dist', 'defaults', defaults, varargin{:});
    
    hand_speed = sqrt(sum(hand_vel.^2, 2));
    [cost, pt_by_pt_cost] = residual_sse(hand_vel, hand_speed);
    
%     pdist = pt_by_pt_cost / sum(pt_by_pt_cost);
%     pdist = pdist .^ greed_factor;
%     error_dist = pdist / sum(pdist);
    
%     error_dist = pt_by_pt_cost / sum(pt_by_pt_cost);
%     cumul_error_dist = cumsum(error_dist);

%     D_max_bins = round(0.5/DT); %20;
%     D_max_diff = cumul_error_dist(D_max_bins+1:end) - cumul_error_dist(1:end-D_max_bins);
%     t0_ind = find(D_max_diff == max(D_max_diff), 1);
%     D_est = hand_speed(t0_ind)*5.15 + 0.365;  % constants derived by post-hoc fitting
%     samples = [max(t0_ind*DT - D_est/2, 0)];
%     if isempty(samples)
%         samples = [0];
%     end    
    
    pdist = pt_by_pt_cost / sum(pt_by_pt_cost);
    pdist = pdist .^ greed_factor;
    pdist = pdist / sum(pdist);
	local_mins = local_extrema(pdist, 'type', 'min');    
    local_min_inds = find(local_mins);
    
    if plot_dist
        figure(); hold on;
        plot(pdist)
        scatter(local_min_inds, pdist(local_min_inds))
    end

    cdist = cumsum(pdist);
    samples = [];
    

    try
        t0_sample_inds = randi([1, length(local_min_inds)-1], [1,n_samples]);
    catch
        t0_sample_inds = randint(1, n_samples, [1, length(local_min_inds)-1]); 
    end
    t0_samples = local_min_inds(t0_sample_inds);
    D_samples = local_min_inds(t0_sample_inds+1) - t0_samples;
    t0_samples = t0_samples * DT;
    D_samples = D_samples * DT;
    
%     while length(samples) < n_samples
%         new_samples = find(histc(rand(1, n_samples), cdist));
%         
%         % t0 samples = biggest local min indices less than the sample
%         
%         D_est = hand_speed(new_samples)*5.15 + 0.365;
%         new_samples = new_samples*DT - D_est'/2;
%         new_samples = new_samples(new_samples > 0);
% %         samples = samples * DT;
%         samples = [samples, new_samples];
%     end

end
