function [movements, submovement_recon, segments, costs, fits, fit_costs, runtime, time_elapsed, n_iterations, n_func_evals] = ...
    decompose_submovements_v2(hand_kin, varargin)
%{
Parameters
----------
hand_kin : matrix
    N x 6 matrix of [pos, vel, acc]
verbose : bool, default = 1
    True to print debugging statements
n_submovements : int, default = -1
    Number of submovements to decompose, default is to iterate starting from 1
method : string, default = 'scattershot'
    Fitting type. Options are 'scattershot' or 'greedy' (not yet yet properly implemented)
prev_decomp : array, default=[]
    Results of previous decomposition, used for resampling parameters if zero_greed == 0
still_threshold : float, default=0.005
    Threshold of endpoint speed in m/s to indicate the start/stop of movement segments
min_ampl : float, default=0.02
    Minimum peak speed of a movement segment to qualify for decomposition
bin_size_ms : float, default=10
    Sampling period, in ms
isi : float, default=0.1
    Constraint specifying minimum time between submovements, in seconds
mse_perc_thresh : float, default=0.02
    % MSE threshold at which to end decomposition
use_cost_diff : bool, default=1
    Flag to indicate that the optimization should terminate if the MSE doesn't improve by 0.1 % by the addition of a submovement
fn_type : string, default='min_jerk'
    Prototype submovement function to fit a sum of
vel_inds : 1-d array, default=[3,4]
    Column indicies of the hand_kin matrix which correspond to the velocity dimension
D_min : float, default=0.150
    Minimum duration of a submovement, in seconds
proc_idx : int, default=1
    Which portion of the decomposition to handle (only used for cluser-based decompositions)
n_procs : int, default=1
    Number of cores over which the decomposition is being split (only used for cluster-based decompositions)
zero_greed : bool, default=0
    If true, sample initial parameters randomly

Returns
-------
movements
    N x M matrix of submovement parameters, M is the number of parameters per submovement
submovement_recon
    K x 2 matrix of reconstructed hand velocity from the submovement parameters
segments
    indicies of the start and end of each of the segment splits from the hand_kin matrix
costs
fits
fit_costs
runtime
    Time required to decompose the entire hand_kin
time_elapsed
    Time required to decompose each segment
n_iterations
    Number of optimization routine iterations required to decompose each segment
n_func_evals
    Number of cost function evaluations required to decompose each segment
    
%}
defaults = {'verbose', 1, ...
    'n_submovements', -1, ...
    'method', 'scattershot', ...
    'prev_decomp', [], ...
    'still_threshold', 0.005, ...
    'min_ampl', 0.02, ...
    'bin_size_ms', 10, ...
    'isi', 0.1, ...
    'mse_perc_thresh', 0.02, ...
    'use_cost_diff', 1, ...
    'fn_type', 'min_jerk', ...
    'vel_inds', [3,4], ...
    'D_min', 0.150, ...
    'proc_idx', 1, ...
    'n_procs', 1 ...
    'zero_greed', 0, ...
};

prev_decomp     = get_var('prev_decomp', 'defaults', defaults, varargin{:});
verbose         = get_var('verbose', 'defaults', defaults, varargin{:});
n_submovements  = get_var('n_submovements', 'defaults', defaults, varargin{:});
method          = get_var('method', 'defaults', defaults, varargin{:});
min_ampl        = get_var('min_ampl', 'defaults', defaults, varargin{:});
still_threshold = get_var('still_threshold', 'defaults', defaults, varargin{:});
isi             = get_var('isi', 'defaults', defaults, varargin{:});
bin_size_ms     = get_var('bin_size_ms', 'defaults', defaults, varargin{:});
mse_perc_thresh = get_var('mse_perc_thresh', 'defaults', defaults, varargin{:});
use_cost_diff   = get_var('use_cost_diff', 'defaults', defaults, varargin{:});
fn_type         = get_var('fn_type', 'defaults', defaults, varargin{:});
vel_inds        = get_var('vel_inds', 'defaults', defaults, varargin{:});
D_min           = get_var('D_min', 'defaults', defaults, varargin{:});
proc_idx        = get_var('proc_idx', 'defaults', defaults, varargin{:});
n_procs         = get_var('n_procs', 'defaults', defaults, varargin{:});
zero_greed      = get_var('zero_greed', 'defaults', defaults, varargin{:});

%% Error checking on parameters
valid_methods = {'scattershot', 'greedy'};
if ~ismember(method, valid_methods)
    error('Method unknown: %s', method)
end


iterate_n_submovements = (n_submovements == -1);
% If n_submovements is specified, then try to fit at most that number of 
% submovements per segment. Otherwise, continue adding submovements until
% the error criterion is reached

bin_size_ms = cast(bin_size_ms, 'double'); % This value tends to get saved as an integer, necessitating the cast
DT = bin_size_ms*1e-3;

%%%%%%%%%%%%%%%%%%%%%%%%
%% Movement segmentation
%%%%%%%%%%%%%%%%%%%%%%%%
hand_vel = hand_kin(:,vel_inds);
segments = segment_hand_vel(hand_vel, 'DT', DT, 'still_threshold', still_threshold, 'min_ampl', min_ampl);
minSegLength = floor(0.200/DT); % minimum segment length

% make sure segments have sufficient length
segLengths = segments(:,2)-segments(:,1);
segments(segLengths < minSegLength,:) = [];

n_segments = size(segments, 1);
segments_per_proc = ceil(n_segments/n_procs);
split = chunkify(1:n_segments, segments_per_proc); % HACK
inds = split{proc_idx};

movements = {};
submovement_recon = zeros(size(hand_vel));

if verbose
    fprintf('# of movement segments: %d\n', n_segments);
end

hand_speed = sqrt(sum(hand_vel.^2, 2));
n_movement_dims = size(hand_vel, 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Begin submovement decompositions
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Decomposing using method: %s\n', method)
start_time = tic;

n_iterations = {};
n_func_evals = {};

for k=inds
    s = max(1, segments(k,1) - 1);
    e = segments(k,2);
    hand_vel_tr = hand_vel( s:e, : );
    hand_speed_tr = hand_speed(s:e);
    if verbose
        fprintf('segment %d/%d, %d:%d\n', k, n_segments, s, e);
    end
    if isempty( find(isnan(hand_vel_tr), 1) )
        T = size(hand_vel_tr,1);
        t = (0:T-1)*DT;
        
        if strcmp(method, 'scattershot')% && iterate_n_submovements
            costs_tr = [];
            bestresult = {};
            bestfitresult = {};
            prev_decomp = [];
            n_iterations_ = [];
            n_func_evals_ = [];
            
            %%% Determine the maximum number of submovements
            seg_len = e-s;
            max_submovements = max(ceil((seg_len*DT - D_min)/max(isi, 0.1)), 1);
            if ~iterate_n_submovements
                max_submovements = min(max_submovements, n_submovements);
            end

            orig_error = residual_sse(hand_vel_tr, hand_speed_tr);
            
            costs_so_far = ones(1, max_submovements)*inf;
            time_elapsed = ones(1, max_submovements)*inf;
            tic

            submov_iter_idx = 0;
            while (submov_iter_idx < 1) || ((submov_iter_idx < max_submovements) && ~iterate_n_submovements) || ((costs_so_far(submov_iter_idx) > mse_perc_thresh) && (submov_iter_idx < max_submovements))
                
                submov_iter_idx = submov_iter_idx + 1;
                m = submov_iter_idx;
                if zero_greed
                    prev_decomp = [];
                    sample_randomly = 1;
                else
                    sample_randomly = 0;
                end

                [costs_tr(m), bestresult{m}, bestfitresult{m}, n_iterations_(m), n_func_evals_(m)] = decompose_lstsq(t, ...
                    hand_vel_tr, submov_iter_idx, 'prev_decomp', prev_decomp, 'term_cond', orig_error*mse_perc_thresh, ...
                    'fn_type', fn_type, 'sample_randomly', sample_randomly, 'isi', isi);
                
                prev_decomp = bestresult{m};
                costs_so_far(submov_iter_idx) = costs_tr(m) / orig_error;
                fprintf('\t# submovements=%d, mse percentage: %g\n', submov_iter_idx, costs_so_far(submov_iter_idx));
                if (submov_iter_idx > 2) && iterate_n_submovements && ((costs_so_far(submov_iter_idx) - costs_so_far(submov_iter_idx-1)) > -0.001) && use_cost_diff
                    fprintf('\t\tcost difference too small! %f\n', (costs_so_far(submov_iter_idx) - costs_so_far(submov_iter_idx-1)))
                    break
                end
                time_elapsed(submov_iter_idx) = toc;
            end
            
            n_iterations{k} = n_iterations_;
            n_func_evals{k} = n_func_evals_;

            %--- Postprocessing of segment decomposition
            fits{k,1} = bestresult;
            
            for ii = 1:length(fits{k,1})
                fits{k,1}{ii}(:,1) = fits{k,1}{ii}(:,1) + (s-1)*DT;
            end
            
            fits{k,1} = fits{k,1}';
            fit_costs{k,1} = costs_tr; 
            
            bestresult = bestresult{m};
            if strcmp(fn_type, 'min_jerk_full')
                best_recon_traj = reconstruct_submovements(prev_decomp, T, DT, 'min_jerk');
            else
                best_recon_traj = reconstruct_submovements(prev_decomp, T, DT, fn_type);
            end
            costs(k) = costs_tr(m);
%             try
%                 best_recon_traj = bestfitresult{m};
%                 costs(k) = costs_tr(m);
%             catch
% %                 bestresult = prev_decomp;
%                 best_recon_traj = reconstruct_submovements(prev_decomp, T, DT, fn_type);
%                 costs(k) = -1;
%             end
            
            bestresult(:,1) = bestresult(:,1) + (s-1)*DT;
            submovement_recon(s:e, :) = best_recon_traj'; 
            movements{k} = bestresult;                  
        elseif strcmp(method, 'greedy')
            [best, bestresult, bestfitresult] = decompose_greedy(t, hand_vel_tr, n_submovements, ...
                'fn_type', fn_type);
            
            bestresult(:,1) = bestresult(:,1) + (s-1)*DT;
            
            submovement_recon(s:e, :) = bestfitresult(1:length(t),:);
            movements{k} = bestresult;
            costs(k) = best;            
        else
            error('Method unknown: %s', method)
        end 
    end
end
runtime = toc(start_time);
fprintf('Submovement decomposition time: %g\n', runtime); 
