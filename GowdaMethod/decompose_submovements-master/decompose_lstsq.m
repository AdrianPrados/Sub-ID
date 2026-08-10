function [best_cost, bestresult, bestfitresult, n_iterations, n_func_evals] = decompose_lstsq(times, vel, numsubmovements, varargin)   
% This function was originally written by Jason Friedman and modified for this work
% http://noisyaccumulation.blogspot.com/2012/02/how-to-decompose-2d-trajectory-data.html
%
% The main decomposition function
%
% Parameters 
% ----------
% times : vector
%    vector of time stamps for each of the velocity samples, evenly spaced
% vel : matrix
%    K x 2 matrix of x-y cartesian velocity
% numsubmovements : int 
%    number of submovements to fit to 'vel'
% prev_decomp : optional, default = []
%    Results from previous decompositions of the same trajectory. Used to resample parameters if certain flags below are set
% D_min : optional, default = 0.10
%    Minimum duration for a submovement, in seconds
% D_max : optional, default = 1.0
%    Maximum duration for submovement, in seconds
% isi : optional, default = 0.100
%    Minimum time between submovements, in seconds
% term_cond : optional, default = -inf
%    MSE threshold to terminate optimization (e.g., 2% MSE)
% fn_type : optional, default = 'min_jerk'
%    Prototype submovement function to fit a sum of
% sigma_min : optional, default = 0.02
%    Minimum value for the sigma parameter (dispersion) of the support-bounded log normal function
% sigma_max : optional, default = 1.0
%    Maximum value for the sigma parameter (dispersion) of the support-bounded log normal function
% mu_min : optional, default = -1.0
%    Minimum value for the mu parameter (skewness) of the support-bounded log normal function
% mu_max : optional, default = 1.0
%    Maximum value for the mu parameter (skewness) of the support-bounded log normal function
% add_new_submovement : optional, default = 1
%    Flag to specify whether a new submovement should be added to the decomposition
% sample_randomly : optional, default = 0
%    Flag to specify whether sampling should be fully random, or if parameters should be resampled from old decompositions
% 
% Returns
% -------
% best_cost
% bestresult
% bestfitresult
% n_iterations
%    Total number of iterations, added up over all 10 restarts
% n_func_evals
%    Total number of iterations, added up over all 10 restarts

    defaults = {'prev_decomp', [], ...
        'D_min', 0.09, ...
        'D_max', 1.0, ...
        'isi', 0.100, ...
        'term_cond', -inf, ...
        'fn_type', 'min_jerk', ...
        'sigma_min', 0.02, ...
        'sigma_max', 1.0, ...
        'mu_min', -1.0, ...
        'mu_max', 1.0, ...
        'add_new_submovement', 1, ...
        'sample_randomly', 0, ...
        };
    
    prev_decomp = get_var('prev_decomp', 'defaults', defaults, varargin{:});
    D_min = get_var('D_min', 'defaults', defaults, varargin{:});
    D_max = get_var('D_max', 'defaults', defaults, varargin{:});
    isi = get_var('isi', 'defaults', defaults, varargin{:});
    term_cond = get_var('term_cond', 'defaults', defaults, varargin{:});
    fn_type = get_var('fn_type', 'defaults', defaults, varargin{:});
    sigma_min = get_var('sigma_min', 'defaults', defaults, varargin{:});
    sigma_max = get_var('sigma_max', 'defaults', defaults, varargin{:});
    mu_min = get_var('mu_min', 'defaults', defaults, varargin{:});
    mu_max = get_var('mu_max', 'defaults', defaults, varargin{:});
    add_new_submovement = get_var('add_new_submovement', 'defaults', defaults, varargin{:});
    sample_randomly = get_var('sample_randomly', 'defaults', defaults, varargin{:});
    
    % one-hot encode the function type
    min_jerk = strcmp(fn_type, 'min_jerk');
    min_jerk_full = strcmp(fn_type, 'min_jerk_full');
    sbln = strcmp(fn_type, 'sbln');

    % define # of parameters per submovement
    if min_jerk
        N_PARAMS_PER_SUBMOVEMENT = 2;
    elseif min_jerk_full
        N_PARAMS_PER_SUBMOVEMENT = 4;
    elseif sbln
        N_PARAMS_PER_SUBMOVEMENT = 4;
    else
        error('Unrecognized submovement function type: %s\n', fn_type);
    end

    prev_decomp_size = size(prev_decomp, 1);
    
    % input error checking
    if numsubmovements < prev_decomp_size
        error('previous decomposition is larger than current')
    end

    if size(times,1)>1
        error('time must be a 1*N vector'); 
    end
    
    if ~(length(times) == max(size(vel)))
        error('times and vel do not match in length')
    end

    DT = times(2) - times(1);
    best_cost = inf;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % create parameter by parameter lower/upper bounds
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % maximum submovement onset time is given by 'end' - min duration
    t0_max = times(end) - D_min;
    vx_min = min(vel(:,1));
    vx_max = max(vel(:,1));
    vy_min = min(vel(:,2));
    vy_max = max(vel(:,2));      

    T_max = length(times);
    lb = []; 
    ub = [];
    for k=1:numsubmovements
        if min_jerk
            lb = [lb, [isi*(k-1), D_min]];
            ub = [ub, [t0_max, min(D_max, T_max*DT)]];
        elseif min_jerk_full
            lb = [lb, [isi*(k-1), D_min, vx_min, vy_min]];
            ub = [ub, [t0_max, min(D_max, T_max*DT), vx_max, vy_max]];            
            % lb_0 = [0                0.150     xrng(1) yrng(1)];
            % ub_0 = [times(end)-0.167 1.0       xrng(2) yrng(2)];
        elseif sbln
            lb = [lb, [isi*(k-1), D_min, mu_min, sigma_min]];
            ub = [ub, [t0_max, D_max, mu_max, sigma_max]];
        end
    end

    % check that there's a feasible set  
    if any(lb > ub)
        error('Lower bounds exceed upper bound - infeasible');
    end

    %% create linear inequalties to create the 'inter-submovement' interval
    if isi > 0
        A = zeros(numsubmovements-1, numsubmovements*N_PARAMS_PER_SUBMOVEMENT);
        b = -isi*ones(numsubmovements-1, 1);
        for k=0:numsubmovements-2
            A(k+1, N_PARAMS_PER_SUBMOVEMENT*k+1) = 1;
            A(k+1, N_PARAMS_PER_SUBMOVEMENT*(k+1)+1) = -1;
        end   
    else
        A = [];
        b = [];
    end

    n_dim_movement = size(vel, 2);
    tv = sqrt(sum(vel.^2, 2));
    
    count = 1; 
    n_iterations = 10;

    if min_jerk || min_jerk_full
        recon = reconstruct_submovements(prev_decomp, T_max, DT, 'min_jerk')';
    end
    res_vel = vel - recon;
    
    res_speed = sqrt(sum(res_vel.^2, 2));
    no_submovements = (prod(recon, 1) == 0);
    if ~sample_randomly
        [t0_samples, D_samples] = greedy_onset_sampling(res_vel, res_speed, ...
            n_iterations, DT, no_submovements, 'greed_factor', prev_decomp_size+1, 'plot_dist', 0);

        [t0_samples, inds] = unique(t0_samples);
        D_samples = D_samples(inds);

        if length(t0_samples) < n_iterations
            n_rand_samples = n_iterations - length(t0_samples);
            t0_samples = [t0_samples; unifrnd(0, t0_max, n_rand_samples, 1)];
            D_samples = [D_samples; unifrnd(D_min, D_max, n_rand_samples, 1)];
        end
    end

    n_opt_iterations = 0;
    n_func_evals = 0;
    
    while count <= n_iterations
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% Initialize estimates of submovement parameters
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if (numsubmovements == 1) && (count == 1)
            if min_jerk
                initialparameters = [0, rand(1)*(D_max-D_min)+D_min];
            elseif min_jerk_full
                D_sample = unifrnd(D_min, D_max); 
                Ax_sample = unifrnd(vx_min, vx_max);
                Ay_sample = unifrnd(vy_min, vy_max);
                initialparameters = [0, D_sample, Ax_sample, Ay_sample];
            elseif sbln
                D_sample = rand(1)*(D_max-D_min)+D_min;
                mu_sample = rand(1)*(mu_max-mu_min)+mu_min;
                sigma_sample = rand(1)*(sigma_max-sigma_min)+sigma_min;
                initialparameters = [0, D_sample, mu_sample, sigma_sample];
            end
        elseif sample_randomly
            % determine number of new submovements
            n_new_submovements = numsubmovements - prev_decomp_size;
                    
            % sample start time and duration of new submovements randomly
            new_submovement_init_params = [];
            for m=1:n_new_submovements
                if min_jerk
                    t0 = unifrnd(0, t0_max);
                    D = unifrnd(D_min, D_max);
                    new_submovement_init_params = [new_submovement_init_params; t0, D];
                elseif min_jerk_full
                    t0 = unifrnd(0, t0_max);
                    D = unifrnd(D_min, D_max);
                    Ax_sample = unifrnd(vx_min, vx_max);
                    Ay_sample = unifrnd(vy_min, vy_max);
                    new_submovement_init_params = [new_submovement_init_params; t0, D, Ax_sample, Ay_sample];
                elseif sbln
                    1./0
                end
            end
            
            if prev_decomp_size > 0
                initialparameters = [prev_decomp(:,1:N_PARAMS_PER_SUBMOVEMENT); new_submovement_init_params];
            else
                initialparameters = new_submovement_init_params;
            end
            
            % sort initial parameters by submovement onset time
            initialparameters_mat = sortrows(initialparameters, 1);
            
            % flatten the initialparameters_mat back to the optimization format
            initialparameters_mat_xpose = initialparameters_mat';
            initialparameters = initialparameters_mat_xpose(:);
        elseif add_new_submovement
            % determine number of new submovements
            n_new_submovements = numsubmovements - prev_decomp_size;
                    
            % sample start time and duration of new submovements randomly
            new_submovement_init_params = [];
            
            for m=1:n_new_submovements
                t0 = t0_samples(count);
                D = D_samples(count);
                if min_jerk
                    new_submovement_init_params = [new_submovement_init_params; t0, D]; %rand(1)*(D_max-D_min)+D_min];
                elseif min_jerk_full
                    new_submovement_init_params = [new_submovement_init_params; t0, D, vx_max*D, vy_max*D]; %rand(1)*(D_max-D_min)+D_min];
                elseif sbln
                    D_sample = rand(1)*(D_max-D_min)+D_min;
                    mu_sample = rand(1)*(mu_max-mu_min)+mu_min;
                    sigma_sample = rand(1)*(sigma_max-sigma_min)+sigma_min;
                    new_submovement_init_params = [new_submovement_init_params; t0, D_sample, mu_sample, sigma_sample];
                end
            end
            
            if prev_decomp_size > 0
                initialparameters = [prev_decomp(:,1:N_PARAMS_PER_SUBMOVEMENT); new_submovement_init_params];
            else
                initialparameters = new_submovement_init_params;
            end
            
            % sort initial parameters by submovement onset time
            initialparameters_mat = sortrows(initialparameters, 1);
            
            % flatten the initialparameters_mat back to the optimization format
            initialparameters_mat_xpose = initialparameters_mat';
            initialparameters = initialparameters_mat_xpose(:);

        else % just use the old parameters and re-optimize
            % sort initial parameters by submovement onset time
            initialparameters_mat = sortrows(prev_decomp(:,1:N_PARAMS_PER_SUBMOVEMENT), 1);
            
            % flatten the initialparameters_mat back to the optimization format
            initialparameters_mat_xpose = initialparameters_mat';
            initialparameters = initialparameters_mat_xpose(:);            
        end
        initialparameters = cast(initialparameters, 'double');

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%% Run optimization routines
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if min_jerk
            options = optimset('GradObj','on', 'LargeScale', 'on', ...
                'Algorithm', 'sqp', ...
                'MaxFunEvals',10^13, 'MaxIter', 5000,...
                'FunValCheck','on','DerivativeCheck','off',...
                'Display','notify');
            
            [result, FVAL, EXITFLAG, OUTPUT] = fmincon(@(parameters) min_jerk_cost_fn(parameters, times, vel, tv), ...
                initialparameters, A, b, [], [], lb, ub, [], options);
            [epsilon, grad, fitresult] = min_jerk_cost_fn(result, times, vel, tv);

        elseif min_jerk_full
            options = optimset('GradObj','on', 'Hessian', 'off', 'LargeScale', 'on', ...
                    'Algorithm', 'sqp', ...
                    'MaxFunEvals',10^13,'MaxIter', 5000,...
                    'FunValCheck','on','DerivativeCheck','off',...
                    'Display','notify');
            
            initialparameters = initialparameters';
            [result, FVAL, EXITFLAG, OUTPUT] = fmincon(@(parameters) calculateerrorMJxy(parameters, times, vel, tv), ...
                initialparameters, A, b, [], [], lb, ub, [], options);

            [epsilon, ~, fitresult] = calculateerrorMJxy(result,times, vel, tv, diff(times(1:2)));

        elseif sbln
            options = optimset('GradObj','off', 'LargeScale', 'on', ...
                'Algorithm', 'active-set', ...
                'MaxFunEvals',10^13, 'MaxIter', 5000,...
                'FunValCheck','on','DerivativeCheck','off',...
                'Display','notify');

            [result, FVAL, EXITFLAG, OUTPUT] = fmincon(@(parameters) sbln_cost_fn(parameters, times, v, tv), ...
                initialparameters, A, b, [], [], lb, ub, [], options);
            [epsilon, fitresult] = sbln_cost_fn(result, times, v, tv);
        end

        %%%%%%%%%%%%
        %%%% Cleanup
        %%%%%%%%%%%%
        n_opt_iterations = n_opt_iterations + OUTPUT.iterations;
        n_func_evals = n_func_evals + OUTPUT.funcCount;
        
        if ~isreal(result(1))
            error('Found an imaginary value');
        end

        if epsilon < best_cost
            best_cost = epsilon;
            bestresult = result;
            bestfitresult = fitresult;
        end
        
        if best_cost < term_cond
            fprintf('\t\tOptimization target met, terminating early\n');
            break
        else
            count = count+1;
        end
    end
    
    % recalculate the optimal Ax, Ay 
    if min_jerk
        [cost, grad, v_pred, A] = min_jerk_cost_fn(bestresult, times, vel, tv);
        bestresult = [reshape(bestresult, N_PARAMS_PER_SUBMOVEMENT, [])', A];
    elseif min_jerk_full
        bestresult = reshape(bestresult, N_PARAMS_PER_SUBMOVEMENT, [])';
    elseif sbln
        [cost, v_pred, A] = sbln_cost_fn(bestresult, times, v, tv);
        bestresult = [reshape(bestresult, N_PARAMS_PER_SUBMOVEMENT, [])', A'];
    end

end
