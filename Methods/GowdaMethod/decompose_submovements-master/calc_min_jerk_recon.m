function [v_pred] = calc_min_jerk_recon(parameters, times)
    N_PARAMS_PER_SUBMOVEMENT = 2;
    n_dim_movement = length(parameters) - N_PARAMS_PER_SUBMOVEMENT;
    v_pred = nan( n_dim_movement , length(times));
    
    t0 = parameters(1);
    D = parameters(2);
    
    thisrng = find(times > t0 & times < t0+D);
    t = times(thisrng);
    A = parameters(N_PARAMS_PER_SUBMOVEMENT+1:end)';
    nt = (t-t0)./D;
    v_pred(:, thisrng) = A * 1./D * (-60 * nt.^3 + 30 * nt.^4 + 30 * nt.^2);