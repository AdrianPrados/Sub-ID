function [v_pred] = calc_sbln_recon(parameters, times)
    N_PARAMS_PER_SUBMOVEMENT = 4;
    n_dim_movement = length(parameters) - N_PARAMS_PER_SUBMOVEMENT;
    v_pred = nan( n_dim_movement , length(times));
    
    t0 = parameters(1);
    D = parameters(2);
    mu = parameters(3);
    sigma = parameters(4);
    thisrng = find(times > t0 & times < t0+D);
    t = times(thisrng);
    
    A = parameters(N_PARAMS_PER_SUBMOVEMENT+1:end)';
    v_pred(:, thisrng) = A*(D./(sqrt(2*pi*sigma^2)*(t-t0).*(t0+D-t)) .* exp(-1./(2*sigma^2) * (log((t-t0)./(t0+D-t)) - mu ).^2));
    