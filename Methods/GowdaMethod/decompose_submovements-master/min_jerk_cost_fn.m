function [cost, grad, v_pred, M] = min_jerk_cost_fn(parameters, times, v, tv)
    N_PARAMS_PER_SUBMOVEMENT = 2;
    [param_dim_x, param_dim_y] = size(parameters);
    parameters = reshape(parameters, N_PARAMS_PER_SUBMOVEMENT, [])';
    n_movements = size(parameters, 1);

    F = zeros(length(times), n_movements);
    for n=1:n_movements
        t0 = parameters(n,1);
        D = parameters(n,2);
        time_inds = find(times>t0 & times<t0+D);
        t = times(time_inds);
        F(time_inds, n) = MJxy(t0, D, 1, 1, t);
    end

    ridge = eye(n_movements)*0.5; % should corespond to |M| < 0.5 (roughly)
    M = ((F'*F + ridge) \ F') * v;

    full_params = [parameters, M];
    full_params = full_params';
    full_params = full_params(:);
    if nargout > 1 % if calculating gradient
        [cost, full_grad, v_pred] = calculateerrorMJxy(full_params, times, v, tv);
        grad_inds = [];
        for k=1:n_movements
            grad_inds = [grad_inds, 4*(k-1)+1, 4*(k-1)+2];
        end
        grad = full_grad(grad_inds, :);
    else % only cost function eval
        [cost] = calculateerrorMJxy(full_params, times, v, tv);
    end
end
