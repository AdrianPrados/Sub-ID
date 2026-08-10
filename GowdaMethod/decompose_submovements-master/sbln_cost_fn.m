function [cost, v_pred, A] = sbln_cost_fn(parameters, times, v, tv)
    N_PARAMS_PER_SUBMOVEMENT = 4;
    parameters = reshape(parameters, N_PARAMS_PER_SUBMOVEMENT, [])';
    n_movements = size(parameters, 1);

    T = zeros( length(times), n_movements );
    for k=1:n_movements
        t0 = parameters(k,1);
        D = parameters(k,2);
        mu = parameters(k,3);
        sigma = parameters(k,4);
        thisrng = find(times>t0 & times<t0+D);
        t = times(thisrng);
        %nt = (t-t0)./D;
        %T(k, thisrng) = 1./D * (-60 * nt.^3 + 30 * nt.^4 + 30 * nt.^2);
        T(thisrng, k) = (D./(sqrt(2*pi*sigma^2)*(t-t0).*(t0+D-t)) .* exp(-1./(2*sigma^2) * (log((t-t0)./(t0+D-t)) - mu ).^2));
    end

    [cost, v_pred, A] = calc_loss(v, tv, T');
end

