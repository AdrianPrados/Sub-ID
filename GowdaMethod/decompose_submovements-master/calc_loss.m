function [cost, v_pred, A] = calc_loss(v, tv, T)
%{  
Calculates loss for all amplitude scaled submovement profiles

Parameters
----------
v : 2 x K matrix
    Observed velocity
tv : 1 x K matrix
    Observed tangential speed
T : K x N matrix
    Matrix of N submovement shapes of unit size, unscaled (each column is a different submovement)

Returns
-------
cost : float
    Cost of the current reconstruction
v_pred : 2 x K matrix
    Predicted velocity
A : 
    Amplitudes of each submovement
%}
    n_movements = size(T, 2);
    r = 0.5;
    ridge = eye(n_movements)*r;
    A = v * (T' / (T*T' + ridge)); 
    v_pred = A * T; 

    cost = calc_error(v, tv, v_pred);
end

