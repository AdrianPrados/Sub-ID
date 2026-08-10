function [cost] = calc_error(v, tv, v_pred)
%{	
Evaluate the error of a velocity reconstruction

Parameters
----------
v : 2 x K matrix
	Observed velocity
tv : 1 x K matrix
	Observed tangential speed
v_pred : 2 x K matrix
	Velocity profile of the velocity reconstructed from submovement parameters

Returns
-------
cost : float
	Cost of the current reconstruction
%}
    v_pred_tang = sqrt(sum(v_pred.^2, 1));
    e = norm((v - v_pred), 'fro').^2;
    e_tang = norm((tv(:) - v_pred_tang(:)), 'fro').^2;
    cost = e + e_tang;
end