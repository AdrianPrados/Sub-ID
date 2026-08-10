function [cost] = cost_fn(v, tv, v_pred, v_pred_tang)

e = norm((v - v_pred), 'fro').^2;
e_tang = norm((tv(:) - v_pred_tang(:)), 'fro').^2;

cost = e + 2*e_tang;