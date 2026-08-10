function [cost, pt_by_pt_cost] = residual_sse(v, tv)
    e = (v ).^2;
    e_tang = (tv(:) ).^2;
    pt_by_pt_cost = sum(e, 2) + e_tang(:);
    cost = sum( pt_by_pt_cost );
end
