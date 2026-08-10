function [recon] = reconstruct_submovements(movements, T_max, DT, fn_type)
    % one-hot encode the function type
    min_jerk = strcmp(fn_type, 'min_jerk');
    sbln = strcmp(fn_type, 'sbln');
    
    if min_jerk
        N_PARAMS_PER_SUBMOVEMENT = 2;
    elseif sbln
        N_PARAMS_PER_SUBMOVEMENT = 4;
    else
        error('Unrecognized submovement function type: %s\n', fn_type);
    end
    
    
    n_dim_movement = size(movements, 2) - N_PARAMS_PER_SUBMOVEMENT;
    if n_dim_movement < 0
        recon = 0;
    else
        t = (0:T_max-1)*DT;
        recon = zeros(n_dim_movement, T_max);
        n_submovements = size(movements, 1);
        for k=1:n_submovements
            if min_jerk
                [v_pred] = calc_min_jerk_recon(movements(k,:), t);
            elseif sbln
                [v_pred] = calc_sbln_recon(movements(k,:), t);
            end
            v_pred(isnan(v_pred)) = 0;
            recon = recon + v_pred;
        end
    end