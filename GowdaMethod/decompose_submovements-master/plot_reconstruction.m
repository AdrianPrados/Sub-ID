function [] = plot_reconstruction(movements, T_max, DT, fn_type, varargin)
    defaults = {'plot_individual_movements', 1};
    plot_individual_movements = get_var('plot_individual_movements', defaults, defaults{:}, varargin{:});
    
    total_recon = reconstruct_submovements(movements, T_max, DT, fn_type);
    size(total_recon)
    n_movements = size(movements, 1);
    
    
    
    
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
    
    
    
    recon = zeros(n_dim_movement, T_max, n_movements);
    for k=1:n_movements
        recon(:, :, k) = reconstruct_submovements(movements(k,:), T_max, DT, fn_type);
    end
    
    time = (0:T_max-1)*DT;
    
    figure()
    for m=1:n_dim_movement
        subplot(n_dim_movement, 1, m)
        hold on
        plot(time, total_recon(m,:), 'color', 'black', 'LineWidth', 3)
        if plot_individual_movements
            for k=1:n_movements
                plot(time, recon(m, :, k), 'color', 'blue', 'LineWidth', 3)
            end
        end
    end
    
%     subplot(2,1,2)
%     hold on
%     plot(time, total_recon(2,:), 'color', 'black', 'LineWidth', 3)
%     if plot_individual_movements
%         for k=1:n_movements
%             plot(time, recon(2, :, k), 'color', 'blue', 'LineWidth', 3)
%         end
%     end
end

function [] = dummy(hand_vel, recon, segments)
% plot_reconstruction(hand_vel, recon, segments)
    figure()
    h(1) = subplot(2,1,1);
    hold on
    plot(hand_vel(:,1), 'b')
    plot(recon(:,1), 'g')
    pts = zeros( size(segments, 1), 1);
    scatter(segments(:,1), pts, 'k.')
    scatter(segments(:,2), pts, 'r.')
    
    h(2) = subplot(2,1,2);
    hold on
    plot(hand_vel(:,2), 'b')
    plot(recon(:,2), 'g')
    scatter(segments(:,1), pts, 'k.')
    scatter(segments(:,2), pts, 'r.')
    linkaxes(h, 'x')
end

