function [segments] = segment_hand_vel(hand_vel, varargin)

defaults = {
    'still_threshold', 0.005, ...
    'min_ampl', 0.02, ...
    'DT', 0.010, ...
};

min_ampl = get_var('min_ampl', 'defaults', defaults, varargin{:});
still_threshold = get_var('still_threshold', 'defaults', defaults, varargin{:});
DT = get_var('min_ampl', 'defaults', defaults, varargin{:}); % 20130509 SAO & SG

still = all( abs(hand_vel) < still_threshold, 2); %(abs(hand_vel(:,1)) < still_threshold) & (abs(hand_vel(:,2)) < still_threshold);
moving = ~still;
try
    segments = segment(moving);
    n_segments = size(segments, 1);

    % Apply the minimum amplitude constraint to the submovements
    keep_inds = [];
    for k=1:n_segments
        s = segments(k,1);
        e = segments(k,2);

        if (max(max(abs(hand_vel(s:e, :)))) > min_ampl) && (e-s)*DT >= 0.2
            keep_inds = [keep_inds, k];
        end
    end
    segments = segments(keep_inds, :);

    % merge segments that are not separated by more than 1 bin
    segments_xpose = segments';
    segments_flat = segments_xpose(:);

    while min(diff(segments_flat)) == 1
        n_segments = length(segments_flat)/2;
        for k = 1:n_segments-1
            if segments_flat(2*k) == segments_flat(2*k+1)-1
                segments_flat(2*k:2*k+1) = [];
                break
            end
        end
    end
    segments = reshape(segments_flat, 2,[])';
    

catch
    segments = [2, size(hand_vel, 1)]; %%% 2 is a dirty hack b/c the decomposer subtracts 1
end