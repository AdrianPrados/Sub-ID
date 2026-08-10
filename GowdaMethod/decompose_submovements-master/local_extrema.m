function [lm] = local_extrema(data, varargin)

defaults = {'type', 'max'};
type = get_var('type', 'defaults', defaults, varargin{:});

lm = zeros(length(data), 1);
max = strcmp(type, 'max');
both = strcmp(type, 'both');
min = strcmp(type, 'min');

if (data(1) < data(2)) && (min || both)
    lm(1) = 1;
elseif (data(1) > data(2)) && (max || both)
    lm(1) = 1;
end

N = length(data);
for k=2:N-1
    if data(k) > data(k-1) && data(k) > data(k+1) && (max || both)
        lm(k) = 1;
    elseif data(k) < data(k-1) && data(k) < data(k+1) && (min || both)
        lm(k) = 1;        
    end 
end

if (data(N) < data(N-1)) && (min || both)
    lm(N) = 1;
elseif (data(N) > data(N-1)) && (max || both)
    lm(N) = 1;
end

end