function [segments] = segment(vec)

vec = vec(:);

segment_starts = find(vec(2:end) & ~vec(1:end-1))+1;
segment_ends = find(~vec(2:end) & vec(1:end-1))+1;

if isempty(segment_starts) || (segment_starts(1) > segment_ends(1))
    segment_starts = [1; segment_starts];
end

if segment_ends(end) < segment_starts(end)
    segment_ends = [segment_ends; length(vec)];
end

segments = [segment_starts, segment_ends];
end
