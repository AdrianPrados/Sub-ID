function c = chunkify(data, chunkSize)
%C = CHUNKIFY(DATA, CHUNKSIZE)
%   Turns a matrix DATA into chunks of data with CHUNKSIZE elements in each
%   cell entry of C.  If DATA's size is not evenly divisible by CHUNKSIZE,
%   the last entry in C will contain fewer items.
%
%Examples:
%   chunkify([1 2 3 4], 2)         --> {[1 2] [3 4]}
%   chunkify([1 2 3 4 5], 2)       --> {[1 2] [3 4] [5]}
%   chunkify({'1' '2' '3' '4'}, 2) --> {{'1' '2'} {'3' '4'}}
%
%Motivating scenario:
%   I wrote this function because I needed to process a lot of data in
%   separate chunks, sometimes to have different machines run each chunk,
%   sometimes to just load in chunks of the data files so I could balance
%   I/O and memory constraints.  In both these cases, I found it handy to
%   have this simple function to break up my index vector.  Because the
%   number of data items was not always an exact multiple of my desired
%   chunk size, I used this function instead of Matlab's built-in RESHAPE
%   (putting one chunk per row, for example).
%
%TODO:
%If DATA is multi-dimensional, the dimension on which to chunkify is given
%by DIM. 
%   chunkify([1 2 3 4], 2, 2)  --> {[1 2] [3 4]}
%   chunkify([1 2 3 4], 2, 1)  --> {[1 2 3 4]}
%   chunkify([1 2; 3 4], 2, 1) --> {[1 2] [3 4]}
%
%by Gerald Dalley

if (nargin < 2), chunkSize = 10; end

c = cell(1, ceil(numel(data)/chunkSize));
for i=0:length(c)-1
    c{i+1} = data(1+i*chunkSize : min(numel(data), (i+1)*chunkSize));
end
