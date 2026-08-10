function [epsilon, grad, sumpredicted] = calculateerrorMJxy(parameters, times, vel, tangvel, timedelta)
% This function was originally written by Jason Friedman and modified for this work
% http://noisyaccumulation.blogspot.com/2012/02/how-to-decompose-2d-trajectory-data.html

% CALCULATEERRORMJXY - calculate the error between the predicted and actual profile (in 2D)
% The predicted trajectory consists of the superposition of one or more minimum jerk velocity profiles
%
% [epsilon,grad,hess,sumpredicted,predictedx,predictedy] =  calculateerrorxy(parameters,times,vel,tangvel,timedelta)
%
% The error is defined by (xvel - xpred)^2 + (yvel-ypred)^2 + (tangvel - tangpred)^2
%
% The function also optionally returns the gradient and Hessian
% (first-order and second-order partial derivatives), for use with
% optimization routines
%
% It can also optionally return the predicted minimum jerk trajectory
% (resulting from the superposition of the submovements)
%
% The parameters should be of length 4 * N (where N is the number submovements)
% each 4 parameters is T0 (onset time in seconds), D (duration in seconds),
% Ax (x amplitude) and Ay (y amplitude)
%
% times should be a 1 * N vector with the times of the recorded movement (in seconds)
%
% vel should be an N * 2 vector with the x and y velocities
%
% tangvel should contain the tangential velocity [i.e. tangvel = sqrt(vel(:,1).^2+vel(:,2).^2)  ]
%
% timedelta (optional, default = 0.005) is the time points to evaluate and
% compare the trajectories. It should match the times data [i.e. timedelta=times(2) - times(1)   ]


numsubmovements = length(parameters)/4;

trajectoryx = vel(:,1);
trajectoryy = vel(:,2);

K = length(times);
predictedx = zeros(numsubmovements, K);
predictedy = zeros(numsubmovements, K);
predicted = zeros(numsubmovements, K);

if nargout > 1
    Jx= zeros(4*numsubmovements, K);
    Jy= zeros(4*numsubmovements, K);
    J= zeros(4*numsubmovements, K);    
end

for k=1:numsubmovements
    % There are 4 parameters per submovement
    T0 = parameters(k*4-3);
    D =  parameters(k*4-2);
    Dx = parameters(k*4-1);
    Dy = parameters(k*4);
    
    % find the appropriate times to calculate this over (T0 <= t <= T0+D)
    thisrng = find(times>T0 & times<T0+D);
    
    if nargout==1
        [predictedx(k,thisrng), predictedy(k,thisrng), predicted(k,thisrng)] ...
            = MJxy(T0, D, Dx, Dy, times(thisrng));
    else
        [predictedx(k,thisrng), predictedy(k,thisrng), predicted(k,thisrng),...
            Jx(k*4-3:k*4, thisrng), Jy(k*4-3:k*4, thisrng), J(k*4-3:k*4, thisrng)] ...
            = MJxy(T0, D, Dx, Dy, times(thisrng));
    end
end
sumpredictedx = sum(predictedx,1)';
sumpredictedy = sum(predictedy,1)';
sumpredicted  = sum(predicted,1)';

if nargout>1 % calculate gradient
    sumtrajsq = sum(trajectoryx.^2) + sum(trajectoryy.^2) + sum(tangvel.^2);
    error_x = (sumpredictedx - trajectoryx)';
    error_y = (sumpredictedy - trajectoryy)';
    error_t = (sumpredicted - tangvel)';
    grad = 2/sumtrajsq * (Jx * error_x' + Jy * error_y' + J * error_t');
end

epsilon = calc_error([trajectoryx, trajectoryy]', tangvel, [sumpredictedx, sumpredictedy]');

sumpredicted = [sumpredictedx sumpredictedy];
