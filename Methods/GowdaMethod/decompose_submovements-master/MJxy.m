function [Bx,By,B,Jx,Jy,J,Hx,Hy,H] = MJxy(t0,D,Ax,Ay,t)
% This function was originally written by Jason Friedman and modified for this work
% http://noisyaccumulation.blogspot.com/2012/02/how-to-decompose-2d-trajectory-data.html
% MJxy - evaluate a minimum jerk curve with seperate displacement for x / y
%
% see Flash and Hogan (1985) for details on the minimum jerk equation
%
%    t0 = movement start time
%    D  = movement duration
%    Ax = displacement resulting from the movement (x)
%    Ay = displacement resulting from the movement (y)
%
% The function is evaluated at times t
%
% The function also optionally returns the first-order and second-order
% partial derivatives, for use with optimization routines
%
% Bx, By and B are the x velocity, y velocity and tangential velocities
% Jx, Jy and J are the gradients (partial derivatives) of the same quantities
% Hx, Hy and H are the Hessian (second-order partial derivatives)
%
% [Bx,By,B,Jx,Jy,J,Hx,Hy,H] = MJxy(t0,D,Ax,Ay,t)


% nt is normalized time (0 <= nt <= 1)
nt = (t-t0)./D;
nt_sq = nt.^2;
nt_cubed = nt.^3;
nt_fourth = nt.^4;

shape = (-60 * nt_cubed + 30 * nt_fourth + 30 * nt_sq);
Bx = Ax/D * shape;
By = Ay/D * shape;

A_tang = sqrt((Ax/D).^2 + (Ay/D).^2);
B = A_tang * shape;

if nargout > 3
    K = length(t);
    Jx = zeros(4, K);
    Jy = zeros(4, K);
    J = zeros(4, K);

    shape_dt0 = -1./D^2*(120*nt_cubed - 180*nt_sq + 60*nt);
    shape_dD = 1./D^2*(-150*nt_fourth + 240*nt_cubed - 90*nt_sq);

    Jx(1,:) = Ax.*shape_dt0;
    Jx(2,:) = Ax.*shape_dD;
    Jx(3,:) = 1./D*shape;

    Jy(1,:) = Ay.*shape_dt0;
    Jy(2,:) = Ay.*shape_dD;    
    Jy(4,:) = 1./D*shape;
    
    J(1,:) = A_tang.*shape_dt0;
    J(2,:) = A_tang.*shape_dD;
    J(3,:) = Ax/A_tang.* shape;
    J(4,:) = Ay/A_tang.* shape;


    if nargout > 6
        Hx = zeros(4, 4, length(t));
        Hy = zeros(4, 4, length(t));
        H = zeros(4, 4, length(t));

        %%% col 1 (first derivative w.r.t. t0)
        shape_dt0dt0 = 1/D^3*(360*nt_sq - 360*nt + 60);
        Hx(1,1,:) = Ax.*shape_dt0dt0;
        Hy(1,1,:) = Ay.*shape_dt0dt0;
        H(1,1,:) = A_tang.*shape_dt0dt0;
        
        shape_dDdt0 = (-1/D^3)*(-600*nt_cubed + 720*nt_sq - 180*nt);
        Hx(2,1,:) = Ax.*shape_dDdt0;
        Hy(2,1,:) = Ay.*shape_dDdt0;
        H(2,1,:) = A_tang.*shape_dDdt0;
        
        Hx(3,1,:) = shape_dt0;
        H(3,1,:) = Ax/A_tang*shape_dt0;
        
        Hy(4,1,:) = shape_dt0;
        H(4,1,:) = Ay/A_tang*shape_dt0;
        

        %%% col 2 (first derivative with respect to D)
        shape_dt0dD = shape_dDdt0;
        Hx(1,2,:) = Ax.*shape_dt0dD;
        Hy(1,2,:) = Ay.*shape_dt0dD;
        H(1,2,:) = A_tang.*shape_dt0dD;
        
        shape_dDdD = 1./D^3 * (900*nt_fourth - 1200*nt_cubed + 360*nt_sq);
        Hx(2,2,:) = Ax.*shape_dDdD;
        Hy(2,2,:) = Ay.*shape_dDdD;
        H(2,2,:) = A_tang.*shape_dDdD;
        
        Hx(3,2,:) = shape_dD;
        H(3,2,:) = Ax/A_tang*shape_dD;
    
        Hy(4,2,:) = shape_dD;
        H(4,2,:) = Ay/A_tang*shape_dD;

        %%%%%%%% col 3 (first derivative with respect to Ax)
        Hx(1,3,:) = shape_dt0;
        H(1,3,:) = Ax./A_tang * shape_dt0;
        
        Hx(2,3,:) = shape_dD;
        H(2,3,:) = Ax./A_tang * shape_dD;
        
        H(3,3,:) = 1./A_tang * shape;
        
        H(4,3,:) = -Ax*Ay/A_tang^3 * shape;
        
        %%%%%% col 4 (first derivative with respect to Ay)
        Hy(1,4,:) = shape_dt0;
        H(1,4,:) = Ay./A_tang * shape_dt0;
        
        Hy(2,4,:) = shape_dD;
        H(2,4,:) = Ay./A_tang * shape_dD;
        
        H(3,4,:) = -Ax*Ay/A_tang^3 * shape;
        
        H(4,4,:) = 1./A_tang * shape;
    end
end
