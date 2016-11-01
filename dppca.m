function [model] = dppca(X, V, E, M, ETA, THRESH, model_init, iter_obj)
% DPPCA        Distributed Probablistic PCA (D-PPCA)
% 
% Description
%  Solve probabilistic PCA problem in a distributed way. The network has 
% max(V) nodes. We assume the network is connected. This function only 
% simulates parameter broadcasts. Local computation is done by dppca_local 
% function.
%
% Input
% X        : D x N matrix for full data from all nodes (N=N1+N2+...+NJ)
% V        : N x 1 vector for each observation's source (node affiliation)
% E        : J x J adjacency matrix where J = max(V)
% M        : 1 x 1 scalar of projection dimension
% ETA      : 1 x 1 scalar of learning rate
% THRESH   : 1 x 1 scalar convergence criterion (Default 10^-5)
% iter_obj : If > 0, print out objective value every (iter_obj) iteration
%            (Default 10)
%
% Output
% model = structure(W, MU, VAR);
% W        : D x M projection matrix
% MU       : D x 1 vector sample means
% VAR      : 1 x 1 scalar estimated variance
% EZ       : M x N x J cube for mean of N latent vectors
% EZZt     : M x M x N x J matrix set for covariance of N latent vectors
% eITER    : Iterations took
% eTIME    : Elapsed time
% objArray : Objective function value change over iterations
% LAMBDAi, GAMMAi, BETAi : Lagrange multipliers for debugging purpose
%
% Implemented
%  by     Sejong Yoon (sjyoon@cs.rutgers.edu)
%  on     2011.10.07 (last modified on 2012/02/01)
%
% References
%  [1] M.E. Tipping and C.M. Bishop, Probablistic principal component 
% analysis, J. Royal Statistical Society B 21(3), pp. 611-622, 1999.
%  [2] Probablistic Modeling Toolkit 3, pmtk3.googlecode.com
%  [3] P.A. Forero, A. Cano and G.B. Giannakis, Distributed clustering
% using wireless sensor networks, IEEE J. Selected Topics in Signal 
% Processing 5(4), August 2011.

COUNTER_MAX = 10000;

% Set default convergence criterion
if nargin < 6
    THRESH = 10^(-5);
elseif nargin < 5
    iter_obj = 10;
end

% D dimensions x N samples
[D, N] = size(X);

% J = number of nodes
J = max(V);

% Check graph is valid
[r,c] = size(E);
if r ~= c
    error('Adjacency matrix is not square!');
elseif r ~= J
    error('Adjacency matrix size does not match number of nodes!');
elseif abs(E' - E) ~= 0
    error('Graph should be indirectional!');
elseif ETA <= 0
    error('Learning rate (ETA) should be positive!');
end

% Find Bi in advance to speed up
Bj = cell(J,1);
for idx = 1 : J
    Bj{idx} = find(E(idx,:) > 0);
end

% Build Xi for faster computation
Xi = cell(J,1);
for idx = 1 : J
    Xi{idx} = X(:, V == idx);
end

% Local parameters and auxiliary variables defined here for simplicity.
% In the real environment, these variables reside in each local sensor.
% Initialize parameters and Lagrange multipliers
if nargin<7
    Wi = rand(D, M, J);
    MUi = rand(D, J);
    PRECi = repmat(1./rand(1), [J, 1]);
else
    % You can further perturb initializations here if you want
    Wi = zeros(D, M, J);
    for idj = 1 : J
        Wi(:,:,idj) = model_init.W(:,:,idj) ;%+ rand(D, M)/1000;
    end
    MUi = repmat(model_init.MU ,[1, J]);% + rand(D, 1)/1000, [1, J]);
    PRECi = repmat(1./model_init.VAR, [J 1]); %+ rand(1)/1000, [J, 1]);
end
LAMBDAi = zeros(D,M,J);
GAMMAi = zeros(D,J);
BETAi = zeros(J,1);
ETAhalf = ETA * 0.5;

% Initialize objective function - Lagrangian (we're minimizing this)
oldObjLR = realmax;
objArray = zeros(COUNTER_MAX, J+1); % last one is for total
Fi = zeros(J, 1);

% Initialize latent variables (for loop implementation)
EZ = cell(J, 1);
EZZt = cell(J, 1);

% Prepare performance measures
converged = 0;
counter = 1;
tic;

%% Main loop
while counter <= COUNTER_MAX
    %% --------------------------------------------------------------------
    % Temporarily store parameters to simulate broadcasting and
    % synchronization. All nodes should update before propagating their 
    % values, i.e. each node cannot see the next iteration of its neighbors 
    % before it computes current iteration.
    Wi_old = Wi;
    MUi_old = MUi;
    PRECi_old = PRECi;
    
    %% --------------------------------------------------------------------
    % In each node: Update parameters
    for idx = 1 : J
        % Update (Eq. 7, 8, 9, 13)
        [Wnew, MUnew, PRECnew, EZn, EZnZnt, Fnew] = ...
            dppca_local(Xi{idx}, M, idx, Bj{idx}, ETA, ...
                Wi_old, MUi_old, PRECi_old, LAMBDAi, GAMMAi, BETAi);
        
        % Assign updated parameters and latent statistics with new values
        Wi(:,:,idx) = Wnew(:,:);
        MUi(:,idx) = MUnew(:);
        PRECi(idx) = PRECnew;
        EZ{idx} = EZn;
        EZZt{idx} = EZnZnt;
        Fi(idx) = Fnew;
    end
    
    %% --------------------------------------------------------------------
    % Broadcast updated parameters to neighbors.
    % This is automatically done as we are passing variables directly.
    % In real settings, sensors should transmit parameters over network.

    %% --------------------------------------------------------------------
    % In each node: Update Lagrange multipliers
    for idx = 1 : J
        % Update (Eq. 10, 11, 12)
        for jn = 1:length(Bj{idx})
            LAMBDAi(:,:,idx) = LAMBDAi(:,:,idx) + ...
                ETAhalf * (Wi(:,:,idx) - Wi(:,:,Bj{idx}(jn)));
            GAMMAi(:,idx) = GAMMAi(:,idx) + ...
                ETAhalf * (MUi(:,idx) - MUi(:,Bj{idx}(jn)));
            BETAi(idx) = BETAi(idx) + ...
                ETAhalf * (PRECi(idx) - PRECi(Bj{idx}(jn)));
        end
    end
    
    %% --------------------------------------------------------------------
    % Stopping criterion checkpoint
        
    % Compute Lagrangian
    objLR = 0;
    for idx = 1 : J
        objLRi = Fi(idx);
        for jn = 1:length(Bj{idx}) 
            objLRi = objLRi ...
                + trace(LAMBDAi(:,:,idx)'*(Wi(:,:,idx)-Wi(:,:,Bj{idx}(jn)))) ...
                + (GAMMAi(:,idx)'*(MUi(:,idx)-MUi(:,Bj{idx}(jn)))) ...
                + (BETAi(idx)'*(PRECi(idx)-PRECi(Bj{idx}(jn)))) ...
                + ETAhalf * norm(Wi(:,:,idx)-Wi(:,:,Bj{idx}(jn)),2)^2 ...
                + ETAhalf * norm(MUi(:,idx)-MUi(:,Bj{idx}(jn)),2)^2 ...
                + ETAhalf * (PRECi(idx)-PRECi(Bj{idx}(jn)))^2;
        end
        objArray(counter, idx) = objLRi;
        objLR = objLR + objLRi;
    end
    objArray(counter,J+1) = objLR;
    relErr = (objLR - oldObjLR) / abs(oldObjLR);
    oldObjLR = objLR;
    
    % Show progress if requested
    if nargin >= 8 && iter_obj > 0 && mod(counter, iter_obj) == 0
        fprintf('Iter = %d:  Cost = %f (rel %3.2f%%) (J = %d, ETA = %f)\n', ...
            counter, objLR, relErr*100, J, ETA);
    end
    
    % Check whether it has converged
    if abs(relErr) < THRESH
        converged = 1;
        break;    
    end

    % Increase counter
    counter = counter + 1;
end

%% Finialize optimization

% Check convergence
if converged ~= 1
    fprintf('Could not converge within %d iterations.\n', COUNTER_MAX);
end

% Compute performance measures
eTIME = toc;
eITER = counter;

% Fill in remaining objective function value slots.
if counter < COUNTER_MAX
    % fill in the remaining objArray
    for idx = 1:J+1
        objArray(counter+1:COUNTER_MAX,idx) = ...
            repmat(objArray(counter,idx), [COUNTER_MAX - counter, 1]);
    end
end

% Remove rotational ambiguity
for idx = 1 : J
    [U, TEMP, V] = svd(Wi(:,:,idx));
    Wi(:,:,idx) = Wi(:,:,idx) * V;
    EZ{idx} = V'*EZ{idx};
    for idy = 1 : size(Xi{idx},2)
        EZZt{idx}(:,:,idy) = V' * EZZt{idx}(:,:,idy) * V;
    end
end

% Assign return values
W = Wi;
MU = MUi;
VAR = 1./PRECi;

% Create structure
model = structure(W, MU, VAR, EZ, EZZt, eITER, eTIME, objArray, ...
    LAMBDAi, GAMMAi, BETAi);

end
