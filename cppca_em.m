function model = cppca_em( X, M, THRESH, model_init, iter_obj )
% CPPCA_EM   Probablistic Principal Component Analysis (PPCA) using EM
% 
% Description
%  Compute estimates of W and VAR given (D dimension)x(N observations) 
% sample matrix X and convergence threshold using EM algorithm. Default 
% is 1e-6. Returns same output to ppca(X, M) but uses EM algorithm thus 
% WW' will be equal not W. We don't compute sample covariance matrix, 
% eigenvalues and eigenvectors here. We used loops deliverately for 
% clearer presentation sacrificing speed. 
%
% Input
% X          : D x N observation matrix
% M          : 1 x 1 scalar for latent space dimension
% THRESH     : 1 x 1 scalar convergence criterion
% model_init : PPCA model to set initialization parameter
% iter_obj   : If > 0, print out objective value every (iter_obj) iteration
%              (Default 10)
%
% Output
% model = structure(W, MU, VAR, EZ, EZZt, eITER, eTIME, objArray)
% W        : D x M projection matrix
% MU       : D x 1 vector sample means
% VAR      : 1 x 1 scalar estimated variance
% EZ       : M x N matrix mean of N latent vectors
% EZZt     : M x M x N cube for covariance of N latent vectors
% eITER    : Iterations took
% eTIME    : Elapsed time
% objArray : Objective function value change over iterations
%
% Implemented
%  by     Sejong Yoon (sjyoon@cs.rutgers.edu)
%  on     2011.10.06 (last modified on 2012/02/01)
%
% References
%  [1] M.E. Tipping and C.M. Bishop, Probablistic principal component 
%      analysis, J. R. Statist. Soc. B (1999), 61 Part 3, pp. 611-622.
%  [2] Probablistic Modeling Toolkit 3, pmtk3.googlecode.com

COUNTER_MAX = 100000;

% Set default convergence criterion
if nargin < 3
    THRESH = 1e-6;
elseif nargin < 5
    iter_obj = 10;
end

% D x N = original dimension x samples
[D, N] = size(X);

% Initialize parameters
% Note that we don't need to initialize MU
if nargin < 4
    W = rand(D, M);
    MU = rand(D, 1);
    VAR = rand(1);
else
    % You can further perturb initializations here if you want
    W = model_init.W ;%+ rand(D, M)/1000;
    MU = model_init.MU ;%+ rand(D, 1)/1000;
    VAR = model_init.VAR ;%+ rand(1)/1000;
end

% Initialize latent variables (for loop implementation)
EZn = zeros(M,N);
EZnZnt = zeros(M,M,N);

% Initialize data log likelihood as 0 and prepare the constant
oldLL = -realmax;
objArray = zeros(COUNTER_MAX,1);
S  = cov(X');

% Prepare performance measures
converged = 0;
counter = 1;
tic;

%% Main loop
while counter <= COUNTER_MAX

    %% E step
    
    % Compute Minv = (W'W + VAR*I)^(-1) first
    Minv = inv(W'*W + VAR*eye(M));
    
    % E[z_n] = Minv * W' * (x_n - MU)
    % Currently M x N
    EZn = Minv * W' * (X - repmat(MU, [1, N]));

    for idx = 1:N    
        % E[z_n z_n'] = VAR * Minv + E[z_n]E[z_n]'
        % Currently M x M
        EZnZnt(:,:,idx) = VAR * Minv + EZn(:,idx) * EZn(:,idx)';
    end
    
    %% M step
    
    % W_new (Eq. 12.56, PRML (Bishop, 2006) pp.578)
    W_new1 = zeros(D,M);
    W_new2 = zeros(M,M);
    for idx = 1:N
        W_new1 = W_new1 + (X(:,idx) - MU) * EZn(:,idx)';
        W_new2 = W_new2 + EZnZnt(:,:,idx);
    end
    W_new = W_new1 /  W_new2 ;
    
    % MU_new  (NOTE: this is only being assigned here to make the recursion 
    % identical to that of the distributed PCCA)
    MU_new = mean( X - W_new*EZn, 2 );
    % MU_new = MU;

    % VAR_new (Eq. 12.57, PRML (Bishop, 2006) pp.578)
    VAR_new1 = 1 / (N * D);
    VAR_new2 = 0;
    VAR_new3 = 0;
    VAR_new4 = 0;
    for idx = 1:N
        VAR_new2 = VAR_new2 + norm((X(:,idx) - MU_new), 2)^2;
        VAR_new3 = VAR_new3 + 2*(EZn(:,idx)' * W_new' * (X(:,idx) - MU_new));
        VAR_new4 = VAR_new4 + trace(EZnZnt(:,:,idx) * (W_new' * W_new));
    end
    VAR_new = VAR_new1 * (VAR_new2 - VAR_new3 + VAR_new4);
    
    %% Check convergence
    
    % Compute data log likelihood (we don't need to compute constant)
    % C = WW' + VAR*I
    % L = (-N/2)(Dln(2pi) + ln|C| + Tr(C^(-1)S))
    C = W_new * W_new' + VAR_new * eye(D);
    if D>M
    	logDetC = (D-M)*log(VAR_new) + log(det(W_new'*W_new + VAR_new*eye(M)));
    else
    	logDetC = log(det(C));
    end
    LL = (-N/2)*(logDetC + trace(C\S));
    
    objArray(counter,1) = LL;
    relErr = (LL - oldLL)/abs(oldLL);
    oldLL = LL;
    
    % Show progress if requested
    if nargin >= 5 && iter_obj > 0 && (mod(counter, iter_obj) == 0)
        fprintf('Iter %d:  LL = %f (rel %3.2f%%)\n', counter, LL, relErr*100); 
    end

    % Update parameters and latent statistics with new values
    W = W_new;
    VAR = VAR_new;
    MU = MU_new;
    EZ = EZn;
    EZZt = EZnZnt;
    
    % Check convergence
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
eITER = counter;
eTIME = toc;

% Fill in remaining objective function value slots. Convert objective to
% minimization function so as to directly compare with distributed version
if counter < COUNTER_MAX
    objArray(counter+1:COUNTER_MAX,1) = ...
        repmat(objArray(counter,1), [COUNTER_MAX - counter, 1]);
end
objArray = objArray .* -1;

% Remove rotational ambiguity
[U, TEMP, V] = svd(W);
W = W * V;
EZ = V'*EZ;
for idx=1:N
    EZZt(:,:,idx) = V'*EZZt(:,:,idx)*V;
end

% Create model structure
model = structure(W, MU, VAR, EZ, EZZt, eITER, eTIME, objArray);

end
