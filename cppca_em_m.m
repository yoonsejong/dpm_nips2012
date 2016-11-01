function model = cppca_em_m( X, MissIDX, M, THRESH, model_init, iter_obj )
% CPPCA_EM_M  Probablistic Principal Component Analysis (PPCA) using EM
%             with Missing Values
% 
% Description
%  Compute estimates of W and VAR given (D dimension)x(N observations)
%  sample matrix X and convergence threshold using EM algorithm. Default is
%  1e-6. Returns same output to ppca(X, M) but uses EM algorithm thus WW'
%  will be equal not W. We don't compute sample covariance matrix,
%  eigenvalues and eigenvectors here. We used loops deliverately for
%  clearer presentation sacrificing speed.
%
%  In addition, this function takes MissIDX, which indicates whether each
%  feature is missing or not. If no feature is missing, i.e. MissIDX =
%  ones(D, N), this function is equivalent to cppca_em.
%
% Input
% X          : D x N observation matrix
% MissIDX    : D x N matrix indicating a feature is missing (0) or not (1)
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
%  on     2012.06.01 (last modified on 2012/06/01)
%
% References
%  [1] M.E. Tipping and C.M. Bishop, Probablistic principal component 
%      analysis, J. R. Statist. Soc. B (1999), 61 Part 3, pp. 611-622.
%  [2] Probablistic Modeling Toolkit 3, pmtk3.googlecode.com

COUNTER_MAX = 100000;

% Set default convergence criterion
if nargin < 4
    THRESH = 1e-6;
elseif nargin < 6
    iter_obj = 1;
end

% D x N = original dimension x samples
[D, N] = size(X);

% Initialize parameters
% Note that we don't need to initialize MU
if nargin < 5
    W = rand(D, M);
    MU = rand(D, 1);
    VAR = rand(1);
else
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
%S = cov_m(X, MissIDX);

% Prepare performance measures
converged = 0;
counter = 1;
tic;

%% Main loop
while counter <= COUNTER_MAX

    %% E step (we need to compute estimates for each sample differently)
    
    for idx = 1:N
        % Get indexes of available features
        DcI = (MissIDX(:,idx) == 1);
        
        % Compute Minv = (W'W + VAR*I)^(-1) first
        % This part now depends on idx since each sample has different set
        % of observable features.
        Wc = W(DcI,:);
        Minv = inv(Wc'*Wc + VAR*eye(M));

        % E[z_n] = Minv * W' * (x_n - MU)
        % Currently M x 1
        EZn(:,idx) = Minv * Wc' * (X(DcI,idx) - MU(DcI));

        % E[z_n z_n'] = VAR * Minv + E[z_n]E[z_n]'
        % Currently M x M
        EZnZnt(:,:,idx) = VAR * Minv + EZn(:,idx) * EZn(:,idx)';
    end
    
    %% M step (we need to update each feature element differently)

    % W_new (Eq. 12.56, PRML (Bishop, 2006) pp.578)
    W_new = zeros(D,M);
    for idd = 1:D
        % Get indexes of samples with current feature
        DcI = (MissIDX(idd,:) == 1);

        W_new1 = (X(idd,DcI) - MU(idd)) * EZn(:,DcI)';
        W_new2 = sum(EZnZnt(:,:,DcI), 3);
        W_new(idd,:) = W_new1 / W_new2 ;
    end

    % MU_new: Now we need to take care of missing values.
    MU_new = zeros(D,1);
    for idx = 1:D
        % Get indexes of samples with current feature
        DcI = (MissIDX(idx,:) == 1);
        
        MU_new(idx) = mean( X(idx,DcI) - W_new(idx,:)*EZn(:,DcI) , 2);
    end

    % VAR_new (Eq. 12.57, PRML (Bishop, 2006) pp.578)
    VAR_new1 = 0;
    VAR_new2 = 0;
    VAR_new3 = 0;
    VAR_new4 = 0;
    for idx = 1:N
        % Get indexes of available features
        DcI = (MissIDX(:,idx) == 1);
        VAR_new1 = VAR_new1 + sum(DcI);
        
        VAR_new2 = VAR_new2 + norm((X(DcI,idx) - MU_new(DcI)), 2)^2;
        VAR_new3 = VAR_new3 + 2*(EZn(:,idx)' * W_new(DcI,:)' * (X(DcI,idx) - MU_new(DcI)));
        VAR_new4 = VAR_new4 + trace(EZnZnt(:,:,idx) * (W_new(DcI,:)' * W_new(DcI,:)));
    end
    VAR_new1 = 1 / VAR_new1;
    VAR_new = VAR_new1 * (VAR_new2 - VAR_new3 + VAR_new4);
    
    %% Check convergence
    
    % Compute data log likelihood (we don't need to compute constant)
    obj_val1 = 0;
    obj_val2 = 0;
    obj_val3 = 0;
    obj_val4 = 0;
    obj_val5 = 0;
    for idx = 1:N
        DcI = (MissIDX(:,idx) == 1);
        Xc = X(DcI,idx);
        MUc = MU_new(DcI);
        Wc = W_new(DcI,:);        
        
        obj_val1 = obj_val1 + 0.5 * sum(DcI) * log(2 * pi * VAR_new);
        obj_val2 = obj_val2 + 0.5 * trace(EZnZnt(:,:,idx));
        obj_val3 = obj_val3 + (1/(2*VAR_new)) * norm(Xc - MUc, 2).^2;
        obj_val4 = obj_val4 + (1/VAR_new) * EZn(:,idx)' * Wc' * (Xc - MUc);
        obj_val5 = obj_val5 + (1/(2*VAR_new)) * trace(EZnZnt(:,:,idx) * (Wc' * Wc));
    end
    LL = -(obj_val1 + obj_val2 + obj_val3 - obj_val4 + obj_val5);
    
    objArray(counter,1) = LL;
    relErr = (LL - oldLL)/abs(oldLL);
    oldLL = LL;
    
    % Show progress if requested
    if nargin >= 6 && iter_obj > 0 && (mod(counter, iter_obj) == 0)
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
% minimization function so as to compare with distributed version
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
