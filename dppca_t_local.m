function [W_new, MU_new, PREC_new, EZn, EZnZnt, F_new] = dppca_t_local(...
    Xi, M, idx, Bj, ETA, Wi, MUi, PRECi, LAMBDAi, GAMMAi, BETAi, EZ, EZZt)
% DPPCA_T_LOCAL Distributed Probablistic CCA (D-PCCA) Local Node
%               (Transposed)
% 
% Description
%  model = dppca_t_local(.) solves local optimization problem using
% iterative forumla and return consensus-enforced parameters so that they 
% can be broadcasted to neighbors. For simpler implementation, this 
% function can access all parameters in the network.
%
% Input
% Xi    : D x Ni matrix for data from idx-th node (N=Ni)
% M     : Projected dimension
% idx   : Current node index
% Bj    : List of indexes in the ball Bi (one-hop neighbors of node i)
% ETA   : Scalar Learning ratio
% Wi, MUi, VARi, LAMBDAi, GAMMAi, BETAi: All parameters in the network
%
% Output
% W_new    : D x M projection matrix
% MU_new   : D x 1 vector sample means
% PREC_new : 1 x 1 scalar estimated variance
% EZ_new   : M x N matrix containing mean of latent space
% EZZt_new : M x M x N cube containing covariances of latent space
% F_new    : 1 x 1 scalar computed optimization forumla (first term only)
%
% Implemented
%  by     Sejong Yoon (sjyoon@cs.rutgers.edu)
%  on     2011.10.07 (last modified on 2012/02/01)

% Get size of this samples and ball of this node
[D, Ni] = size(Xi);
cBj = length(Bj);

% Initialize latent variables (for loop implementation)
EZn = zeros(M, Ni);
EZnZnt = zeros(M, M, Ni);

%% E-step

% Compute Mi^(-1) = (Wi'Wi + VARi*I)^(-1) first
Miinv = inv( Wi(:,:,idx)' * Wi(:,:,idx) + 1/PRECi(idx) * eye(M) );

% (Eq. 18): E[Zn] = Mi^(-1) * Wi' * (Xin - MUi)
% Currently M x N
EZn = Miinv * Wi(:,:,idx)' * (Xi - repmat(MUi(:,idx), [1, Ni]));

for n = 1:Ni
    % (Eq. 19): E[z_n z_n'] = VAR * Minv + E[z_n]E[z_n]'
    % Currently M x M
    EZnZnt(:,:,n) = 1/PRECi(idx) * Miinv + EZn(:,n) * EZn(:,n)';
end
    
%% M-step

% (Eq. 37): Update Wi
W_new1 = sum(EZnZnt, 3) * PRECi(idx) + 2*ETA*cBj*eye(M);
W_new2 = zeros(D, M);
for n = 1:Ni
    W_new2 = W_new2 + (Xi(:,n) - MUi(:,idx)) * EZn(:,n)';
end
W_new2 = W_new2 * PRECi(idx);
W_new3 = 2 * LAMBDAi(:,:,idx);
W_new4 = zeros(D, M);
for jn = 1:cBj
    W_new4 = W_new4 + (Wi(:,:,idx) + Wi(:,:,Bj(jn)));
end
W_new = (W_new2 - W_new3 + ETA * W_new4) / W_new1 ;

% (Eq. 36): Don't update MUi (we fix zero mean)
MU_new = MUi(:,idx);

% (Eq. 34): Solve for VARi^(-1)
PREC_new1 = 2*ETA*cBj;
PREC_new21 = 2*BETAi(idx);
PREC_new22 = 0;
for jn = 1:cBj
    PREC_new22 = PREC_new22 + ETA*(PRECi(idx) + PRECi(Bj(jn)));
end
PREC_new23 = 0;
PREC_new24 = 0;
for n = 1:Ni
    PREC_new23 = PREC_new23 + EZn(:,n)'*W_new'*(Xi(:,n)-MU_new);
    PREC_new24 = PREC_new24 + 0.5 * ( norm(Xi(:,n)-MU_new,2)^2 ...
        + trace( EZnZnt(:,:,n) * W_new' * W_new ) );
end
PREC_new2 = PREC_new21 - PREC_new22 - PREC_new23 + PREC_new24;
PREC_new3 = -Ni * D / 2;
PREC_new = roots([PREC_new1, PREC_new2, PREC_new3]);

% We follow larger, real solution.
if length(PREC_new) > 1
    PREC_new = max(PREC_new);
end
if abs(imag(PREC_new)) ~= 0i
    error('No real solution!');
    % We shouldn't reach here since both solutions are not real...
end
if PREC_new < 0
    error('Negative precicion!');
end

%% Compute local objective function
S = cov(Xi');
C = W_new * W_new' + (1/PREC_new) * eye(D);
if D>M
    logDetC = (D-M)*log(1/PREC_new) + log(det(W_new'*W_new + (1/PREC_new)*eye(M)));
else
    logDetC = log(det(C));
end
F_new = (Ni/2)*(logDetC + trace(C\S));
