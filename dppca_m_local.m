function [W_new, MU_new, PREC_new, EZn, EZnZnt, F_new] = dppca_m_local(...
    Xi, MissIDXi, M, idx, Bj, ETA, Wi, MUi, PRECi, LAMBDAi, GAMMAi, BETAi)
% DPPCA_M_LOCAL  Distributed Probablistic PCA (D-PPCA) Local Node with
%                Missing Values
% 
% Description
%  model = dppca_m_local(.) solves local optimization problem using
% iteration forumla and return consensus-enforced parameters so that they
% can be broadcasted to neighbors. For simpler implementation, this
% function can access all parameters in the network although it will only
% use values actually accessible in the real environments.
%
% Input
% Xi       : D x Ni matrix for data from idx-th node (N=Ni)
% MissIDXi : D x Ni matrix for missing value indication
% M        : Projected dimension
% idx      : Current node index
% Bj       : List of indexes in the ball Bi (one-hop neighbors of node i)
% ETA      : Scalar Learning ratio
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
%  on     2012.06.01 (last modified on 2012/06/01)

% Get size of this samples and ball of this node
[D, Ni] = size(Xi);
cBj = length(Bj);

% Initialize latent variables (for loop implementation)
EZn = zeros(M, Ni);
EZnZnt = zeros(M, M, Ni);

%% E-step

for n = 1:Ni
    % Get indexes of available features
    DcI = (MissIDXi(:,n) == 1);

    % Compute Mi^(-1) = (Wi'Wi + VARi*I)^(-1) first
    Wc = Wi(DcI,:,idx);
    Miinv = inv( Wc' * Wc + 1/PRECi(idx) * eye(M) );

    % E[Zn] = Mi^(-1) * Wi' * (Xin - MUi)
    % Currently M x N
    EZn(:,n) = Miinv * Wc' * (Xi(DcI,n) - MUi(DcI,idx));

    % E[z_n z_n'] = VAR * Minv + E[z_n]E[z_n]'
    % Currently M x M
    EZnZnt(:,:,n) = 1/PRECi(idx) * Miinv + EZn(:,n) * EZn(:,n)';
end   
    
%% M-step

% Update Wi
W_new = zeros(D,M);
for d = 1:D
    % Get indexes of samples with current feature
    NcI = (MissIDXi(d,:) == 1);
    
    W_new1 = (Xi(d, NcI) - MUi(d, idx)) * EZn(:, NcI)';
    W_new2 = 2 * LAMBDAi(d,:,idx);
    W_new3 = zeros(1, M);
    for jn = 1:cBj
        W_new3 = W_new3 + (Wi(d,:,idx) + Wi(d,:,Bj(jn)));
    end
    W_new4 = sum(EZnZnt(:,:,NcI), 3) * PRECi(idx) + 2*ETA*cBj*eye(M);
    W_new(d,:) = (PRECi(idx) * W_new1 - W_new2 + ETA * W_new3) / W_new4;
end

% Update MUi
MU_new = zeros(D, 1);
for d = 1:D
    % Get indexes of samples with current feature
    NcI = (MissIDXi(d,:) == 1);
    
    MU_new1 = PRECi(idx) * sum( Xi(d,NcI) - W_new(d,:)*EZn(:,NcI), 2 );
    MU_new2 = 2 * GAMMAi(d,idx);
    MU_new3 = 0;
    for jn = 1:cBj
        MU_new3 = MU_new3 + ( MUi(d,idx) + MUi(d,Bj(jn)) );
    end
    MU_new4 = sum(NcI) * PRECi(idx) + 2 * ETA * cBj;
    
    MU_new(d) = (MU_new1 - MU_new2 + ETA * MU_new3) / MU_new4;
end

% Solve for VARi^(-1)
PREC_new1 = 2*ETA*cBj;
PREC_new21 = 2*BETAi(idx);
PREC_new22 = 0;
for jn = 1:cBj
    PREC_new22 = PREC_new22 + ETA*(PRECi(idx) + PRECi(Bj(jn)));
end
PREC_new23 = 0;
PREC_new24 = 0;
PREC_new4 = 0;
for n = 1:Ni
    % Get indexes of available features
    DcI = (MissIDXi(:,idx) == 1);
    PREC_new4 = PREC_new4 + sum(DcI);
    Wc = W_new(DcI,:);
    
    PREC_new23 = PREC_new23 + EZn(:,n)' * Wc' * ( Xi(DcI,n) - MU_new(DcI) );
    PREC_new24 = PREC_new24 + ...
        0.5 * ( norm(Xi(DcI,n) - MU_new(DcI,:), 2)^2 + trace( EZnZnt(:,:,n) * (Wc' * Wc) ) );
end
PREC_new2 = PREC_new21 - PREC_new22 - PREC_new23 + PREC_new24;
PREC_new3 = -PREC_new4 / 2;
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

%% Compute objective formula
obj_val1 = 0;
obj_val2 = 0;
obj_val3 = 0;
obj_val4 = 0;
obj_val5 = 0;
for n = 1:Ni
    DcI = (MissIDXi(:,n) == 1);
    Xc = Xi(DcI,n);
    MUc = MU_new(DcI);
    Wc = W_new(DcI,:);

    obj_val1 = obj_val1 + 0.5 * sum(DcI) * log(2 * pi / PREC_new);
    obj_val2 = obj_val2 + 0.5 * trace(EZnZnt(:,:,n));
    obj_val3 = obj_val3 + (0.5*PREC_new) * norm(Xc - MUc, 2).^2;
    obj_val4 = obj_val4 + PREC_new * EZn(:,n)' * Wc' * (Xc - MUc);
    obj_val5 = obj_val5 + (0.5*PREC_new) * trace(EZnZnt(:,:,n) * (Wc' * Wc));
end
F_new = (obj_val1 + obj_val2 + obj_val3 - obj_val4 + obj_val5);
