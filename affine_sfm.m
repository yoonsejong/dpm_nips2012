function [sfm_M, sfm_S, sfm_b, U3, W3, V3] = affine_sfm(measurement_matrix)
% AFFINE_SFM  Affine Structure from Motion given measurement matrix. 
%
% Description
%  We first translate points to their center of mass of points in each
% frame, i.e.
%             (x,y) = (x,y) - mean(measurement_matrix,2)
% Then we compute SVD of measurement_matrix as measurement_matrix = U W V'
% Our affine solution for SfM is
%             U   = motion (A)
%             VW' = structure (P)
% We ignore rotational ambiguity for the time being.
%
% IN
%  measurement_matrix - 2M x N measurement matrix with M farmes of N points
%
% OUT
%  sfm_M    - Motion (rotation) matrix (U * sqrt(W) * Q)
%  sfm_S    - Structure matrix (Q' * sqrt(W) * V)
%  sfm_b    - Vector b
%  U3,W3,V3 - Result of SVD on measurement_matrix
%  Q        - Orthogonal constraint

% Compute centroid and set fixed coordinate
N = size(measurement_matrix, 2);

centroid = mean(measurement_matrix, 2);
mm_trans = measurement_matrix - repmat(centroid, [1, N]);

% Do SVD
[U, W, V] = svd(mm_trans);

% We throw away null space
U3 = U(:, 1:3);
W3 = W(1:3, 1:3);
V3 = V(:, 1:3);

% Find an affine SfM solution
sfm_M = U3 * sqrt(W3);
sfm_S = sqrt(W3) * V3';

% Remove ambiguity by finding nullspace (= solving the least square)
[sfm_M,sfm_S] = remove_rot_amb( sfm_M, sfm_S );

% Reconstruct back
sfm_b = sfm_M * sfm_S + repmat(centroid, [1 N]);

end

%% remove_rot_amb
function [M, S] = remove_rot_amb(M, S)

% Remove ambiguity by finding nullspace (= solving the least square)
QQt = getQQt(M);
[Uq, Dq, Vq] = svd(QQt);
Q = Uq * sqrt(Dq);

% find final SfM solution
M = M * Q;
S = inv(Q) * S;

end

%% getQQt
function QQt = getQQt(sfm_M, F)

if nargin < 2
    F = size(sfm_M, 1);
end

A = [];
b = repmat([1; 1; 0], [F/2, 1]);

for idx = 1:ceil(F/2)
    r11 = sfm_M(idx*2-1,1:3)'*sfm_M(idx*2-1,1:3);  % r1*r1'
    r22 = sfm_M(idx*2,1:3)'*sfm_M(idx*2,1:3);      % r2*r2'
    r12 = sfm_M(idx*2-1,1:3)'*sfm_M(idx*2,1:3);    % r1*r2'

    A = [ A; r11(:)'; r22(:)'; r12(:)' ];
end

% Since QQt is symmetric, only 6 variables are independent
idx = [1 2 3; 2 4 5; 3 5 6];
idx = idx(:);

% Form the matrix of independent coefficients
B = zeros(3*F/2,6);
for i=1:6,
  B(:,i) = sum(A(:,idx==i),2);
end

% find QQt
Y = B \ b;
QQt = reshape(Y(idx), [3, 3])';

end
