function [Earr] = get_adj_graph(J)
% GET_ADJ_GRAPH     Get a set of square adjacent graphs for some topologies
%
% INPUT
%   J       Number of nodes
%
% OUTPUT
%   Earr    Set of adjacent graphs
%
% Implemented/Modified
%  by     Sejong Yoon (sjyoon@cs.rutgers.edu)
%  on     2012.01.30 (last modified on 2012/02/01)

% E: Network topology
E1 = ones(J) - eye(J); % complete
E2 = diag(ones(1,J-1),1); E2(J,1) = 1; E2 = E2+E2'; % ring
E3 = [ones(1,J);zeros(J-1,J)]; E3 = E3+E3'; % star
E4 = diag(ones(1,J-1),1); E4 = E4+E4'; % chain
if J > 4 && mod(J, 4) == 0
    E5 = [ones(1,J/2) zeros(1,J/2); zeros(J/2-1,J); ...
          zeros(1,J/2) ones(1,J/2) ; zeros(J/2-1,J)]; 
    E5(1,J/2+1)=1; E5=E5+E5'; % cluster
    Earr = {E1, E2, E3, E4, E5};
else
    Earr = {E1, E2, E3, E4};
end

end
