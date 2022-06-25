function [W,R,P] = LRAGE(X, lambda, d, k, NITER)        
%% LRAGE: Low-Rank Adaptive Graph Embedding for Unsupervised Feature Extraction
% 
%% Objective Funtion:
%   min_{P,R,W}  sum_ij ||x_i^T-x_j^TPR||_2^2W_ij + alpha ||W||_F^2 + lambda ||PR||_2,1   
%   s.t. P^TP = I_c, W_i >= 0, sum_j W_ij = 1   
% 
%% MATLAB Code:
%   [id,A,BBBBB,AAAAA] = LRAGE(X,gamma,d,k,NITER)        
%       Input£º
%           X: Data matrix with size dim * num. Each column vector represents a data sample.
%           lambda: nonnegative balanced parameter of L2,1 regularization
%           d: subspace dimentionality
%           k: the number of the nearest neighobrs 
%           NITER: the number of iterations
%       Output£º
%           W: adaptive similarity matrix
%           P: projection matrix    
%           R: regression matrix     
% 
%% Reference:
%	Jianglin Lu, Hailing Wang, Jie Zhou, Yudong Chen, Zhihui Lai, Qinghua Hu
%   "Low-Rank Adaptive Graph Embedding for Unsupervised Feature Extraction"
%	Pattern Recognition, 2021
%   Corresponding Author: Jie Zhou (jie_jpu@163.com)

%% Data Normalization
num = size(X,2);
dim = size(X,1);
XT = X';
meanValue = mean(XT);
X_tmp = XT - ones(num,1) * meanValue;
scal = 1./sqrt(sum(X_tmp.*X_tmp) + eps);
scalMat = sparse(diag(scal));
X = X_tmp * scalMat;
X = X';
%% Variables Initialization 
R = eye(d, dim); 
P = zeros(dim, d);
W = zeros(num);
U1 = eye(dim, dim);
distX = L2_distance_1(X, X);
[distX1, idx] = sort(distX, 2);
alpha0 = zeros(num, 1);
for i = 1:num
    di = distX1(i, 2:k+2);
    alpha0(i) = 0.5 * (k * di(k+1) - sum(di(1:k)));
end
alpha = mean(alpha0);
%% Iterations  
for iter = 1:NITER
    %% Update W
    XABXAB = X' * P * R;
    distx = L2_distance_1(X, XABXAB'); 
    if iter>5%1
        [~, idx] = sort(distx, 2);
    end
    W = zeros(num);
    for i=1:num
        idxa0 = idx(i,2:k+1);
        dxi = distx(i,idxa0);   
        ad = -(dxi) / (2 * alpha);
        W(i,idxa0) = EProjSimplex_new(ad);
    end  
    W = (W + W') / 2;
    G = diag(sum(W)); 
    
    %% Update P
    XGX=X * G * X';
    XSX=X * W' * X';
    first=(XGX + lambda .* U1);      
    [P]=Find_K_Max_Eigen(first \ (XSX * XSX'), d);

    %% Update R
    R =(P' * first * P) \ (P' * XSX); 

    %% Update U    
    AB=P * R;
    Xi2 = sqrt(sum(AB.*AB,2) + eps);   
    d2 = 0.5./Xi2;  
    U1 = spdiags(d2, 0, dim, dim);
end

