function [W,Xhat] = TIME_FS(X,c,options)
%% References
    % Y.Y. Huang, M.H. Lu, W. Huang, X.W. Yi, T.R. Li, TIME-FS: Joint Learning of Tensorial Incomplete Multi-View Unsupervised Feature Selection and Missing-View Imputation,
    % in Proceedings of the AAAI Conference on Artificial Intelligence, 2025.
%% Input
    %X=cell(1,V) multi-view dataset, each cell is a view, each column of represents a sample
    %options: struct with parameters (gamma, lambda, eta, r, k)
    %c: the number of clusters
%% Output
    %W: Feature selection matrix
    %Xhat: the imputed multi-view data

options.r = c;% The number of anchors
options.k = c;% The dimension of the low-dimensional subspace
V = size(X,2);% The number of views
[~,n] = size(X{1});% The number of samples

% -------------- Initialize E-------------- %
G = cell(1,V);
E = cell(1,V);
Xhat = cell(1,V);
for v = 1:V
    % Construct the missing sample indicator matrix for v-th view
    idx_v = find(ismissing(X{v}(1,:))); 
    ns = numel(idx_v); 
    G{v} = sparse(1:ns, idx_v, 1, ns, n); 

    %Initialize the missing values by mean values   
    dv = size(X{v}, 1); 
    Xv = X{v}; 
    Xv(:, idx_v) = 0; 
    E{v} = (sum(Xv, 2) / (n - ns)).* ones(dv,ns);  
    X{v} = Xv; 
    Xhat{v} = Xv + E{v} * G{v};
end
clear Ev Evi dv

% -------------- Initialize Z W-------------- %
W = cell(1,V);
Z = zeros(options.k,n,V);
QW = cell(1,V);
for v = 1:V
    tmp = rand(n,options.k);
    [zr,~] = qr(tmp,0);
    zr = zr(:,1:options.k);
    Z(:,:,v) = zr';

    W{v} = Xhat{v}*(Z(:,:,v)');
    QW{v} = 1./(2*sqrt(sum(W{v}.*W{v},2))+eps);
end
clear zr 

% -------------- Initialize A M P -------------- %
A = rand(options.k,options.r);
P = rand(V,options.r);
M = updateM (Z,A,P,options); 
% -------------- Initialize alpha -------------- %
alpha =  updateAlpha(Xhat,W,Z,QW,options);

% -------------- Optimization -------------- %
max_iter = 50; %The maximum number of iterations can be adjusted depending on the situation.
for  iter = 1:max_iter
    Z = updateZ(Xhat,W,alpha,A,M,P,options);
    [W,QW] = updateW(Xhat,Z,QW,options);
    A = updateA(Z,M,P);
    P = updateP(Z,A,M);
    M = updateM (Z,A,P,options);
    E = updateE(W,Z,G);
    for v = 1:V
       Xhat{v} = X{v}+E{v}*G{v}; 
    end
    alpha = updateAlpha(Xhat,W,Z,QW,options);

    % -------------- calculate obj -------------- %
    obj(iter) = obj_fun(Xhat,W,Z,QW,alpha,A,M,P,options);
    if  iter > 1 && abs(obj(iter)-obj(iter-1))<1e-3
        break
    end
end

end

function f=obj_fun(Xhat,W,Z,QW,alpha,A,M,P,options)
%% input 
% Xhat: The imputed multi-view dataset
% W: Feature selection matrix 
% Z: Tensor formed by stacking low-dimensional representations of views.
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W
% alpha: Adaptive weights for each views 
% A: Consensus anchor matrix
% M: Concensus anchor graph
% P: Anchor preference weight matrix;
% options: Parameters 
%% ouput
% f: objective function value

V=size(Xhat,2);

f1=0;
f2=0;
f3 = 0;
for v = 1:V
    Xv_hat = Xhat{v}; 
    f1 = f1 + (alpha(v)^options.gamma)*(norm(Xv_hat-W{v}*Z(:,:,v),'fro')^2);  
    f2 = f2+(alpha(v)^options.gamma)*trace((W{v}.*QW{v})'*W{v});
    f3 = f3 + norm(Z(:,:,v)-A*diag(P(v,:))*M','fro')^2;
end

f4 = norm(M,'fro')^2;

f = f1+options.lambda*f2+f3+options.eta*f4;
clear f1 f2 f3 f4
end

function A = updateA(Z,M,P)
%% input 
% Z: Tensor formed by stacking low-dimensional representations of views
% M: Concensus anchor graph
% P: View preference weight matrix

%% ouput
% A: Consensus anchor matrix

% Obtain the mode-1 unfolding of tensor Z
tensor_size = size(Z);
Z_mode_1 = reshape(Z, tensor_size(1), []);
% Calculate P*M ,where * is khatri-rao product
PM = khatrirao(P,M);
% Update A
A = Z_mode_1*PM/(PM'*PM);
clear tensor_size Z_mode_1 PM
end

function [M] = updateM (Z,A,P,options)
%% input 
% Z: Tensor formed by stacking low-dimensional representations of views
% A: Consensus anchor matrix
% P: View preference weight matrix
% options: Parameters 

%% ouput
% M: Consensus anchor graph

tensor_size = size(Z);
n = tensor_size(2);

% Obtain the horizontal mode-2 unfolding of tensor Z
Z_mode_2 = reshape(permute(Z, [2 1 3]), tensor_size(2), []);
I = eye(options.r); % r is the number of anchor

% Calculate P*A ,where * is khatri-rao product
PA = khatrirao(P,A);

Y = -1*Z_mode_2*(PA)/(options.eta*I+(PA')*PA);

% Update M
v1 = ones(options.r,1); 
tmp = ((1+Y*v1)/options.r).*ones(n,options.r)-Y;
M = max(tmp,0);
clear tensor_size Z_mode_2 PA Y tmp
end

function P = updateP(Z,A,M)
%% input 
% Z: Tensor formed by stacking low-dimensional representations of views
% A: Consensus anchor matrix
% M: Concensus anchor graph
%% ouput
% P: View preference weight matrix

% Obtain the horizontal mode-3 unfolding of tensor Z
tensor_size = size(Z);
Z_mode_3 = reshape(permute(Z, [3 1 2]), tensor_size(3), []);

% Calculate P*M ,where * is khatri-rao product
MA = khatrirao(M,A);
% Update P
P = Z_mode_3*MA/(MA'*MA);   
clear Z_mode_3 tensor_size MA
end 

function [alpha] = updateAlpha(Xhat,W,Z,QW,options)
%% input 
% Xhat: The imputed multi-view dataset
% W: Feature selection matrix 
% Z: Tensor formed by stacking low-dimensional representations of views
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W
% options: parameters 
%% ouput
% alpha: Adatptive weights of each views

V = size(Xhat,2); 

d = zeros(1,V); 
for v = 1:V
   d(v) = norm(Xhat{v}-W{v}*Z(:,:,v),'fro')^2 + ...
        options.lambda*trace((W{v}.*QW{v})'*W{v});   
end

% Update alpha
alpha = d.^(1/(1-options.gamma))/sum(d.^(1/(1-options.gamma)));
end

function [E] = updateE(W,Z,G)
%% input 
% W: Feature selection matrix 
% Z: Tensor formed by stacking low-dimensional representations of views
% G: Missing samples indicator matrix for each view 
% options: Parameters 
%% ouput
% E: Adaptive imputation matrix for missing samples in each view

V = size(W,2); 

% Update E
E = cell(1,V);
for v =1:V
    E{v} = full(W{v}*(Z(:,:,v)*G{v}'));
end

end

function [W,QW] = updateW(Xhat,Z,QW,options)
%% input 
% Xhat: The imputed multi-view dataset
% Z: Tensor formed by stacking low-dimensional representations of views
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W
% options: Parameters

%% ouput
% W: Feature selection matrix
% QW: Auxiliary matrix for computing the 2,1 norm of matrix W

V = size(Xhat,2);
W = cell(1,V);

for v = 1:V
    Xv_hat = Xhat{v};
    W{v} = (Xv_hat*Z(:,:,v)')./(1-options.lambda*QW{v});
    QW{v} = 1./(2*sqrt(sum(W{v}.*W{v},2)+eps));
end

end

function [Z] = updateZ(Xhat,W,alpha,A,M,P,options)
%% input 
% Xhat: The imputed multi-view dataset
% W: Feature selection matrix 
% alpha: Adaptive weights for each view
% A: Consensus anchor matrix
% M: Concensus anchor graph
% P: Anchor preference weight matrix;
% options: Parameters 
%% ouput
% Z: Tensor formed by stacking low-dimensional representations of views.

V = size(Xhat,2);

[~,n] = size(Xhat{1});

% Update Z
Z = zeros(options.k,n,V);
for v = 1:V
    Cv = A*diag(P(v,:))*M';
    Qv = (alpha(v)^options.gamma)*W{v}'*Xhat{v}+Cv;% intermediate product
    [Uv, ~, Vv] = svd(Qv,'econ');
    Z(:,:,v) = Uv*Vv';
end

end
