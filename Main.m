clc;clear
warning('off', 'all');
addpath('TIME-FS')
addpath('CalcMeasures')

%% load data
load('COIL20_MissingRatio0.4.mat'); %The COIL20 dataset contains 40% of the samples randomly removed from each view, serving as missing data.
c = length(unique(Y)); % The number of clusters
V = numel(X); % The number of views
n = size(X{1},2); % The number of samples

options = [];
options.gamma = 6; options.lambda = 0.001; options. eta = 1;

[W,Xhat] = TIME_FS(X,c,options);

% Use the feature selection matrices W to select features
% and evalute the seleced features by k-means clustering
Xhat = cell2mat(Xhat');
W = cell2mat(W'); %Concatenate W{v} vertically
[~,idx] = sort(sum(W.*W,2),'descend');
d = size(Xhat,1); %The number of total features.
fea_ratio = 0.4; %The proportion of selected features.
feaSet = Xhat(idx(1 : round(fea_ratio*d)),:)';
Metric = zeros(30,2);
for i = 1:30
    yhat = kmeans(feaSet,c,'Start', 'kmeans++');
    yhat=bestMap(Y,yhat);
    acc_ = length(find(yhat == Y))/length(yhat); % Calculate ACC
    nmi_ = MutualInfo(Y,yhat); % Calculate NMI
    Metric(i,:) = [acc_ nmi_]; 
end
fprintf('ACC(mean) = %f, NMI(mean) = %f\n',mean(Metric(:,1)),mean(Metric(:,2)))




