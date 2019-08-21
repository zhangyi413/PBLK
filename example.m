clc
clear

load('polyL1_2d_10.mat'); % training samples Branin-Hoo function
load('polyL1_2d_5000_te.mat'); % testing samples Branin-Hoo function

X = trX{1}; % training samples X 
Y = trY{1}; % training samples Y
Xtest = teX{1}; % testing samples X
Ytest = teY{1}; % testing samples Y

lb = 0.1*ones(1, size(trX{1}, 2)); % The lower bound of the correlation parameters
ub = 5*ones(1, size(trX{1}, 2)); % The upper bound of the correlation parameters
theta0 = 2.5*ones(1, size(trX{1}, 2)); % The initial values of the correlation parameters

[ Yhat ] = PBLK( X, Y, Xtest, theta0, lb, ub); % Fitting the model

errors = Yhat - Ytest;

rrmse = sqrt( mean( (errors.^2) ) )/std(Ytest, 1);
rmae = max(abs(errors))/std(Ytest, 1);