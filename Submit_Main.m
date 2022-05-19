%% Initialization
%     clearvars -except
clc; close all;
%
Convergence_criterion = 1;
if Convergence_criterion == 1
    % The number of MCMC samples
    sn_MCMC = 2e3;
    % The number of surrogate model uncertainty variables
    sn_Func = 5e4;
    % iteration: # of final data
    x_MCMC = [];
    % The number of quadrature points
    n = 5;
end
%
mu = [4.6876,2.0512]; sigma = [0.5 0.5]; Var = sigma.^2;
NN = size(mu,2);
lower = [0 0]; upper = [10,10];
g = @(x) -1 + (0.9063*x(:,1) + 0.4226*x(:,2) -6).^2+ (0.9063*x(:,1) + 0.4226*x(:,2) -6).^3 - 0.6 * (0.9063*x(:,1) + 0.4226*x(:,2) -6).^4 - (-0.4226*x(:,1) + 0.9063*x(:,2));
Example = 'Example_IOWA';
fprintf('The Example_IOWA example. \n');
theta_Krig =1*ones(1,NN); % Kriging hyper-parameter
t_start = tic; % for (CPU) time
% iter=1; % # of iteration
% Select the initial sample
% % load(strcat(pwd, filesep, 'DATA_save', filesep, 'X_t.mat'))
% % X_t = samp_x_init{data_number};
% % Y_t = g(X_t);
data_number = kk;
X_t = samp_x;
Y_t = samp_y;
reg = "reg0";
%% Calculate true reliability
load(['D:\MjKim_study\Research_MjKim\EURR\Example 1\DATA_save\x_samp_Re_save'])
x_samp = x_samp_Re_save{data_number};
clearvars x_samp_Re_save
Re_true = sum(g(x_samp)<0)/size(x_samp,1)
% delete the samples which are not in boundary (for training)
aa = [];
for jj = 1:size(x_samp,1)
    if all(x_samp(jj,:) > lower) && all(x_samp(jj,:) <= upper), aa(end+1) = jj; end
end
x_samp = x_samp(aa,:);
n_MCS = size(x_samp,1);

fprintf(' probability of failure (true) is %f. \n', 1-Re_true);
fprintf(' COV of probability of failure is %f (percent). \n', sqrt((1-(sum(g(x_samp)<0)/n_MCS))/(n_MCS*(sum(g(x_samp)<0)/n_MCS)))*100);

seeds = randi([1,1e2]);
if reg == "reg0"
    [dmodel,~]=dacefit(X_t, Y_t, @regpoly0, @corrgauss, theta_Krig, 10^(-1)*ones(1,NN) ,10^(1)*ones(1,NN));
elseif reg == "reg1"
    [dmodel,~]=dacefit(X_t, Y_t, @regpoly1, @corrgauss, theta_Krig, 10^(-1)*ones(1,NN) ,10^(1)*ones(1,NN));
else
    [dmodel,~]=dacefit(X_t, Y_t, @regpoly2, @corrgauss, theta_Krig, 10^(-1)*ones(1,NN) ,10^(1)*ones(1,NN));
end
%% sequential sampling process
while 1
    Info_Krig.dmodel = dmodel; Info_Krig.X_t = X_t; Info_Krig.Y_t = Y_t;
    [M_for_use,MSE_for_use] = predictor(x_samp,Info_Krig.dmodel);
    hist_Re(iter) = round(sum(M_for_use<0)/size(x_samp,1),4);
    hist_error(iter) = round(abs(Re_true - hist_Re(iter))/Re_true,4);
    fprintf(' history of relative error: %f. \n', abs(Re_true - hist_Re(iter))/Re_true);
    ns = size(X_t,1); % number of samples
    X_new=[]; [X_new(1,:),convergence] = EFF_mjk(x_samp,dmodel,X_t);
    %% convergence criterion, #0: EUC, #1: EURR, #2: EFF
    if Convergence_criterion == 0
        [EUC,CCV] = EUC_mjk(dmodel,X_t,Y_t,x_samp,X_new,reg);
        hist_CCV(:,iter) = CCV;
        hist_EUC(:,iter) = EUC
        if EUC < 0.001,disp("each criterion: Sampling is over. \n"), break; end
    elseif Convergence_criterion == 1
       if iter >= 10
            %% Realization of reliability using current sample set
            [~, CI, std_Re, Mterms] = Re_realization(Info_Krig, sn_Func , hist_Re(iter), mu, sigma.^2, x_samp, sn_MCMC, seeds, reg);
            CIR = CI(end) - CI(1); %%%%%%%%%%%%
            hist_CI(iter,:) = CI;
            if CI(end) - CI(1) <= 3e-2 && CI(1) < 0.99 %%%%%%%%
                [~, QER] = Expected_Re_realization(Info_Krig, sn_Func , X_new, n, mu, sigma.^2, x_samp, sn_MCMC, seeds, Mterms, reg);
                CIER = QER(:,end) - QER(:,1); %%%%%%%%%%%%%%%%%
                if (CIR - CIER < 0.01) && (QER(1) > 0.005),disp("Proposed criterion: Sampling is over. \n"), break; end
            else
                QER = nan * ones(1, length(CI));
            end
            hist_CIER(iter,:) = QER;
       end
    end
    X_t = [X_t;X_new];
    Y_t = [Y_t;g(X_new)];
    if reg == "reg0"
        [dmodel,~]=dacefit(X_t, Y_t, @regpoly0, @corrgauss, theta_Krig, 10^(-1)*ones(1,NN) ,10^(1)*ones(1,NN));
    elseif reg == "reg1"
        [dmodel,~]=dacefit(X_t, Y_t, @regpoly1, @corrgauss, theta_Krig, 10^(-1)*ones(1,NN) ,10^(1)*ones(1,NN));
    else
        [dmodel,~]=dacefit(X_t, Y_t, @regpoly2, @corrgauss, theta_Krig, 10^(-1)*ones(1,NN) ,10^(1)*ones(1,NN));
    end
    iter = iter + 1;
    fprintf(' Sampling is added. The number of added sample is %d. Elapsed time is %d. \n',iter-1, toc(t_start) );
end
%% after sequential sampling,
Info_Krig.dmodel = dmodel;
Info_Krig.X_t = X_t;
Info_Krig.Y_t = Y_t;

[M_for_use,~] = predictor(x_samp,Info_Krig.dmodel);
Re_final_surrogate = sum(M_for_use<0)/size(x_samp,1);

error = abs(Re_true - Re_final_surrogate)/Re_true

%% save data
if Convergence_criterion == 0
    save(['D:\MjKim_study\Research_MjKim\EURR\Example 1\EFF\', sprintf('(0.001)_EFF_reg0_%d',data_number),'.mat'])
else
    save(['D:\MjKim_study\Research_MjKim\EURR\Example 1\EFF\', sprintf('(rev4)_EURR_reg0_%d',data_number),'.mat'])
end