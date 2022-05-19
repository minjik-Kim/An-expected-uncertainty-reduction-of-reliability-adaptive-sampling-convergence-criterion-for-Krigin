function [ERe, Exp_CI] = Submit_Expected_Re_realization(Info_Krig, sn_Func , CP, n, x_DV, Var, x_samp, sn_MCMC, seeds, Mterms, reg)
    [roots_x,C]=gausshermi(n);
    rng(seeds)
    xi = randn(Mterms,sn_Func); % xi will be updated later.
    dmodel = Info_Krig.dmodel; X_t = Info_Krig.X_t; Y_t = Info_Krig.Y_t;

    dim = size(x_DV,2);
    theta_Krig =1*ones(1,dim);
    theta = dmodel.theta;
    theta_save = dmodel.theta; 

    a = []; ns = size(X_t,1);
    
    X_t_use = X_t;
    X_t_use(ns+1,:) = CP;
    
    [M_for_use,~,MSE_for_use] = predictor(X_t_use(ns+1,:),Info_Krig.dmodel);
    
    Pf_IS_use = zeros(1,sn_Func); Exp_output = zeros(sn_Func,sn_MCMC); % [Pf_IS_use] will be updated later.
    
    for iii = 1:n
        clearvars R2_mod F_mod hpdfval_mod x_test
        Y_t(ns+1,:) = sqrt(2) * sqrt(MSE_for_use) * roots_x(iii) + M_for_use;
        
        if reg == "reg0"
            [dmodel,~]=dacefit(X_t_use, Y_t, @regpoly0, @corrgauss, theta_save); % original hyper-parameter
%             [dmodel,~]=dacefit(X_t_use, Y_t, @regpoly0, @corrgauss, theta_Krig, 10^(-1)*ones(1,dim) ,10^(1)*ones(1,dim));
        elseif reg == "reg1"
            [dmodel,~]=dacefit(X_t_use, Y_t, @regpoly1, @corrgauss, theta_save); % original hyper-parameter
%             [dmodel,~]=dacefit(X_t_use, Y_t, @regpoly1, @corrgauss, theta_Krig, 10^(-1)*ones(1,dim) ,10^(1)*ones(1,dim));
        else
            [dmodel,~]=dacefit(X_t_use, Y_t, @regpoly2, @corrgauss, theta_save); % original hyper-parameter
%             [dmodel,~]=dacefit(X_t_use, Y_t, @regpoly2, @corrgauss, theta_Krig, 10^(-1)*ones(1,dim) ,10^(1)*ones(1,dim));
        end
        Info_Krig_mod{iii}.dmodel = dmodel; Info_Krig_mod{iii}.X_t = X_t_use; Info_Krig_mod{iii}.Y_t = Y_t; % update Kriging model
        [Re(iii,:), ~, ~, ~] = Re_realization(Info_Krig_mod{iii}, sn_Func , 0, x_DV, Var, x_samp, sn_MCMC, seeds, reg); % 
    end
    ERe = sum(Re.*C',1)/sqrt(pi);
    Exp_CI = quantile(ERe,[0.025:0.025:0.975]);
end