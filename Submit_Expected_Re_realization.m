function [ERe, Exp_CI] = Expected_Re_realization(Info_Krig, sn_Func , CP, n, x_DV, Var, x_samp, sn_MCMC, seeds, Mterms, reg)
    
    [roots_x,C]=gausshermi(n);
    %%
    rng(seeds)
    xi = randn(Mterms,sn_Func);
    %%
    dmodel = Info_Krig.dmodel;
    X_t = Info_Krig.X_t;
    Y_t = Info_Krig.Y_t;

    % theta는 그대로 이용
    dim = size(x_DV,2);
    theta_Krig =1*ones(1,dim);
    
    theta = dmodel.theta;
    theta_save = dmodel.theta; % 중복인가?
%     x_test = x_test_save;

    a = [];

    ns = size(X_t,1);
    X_t_use = X_t;
    X_t_use(ns+1,:) = CP;
    
    [M_for_use,~,MSE_for_use] = predictor(X_t_use(ns+1,:),Info_Krig.dmodel);
    
    Pf_IS_use = zeros(1,sn_Func);
    Exp_output = zeros(sn_Func,sn_MCMC);
    for iii = 1:length(roots_x)
        tic
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
        % Info_Krig_mod
        Info_Krig_mod{iii}.dmodel = dmodel; Info_Krig_mod{iii}.X_t = X_t_use; Info_Krig_mod{iii}.Y_t = Y_t;
        
%         [M_samp,MSE_samp] = predictor(x_samp,Info_Krig_mod{iii}.dmodel); % U-function 고려?
%         Deno_Pf_mod = mean(normcdf(M_samp./sqrt(MSE_samp)));

        h = 0;
        [Re(iii,:), ~, ~, ~] = Re_realization(Info_Krig_mod{iii}, sn_Func , h, x_DV, Var, x_samp, sn_MCMC, seeds, reg); % h: standard deviation
        
% 
%         clearvars x_test
%         
% %         x_test = MCMC_sampling(x_DV,Var,Info_Krig_mod{iii},sn_MCMC,1);
%         if type == 0
%             x_test = x_MCMC;
%         else
%             x_test = MCMC_sampling(x_DV,Var,Info_Krig_mod{iii},sn_MCMC,1);
%         end
%         
%         norm_x_mod = (X_t_use - mean(X_t_use))./std(X_t_use);
%         norm_y_mod = (Y_t - mean(Y_t))./std(Y_t);
%         norm_tp_mod = (x_test - mean(X_t_use))./std(X_t_use);
% 
%         if reg == "reg0"
%             F_mod = regpoly0(norm_x_mod);
%         elseif reg == "reg1"
%             F_mod = regpoly1(norm_x_mod);
%         else
%             F_mod = regpoly2(norm_x_mod);
%         end
%         
%         % fixed theta
%         for i = 1 : size(norm_x_mod,1)
%             for j = i : size(norm_x_mod,1)
%                 R2_mod(i,j) = prod(exp(-theta.*(norm_x_mod(i,:)-norm_x_mod(j,:)).^2)); 
%             end
%         end
%         R2_mod = (R2_mod+R2_mod') - diag(ones([size(norm_x_mod,1),1]));
%         
%         % dmodel.beta
%         estimated_beta_mod = (F_mod'/(R2_mod)*F_mod)\F_mod'/R2_mod*norm_y_mod; 
% 
%         for nn = 1:length(norm_tp_mod)
%             test_point = norm_tp_mod(nn,:);
%             [r_mod(:,nn),~] = corrgauss(theta, norm_x_mod - repmat(test_point,size(norm_x_mod,1),1));
%             
%             if reg == "reg0"
%                 F_n_mod(:,:,nn) = regpoly0(test_point);
%                 Predicted_mean_mod(nn) = (regpoly0(test_point)*estimated_beta_mod+r_mod(:,nn)'*(R2_mod\(norm_y_mod-F_mod*estimated_beta_mod))) * std(Y_t) + mean(Y_t);
%             elseif reg == "reg1"
%                 F_n_mod(:,:,nn) = regpoly1(test_point);
%                 Predicted_mean_mod(nn) = (regpoly1(test_point)*estimated_beta_mod+r_mod(:,nn)'*(R2_mod\(norm_y_mod-F_mod*estimated_beta_mod))) * std(Y_t) + mean(Y_t);
%             else
%                 F_n_mod(:,:,nn) = regpoly2(test_point);
%                 Predicted_mean_mod(nn) = (regpoly2(test_point)*estimated_beta_mod+r_mod(:,nn)'*(R2_mod\(norm_y_mod-F_mod*estimated_beta_mod))) * std(Y_t) + mean(Y_t);
%             end
%         end
% 
%         %% undeleteable (numerical error 고려한 코드 때문에 x_test 샘플 달라짐)
%         clearvars Cor F_ele_comp A_mod B_mod eigvec_mod eigval_mod
%         Cor = prod(exp(-theta.*(repmat(norm_tp_mod,size(norm_tp_mod,1),1)-repelem(norm_tp_mod,size(norm_tp_mod,1),1)).^2),2);
%         Cor = reshape(Cor,size(norm_tp_mod,1),size(norm_tp_mod,1));
% 
%         F_ele_comp = transpose(reshape(F_n_mod,size(F_n_mod,2),size(F_n_mod,3)));
% 
%         A_mod = transpose(r_mod)*(R2_mod\r_mod);
%         B_mod = transpose(transpose(F_ele_comp)-transpose(F_mod)*(R2_mod\r_mod)) * ((transpose(F_mod)*(R2_mod\F_mod)) \ (transpose(F_ele_comp)-transpose(F_mod)*(R2_mod\r_mod)));
% 
%         Predicted_cov_mod = dmodel.sigma2 * (Cor - A_mod + B_mod);
% 
%         diagm_mod = repmat(diag(Predicted_cov_mod),1,size(Predicted_cov_mod,1));
%         cov_mat_mod = Predicted_cov_mod./(sqrt(diagm_mod).*sqrt(diagm_mod'));
% 
%         NN = size(Predicted_cov_mod,1);
% 
%         E = eye(NN);
% 
%         [eigvec_mod, eigval_mod] = eigs(cov_mat_mod,E,50);
%         [eigval_mod,idx_mod]  = sort(diag(real(eigval_mod)),'descend');
% 
%         eigvec_mod = eigvec_mod(:,idx_mod);
%         eigvec_mod = real(eigvec_mod);
% 
%         Hrep_mod = zeros(sn_Func,NN);
%         
%         for i = 1:Mterms
%             Hrep_mod = Hrep_mod + transpose(xi(i,:)./sqrt(eigval_mod(i)))*eigvec_mod(:,i)';
%         end
% 
%         Hrep_mod = Hrep_mod*cov_mat_mod;
% 
%         mu_mod = Predicted_mean_mod;
%         Sig_mod = transpose(sqrt(diag(Predicted_cov_mod)));
% 
%         y_samp_mod = repmat(mu_mod,sn_Func,1) + repmat(Sig_mod,sn_Func,1).*Hrep_mod;

%         %% Deno_Pf 수정
%         [M_samp,MSE_samp] = predictor(x_samp,Info_Krig_mod{iii}.dmodel); % U-function 고려?
%         Deno_Pf_mod = mean(normcdf(M_samp./sqrt(MSE_samp)));
% 
%         hpdfval_mod = TargetPDF2(x_test, Info_Krig_mod{iii}.dmodel, x_DV, Var, Deno_Pf_mod, 2);
%         w_mod{iii} = 1./hpdfval_mod';
% 
%         Exp_output = Exp_output + (y_samp_mod>0).*w_mod{iii} * C(iii);
    end
    ERe = sum(Re.*C',1)/sqrt(pi);
%     Expected_Pof = sum(Exp_output,2)/sqrt(pi)/sn_MCMC;
%     ERe = 1 - Expected_Pof;
    
    Exp_CI = quantile(ERe,[0.025:0.025:0.975]);
%     Exp_CI = quantile(ERe,[0.005:0.005:0.995]);
end