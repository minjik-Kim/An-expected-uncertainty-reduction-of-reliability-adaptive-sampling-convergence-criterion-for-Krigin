function [Reliability, CI, std_Re, Mterms] = Re_realization(Info_Krig, sn_Func , Re, x_DV, Var, x_samp, sn_MCMC, seeds, reg)

        %% MCMC samping using current sample set
        x_test = MCMC_sampling(x_DV,Var,Info_Krig,sn_MCMC,1);
        %% Info_Krig; Kriging 정보
        dmodel = Info_Krig.dmodel; X_t = Info_Krig.X_t; Y_t = Info_Krig.Y_t;
        
        theta = dmodel.theta;
        %% normalization
        Y_t = Info_Krig.Y_t;

        norm_x = (X_t - mean(X_t))./std(X_t);
        norm_y = (Y_t - mean(Y_t))./std(Y_t);
        norm_tp = (x_test - mean(X_t))./std(X_t);

        sn_MCMC = size(norm_tp,1);
        
        if reg == "reg0"
            F = regpoly0(norm_x);
        elseif reg == "reg1"
            F = regpoly1(norm_x);
        else
            F = regpoly2(norm_x);
        end
        
        for i = 1 : size(norm_x,1)
            for j = i : size(norm_x,1)
                R2(i,j) = prod(exp(-theta.*(norm_x(i,:)-norm_x(j,:)).^2));
            end
        end
        R2 = (R2+R2') - diag(ones([size(norm_x,1),1]));
        
        estimated_beta = (F'/(R2)*F)\F'/R2*norm_y;

        for nn = 1:length(norm_tp)
            test_point = norm_tp(nn,:);
            [r(:,nn),~] = corrgauss(theta, norm_x - repmat(test_point,size(norm_x,1),1));
            
            if reg == "reg0"
                F_n(:,:,nn) = regpoly0(test_point);
                Predicted_mean(nn) = (regpoly0(test_point)*estimated_beta+r(:,nn)'*(R2\(norm_y-F*estimated_beta))) * std(Y_t) + mean(Y_t);
            elseif reg == "reg1"
                F_n(:,:,nn) = regpoly1(test_point);
                Predicted_mean(nn) = (regpoly1(test_point)*estimated_beta+r(:,nn)'*(R2\(norm_y-F*estimated_beta))) * std(Y_t) + mean(Y_t);
            else
                F_n(:,:,nn) = regpoly2(test_point);
                Predicted_mean(nn) = (regpoly2(test_point)*estimated_beta+r(:,nn)'*(R2\(norm_y-F*estimated_beta))) * std(Y_t) + mean(Y_t);
            end
            
        end
        
        Cor = prod(exp(-theta.*(repmat(norm_tp,size(norm_tp,1),1)-repelem(norm_tp,size(norm_tp,1),1)).^2),2);
        Cor = reshape(Cor,size(norm_tp,1),size(norm_tp,1));
        F_ele = transpose(reshape(F_n,size(F_n,2),size(F_n,3)));

        A = transpose(r)*(R2\r);
        B = transpose(transpose(F_ele)-transpose(F)*(R2\r)) * ((transpose(F)*(R2\F)) \ (transpose(F_ele)-transpose(F)*(R2\r)));
        Predicted_cov = dmodel.sigma2 * (Cor - A + B);
        
        diagm = repmat(diag(Predicted_cov),1,size(Predicted_cov,1));
        cov_mat = Predicted_cov./(sqrt(diagm).*sqrt(diagm'));
        
        % eigenvalues, eigenvectors
        NN = size(Predicted_cov,1);
        E = eye(NN);
        [eigvec, eigval] = eigs(cov_mat,E,50);
        [eigval,idx]  = sort(diag(real(eigval)),'descend');
        eigvec        = eigvec(:,idx);
        eigvec = real(eigvec);

        % 10^(-6) 이하의 고유값은 무시,
        Mterms =  max(find(real(eigval)>1e-6));

        % Standard Gaussian random variables for Surrogate model uncertainty
        rng(seeds)
        xi   = randn(Mterms,sn_Func);

        Hrep = zeros(sn_Func,NN);
        for i = 1:Mterms
            Hrep = Hrep + transpose(xi(i,:)./sqrt(eigval(i)))*eigvec(:,i)';
        end
        Hrep = Hrep*cov_mat;

        Sig = transpose(sqrt(diag(Predicted_cov)));
        mu = Predicted_mean;

        y_samp = repmat(mu,sn_Func,1) + repmat(Sig,sn_Func,1).*Hrep;

        % Weights of IS
        [M_samp,MSE_samp] = predictor(x_samp,Info_Krig.dmodel);
        Deno_Pf = mean(normcdf(M_samp./sqrt(MSE_samp)));

        hpdfval = TargetPDF(x_test,Info_Krig.dmodel,x_DV,Var,Deno_Pf,0);
        w = 1./hpdfval';

        Pof = (sum((y_samp>0).*w,2)/sn_MCMC)';
        Reliability = 1 - Pof;

        CI = quantile(Reliability,[0.025:0.025:0.975]); % confidence limit of reliability 
%         CI = quantile(Reliability,[0.005:0.005:0.995]); % confidence limit of reliability 
        std_Re = sqrt( mean(((sum((y_samp>0).*w.^2,2)/sn_MCMC) - (1-Re)^2)/(sn_MCMC-1)') ); % standard deviation of reliability (by surrogate model uncertainty)
        
end