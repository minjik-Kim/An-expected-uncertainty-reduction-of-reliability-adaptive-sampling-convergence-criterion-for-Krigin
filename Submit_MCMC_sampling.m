function [x_MCMC] = Submit_MCMC_sampling(x_DV,MCMC_Var, Info_Krig,sn_MCMC,Deno_Pf)
    %% Reweighting Scheme
    % Parameters
    NN=size(x_DV,2);
    burnin  = 10000;      % Number of runs until the chain approaches stationarity (user defined)
    lag     = 3;        % Thinning or lag period: storing only every lag-th point 
    
    % Storage
    theta   = zeros(NN,sn_MCMC);      % Samples drawn from the Markov chain (States)
    acc     = 0;               % Accepted samples
    %% Proposal PDF
    % Bivariate normal parameters
    cov_proposal_PDF = 'user defined proposal distribution'; % 1*diag(MCMC_Var)
    % Proposal PDF
    proposal_PDF = @(X,mu) mvnpdf(X,mu,cov_proposal_PDF);          % Proposal PDF (user defined)
    sample_from_proposal_PDF = @(mu) mvnrnd(mu,cov_proposal_PDF);  % Function that samples from proposal PDF (user defined)
    %% Target PDF
    p = @(x) TargetPDF(x,Info_Krig.dmodel,x_DV,MCMC_Var,Deno_Pf,1); % Deno Pf 마지막거 삭제 필요
    %% MH algorithm
%     aa = eps;   bb = 9;
    tt    = x_DV;      % Start points or initial states of the chain in X and Y 

    for i = 1:burnin   % First make the burn-in stage
       [tt, a] = MH_routine(tt,p,proposal_PDF,sample_from_proposal_PDF); 
    end
    
    for i = 1:sn_MCMC   % Cycle to the number of samples
        for j = 1:lag   % Cycle to make the thinning
            [tt, a] = MH_routine(tt,p,proposal_PDF,sample_from_proposal_PDF);
        end
        theta(:,i) = tt;        % Store the chosen states
        acc        = acc + a;   % Accepted ?
    end
    accrate = acc/sn_MCMC;           % Acceptance rate
    disp(accrate);
    x_MCMC = transpose(theta);
end