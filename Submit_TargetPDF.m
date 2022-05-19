function PDF = Submit_TargetPDF(X, dmodel, x_DV, Var, Deno_Pf, MCMC)
    if MCMC == 1 %
        if size(X,1) == 1
            [M,~,MSE] = predictor(X,dmodel);
            PDF = normcdf(M/sqrt(MSE)) * prod(normpdf(X,x_DV,sqrt(Var)),2);
            PDF = PDF/Deno_Pf;
        else
            [M,MSE] = predictor(X,dmodel);
            PDF = normcdf(M/sqrt(MSE)) * prod(normpdf(X,x_DV,sqrt(Var)),2);
            PDF = PDF/Deno_Pf;
        end
    else
        if size(X,1) == 1
            [M,~,MSE] = predictor(X,dmodel);
            PDF = normcdf(M/sqrt(MSE));
            PDF = PDF/Deno_Pf;
        else
            [M,MSE] = predictor(X,dmodel);
            PDF = normcdf(M./sqrt(MSE));
            PDF = PDF/Deno_Pf;
        end
    end
end