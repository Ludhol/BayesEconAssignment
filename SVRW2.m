% univariate stochastic volatility with random walk transition eq
% Function slightly adapted from Joshua CC Chan's function

% Called by:
% [htdraw(:,j) , statedraw(:,j)] = SVRW2(yss(:,j),htdraw(:,j),... 
%           Wdraw(j,:),h_prmean(j),h_prvar(j,j),1); 
% in draw_sigma

% Output and input variables:
% h = htdraw(:,j)
% S = statedraw(:,j)

% Ystar = yss(:,j)      dependent variable in ME modified for current SE
% h = htdraw(:,j)       previous h_j,t draw
% sig = Wdraw(j,:)      W is varcovar matrix of eta (error term for current SE)
% h_prmean = h_prmean(j) from the OLS in ts_prior
% h_prvar = h_prvar(j,j) from the OLS in ts_prior
% TVP_Sigma = 1

% Here we are drawing states for
% ME: yss_t = h_t + e_t where e_t ~ log(chi^2(1))
% SE: h_t+1 = h_t + eta_t where eta_t ~ N(0, W)
% Because the ME is non-gaussian we cannot use CK directly.
% Therefore we estimate log(chi^2(1)) using a mixture of 7 normal
% distributions.

% Steps based on Kim et al section 3.2
function [h, S] = SVRW2(Ystar,h,sig,h_prmean,h_prvar,TVP_Sigma)

T = length(h);
% normal mixture
% pi is the probability of using mixture i
% mi is the mean of mixture i
% sigi is the variance of mixture i
pi = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
mi = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
sigi = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
sqrtsigi = sqrt(sigi);

% sample S from a 7-point distrete distribution
temprand = rand(T,1);
q = repmat(pi,T,1).*normpdf(repmat(Ystar,1,7),repmat(h,1,7)+repmat(mi,T,1), repmat(sqrtsigi,T,1));
% one element 1<=i<=7 of one row 1<=j<=173 of q:
% q(j,i) = pi(i)*normpdf(Ystar(j),h(i)+mi(i),sqrtsigi(i))
% size(q) = [173 7]
q = q./repmat(sum(q,2),1,7); % Normalise q between 0 and 1
S = 7 - sum(repmat(temprand,1,7)<cumsum(q,2),2)+1;
% cumsum(q,2) sums columns, cumsum([1 2 3],2) = [1 3 6],
% repmat(temprand,1,7)<cumsum(q,2) returns a binary 173x7 matrix
% 1 if inequality holds otherwise 0
% sum(repmat(temprand,1,7)<cumsum(q,2),2) sums the 1s and 0s in a row
% and thus returns a 173x1 vector
% elements of temprand and q are between 0 and 1


% S is a Tx1 vector of values between 1 and 7, indicating which normal
% distribution to use

vart = zeros(1,1,T);
yss1 = zeros(T,1);
for i = 1:T
    imix = S(i);
    vart(1,1,i) = sigi(imix);
    yss1(i,1) = Ystar(i,1) - mi(imix);
end
% We now have (mixing math and code) a linear and gaussian state space model
% ME: yss1 = h + (e - mi) where (e - mi) ~ N(0, vart)
% SE: h_t+1 = h_t + eta_t where eta_t ~ N(0, W)
% This means that we can use a version of carter & kohn to draw h
[h,log_lik3] = carter_kohn2(yss1',ones(T,1),vart,sig,1,1,T,h_prmean,h_prvar,TVP_Sigma*ones(T,1));


