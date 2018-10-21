% TVP-VAR Time varying structural VAR with stochastic volatility
% ------------------------------------------------------------------------------------
% This code implements the TVP-VAR model as in Primiceri (2005). See also
% the monograph, Section 4.2 and Section 3.3.2.
% 
% Some comments in this code has been created by Anne-Marie Hammond, Sante
% Carbone and Ludvig Holm?r for an assignment in course 5326 at SSE. This
% course is held by Daniel Buncic.
%
% ************************************************************************************
% The model is:
%
%     Y(t) = B0(t) + B1(t)xY(t-1) + B2(t)xY(t-2) + e(t) 
% 
%  with e(t) ~ N(0,SIGMA(t)), and  L(t)' x SIGMA(t) x L(t) = D(t)*D(t)',
%             _                                          _
%            |    1         0        0       ...       0  |
%            |  L21(t)      1        0       ...       0  |
%    L(t) =  |  L31(t)     L32(t)    1       ...       0  |
%            |   ...        ...     ...      ...      ... |
%            |_ LN1(t)      ...     ...    LN(N-1)(t)  1 _|
% 
% 
% and D(t) = diag[exp(0.5 x h1(t)), .... ,exp(0.5 x hn(t))].
%
% The state equations are
%
%            B(t) = B(t-1) + u(t),            u(t) ~ N(0,Q)
%            l(t) = l(t-1) + zeta(t),      zeta(t) ~ N(0,S)
%            h(t) = h(t-1) + eta(t),        eta(t) ~ N(0,W)
%
% where B(t) = [B0(t),B1(t),B2(t)]', l(t)=[L21(t),...,LN(N-1)(t)]' and
% h(t) = [h1(t),...,hn(t)]'.
%
% ************************************************************************************
%   NOTE: 
%      There are references to equations of Primiceri, "Time Varying Structural Vector
%      Autoregressions & Monetary Policy",(2005),Review of Economic Studies 72,821-852
%      for your convenience. The definition of vectors/matrices used to be based on this
%      paper but has been amended here to fit the definitions in the
%      monograph.
% ------------------------------------------------------------------------------------

clear all;
clc;
%Set rng mode:
randn('state',sum(100*clock)); %#ok<*RAND>
rand('twister',sum(100*clock)); 
%----------------------------------LOAD DATA----------------------------------------
% Load Korobilis (2008) quarterly data
load ydata.dat;
load yearlab.dat;

% % Demean and standardize data
% t2 = size(ydata,1);
% stdffr = std(ydata(:,3));
% ydata = (ydata- repmat(mean(ydata,1),t2,1))./repmat(std(ydata,1),t2,1);

%Matrix Y from koop section 2.1 equation (3) (page 5)
Y=ydata;

% Number of observations and dimension of X and Y
t=size(Y,1); % t is the time-series observations of Y
M=size(Y,2); % M is the dimensionality of Y

% Number of factors & lags:
tau = 40; % tau is the size of the training sample
% tau and training is discussed in koop section 4.1.1 (page 41)
p = 2; % p is number of lags in the VAR part
numa = M*(M-1)/2; % Number of lower triangular elements of L_t (other than 0's and 1's)
%L_t is from Primiceri section 2 eqn (2), used to create diagonalized
%variance convariance matrices Sigma_t
% ===================================| VAR EQUATION |==============================
% Generate lagged Y matrix. This will be part of the X matrix
ylag = mlag2(Y,p); % Y is [T x M]. ylag is [T x (Mp)]
% Form RHS matrix X_t = [1 y_t-1 y_t-2 ... y_t-k] for t=1:T
ylag = ylag(p+tau+1:t,:);

K = M + p*(M^2); % K is the number of elements in the state vector
% In koop K is defined as = 1 + Mp, i.e. the number of coefficients in each
% equation. However in the code it is defined as this times M, which is the
% number of coefficients in all equations or the number of elements in
% alpha from equation (4) (in koop)
% Create Z matrix, the matrix containing all the lagged variables.
Z = zeros((t-tau-p)*M,K);
for i = 1:t-tau-p
    ztemp = eye(M);
    for j = 1:p        
        xtemp = ylag(i,(j-1)*M+1:j*M);
        xtemp = kron(eye(M),xtemp);
        ztemp = [ztemp xtemp];  %#ok<AGROW>
    end
    Z((i-1)*M+1:i*M,:) = ztemp;
end

% Redefine FAVAR variables y, FAVAR = Factor Augmented VAR see koop section
% 5.3. Seems like they are just chaning y to exclude the part used for OLS
% priors
y = Y(tau+p+1:t,:)';
yearlab = yearlab(tau+p+1:t);
% Time series observations
t=size(y,2);   % t is now 215 - p - tau = 173

%----------------------------PRELIMINARIES---------------------------------
% Set some Gibbs - related preliminaries
nrep = 5000;  % Number of replications
nburn = 2000;   % Number of burn-in-draws
it_print = 100;  %Print in the screen every "it_print"-th iteration

%========= PRIORS:
% To set up training sample prior a-la Primiceri, use the following subroutine
[B_OLS,VB_OLS,L_OLS,h_OLS,VL_OLS]= ts_prior(Y,tau,M,p);

% % Or use uninformative values
% A_OLS = zeros(numa,1);
% B_OLS = zeros(K,1);
% VA_OLS = eye(numa);
% VB_OLS = eye(K);
% sigma_OLS = 0*ones(M,1);

% Set some hyperparameters here (see Primeceri page 831, end of section 4.1)
k_Q = 0.01;
k_S = 0.1;
k_W = 1;

% We need the sizes of some matrices as prior hyperparameters (see Primeceri page
% 831 again, lines 2-3 and line 6)
sizeW = M; % Size of matrix W
sizeS = 1:M; % Size of matrix S

%-------- Now set prior means and variances (_prmean / _prvar)
% These are the Kalman filter initial conditions for the time-varying
% parameters B(t), L(t) and (log) D(t). These are the mean VAR
% coefficients, the lower-triangular VAR covariances and the diagonal
% log-volatilities, respectively 
% See Primeceri page 831

% B_0 ~ N(B_OLS, 4Var(B_OLS))
B_0_prmean = B_OLS;
B_0_prvar = 4*VB_OLS;

% L_0 ~ N(A_OLS, 4Var(L_OLS))
L_0_prmean = L_OLS;
L_0_prvar = 4*VL_OLS;

% log(sigma_0) ~ N(log(sigma_OLS),4I_n)
h_prmean = h_OLS;
h_prvar = 4*eye(M);

% Note that for IW distribution I keep the _prmean/_prvar notation....
% Q is the covariance of B(t), S is the covariance of A(t) and W is the
% covariance of (log) SIGMA(t)

% Q ~ IW(k2_Q*size(subsample)*Var(B_OLS),size(subsample))
Q_prmean = ((k_Q)^2)*tau*VB_OLS;
Q_prvar = tau;

% W ~ IG(k2_W*(1+dimension(W))*I_n,(1+dimension(W)))
W_prmean = ((k_W)^2)*ones(M,1);
W_prvar = 2;

% S ~ IW(k2_S*(1+dimension(S)*Var(A_OLS),(1+dimension(S)))
S_prmean = cell(M-1,1);
S_prvar = zeros(M-1,1);

ind = 1;
for ii = 2:M
    % S is block diagonal as in Primiceri (2005)
    S_prmean{ii-1} = ((k_S)^2)*(1 + sizeS(ii-1))*VL_OLS(((ii-1)+(ii-3)*(ii-2)/2):ind,((ii-1)+(ii-3)*(ii-2)/2):ind);
    S_prvar(ii-1) = 1 + sizeS(ii-1);
    ind = ind + ii;
end


%========= INITIALIZE MATRICES:
% Specify covariance matrices for measurement and state equations
consQ = 0.0001; % Small values to initialse the Gibbs sampler
consS = 0.0001;
consSigma = 0.01;
consW = 0.0001;
Sigmat = kron(ones(t,1),consSigma*eye(M));   % Initialize Htdraw, a draw from the VAR covariance matrix
Sigmatchol = kron(ones(t,1),sqrt(consSigma)*eye(M)); % Cholesky of Htdraw defined above
Qdraw = consQ*eye(K);   % Initialize Qdraw, a draw from the covariance matrix Q
Sdraw = consS*eye(numa);  % Initialize Sdraw, a draw from the covariance matrix S
Sblockdraw = cell(M-1,1); % ...and then get the blocks of this matrix (see Primiceri)
ijc = 1;
for jj=2:M
    Sblockdraw{jj-1} = Sdraw(((jj-1)+(jj-3)*(jj-2)/2):ijc,((jj-1)+(jj-3)*(jj-2)/2):ijc);
    ijc = ijc + jj;
end
Wdraw = consW*ones(M,1);    % Initialize Wdraw, a draw from the covariance matrix W
Btdraw = zeros(K,t);     % Initialize Btdraw, a draw of the mean VAR coefficients, B(t)
Ltdraw = zeros(numa,t);  % Initialize Atdraw, a draw of the non 0 or 1 elements of A(t)
htdraw = zeros(t,M);   % Initialize Sigtdraw, a draw of the log-diagonal of SIGMA(t)
Dt = kron(ones(t,1),0.01*eye(M));   % Matrix of the exponent of Sigtdraws (SIGMA(t))
statedraw = 5*ones(t,M);       % initialize the draw of the indicator variable 
                               % (of 7-component mixture of Normals approximation)
Zs = kron(ones(t,1),eye(M));

% Storage matrices for posteriors and stuff
Bt_postmean = zeros(K,t);    % regression coefficients B(t)
Lt_postmean = zeros(numa,t); % lower triangular matrix A(t)
Dt_postmean = zeros(t,M);  % diagonal std matrix SIGMA(t)
Qmean = zeros(K,K);          % covariance matrix Q of B(t)
Smean = zeros(numa,numa);    % covariance matrix S of A(t)
Wmean = zeros(M,1);          % covariance matrix W of SIGMA(t)

sigmean = zeros(t,M);    % mean of the diagonal of the VAR covariance matrix
cormean = zeros(t,numa); % mean of the off-diagonal elements of the VAR cov matrix
sig2mo = zeros(t,M);     % squares of the diagonal of the VAR covariance matrix
cor2mo = zeros(t,numa);  % squares of the off-diagonal elements of the VAR cov matrix

%========= IMPULSE RESPONSES:
% Note that impulse response and related stuff involves a lot of storage
% and, hence, put istore=0 if you do not want them
istore = 1;
if istore == 1
    nhor = 21;  % Impulse response horizon
    imp75 = zeros(nrep,M,nhor);
    imp81 = zeros(nrep,M,nhor);
    imp96 = zeros(nrep,M,nhor);
    bigj = zeros(M,M*p);
    bigj(1:M,1:M) = eye(M);
end
%----------------------------- END OF PRELIMINARIES ---------------------------

%====================================== START SAMPLING ========================================
%==============================================================================================
tic; % This is just a timer
disp('Number of iterations');

for irep = 1:nrep + nburn    % GIBBS iterations starts here
    % Print iterations
    if mod(irep,it_print) == 0
        disp(irep);toc;
    end
    % -----------------------------------------------------------------------------------------
    %   STEP I: Sample B from p(B|y,L,D,V) (Drawing coefficient states, pp. 844-845)
    % -----------------------------------------------------------------------------------------

    % First we use Carter & Kohn's algorithm to first filter the states of
    % the state vector and then to draw the state vector.
    % Step A.1 in Primiceri
    draw_beta
    
    %-------------------------------------------------------------------------------------------
    %   STEP II: Draw A(t) from p(Lt|y,B,D,V) (Drawing coefficient states, p. 845)
    %-------------------------------------------------------------------------------------------
    
    % Then we draw covariance states
    draw_l
    
    
    %------------------------------------------------------------------------------------------
    %   STEP III: Draw diagonal VAR covariance matrix log-D(t)
    %------------------------------------------------------------------------------------------
    
    % Then we draw volatility states
    draw_h
    

    % Create the VAR covariance matrix SIGMA(t). It holds that:
    %           L(t) x SIGMA(t) x L(t)' = D(t) x D(t) '
    Sigmat = zeros(M*t,M);
    Sigmatsd = zeros(M*t,M);
    for i = 1:t
        inva = inv(capLt((i-1)*M+1:i*M,:));
        stem = diag(Dt(i,:));
        Sigmasd = inva*stem;
        Sigmadraw = Sigmasd*Sigmasd';
        Sigmat((i-1)*M+1:i*M,:) = Sigmadraw;  % H(t)
        Sigmatsd((i-1)*M+1:i*M,:) = Sigmasd;  % Cholesky of H(t)
    end
    
    %----------------------------SAVE AFTER-BURN-IN DRAWS AND IMPULSE RESPONSES -----------------
    if irep > nburn              
        % Save only the means of parameters. Not memory efficient to
        % store all draws (at least for the time-varying parameters vectors,
        % which are large). If you want to store all draws, it is better to
        % save them in a file at each iteration. Use the MATLAB command 'save'
        % (type 'help save' in the command window for more info)
        Bt_postmean = Bt_postmean + Btdraw;   % regression coefficients B(t)
        Lt_postmean = Lt_postmean + Ltdraw;   % lower triangular matrix A(t)
        Dt_postmean = Dt_postmean + htdraw;  % diagonal std matrix SIGMA(t)
        Qmean = Qmean + Qdraw;     % covariance matrix Q of B(t)
        ikc = 1;
        for kk = 2:M
            Sdraw(((kk-1)+(kk-3)*(kk-2)/2):ikc,((kk-1)+(kk-3)*(kk-2)/2):ikc)=Sblockdraw{kk-1};
            ikc = ikc + kk;
        end
        Smean = Smean + Sdraw;    % covariance matrix S of A(t)
        Wmean = Wmean + Wdraw;    % covariance matrix W of SIGMA(t)
        % Get time-varying correlations and variances
        stemp6 = zeros(M,1);
        stemp5 = [];
        stemp7 = [];
        for i = 1:t
            stemp8 = corrvc(Dt((i-1)*M+1:i*M,:));
            stemp7a = [];
            ic = 1;
            for j = 1:M
                if j>1
                    stemp7a = [stemp7a ; stemp8(j,1:ic)']; %#ok<AGROW>
                    ic = ic+1;
                end
                stemp6(j,1) = sqrt(Dt((i-1)*M+j,j));
            end
            stemp5 = [stemp5 ; stemp6']; %#ok<AGROW>
            stemp7 = [stemp7 ; stemp7a']; %#ok<AGROW>
        end
        sigmean = sigmean + stemp5; % diagonal of the VAR covariance matrix
        cormean =cormean + stemp7;  % off-diagonal elements of the VAR cov matrix
        sig2mo = sig2mo + stemp5.^2;
        cor2mo = cor2mo + stemp7.^2;
         
        if istore==1
            
            
            IRA_tvp
            
            
        end %END the impulse response calculation section   
    end % END saving after burn-in results 
end %END main Gibbs loop (for irep = 1:nrep+nburn)
clc;
toc; % Stop timer and print total time
%=============================GIBBS SAMPLER ENDS HERE==================================
Bt_postmean = Bt_postmean./nrep;  % Posterior mean of B(t) (VAR regression coeff.)
Lt_postmean = Lt_postmean./nrep;  % Posterior mean of A(t) (VAR covariances)
Dt_postmean = Dt_postmean./nrep;  % Posterior mean of SIGMA(t) (VAR variances)
Qmean = Qmean./nrep;   % Posterior mean of Q (covariance of B(t))
Smean = Smean./nrep;   % Posterior mean of S (covariance of A(t))
Wmean = Wmean./nrep;   % Posterior mean of W (covariance of SIGMA(t))

sigmean = sigmean./nrep;
cormean = cormean./nrep;
sig2mo = sig2mo./nrep;
cor2mo = cor2mo./nrep;

% Standard deviations of residuals of Inflation, Unemployment and Interest Rates
figure
set(0,'DefaultAxesColorOrder',[0 0 0],...      
    'DefaultAxesLineStyleOrder','-')
subplot(3,1,1)
plot(yearlab,sigmean(:,1))
title('Posterior mean of the standard deviation of residuals in Inflation equation')
subplot(3,1,2)
plot(yearlab,sigmean(:,2))
title('Posterior mean of the standard deviation of residuals in Unemployment equation')
subplot(3,1,3)
plot(yearlab,sigmean(:,3))
title('Posterior mean of the standard deviation of residuals in Interest Rate equation')

if istore == 1 
    qus = [.16, .5, .84];
    imp75XY=squeeze(quantile(imp75,qus));
    imp81XY=squeeze(quantile(imp81,qus));
    imp96XY=squeeze(quantile(imp96,qus));
            
    % Plot impulse responses
    figure       
    set(0,'DefaultAxesColorOrder',[0 0 0],...
        'DefaultAxesLineStyleOrder','--|-|--')
    subplot(3,3,1)
    plot(1:nhor,squeeze(imp75XY(:,1,:)))
    title('Impulse response of inflation, 1975:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)
    subplot(3,3,2)
    plot(1:nhor,squeeze(imp75XY(:,2,:)))
    title('Impulse response of unemployment, 1975:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,3)
    plot(1:nhor,squeeze(imp75XY(:,3,:)))
    title('Impulse response of interest rate, 1975:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,4)
    plot(1:nhor,squeeze(imp81XY(:,1,:)))
    title('Impulse response of inflation, 1981:Q3')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,5)
    plot(1:nhor,squeeze(imp81XY(:,2,:)))
    title('Impulse response of unemployment, 1981:Q3')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,6)
    plot(1:nhor,squeeze(imp81XY(:,3,:)))
    title('Impulse response of interest rate, 1981:Q3')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,7)
    plot(1:nhor,squeeze(imp96XY(:,1,:)))
    title('Impulse response of inflation, 1996:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)    
    subplot(3,3,8)
    plot(1:nhor,squeeze(imp96XY(:,2,:)))
    title('Impulse response of unemployment, 1996:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)
    subplot(3,3,9)
    plot(1:nhor,squeeze(imp96XY(:,3,:)))
    title('Impulse response of interest rate, 1996:Q1')
    xlim([1 nhor])
    set(gca,'XTick',0:3:nhor)
end

disp('             ')
disp('To plot impulse responses, use:')
disp('plot(1:nhor,squeeze(imp75XY(:,VAR,:))), for impulse responses at 1975:Q1')
disp('plot(1:nhor,squeeze(imp81XY(:,VAR,:))), for impulse responses at 1981:Q3')
disp('plot(1:nhor,squeeze(imp96XY(:,VAR,:))), for impulse responses at 1996:Q1')
disp('             ')
disp('where VAR=1 for impulses of inflation, VAR=2 for unemployment and VAR=3 for interest rate')



