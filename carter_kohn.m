function [bdraw,log_lik] = carter_kohn(y,Z,Ht,Qt,m,p,t,B0,V0)
% Carter and Kohn (1994), On Gibbs sampling for state space models.
% Kalman filter used to extract estimates of the state vector, in the case
% of drawbeta: B_t
% Parameters from drawbeta:
% Qt = Qdraw in main code
% m = K
% p = M
% V0 called B_0_prvar in drawbeta code, also from training sample
% B0 called B_0_prmean in drawbeta code which comes from the training sample

% When drawing the blocks of A_t in step 2, the names of the ingoing
% variables are a bit less workable:
% [Atblockdraw,log_lik2a] = carter_kohn(yhat(ii,:),Zc(:,1:ii-1),sigma2temp(:,ii),...
% Sblockdraw{ii-1},sizeS(ii-1),1,t,A_0_prmean(((ii-1)+(ii-3)*(ii-2)/2):ind,:)...
% ,A_0_prvar(((ii-1)+(ii-3)*(ii-2)/2):ind,((ii-1)+(ii-3)*(ii-2)/2):ind));

% This is what we have then:
% y = yhat(ii,:)
% Z = Zc(:,1:ii-1)
% Ht = sigma2temp(:,ii)
% Qt = Sblockdraw{ii-1}
% m = sizeS(ii-1)
% p = 1
% t = t
% B0 = A_0_prmean(((ii-1)+(ii-3)*(ii-2)/2):ind,:)
% V0 = A_0_prvar(((ii-1)+(ii-3)*(ii-2)/2):ind,((ii-1)+(ii-3)*(ii-2)/2):ind))
% The idea of it is still the same though, but now we are filtering and drawing states
% of the matrix A_t. Since A_t is a matrix and this function draws vectors we need to draw
% each column seperately. Which is why this function is in a loop from 1 to
% M when drawing A_t. A_t is turned into a lower triangular matrix in
% draw_sigma.


% Initialize values for Kalman Filter
bp = B0; 
Vp = V0; 
bt = zeros(t,m);
Vt = zeros(m^2,t);
log_lik = 0;
% Kalman filter recursion slide 109 topic 3 for overview
% btt is a_t|t in slides (a filtered state)
% Vtt is P_t|t in slides (mean squared error of same state)
% H
for i=1:t
    R = Ht((i-1)*p+1:i*p,:); % Extract rows (i-1)*p+1 to i*p, always p-1 rows
    H = Z((i-1)*p+1:i*p,:); % Same as above but from Z
    cfe = y(:,i) - H*bp;   % conditional forecast error, col i - 
    f = H*Vp*H' + R;    % variance of the conditional forecast error
    inv_f = inv(f);
    % log_lik = log_lik + log(det(f)) + cfe'*inv_f*cfe;
    btt = bp + Vp*H'*inv_f*cfe; 
    Vtt = Vp - Vp*H'*inv_f*H*Vp;
    % Kalman gain is 
    if i < t % Update bp and vp
        bp = btt;
        Vp = Vtt + Qt;
    end
    bt(i,:) = btt';
    Vt(:,i) = reshape(Vtt,m^2,1);
end
% After Kalman Filtering we use the estimation of the state (a_t|t) and its
% associated mean squared error (P_t|t) to draw the state vector.

% draw Sdraw(T|T) ~ N(S(T|T),P(T|T))
bdraw = zeros(t,m);
bdraw(t,:) = mvnrnd(btt,Vtt,1);

% Backward recurssions see slide 110
% bmean is EA_t from slides
% bvar is VA_t from slides
for i=1:t-1
    bf = bdraw(t-i+1,:)';
    btt = bt(t-i,:)';
    Vtt = reshape(Vt(:,t-i),m,m);
    f = Vtt + Qt;
    inv_f = inv(f);
    cfe = bf - btt;
    bmean = btt + Vtt*inv_f*cfe;
    bvar = Vtt - Vtt*inv_f*Vtt;
    bdraw(t-i,:) = mvnrnd(bmean,bvar,1); %bmean' + randn(1,m)*chol(bvar);
end
bdraw = bdraw';