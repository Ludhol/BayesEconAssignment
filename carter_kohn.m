function [bdraw,log_lik] = carter_kohn(y,Z,Ht,Qt,m,p,t,B0,V0)
% Carter and Kohn (1994), On Gibbs sampling for state space models.
% Kalman filter used to extract estimates of the state vector, in the case
% of drawbeta: B_t

% Parameters from drawbeta:
% y = y
% Z = Z
% Qt = Qdraw
% Ht = Sigmat
% m = K
% p = M
% t = t
% V0 called B_0_prvar in drawbeta code, also from training sample
% B0 called B_0_prmean in drawbeta code which comes from the training sample

% When drawing the blocks of L_t in step 2, the names of the ingoing
% variables are a bit less workable:
% [Ltblockdraw,log_lik2a] = carter_kohn(yhat(ii,:),Zc(:,1:ii-1),h2temp(:,ii),...
%            Sblockdraw{ii-1},sizeS(ii-1),1,t,L_0_prmean(((ii-1)+(ii-3)*(ii-2)/2):ind,:)...
%            ,L_0_prvar(((ii-1)+(ii-3)*(ii-2)/2):ind,((ii-1)+(ii-3)*(ii-2)/2):ind));


% This is what we have then:
% y = yhat(ii,:)
% Z = Zc(:,1:ii-1)
% Ht = h2temp(:,ii)
% Qt = Sblockdraw{ii-1}
% m = sizeS(ii-1)
% p = 1
% t = t
% B0 = L_0_prmean(((ii-1)+(ii-3)*(ii-2)/2):ind,:)
% V0 = L_0_prvar(((ii-1)+(ii-3)*(ii-2)/2):ind,((ii-1)+(ii-3)*(ii-2)/2):ind))

% The idea of it is still the same though, but now we are filtering and drawing states
% of the matrix L_t. Since the measurement equation for L_t is non-linear we need to draw
% each equation seperately. Which is why this function is in a loop from 1 to
% M when drawing L_t. L_t is turned into a lower triangular matrix in
% draw_h.


% Initialize values for Kalman Filter
bp = B0; 
Vp = V0; 
bt = zeros(t,m);
Vt = zeros(m^2,t);
log_lik = 0;

% Kalman filter recursion slide 109 topic 3 for overview
% btt is a_t|t in slides (a filtered state)
% Vtt is P_t|t in slides (mean squared error of same state)
% H is Z_t in slides (measurement equation coeffecient matrix)
% R is H_t in slides (variance matrix of epsilon_t (measurement equation))
% cfe is uhat_t in slides
% f is F_t in slides
% Kalman gain is not expressed explicitly in the code but would be
% Vp*H'*inv_f, see lines 60 and 61
for i=1:t
    R = Ht((i-1)*p+1:i*p,:); % Extract rows (i-1)*p+1 to i*p, always p-1 rows
    H = Z((i-1)*p+1:i*p,:); % Same as above but from Z
    cfe = y(:,i) - H*bp;   % conditional forecast error (uhat_t in slides)
    f = H*Vp*H' + R;    % variance of the conditional forecast error
    inv_f = inv(f);
    % log_lik = log_lik + log(det(f)) + cfe'*inv_f*cfe;
    btt = bp + Vp*H'*inv_f*cfe; % Update bp (from a_t|t-1 to a_t|t)
    Vtt = Vp - Vp*H'*inv_f*H*Vp; % Update Vp (from P_t|t-1 to P_t|t)
    if i < t % move changes into next iteration
        bp = btt;
        Vp = Vtt + Qt;
    end
    % Store results
    bt(i,:) = btt';
    Vt(:,i) = reshape(Vtt,m^2,1);
end
% After Kalman Filtering we use the estimation of the state (a_t|t) and its
% associated mean squared error (P_t|t) to draw the state vector.

% draw Sdraw(T|T) ~ N(S(T|T),P(T|T))
bdraw = zeros(t,m);
bdraw(t,:) = mvnrnd(btt,Vtt,1); % Draw the last state T

% Backward recurssions see slide 110
% bmean is EA_t from slides
% bvar is VA_t from slides, apprently able to pre-draw this
for i=1:t-1
    bf = bdraw(t-i+1,:)'; % bf is the state drawn in previous iteration
    btt = bt(t-i,:)'; % btt is here the current filtered state
    Vtt = reshape(Vt(:,t-i),m,m); % current filtered state variance
    f = Vtt + Qt; % used to compute EA_t
    inv_f = inv(f); % used to compute EA_t and VA_t
    cfe = bf - btt; % used to compute EA_t and VA_t
    bmean = btt + Vtt*inv_f*cfe; % compute EA_t
    bvar = Vtt - Vtt*inv_f*Vtt; % compute VA_t
    bdraw(t-i,:) = mvnrnd(bmean,bvar,1); %bmean' + randn(1,m)*chol(bvar);
    % line above draws current iterations state
end
bdraw = bdraw'; % ' here (and on line 80) since mvnrnd function returns row vector