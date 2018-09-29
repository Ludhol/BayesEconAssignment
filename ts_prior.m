function [aols,vbar,a0,ssig1,a02mo] = ts_prior(rawdat,tau,p,plag)
% From hetero-code we have that: 
% rawdat = Y
% tau = tau size of training sample (40)
% p = M number of independent variables (3)
% plag = p number of lags (2)
% This makes the code slightly confusing

yt = rawdat(plag+1:tau+plag,:)';

%m is the number of elements in the state vector
% Equivalent to K in the hetero-code
m = p + plag*(p^2);
Zt=[];
for i = plag+1:tau+plag % iterate over our data range
    ztemp = eye(p);
    for j = 1:plag % iterate over our lags
        xlag = rawdat(i-j,1:p); % xlag is the lagged independent variables
        xtemp = zeros(p,p*p);
        for jj = 1:p
            xtemp(jj,(jj-1)*p+1:jj*p) = xlag; %place the lags on the diagonalish
        end
        ztemp = [ztemp   xtemp];
    end
    Zt = [Zt ; ztemp];
end
% Z_t is similar to the Z_t in the main code and used to produce aols, see
% below
vbar = zeros(m,m); % X'X
xhy = zeros(m,1); % X'y
for i = 1:tau
    zhat1 = Zt((i-1)*p+1:i*p,:); % Extract the rows from (i-1)*p+1 to i*p, always p-1 rows
    vbar = vbar + zhat1'*zhat1; % Add zhat1'*zhat1 which is mxm (in usual ols notation X'X)
    xhy = xhy + zhat1'*yt(:,i); % Add zhat1'*yt(:,i) which is mx1 (in usual ols notation X'y)
end

vbar = inv(vbar); % usual ols notation (X'X)^-1
aols = vbar*xhy; % usual ols notation (X'X)^-1 * X'y = beta

sse2 = zeros(p,p); 
for i = 1:tau
    zhat1 = Zt((i-1)*p+1:i*p,:);
    sse2 = sse2 + (yt(:,i) - zhat1*aols)*(yt(:,i) - zhat1*aols)';
end
hbar = sse2./tau;

vbar = zeros(m,m);
for i = 1:tau
    zhat1 = Zt((i-1)*p+1:i*p,:);
    vbar = vbar + zhat1'*inv(hbar)*zhat1;
end
vbar = inv(vbar);
achol = chol(hbar)';
ssig = zeros(p,p);
for i = 1:p
    ssig(i,i) = achol(i,i); 
    for j = 1:p
        achol(j,i) = achol(j,i)/ssig(i,i);
    end
end
achol = inv(achol);
numa = p*(p-1)/2;
a0 = zeros(numa,1);
ic = 1;
for i = 2:p
    for j = 1:i-1
        a0(ic,1) = achol(i,j);
        ic = ic + 1;
    end
end
ssig1 = zeros(p,1);
for i = 1:p
    ssig1(i,1) = log(ssig(i,i)^2);
end

hbar1 = inv(tau*hbar);
hdraw = zeros(p,p);
a02mo = zeros(numa,numa);
a0mean = zeros(numa,1);
for irep = 1:4000
    hdraw = wish(hbar1,tau);
    hdraw = inv(hdraw);
    achol = chol(hdraw)';
    ssig = zeros(p,p);
    for i = 1:p
        ssig(i,i) = achol(i,i); 
        for j = 1:p
            achol(j,i) = achol(j,i)/ssig(i,i);
        end
    end
    achol = inv(achol);
    a0draw = zeros(numa,1);
    ic = 1;
    for i = 2:p
        for j = 1:i-1
            a0draw(ic,1) = achol(i,j);
            ic = ic + 1;
        end
    end
    a02mo = a02mo + a0draw*a0draw';
    a0mean = a0mean + a0draw; 
end

a02mo = a02mo./4000;
a0mean = a0mean./4000;
a02mo = a02mo - a0mean*a0mean';