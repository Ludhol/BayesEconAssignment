% This is just a visual demonstration of the logchi2 normal approximation. File not included in original code.

mi = [-10.12999 -3.97281 -8.56686 2.77786 .61942 1.79518 -1.08819] - 1.2704;  %% means already adjusted!! %%
sigi = [5.79596 2.61369 5.17950 .16735 .64009 .34023 1.26261];
sqrtsigi = sqrt(sigi);
pi = [0.0073 .10556 .00002 .04395 .34001 .24566 .2575];
x = linspace(0,10)';
v = ones(100,1);
chi2 = chi2pdf(x,v);
y = exp(x);
logchi2 = chi2pdf(y,v).*y;
norms = zeros(100,7);
for i =1:7
    norms(:,i) = pi(i)*normpdf(x,mi(i),sqrtsigi(i));
end
normsum=sum(norms,2);

plot(x,logchi2,'--',x,normsum,':',x,norms);
legend('logchi2','normsum','norm1','norm2','norm3','norm4','norm5','norm6','norm7')
