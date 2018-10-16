    % First create capAt, the lower-triangular matrix A(t) with ones on the
    % main diagonal. This is because the vector Atdraw has a draw of only the
    % non-zero and non-one elements of A(t) only.
    capLt = zeros(M*t,M);
    for i = 1:t
        capltemp = eye(M);
        lltemp = Ltdraw(:,i);
        ic=1;
        for j = 2:M
            capltemp(j,1:j-1) = lltemp(ic:ic+j-2,1)';
            ic = ic + j - 1;
        end
        capLt((i-1)*M+1:i*M,:) = capltemp;
    end
    
    % yhat is the vector y(t) - Z x B(t) defined previously. Multiply yhat
    % with capAt, i.e the lower triangular matrix A(t). Then take squares
    % of the resulting quantity (saved in matrix y2)
    y2 = [];
    for i = 1:t
        ytemps = capLt((i-1)*M+1:i*M,:)*yhat(:,i);
        y2 = [y2  (ytemps.^2)]; %#ok<AGROW>
    end
    % See equation A.3 and A.4 in Primiceri
    % SVRW2 involves mixtures, see paragraphs below eqn A.4
    % Since y2 may be small the offset constant 1e-6 is used to make the estimation more robust
    yss = log(y2 + 1e-6)';
    for j=1:M
        [htdraw(:,j) , statedraw(:,j)] = SVRW2(yss(:,j),htdraw(:,j),... 
            Wdraw(j,:),h_prmean(j),h_prvar(j,j),1); 
    end
    
    sigt = exp(.5*htdraw);
    
    e2 = htdraw(2:end,:) - htdraw(1:end-1,:);
    W1 = W_prvar + t - p - 1;
    W2 = W_prmean + sum(e2.^2)';
    Winvdraw = gamrnd(W1./2,2./W2);
    Wdraw = 1./Winvdraw;
    