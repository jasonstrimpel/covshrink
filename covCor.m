function [sigma,shrinkage]=covCor(x,shrink)

% function sigma=covcorr(x)
% x (t*n): t iid observations on n random variables
% sigma (n*n): invertible covariance matrix estimator
%
% Shrinks towards constant correlation matrix
% if shrink is specified then this const. is used for shrinkage

% The notation follows Ledoit and Wolf (2004)
% This version: 06/2009

% de-mean returns
[t,n]=size(x);
meanx=mean(x);
x=x-meanx(ones(t,1),:);

% compute sample covariance matrix
sample=(1/t).*(x'*x);

% compute prior
var=diag(sample);
sqrtvar=sqrt(var);
rho=(sum(sum(sample./(sqrtvar(:,ones(n,1)).*sqrtvar(:,ones(n,1))')))-n)/(n*(n-1));
prior=rho*sqrtvar(:,ones(n,1)).*sqrtvar(:,ones(n,1))';
prior(logical(eye(n)))=var;

if (nargin < 2 | shrink == -1) % compute shrinkage parameters
  c=norm(sample-prior,'fro')^2;
  y=x.^2;
  p=1/t*sum(sum(y'*y))-sum(sum(sample.^2));
  rdiag=1/t*(sum(sum(y.^2)))-sum(var.^2);
  v=((x.^3)'*x)/t-(var(:,ones(1,n)).*sample);
  v(logical(eye(n)))=zeros(n,1);
  roff=sum(sum(v.*(sqrtvar(:,ones(n,1))'./sqrtvar(:,ones(n,1)))));
  r=rdiag+rho*roff;
  % compute shrinkage constant
  k=(p-r)/c;
  shrinkage=max(0,min(1,k/t));
else % use specified number
  shrinkage = shrink;
end

% compute the estimator
sigma=shrinkage*prior+(1-shrinkage)*sample;

 
	
