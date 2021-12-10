function [evals,evects] = nmFold(nmEvs,vectT)
%NMFOLD splits the single vector into eigenvalue and eigenvector parts, and
%combines real and imaginary values of both 

evLen=length(vectT)/nmEvs/2-1; 
evals=zeros(nmEvs,1); 
evects=zeros(evLen,nmEvs); 

for iev=1:nmEvs
    evals(iev)=vectT(2*(iev-1)+1)+1i*vectT(2*iev);
    evects(:,iev)=vectT(2*nmEvs+2*(iev-1)*evLen+(1:evLen))+...
        1i*vectT(2*nmEvs+2*(iev-1)*evLen+evLen+(1:evLen));
end
end

