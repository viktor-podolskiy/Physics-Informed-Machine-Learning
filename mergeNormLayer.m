classdef mergeNormLayer < nnet.layer.Layer 
% layer that combines the network output (containing eigenvalue/eigenvector
% combinations) and the initial configuration array to be used in the loss
% function
% the layer also normalizes eigenvector 

    properties (Access = public)
        % (Optional) Layer properties.

        % Layer properties go here.
    end

    properties (Learnable)
        % (Optional) Layer learnable parameters.

        % Layer learnable parameters go here.
    end
    
    methods
        function layer = mergeNormLayer(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Layer constructor function goes here.
            layer.Name = name;
            layer.Description='Normalizaion of eigenvectors'; 
            layer.NumInputs=2; 
            layer.InputNames={'data','config'}; 
        end
        
        function [Z1] = predict(~, X1,X2)
            % Forward input data through the layer at prediction time and
            % output the result.
            %
            % Inputs:
            %         layer       - Layer to forward propagate through
            %         X1, ..., Xn - Input data
            % input format: c-by-N*S, where c is the number of features of the sequences, N is the number of observations, and S is the sequence length
            % Outputs:
            %         Z1, ..., Zm - Outputs of layer forward function
            % Layer forward function for prediction goes here.

            S=size(X1,1); 
            evVal=[1,2]; %nVal=2; 
            evVect=(3:S); %nVect=S-2; 
            
            %split the eigenvector off and calculate its norm
            xVect=X1(evVect,:,:); 
            x2=xVect.*xVect; 
            xA=sqrt(sum(x2,1)); %+1e-8; 

            % don't touch the eigenvalue
            tmp=zeros(S,1, 'like',X1);
            tmp(evVect)=1; 
            xNorm=tmp.*xA;
            xNorm(evVal,:)=1; 
            
            % normalize the eigenvector
            X1=X1./xNorm; %normalize the eigenvector

            Z1=[X1;X2]; 
        end
        
        function [dLdX1,dLdX2]=backward(~,X1,~,~,dLdZ1,~)
%         function [dLdX1,dLdX2]=ttt(~,X1,~,~,dLdZ1,~)
            S=size(X1,1); O=size(X1,2); 
            evVal=[1,2]; %nVal=2; 
            evVect=(3:S); %nVect=S-2; 
            
            %split the eigenvector off and calculate its norm
            xVect=X1(evVect,:,:); 
            x2=xVect.*xVect; 
            xA=sqrt(sum(x2,1)); %+1e-8; 

            dLdX1=zeros(S,O,'like',X1); 
            dLdX1(evVal,:)=dLdZ1(evVal,:); 
            
            dLdZvec=reshape(dLdZ1(evVect,:),1,S-2,O); 
            dZdX=reshape(1./xA,1,1,O).*eye(S-2)-...
                reshape(1./xA.^3,1,1,O).*reshape(xVect,1,S-2,O).*reshape(xVect,S-2,1,O); 
            dLdXvec=reshape(sum(dLdZvec.*dZdX,2),S-2,O); 
            dLdX1(evVect,:)=dLdXvec; 
            
            dLdX2=dLdZ1(S+1:end,:)*0; 
        end 
            
    end
end