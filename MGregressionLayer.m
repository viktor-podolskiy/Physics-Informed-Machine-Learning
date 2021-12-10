classdef MGregressionLayer < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    properties (Access = public)
%         % EM properties 
%         lam0; %vacuum wavelength
%         Lam; %period of the structure
%         
        % geometry parameterization
        configLen; % length of the configuration array [includes angle and permittivity]
        OVRwt; %weight of overlap integral
        EVLwt; %weight representing eigenvalue difference
%         GPUrun; % run loss on the GPU
%         
%         % physics-guided loss parameters
%         % all weights are further normalized by the proportion of labeled
%         % vs unlabeled data
%         
%         EVUwt; %weight representing nz->infty "pull", unlabeled data only
%         EVUtm; %number of iterations controlling the cut-off of the nz->infty "pull"
% 
%         PGLwt; %weight of physics-guided "eigenvalue" constraint, labeled data
%         PGUwt; %weight of physics-guided "eigenvalue" constraint, unlabeled data
%         PGtm; %number of iterations controlling the cut-on of the PG loss
%         PGtm0; %initial kick-on of the PG loss
        
    end 
    
    methods
        function layer = MGregressionLayer(name, configLen,OVRwt,EVLwt)
%             ,lam0,Lam,...
%                 configLen, GPUrun,...
%                 ,EVUwt,EVUtm,PGLwt,PGUwt,PGtm,PGtm0)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Meaning-informed loss layer';

%             % physics parameters
%             layer.lam0=lam0; 
%             layer.Lam=Lam; 
%             
            %config parameters
            layer.configLen=configLen;
%             layer.GPUrun=GPUrun; 
%             
%             %PG parameters
            layer.OVRwt=OVRwt; layer.EVLwt=EVLwt; 
%             layer.EVUwt=EVUwt; layer.EVUtm=EVUtm; 
%             layer.PGLwt=PGLwt; layer.PGUwt=PGUwt; layer.PGtm=PGtm; layer.PGtm0=PGtm0; 
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the loss between
            % the predictions Y and the training targets T.
            % both T and Y have dimensions of NFeatures x 1 x NTrain
            

            % reshape input matrices
            iniSz=size(Y); 
            S=iniSz(1); 
            O=iniSz(3); 
            Y=reshape(Y,[S,O]); 
            T=reshape(T,[S,O]); %Now Y,T are NFeatures x NTrain
            
            % split input data into eigenvalues/eigenvectors/configs term
            evalY=Y([1,2],:); 
            evalT=T([1,2],:); 
            
            
            evLen=(size(Y,1)-2-layer.configLen)/2; 
            rvecY=Y((1:evLen)+2,:); rvecT=T((1:evLen)+2,:); %real part of eivenvector
            ivecY=Y((1:evLen)+2+evLen,:); ivecT=T((1:evLen)+2+evLen,:); %imaginary part of eivenvector

%             configT=T((1:layer.configLen)+2+2*evLen,:); %config term, T
            
            % eigenvalue difference
            dif=sum((evalY-evalT).^2); 
            
            % overlap
            overlap=(sum(rvecY.*rvecT+ivecY.*ivecT).^2+...
                sum(ivecY.*rvecT-rvecY.*ivecT).^2);
            overlap=(overlap-1).^2; 

            % total loss
            loss=sum(layer.EVLwt.*dif+layer.OVRwt.*overlap,2)/O; 
            
        end
        
%         function dLdY= backwardLoss(layer, Y, T)
%             iniSz=size(Y); 
%             S=iniSz(1); 
%             O=iniSz(3); 
%             Y=reshape(Y,[S,O]); 
%             T=reshape(T,[S,O]); %Now Y,T are NFeatures x NTrain
%             
%             % split input data into eigenvalues/eigenvectors/configs term
%             evalY=Y([1,2],:); 
%             evalT=T([1,2],:); 
%             evLen=(size(Y,1)-2-layer.configLen)/2; 
%             rvecY=Y((1:evLen)+2,:); rvecT=T((1:evLen)+2,:); %real part of eivenvector
%             ivecY=Y((1:evLen)+2+evLen,:); ivecT=T((1:evLen)+2+evLen,:); %imaginary part of eivenvector
%             
%             configT=T((1:layer.configLen)+2+2*evLen,:); %config term, T
% 
%             % setting up weights for different components of the loss function
% %             Uwt=zeros(1,O); Lwt=ones(1,O); 
%             evT1=T(1,:); %Uwt(evT1==0)=1; Lwt(evT1==0)=0; 
% 
%             % initialize the loss gradient
%             dLdY=0*Y; 
%             
%             % eigenvalue-difference based loss 
%             dLdYdif=2*(evalY-evalT)/O; 
%             dLdY([1,2],evT1~=0)=dLdY([1,2],evT1~=0)+layer.EVLwt*dLdYdif(:,evT1~=0); 
%             
%             % overlap-based loss
%             overlap=(sum(rvecY.*rvecT+ivecY.*ivecT).^2+...
%                 sum(ivecY.*rvecT-rvecY.*ivecT).^2);
%             
%             dLOdYr=2*(overlap-1).*(rvecT.*sum(rvecY.*rvecT+ivecY.*ivecT)-ivecT.*sum(ivecY.*rvecT-rvecY.*ivecT))/O; 
%             dLOdYi=2*(overlap-1).*(ivecT.*sum(rvecY.*rvecT+ivecY.*ivecT)+rvecT.*sum(ivecY.*rvecT-rvecY.*ivecT))/O;
%             dLOdY=[dLOdYr;dLOdYi]; 
%             
%             dLdY(2+(1:2*evLen),evT1~=0)=dLdY(2+(1:2*evLen),evT1~=0)+dLOdY(:,evT1~=0)*layer.OVRwt; 
%             
%             % PG-based loss
%             [AMatArr,callInd]=calcAArr(layer,configT,0,(evLen-1)/2); 
%             
%             % smooth the call number 
%             callNum=ceil(callInd/5)*5; 
% 
%             % pre-calculate the matrices
%             tmp=zeros(1,O,'like',Y); 
%             
%             evC=double(gather([rvecY;ivecY]));
%             evalC=double(gather(evalY)); 
%             if layer.GPUrun
%                 
%                 evC=gpuArray(evC); 
%                 evalC=gpuArray(evalC); 
%             end 
% 
%             evCreshape=reshape(evC,1,2*evLen,O);
%             avC=reshape(sum(AMatArr.*evCreshape,2),2*evLen,O); 
%             avLen=sum(avC.^2); 
%            
%             % calculate lambda*v
%             % split evectors into real/imag parts 
%             evR=evC(1:end/2,:); evI=evC(end/2+1: end,:); 
%             avR=avC(1:end/2,:); avI=avC(end/2+1: end,:); 
%                 
%             % calculate kz based on effective indices and calculate kz2*eV
%             omg0=2*pi/layer.lam0;
%             kz2r=omg0^2*(evalC(1,:).^2-evalC(2,:).^2); kz2i=omg0^2*(2*evalC(1,:).*evalC(2,:)); 
%             lvR=kz2r.*evR-kz2i.*evI; 
%             lvI=kz2r.*evI+kz2i.*evR; 
% 
%             cang=(sum((lvR-avR).^2+(lvI-avI).^2)); % calculating |lam*v-A*v|
%             angF=1/(1+exp(-(callNum-layer.PGtm0)/layer.PGtm)); 
% 
%             kz2rArr=reshape(kz2r,1,1,O).*eye(evLen); kz2iArr=reshape(kz2i,1,1,O).*eye(evLen);
%             KMatArr=[kz2rArr,-kz2iArr;kz2iArr,kz2rArr]-AMatArr; 
%             
%             dCdY=tmp+...
%                 2*reshape(sum(KMatArr.*sum(KMatArr.*evCreshape,2),1),2*evLen,O)./avLen/O...
%                 -2*cang./avLen.^2.*reshape(...
%                 sum(AMatArr.*sum(AMatArr.*evCreshape,2),1),2*evLen,O)/O; % calculating d|lam*v-A*v|/dv
%             dLdY(2+(1:2*evLen),evT1~=0)=dLdY(2+(1:2*evLen),evT1~=0)+dCdY(:,evT1~=0)*layer.PGLwt.*angF; 
%             dLdY(2+(1:2*evLen),evT1==0)=dLdY(2+(1:2*evLen),evT1==0)+dCdY(:,evT1==0)*layer.PGUwt.*angF; 
% 
%             dCdkz2r=2*sum((lvR-avR).*evR+(lvI-avI).*evI)./avLen/O; 
%             dCdkz2i=2*sum(-(lvR-avR).*evI+(lvI-avI).*evR)./avLen/O; 
%             dLdY(1,evT1~=0)=dLdY(1,evT1~=0)+2*omg0^2*(dCdkz2r(evT1~=0).*evalC(1,evT1~=0)+dCdkz2i(evT1~=0).*evalC(2,evT1~=0))*layer.PGLwt.*angF; 
%             dLdY(1,evT1==0)=dLdY(1,evT1==0)+2*omg0^2*(dCdkz2r(evT1==0).*evalC(1,evT1==0)+dCdkz2i(evT1==0).*evalC(2,evT1==0))*layer.PGUwt.*angF; 
%             dLdY(2,evT1~=0)=dLdY(2,evT1~=0)+2*omg0^2*(-dCdkz2r(evT1~=0).*evalC(2,evT1~=0)+dCdkz2i(evT1~=0).*evalC(1,evT1~=0))*layer.PGLwt.*angF; 
%             dLdY(2,evT1==0)=dLdY(2,evT1==0)+2*omg0^2*(-dCdkz2r(evT1==0).*evalC(2,evT1==0)+dCdkz2i(evT1==0).*evalC(1,evT1==0))*layer.PGUwt.*angF; 
% 
%             % eigenvalue "pull" loss
%             devMdY1=-exp(-evalY(1,:))/O; 
%             evWt=exp(-callNum/layer.EVUtm); 
%             dLdY(1,evT1==0)=dLdY(1,evT1==0)+layer.EVUwt.*evWt.*devMdY1(:,evT1==0); 
%             devMdY2=4*evalY(2,:).^3/O; 
%             dLdY(2,evT1==0)=dLdY(2,evT1==0)+layer.EVUwt.*evWt.*devMdY2(:,evT1==0); 
% 
%             % reshape the gradient matrix 
%             dLdY=reshape(dLdY,iniSz);
% %             if layer.GPUrun
% %                 clear evC;
% %                 clear evalC;
% %                 clear AMatArr;
% %                 wait(gpuDevice());
% %             end 
%         end 
        
    end
end