classdef MGregressionLayer < nnet.layer.RegressionLayer
    % Example custom regression layer with mean-absolute-error loss.
    properties (Access = public)
        % geometry parameterization
        configLen; % length of the configuration array [includes angle and permittivity]
        OVRwt; %weight of overlap integral
        EVLwt; %weight representing eigenvalue difference
        
    end 
    
    methods
        function layer = MGregressionLayer(name, configLen,OVRwt,EVLwt)
            % layer = maeRegressionLayer(name) creates a
            % mean-absolute-error regression layer and specifies the layer
            % name.
			
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Meaning-informed loss layer';

            %config parameters
            layer.configLen = configLen;
%             
            layer.OVRwt = OVRwt; layer.EVLwt = EVLwt; 
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the loss between
            % the predictions Y and the training targets T.
            % both T and Y have dimensions of NFeatures x 1 x NTrain
            

            % reshape input matrices
            iniSz = size(Y); 
            S = iniSz(1); 
            O = iniSz(3); 
            Y = reshape(Y,[S,O]); 
            T = reshape(T,[S,O]); %Now Y,T are NFeatures x NTrain
            
            % split input data into eigenvalues/eigenvectors/configs term
            evalY = Y([1,2],:); 
            evalT = T([1,2],:); 
            
            
            evLen = (size(Y,1)-2-layer.configLen)/2; 
            rvecY = Y((1:evLen)+2,:); rvecT = T((1:evLen)+2,:); %real part of eivenvector
            ivecY = Y((1:evLen)+2+evLen,:); ivecT = T((1:evLen)+2+evLen,:); %imaginary part of eivenvector

            % eigenvalue difference
            dif = sum((evalY-evalT).^2); 
            
            % overlap
            overlap = (sum(rvecY.*rvecT+ivecY.*ivecT).^2+...
                sum(ivecY.*rvecT-rvecY.*ivecT).^2);
            overlap = (overlap-1).^2; 

            % total loss
            loss = sum(layer.EVLwt.*dif+layer.OVRwt.*overlap,2)/O; 
            
        end
        
        
    end
end