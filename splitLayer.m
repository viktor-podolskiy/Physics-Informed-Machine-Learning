classdef splitLayer < nnet.layer.Layer

    methods
        function layer = splitLayer(name)
            % (Optional) Create a myLayer.
            % This function must have the same name as the class.

            % Layer constructor function goes here.
            layer.Name = name;
            layer.Description='Splitting data in two';
            layer.NumInputs=1; 
            layer.OutputNames={'data','config'}; 

        end

        % splits intput into two, in order to pass config array into
        % PG-layer
        function [Z1, Z2] = predict(layer, X1)
            Z1=zeros(size(X1),'like',X1)+X1; 
            Z2=zeros(size(X1),'like',X1)+X1; 
        end
        
    end
end