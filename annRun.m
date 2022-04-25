% -------------------------------------------------------------------------
% part of the Physics-Informed Machine Learning study 
% see manuscript by A.Ghosh et.al for details 
%
% the script trains artificial neural network based on the previously
% generated RCWA data
% 
% (c) 2021, A. Ghosh and V.A. Podolskiy, University of Massachusetts Lowell
% 
% -------------------------------------------------------------------------

clear classes
reset(gpuDevice(1))


fname ='./m=50/ANNtest.mat'; % file name used for string the resulting ANN

rng(3988) % initialize random number generator to enforce repeatability of train/test/val splits

NNtype = 4; % 1=Black Box 2=Meaning Informed; 3=Physics Informed, labeled training only; 4=Physice Informed, add unlabeled data
angleLim = false; %limit training data to \theta=20 degree only 

load("./m=50/dataFull.mat"); % load training data
targetTbl =[targetTbl,geomTbl]; % add configArr to target table
geomSz =size(geomTbl,2); 

% setup the ANN
layers =layerGraph();

tempLayers =[
    sequenceInputLayer(geomSz,"Name","sequence")
    sequenceFoldingLayer("Name","fold")
];
layers =addLayers(layers,tempLayers);
tempLayers =[
    splitLayer("split")
];
layers =addLayers(layers,tempLayers);

tempLayers = [
    fullyConnectedLayer(geomSz,"Name","fc_1")
    eluLayer("Name","relu1")
    fullyConnectedLayer(50,"Name","fc_2")
    lstmLayer(50,"Name","lstm_1",'OutputMode','sequence')
    eluLayer("Name","relu21")
    fullyConnectedLayer(150,"Name","fc_31")
    tanhLayer("Name","relu22")
    fullyConnectedLayer(150,"Name","fc_36")
    eluLayer("Name","relu2")
    fullyConnectedLayer(150,"Name","fc_3")
    fullyConnectedLayer(size(targetTbl,2)-geomSz,"Name","fc_out")
];
layers = addLayers(layers,tempLayers);

tempLayers = [
    mergeNormLayer("merge")
];
layers = addLayers(layers,tempLayers);

switch NNtype 
    case 1
        regLr = regressionLayer("Name","regressionoutput");
    case 2
        regLr = MGregressionLayer("MGregressionoutput",geomSz,5,1);
    case {3,4}
        regLr = PGregressionLayer("PGregressionoutput",lam0,Lam,geomSz, true,...
            5,1,75,100,0.5,1,300,150);
    otherwise 
        disp('error: unknown training regime')
        return
end 

tempLayers = [
    sequenceUnfoldingLayer("Name", "unfold")
    regLr
];

layers = addLayers(layers,tempLayers);
clear tempLayers; 

% connect network
layers = connectLayers(layers,"fold/out","split");
layers = connectLayers(layers,"split/data","fc_1");
layers = connectLayers(layers,"split/config","merge/config");
layers = connectLayers(layers,"fc_out","merge/data");
layers = connectLayers(layers,"merge","unfold/in");
layers = connectLayers(layers,'fold/miniBatchSize','unfold/miniBatchSize');


% Plot the ANN
figure(2)
set(gca,'FontSize',18)
plot(layers);

% Prepare the Data
numDat = size(geomTbl,1);  
[train,val,test] = dividerand(numDat,0.1,0.1,0.8); 

% further limit training data to particular value of parameter theta
if angleLim
    tmp = geomTbl(train,1); 
    val = [val,train(tmp~=20)]; 
    train = train(tmp==20); 
end 

xTrain = geomTbl(train,:).'; xVal = geomTbl(val,:).'; xTest = geomTbl(test,:).'; 
yTrain = targetTbl(train,:).';yVal = targetTbl(val,:).'; yTest = targetTbl(test,:).';

xTrainI = xTrain; 
xTestI = xTest; 
xValI = xVal; 

if NNtype ==4
    % add validation data as unlabeled data to the training set
    xTrainE =[xTrainI, xValI];
    yTmp = yVal.*rand(size(yVal,1), size(yVal,2));  %scramble "unlabeled" data
    yTmp((1),:) = 0; % marks "unlabeled" data 
    yTmp((end-geomSz:end),:) = yVal((end-geomSz:end),:); %restore correct configuration data 
    yTrainE =[yTrain, yTmp]; 
else 
    xTrainE =xTrainI; yTrainE =yTrain; 
end 
    
% Train and validate the network
options = trainingOptions('adam', ...
    'GradientDecayFactor',0.9,...
    'SquaredGradientDecayFactor',0.99,...
    'MaxEpochs',5000, ...
    'InitialLearnRate',10e-4, ...
    'Shuffle','never', ...
    'GradientThreshold',1, ...
    'LearnRateDropFactor',0.9, ...
    'LearnRateDropPeriod',1000,...
    'LearnRateSchedule','piecewise',...
    'Plots','training-progress', ...
    'ExecutionEnvironment',"gpu",...
    'Verbose',0);

tic
net = trainNetwork(xTrainE,yTrainE,layers,options);
toc 

% save the results 
save(fname,'net','xTest','xTrain','xVal','yTest','yTrain','yVal','nmEvs',...
    'layers',...
    'lam0','Lam')
