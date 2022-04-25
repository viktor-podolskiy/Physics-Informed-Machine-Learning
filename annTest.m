% -------------------------------------------------------------------------
% part of the Physics-Informed Machine Learning study 
% see manuscript by A.Ghosh et.al for details 
%
% the script tests artificial neural network based on the previously
% generated RCWA data
% 
% (c) 2021, A. Ghosh and V.A. Podolskiy, University of Massachusetts Lowell
% 
% -------------------------------------------------------------------------

clear

%load data
annName ='./m=50/ANNtest.mat'; % name of the file containing trained neural net
dataName ="./m=50/dataFull.mat"; % filename of the dataset

load(annName,'net','nmEvs'); 
data =load(dataName); 

%setup predictions and ground truth
yExact = [data.targetTbl.';data.geomTbl.']; %ground thruth data
tic 
tTest = predict(net,data.geomTbl.'); % predicted data
% tTest = predict(net,xTest); % predicted data
toc

%separate the configuration array from both predicted data and ground truth
configLen = size(data.geomTbl,2); 
sz = size(tTest); 
tTest = mat2cell(tTest,[sz(1)-configLen, configLen],sz(2)); 
tTest = tTest{1}; 
yExact = mat2cell(yExact,[sz(1)-configLen, configLen],sz(2)); 
yExact = yExact{1}; 

%calculate difference and overlap FOMs
evDiff = zeros(sz(2),1); 
ovr = 0*evDiff; 

for is =1:size(tTest,2)
    [evExact,hvecExact] = nmFold(nmEvs,yExact(:,is)); 
    [evTst,hvecTst] = nmFold(nmEvs,tTest(:,is)); 
    evDiff(is) = abs(evExact-evTst)./abs(evExact); %calculating parameter delta
    ovr(is) = abs(hvecExact'*hvecTst)/sqrt(hvecExact'*hvecExact)/sqrt(hvecTst'*hvecTst); %calculating parameter O
end


%% plot data for testing the performance of the resulting ANN
figure(3)
clf

subplot(1,2,1)
histogram(evDiff, (0:0.025:.5), 'Normalization','probability', 'EdgeColor','none')
xlabel('$\delta$', 'Interpreter', 'latex')
ylabel('$P(\delta)$', 'Interpreter', 'latex')
set(gca,'FontSize',16)
xlim([0 0.5])

subplot(1,2,2)
histogram(ovr, (0:0.05:1)+eps,   'Normalization','probability', 'EdgeColor','none') 
xlabel('$O$', 'Interpreter', 'latex')
ylabel('$P(O)$', 'Interpreter', 'latex')
set(gca,'FontSize',16)
xlim([0 1])

