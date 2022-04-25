clear 
clc
if ~exist('m=50', 'dir')
    mkdir('m=50')
end 
run 'generateRCWAdata.m'
run 'annRun.m'
run 'annTest.m'
