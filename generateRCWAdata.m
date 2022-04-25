% -------------------------------------------------------------------------
% part of the Physics-Informed Machine Learning study 
% see manuscript by A.Ghosh et.al for details 
%
% the script generates data for the highest-index propagating mode within
% the composite  
% 
% (c) 2021, A. Ghosh and V.A. Podolskiy, University of Massachusetts Lowell
% 
% -------------------------------------------------------------------------

clear
rngStart = 112277711; 
rng(rngStart)

% initial setup 
lam0 = 1; %operating wavelength
Lam = 5; %period of the stack

angArr = [20 40 60 80]; %array of parameters theta

mMax = 50; %controls the size of A matrix 

Npix = 10; %number of layers; not more than 2*maxM+1
Nlvs = 2; % maximum refractive index; either 2 or 4 in this work

prbPlasm =0; %0 or 0.25; %probability of having plasmonic layers


%% setting up permittivity arrays

numIter = 2000; % number of unique configurations to be generated
fullData = cell(1,numIter*length(angArr)); 
sigArr = zeros(1,numIter*length(angArr),'uint64'); % array of indices to keep track of unique configurations
iiter =0; %counters
iitot =0; 

tic

while iiter < numIter*length(angArr)
    iitot =iitot+1; 

    xPix = linspace(0,Lam,Npix+1); 
    fsig = zeros(1,Npix); 
    fsig(:) = 1; 

    sig = uint64(0); 

    nMaxCur = Nlvs;  

    for ipix = 1:Npix
        lvl = max([nMaxCur^2*rand(),1])+0.1i*rand();
        fsig(ipix) = sqrt(lvl); 
        sig = sig*100+ceil(real(lvl)); 
    end 


    % add plasmonic layers; Note: does not change signatures of the structures
    if rand()<= prbPlasm
        NplasmPix = min([Npix, ceil(rand()*Npix)]); 
        for ip = 1:NplasmPix
            ipix = min([Npix, ceil(rand()*Npix)]);
            lvl = (-1+0.25i)*10^2;
            fsig(ipix) = sqrt(lvl); 
        end 
    end 


    if isempty(find(sigArr==sig,1))
    
        % unique configuration - proceed with iteration
        for ang0 = angArr
            AMat = rcwaFun(lam0,Lam,mMax,[ang0, real(fsig.^2), imag(fsig.^2)]); 

            % eigenvalue analysis
            [hMat,kz2] = eigs(AMat,1,'largestreal'); 
            kz2 = diag(kz2); 
            kz = sqrt(kz2);

            nz=kz*lam0/2/pi; 
            if sum(isnan(nz))== 0 
                iiter = iiter+1; 

                tmp = struct('fsig',fsig, ...
                    'lam0',lam0,...
                    'Lam',Lam,...
                    'ang0',ang0,...
                    'Npix',Npix,...
                    'AMat',AMat,...
                    'kz',kz,...
                    'nz',nz,...
                    'hMat',hMat,...
                    'Nlvs',Nlvs); 
                fullData{iiter} = tmp; 
                sigArr(iiter) = sig; 
                if mod(iiter,200) == 0 
                    disp(['iiter=',num2str(iiter),',nMaxCur=',num2str(nMaxCur)])
                end 
            end 
        end 
    end 
end
toc

% post-process data 
lam0 =fullData{1}.lam0; 
Lam =fullData{1}.Lam; 

nmEvs =1; % each ev has a complex nz and a complex vector hVec associated with it 
lenhVec = size(fullData{1}.hMat,1); 
len0 = length(fullData);
geomTbl = zeros(len0, 2*Npix+1);
targetTbl = zeros(len0, 2*(nmEvs+lenhVec)); 
for il = 1:len0
    geomTbl(il,:) = [fullData{il}.ang0, real(fullData{il}.fsig.^2), imag(fullData{il}.fsig.^2)];
    tmpNZ = fullData{il}.nz; 
    tmpEV = fullData{il}.hMat; 
    [~, perm] = sort(real(tmpNZ), 'descend'); %sort evs
    tmpNZ = tmpNZ(perm(1:nmEvs));
    targetTbl(il,(1: nmEvs)) = real(tmpNZ);
    targetTbl(il,(nmEvs+1: 2*nmEvs)) = imag(tmpNZ);

    tmpEV = tmpEV(:,perm(1:nmEvs)); 
    tmpEV = tmpEV(:);
    targetTbl(il, 2*nmEvs+(1:length(tmpEV))) = real(tmpEV);
    targetTbl(il, 2*nmEvs+length(tmpEV)+(1:length(tmpEV))) = imag(tmpEV);
    
end 
save('m=50/dataFull.mat','nmEvs','Lam','geomTbl','lam0','lenhVec',...
    'nmEvs','targetTbl'); 


