function [AMat] = rcwaFun(lam0,Lam,mMax,configArr)
%RCWAFUN implements 1D RCWA; 
%   Input: operating wavelength, period of the structure, and configuration
%   array containing parameter theta and array of real/imaginary parts of
%   permittivity of each layer

mArr=(-mMax:mMax); 
mArrLong=(-2*mMax:2*mMax); 
[m2,l2]=meshgrid(mArr); 
m2Ml2=m2-l2+2*mMax+1;  


dx=Lam/length(mArrLong); 
xArr=(0:dx:Lam-dx); 

Npix=(length(configArr)-1)/2; 
xPix=(linspace(0,Lam,Npix+1)); 
epsXXarr=12+0*xArr;

ePixArr=configArr(2:end); 
                 
for ipix=1:Npix
    epsXXarr(xArr>=xPix(ipix)&xArr<=xPix(ipix+1))=ePixArr(ipix)+ePixArr(ipix+Npix)*1i; 
end 
epsZZarr=epsXXarr; 

% setting up Fourier transform
ang0=configArr(1); 
omg0=2*pi/lam0; %omega/c
q0=2*pi/Lam; 

FTMat=exp(1i*q0*(mArrLong.')*xArr)/length(mArrLong); 
epsXXf=FTMat*epsXXarr.'; 
epxZZf=FTMat*(1./epsZZarr.'); 

% setting up RCWA
epsXXMat=epsXXf(m2Ml2); 
epsZZTild=epxZZf(m2Ml2); 

kx0=omg0*sind(ang0); 
kxArr=kx0+q0*mArr; 
xiMat=epsZZTild.*(kxArr.'*kxArr); 
AMat=epsXXMat*(omg0^2*eye(length(mArr))-xiMat); 


end

