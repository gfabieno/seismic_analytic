% MATLABScript for obtaning the analytical solution for 2D wave propagation
% in a viscoelastic medium, based on the Appendix B of paper of Carcione et 
% al. (1988)(1), corrected in (2).
%
% The source used is a Ricker wavelet in the y velocity component.
%
% A figure of the source temporal function and its inverse Fourier
% transform are provided to verify the good working of the method. Also
% works for elastic variables by substituting the values of vp and vs by 
% real constant ones (vp_0 & vs_0).
%
% Attenuation is implemented by GMB-EK rheology presented in (4) and
% thoroughfully explained in (3).
%
% In order to make a proper comparison with numerical data it is
% recommended to either use the same rheology in both the numerical and the
% analytical. It is also possible to get an almost constant Q value by 
% increasing a lot the number of mechanisms used in both the numerical and
% the analytical solutions, thus minimizing the effects of the Q fitting in
% the comparison. 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%  By Josep de la Puente, LMU Geophysics, 2005  %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

home
clear all;
close all;

disp('======================================')
disp('COMPUTATION OF VISCOELASTIC ANALYTICAL')
disp('             SOLUTION                 ')
disp('======================================')
disp('   ((c) Josep de la Puente Alvarez  ')
disp('        LMU Geophysics 2005)')
disp('======================================')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Source parameters (Ricker wavelet):
f0=10;               % cutoff frequency
t0=0.1;              % time shift
nu=0.5;              % weight of exponential (for Carcione's source)
epsilon=1.0;         % weight of cosinus (for Carcione's source)

%Frequency domain:
w_max=1001;          % Maximum frequency (minimum freq = -w_max). Is better to keep an odd number 
w_inc=0.1;           % Increment in frequency

%Position of receiver:
x=0.5;
y=0.5;

%Force of the source:
F=1.0;               % Constant magnitude. Only alters amplitude of the seismograms

%Material parameters:
rho=1.0;             % Density
vp_0=sqrt(3);        % P-wave velocity (elastic)
vs_0=1;              % S_wave velocity (elastic)

%Attenuation Q-Solver parameters
n=3;                 % Number of mechanisms we want to have
QPval=20;            % Desired QP constant value
QSval=10;            % Desired QS constant value
freq=f0;             % Central frequency of the absorption band (in Hertz). Good to center it
                     %  at the source's central frequency
f_ratio=100;         % The ratio between the maximum and minimum frequencies of our bandwidth
                     % (Usually between 10^2 and 10^4)
                        
%Outputting variables
toutmax=1.0;         % Maximum value of t desired as output


disp('--------------------------------------')
disp(' GEOMETRY AND MATERIAL SETTINGS:');
disp('--------------------------------------')
disp(strcat('Density=  ',num2str(rho)));
disp(strcat('VP= ',num2str(vp_0)));
disp(strcat('VS= ',num2str(vs_0)));
disp('Source at coordinates: (0,0)');
disp(strcat('Receiver at coordinates: (',num2str(x),',',num2str(y),')'));
disp(' ');

disp('--------------------------------------')
disp(' ATTENUATION SETTINGS:');
disp('--------------------------------------')
disp(strcat('Number of mechanisms=  ',num2str(n)));
disp(strcat('Q for P-wave=  ',num2str(QPval)));
disp(strcat('Q for S-wave=  ',num2str(QSval)));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% PROBLEM SETUP %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Derived quantities initialization
r=sqrt(x^2+y^2);     % Distance to the receiver
w0=f0/(2*pi);        % Conversion to angular velocity
w=[0:w_inc:w_max];
w=2*w-w_max;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% GMB-EK COMPLEX M SOLUTION %%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Equation system initialization %%
kmax=2*n-1;             %Number of equations to solve (the system is overdetermined)
AP=zeros(kmax,n);       %Initialization of system matrix (P wave)
AS=zeros(kmax,n);       %Initialization of system matrix (S wave)
QP=ones(kmax,1)/QPval;  %Desired values of Q for each mechanism inverted (P wave)
QS=ones(kmax,1)/QSval;  % " " (S wave)
YP=zeros(n,1);
YS=zeros(n,1);

%% Selection of the logarithmically equispaced frequencies
wmean=2*pi*freq;
wmin_disc=wmean/sqrt(f_ratio); 

for j=1:kmax
    w_disc(j)=exp(log(wmin_disc)+(j-1)/(kmax-1)*log(f_ratio));
end

%% Filling of the linear system matrix %%
for m=1:kmax
    for j=1:n
        AP(m,j)=(w_disc(2*j-1).*w_disc(m)+w_disc(2*j-1).^2/QPval)./(w_disc(2*j-1).^2+w_disc(m).^2);
    end
end

for m=1:kmax
    for j=1:n
        AS(m,j)=(w_disc(2*j-1).*w_disc(m)+w_disc(2*j-1).^2/QSval)./(w_disc(2*j-1).^2+w_disc(m).^2);
    end
end

%% Solving of the system %%
YP=AP\QP;
YS=AS\QS;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% VISUALIZATION OF Q IN FREQUENCY %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Getting values for the continuous representation of Q %%
wmin=wmean/sqrt(f_ratio); 
wmax=wmean*sqrt(f_ratio); 
xfrq=[wmin:(wmax-wmin)/(10000-1):wmax];

%% P-wave Q continuous values
numP=0;
denP=1;
for j=1:n
    numP=numP+(YP(j)*w_disc(2*j-1)*xfrq(:))./(w_disc(2*j-1)^2+xfrq(:).^2);
    denP=denP-(YP(j)*w_disc(2*j-1).^2)./(w_disc(2*j-1)^2+xfrq(:).^2);
end
Q_contP=denP./numP;

%% S-wave Q continuous values
numS=0;
denS=1;
for j=1:n
    numS=numS+(YS(j)*w_disc(2*j-1)*xfrq(:))./(w_disc(2*j-1)^2+xfrq(:).^2);
    denS=denS-(YS(j)*w_disc(2*j-1).^2)./(w_disc(2*j-1)^2+xfrq(:).^2);
end
Q_contS=denS./numS;

%% Computing fitting quality (RMS and maximum difference)
maxPdif=0;
maxSdif=0;

for j=1:length(Q_contP)
    tempP=abs(Q_contP(j)-QPval);
    if tempP >= maxPdif
        maxPdif=tempP;
    end
    tempS=abs(Q_contS(j)-QSval);
    if tempS >= maxSdif
        maxSdif=tempS;
    end
end

subplot(1,2,1),
semilogx(xfrq/(2*pi),Q_contP,xfrq/(2*pi),QPval*ones(length(xfrq),1)),
title('Q for the P-wave'),legend('computed','desired')
xlabel('frequency (Hz)'),ylabel('Q'),axis([wmin/(2*pi) wmax/(2*pi) 0 QPval*1.25])
subplot(1,2,2),
semilogx(xfrq/(2*pi),Q_contS,xfrq/(2*pi),QSval*ones(length(xfrq),1)),
title('Q for the S-wave'),legend('computed','desired'),
xlabel('frequency (Hz)'),ylabel('Q'),axis([wmin/(2*pi) wmax/(2*pi) 0 QSval*1.25])

disp(strcat('Attenuation bandwidth= [',num2str(wmin/(2*pi)),',',num2str(wmax/(2*pi)),'] Hz'));
disp('');
disp(strcat('Maximum QP fitting error=  ',num2str(maxPdif/QPval*100),' %'));
disp('');
disp(strcat('Maximum QS fitting error=  ',num2str(maxSdif/QSval*100),' %'));
disp(' ');
disp('--------------------------------------')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%% VELOCITIES COMPUTATION %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Translating into Y_kappa and Y_mu %%
YK=YP*0;
YM=YP*0;
YK(:)=(vp_0^2*YP(:)-4/3*vs_0^2*YS(:))/(vp_0^2-4/3*vs_0^2);
YM(:)=YS(:);

% Complex modulus
mu_0=vs_0^2*rho;
lam_0=vp_0^2*rho-2*mu_0;

MUK=(lam_0+2/3*mu_0);    % Elastic bulk modulus
MUM=2*mu_0;              % Elastic shear modulus

MK=MUK*ones(length(w),1);
MM=MUM*ones(length(w),1);
lambda=zeros(length(w),1);
mu=zeros(length(w),1);
vp=zeros(length(w),1);
vs=zeros(length(w),1);

for j=1:n
    MK(:)=MK(:)-MUK*(YK(j)*(w_disc(2*j-1)./(w_disc(2*j-1)+i*w(:))));
    MM(:)=MM(:)-MUM*(YM(j)*(w_disc(2*j-1)./(w_disc(2*j-1)+i*w(:))));
end

lambda(:)=MK(:)-1/3*MM(:);     
mu(:)=MM(:)/2;                 

% Complex wave velocities (NOTE: Substitute by vp_0 and vs_0 to get the
% elastic solution)
vp(:)=sqrt((lambda(:)+2*mu(:))/rho);
vs(:)=sqrt(mu(:)/rho);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% PROBLEM FUNCTIONS INITIALIZATION %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Green's functions (CORRECTED, as appear in Carcione (2002))
for j=1:length(w)
    if w(j)>0
        G1=-i*pi/2*(1/vp(j)^2*besselh(0,2,(w(j)*r/(vp(j))))+ ...
            1./(w(j)*r*vs(j))*besselh(1,2,(w(j)*r/(vs(j))))- ...
            1./(w(j)*r*vp(j))*besselh(1,2,(w(j)*r/(vp(j)))));
        G2=i*pi/2*(1/vs(j)^2*besselh(0,2,(w(j)*r/(vs(j))))- ...
            1./(w(j)*r*vs(j))*besselh(1,2,(w(j)*r/(vs(j))))+ ...
            1./(w(j)*r*vp(j))*besselh(1,2,(w(j)*r/(vp(j)))));
        u1(j)=F/(2*pi*rho)*(x*y/r^2)*(G1+G2);
        u2(j)=F/(2*pi*rho)*(1/r^2)*(y^2*G1-x^2*G2);
    end
end

%% Geting the symmetric conjugate of the u1 and u2 functions
for j=1:length(w)
        u1(j)=conj(u1(length(w)+1-j));
        u2(j)=conj(u2(length(w)+1-j));
end

%Ricker type source
S=sqrt(pi)*w.^2/(4*(pi*f0)^3).*exp(-i*w*t0).*exp(-w.^2/(4*(pi*f0)^2));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% PROBLEM SOLUTION %%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Construction of the w-space displacements
PHI_DX=u1.*S;
PHI_DY=u2.*S;

%% Construction of the w-space velocities
PHI_VX=PHI_DX.*i.*w;
PHI_VY=PHI_DY.*i.*w;

%% Time axis scaling by the Nyquist frequency
dt=1/(2*w_max)*2*pi;        

%% FFT of the PHI functions to obtain solution
Sol_x=(ifft(fftshift(PHI_DX)))/dt;  
Sol_y=(ifft(fftshift(PHI_DY)))/dt;

%% Same for the velocities 
Vel_x=(ifft(fftshift(PHI_VX)))/dt;  
Vel_y=(ifft(fftshift(PHI_VY)))/dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% VARIABLES' SCALING %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t=(1:length(w))*dt-dt ;  %The substraction is to have a t=0 value

%% Rescaling of the time variable to adjust to the t-space sampling rate
for j=1:length(w)/2
    t_res(j)=t(2*j-1);
end

%% Rescaling of the sampling of our function to compare with a t-domain
%% function
for j=1:length(w)/2
    SX_res(j)=Sol_x(2*j-1);
    SY_res(j)=Sol_y(2*j-1);
    VX_res(j)=Vel_x(2*j-1);
    VY_res(j)=Vel_y(2*j-1);
end

%% MATLAB's "ifft" routines don't give us a purely real function for the
%% inverse transform of a conjugate symmetric function. Therefore we have
%% to correct the amplitudes with the small complex residues obtained.
SX_res=abs(SX_res).*sign(real(SX_res));
SY_res=abs(SY_res).*sign(real(SY_res));;
VX_res=abs(VX_res).*sign(real(VX_res));
VY_res=abs(VY_res).*sign(real(VY_res));;

%% Looking for the t value closest to our desired tmax
found=0;
for j=1:length(t_res)
    if found==0
    if t_res(j)>=toutmax
        countt=j;
        found=1;
    end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%  PLOTTING     %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Displacements displayed in desired time window
minx=(min(real(SX_res(1:countt)))-eps)*1.2;
maxx=(max(real(SX_res(1:countt)))+eps)*1.2;
miny=(min(real(SY_res(1:countt)))-eps)*1.2;
maxy=(max(real(SY_res(1:countt)))+eps)*1.2;
figure,
subplot(121),plot(t_res(1:countt),SX_res(1:countt),'-r'),xlabel('Time (sec)'),
legend('x component'),
axis([t_res(1) t_res(countt) minx maxx]),ylabel('x-displacement')
title(strcat('Receiver at x= ',num2str(x),',y= ',num2str(y)))
subplot(122),plot(t_res(1:countt),SY_res(1:countt),'-r'),xlabel('Time (sec)'),
legend('y component'),
axis([t_res(1) t_res(countt) miny maxy]),ylabel('y-dispplacement')
title(strcat('Source frequency= ',num2str(f0),' Hz; Source delay= ',num2str(t0),'s'))

% Velocities displayed in desired time window
minx=(min(real(VX_res(1:countt)))-eps)*1.2;
maxx=(max(real(VX_res(1:countt)))+eps)*1.2;
miny=(min(real(VY_res(1:countt)))-eps)*1.2;
maxy=(max(real(VY_res(1:countt)))+eps)*1.2;
figure,
subplot(121),plot(t_res(1:countt),VX_res(1:countt),'-k'),xlabel('Time (sec)'),
legend('x component'),
axis([t_res(1) t_res(countt) minx maxx]),ylabel('u velocity')
title(strcat('Receiver at x= ',num2str(x),',y= ',num2str(y)))
subplot(122),plot(t_res(1:countt),VY_res(1:countt),'-k'),xlabel('Time (sec)'),
legend('y component'),
axis([t_res(1) t_res(countt) miny maxy]),ylabel('v velocity'),
title(strcat('Source frequency= ',num2str(f0),' Hz; Source delay= ',num2str(t0),'s'))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% SOURCE COMPARISON %%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% As a check for the scalings chosen, we can compare the ifft of the
%% Ricker source to the analytical expression of the inverse Fourier
%% transform:
B=pi*pi*f0*f0;
 
%% Real time-domain expression of the Ricker wavelet source
SOrig=t*0;
SOrig=(0.5-B*(t-t0).*(t-t0)).*exp(-B*(t-t0).*(t-t0)); 

SS=ifft(fftshift(S))/dt;

for j=1:length(w)/2
    SS_res(j)=SS(2*j-1);
end

SS_res=abs(SS_res).*sign(real(SS_res));
minS=(min(real(SS_res(1:countt)))-eps)*1.2;
maxS=(max(real(SS_res(1:countt)))+eps)*1.2;

figure,
plot(t(1:countt),SOrig(1:countt),'-b',t_res(1:countt),SS_res(1:countt),'.k'),
title('MATLAB Check: ifft of the source')
xlabel('time (s)'),axis([t_res(1) t_res(countt) minS maxS]),
legend('Analytical source','ifft of the Analytical FT of the source')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% END OF THE PROGRAM !!! %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
%%%%%%%%%%%%%%%% 
%% REFERENCES %%
%%%%%%%%%%%%%%%%
% (1) "WAVE PROPAGATION SIMULATION IN A LINEAR VISCOELASTIC MEDIUM", Carcione et al. (1988)
%
% (2) "WAVE FIELDS IN REAL MEDIA: WAVE PROPAGSTION IN ANISOTROPIC, ANELASTIC
% AND POROUS MEDIA", Carcione (2002)
%
% (3) "THE FINITE DIFFERENCE METHOD FOR SEISMOLOGISTS: AN INTRODUCTION",
% Moczo et al. (2005)
%
% (4) "INCORPORATION OF ATTENUATION INTO TIME-DOMAIN COMPUTATIONS OF
% SEISMIC WAVE FIELDS", Emmerich & Korn (1987)
