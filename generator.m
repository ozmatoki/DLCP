% This function makes a randomized worm-like pattern with two (or one reference holes for holography

N=2^11; 
x=1:N; y=1:N; 
[X,Y]=meshgrid(x,y); 
list=[]; % list will have all the parame

pxs=0.02*3.2/4; % um per pixel 
dist1=3.2/pxs; % so me measure for the spacing of the ref holes
dist2=2.5/pxs;
temp=zeros(N,N);

Nr=3; %number of aux holes per line
Extra_lines=0; % extra lines on one dimension
L=Nr*9;
load_const_aux_list=0; % this will load

% parameter for central aperture
x0=N/2; y0=N/2-L/2; 
r=1.2./pxs;  % [px][um/(um/px)]
cond_round=(((X-x0).^2+(Y-y0).^2)./r.^2)<1;
cond_nonsymmetric = ((Y-y0)>(-3*r/4)); % makes a flat bottom
cond1 = cond_round  .*  cond_nonsymmetric;
list=[list;[x0,y0,2*r]];

% parameter for ref. hole 1
x0=N/2+dist2; y0=N/2 + dist1-40; r=0.270/2/pxs; % Diameter/2/um_per_px
cond2=(((X-x0).^2+(Y-y0).^2)./r.^2)<1;
temp(cond2)=1; 
list=[list;[x0,y0,2*r]];

% % parameter for ref. hole 2
% x0=N/2-dist2; y0=N/2 + dist1-40; r=0.270/2/pxs; 
% cond3=(((X-x0).^2+(Y-y0).^2)./r.^2)<1;
% temp(cond3)=1; 
% list=[list;[x0,y0,2*r]];
cond3=zeros(N);

%setting the support for the reconstruction
support=cond1+cond2+cond3;

data = [];
for i=1:100
    % Making a random worm-like domain pattern in the central aperture
    a=rand(N); 
    b=imgaussfilt(a,2); 
    b=(b-min(b(:)))/(max(b(:)-min(b(:))));
    b=imgaussfilt(a,5);
    b=round(b);
    c=0.1*exp((1+1i)*b);
    FT=ifftshift(fft2(fftshift(c.*cond1+cond2+cond3)));
    IFT=abs(FT).^2;

    input=log10(IFT);
    %output=(2*b-1).*cond1+cond2+cond3;
    tag=((b+1).*cond1+cond2+cond3);

    Hrec=fftshift(ifft2(ifftshift(IFT)));
    input2a = log10(abs(Hrec));
    input2b = angle(Hrec);

    save(['data2/data' int2str(i-1) '.mat'],'input', 'input2a', 'input2b', 'tag');
end



%figure(1); 
%imagesc(output); zoom(8);

%figure(2); 
%imagesc(input); caxis([10, 1000]);
