% This function calls Bellhop for Large-Scale (L-S) channel 
% simulation and is called if Bellhop is selected as the L-S 
% simulator. 
% Edit the rows specified with !!! to set the desired channel 
% properties. 

function [lmean,taumean,theta,ns,nb,hp,lmin_ind]=runBellhop(zmax, TX_depth, RX_depth, RX_range, freq, c, c_bottom,file_name)
bellhop_dir = getenv('BELLHOP_DIR');
if ~isempty(bellhop_dir) && exist(bellhop_dir, 'dir')
    addpath(genpath(bellhop_dir));
end

local_bellhop_dir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'Bellhop');
if exist(local_bellhop_dir, 'dir')
    addpath(genpath(local_bellhop_dir));
end

if exist('bellhop', 'file') ~= 2
    error(['Bellhop executable/function not found. Add it to the MATLAB path, ', ...
           'set BELLHOP_DIR, or place it under generate_bellhop_water_channel/Bellhop.']);
end

format long;

title=file_name;


%% write the .env file
options1 = 'SVM'; % !!! 'SVW *' for nonflat surface 
options2 = 'A';
options3 = 'A'; 
nTX_depth= 1; 
nRX_depth= 1; 
nRX_range= 1;
range_box = RX_range*1.1; 
depth_box = zmax+1;
nbeams = 0; % !!! number of beams
launching_angles = [-20 20]; % !!! range of launching angles

SSP= [1540, 1539, 1535, 1530, 1529]; %c*ones(1, 5); % !!! SSP values % [1540, 1539, 1535, 1530, 1529]; 
SSP_z= linspace(0, zmax, 5); % !!! SSP depths

cs_bottom = 219.106; % !!!
density_bottom = 1.800; % !!! in g/cm3
alpha_bottom = [0.668 1.3]; % !!! in dB/m 

%% write the env. file 1: 
f = fopen([title '.env'], 'w');
fprintf(f, '''%s''\n', title);
fprintf(f, '%f ! Frequency (Hz)\n', freq);
fprintf(f, '1 ! nmedia\n');
fprintf(f, '''%s'' ! OPTIONS1\n', options1);
fprintf(f, '51 0 %f \n', zmax);

fprintf(f, '0  %f / \n', SSP(1));
fprintf(f, '%f %f / \n', SSP_z(2), SSP(2));
fprintf(f, '%f %f / \n', SSP_z(3), SSP(3));
fprintf(f, '%f %f / \n', SSP_z(4), SSP(4));
fprintf(f, '%f %f / \n', zmax, SSP(5));

fprintf(f, '''%s'' 0.0 ! OPTIONS2\n', options2);
% fprintf(f, '%d 1600 110 1.9 0.8 2.5 / ! bottom line \n', zmax);
fprintf(f, '%f %f %f %f %f %f/\n', zmax, c_bottom, cs_bottom, density_bottom, alpha_bottom(1), alpha_bottom(2));
fprintf(f, '%d ! number of sources\n', nTX_depth);
fprintf(f, '%d / ! sources depth\n', TX_depth);
fprintf(f, '%d ! number of rec in depth\n', nRX_depth);
if(length(RX_depth) == 2)
    fprintf(f, '%f %f / ! receivers depth\n', RX_depth(1), RX_depth(2));
else
    fprintf(f, '%f / ! receivers depth\n', RX_depth);
end

fprintf(f, '%d ! number of rec in range\n', nRX_range);
if(length(RX_range) == 2)
    fprintf(f, '%f %f / ! receivers range\n', RX_range(1)/1000, RX_range(2)/1000);
else
    fprintf(f, '%f / ! receivers range\n', RX_range/1000);
end
fprintf(f, '''%s'' ! OPTIONS3\n', options3);
fprintf(f, '%d ! nbeams(number of launching angles)\n', nbeams);
fprintf(f, '%f %f / ! angles [min max] degrees \n', launching_angles);
fprintf(f, '0.0 %f %f  ! STEP (m), ZBOX (m), RBOX (km)\n', depth_box, range_box/1000);
fclose(f);

%% write the .ati file
if options1(end)=='*'
    f = fopen([title '.ati'], 'w');
    fprintf(f, '''L''\n');
    
    nR=1001; % !!! number of surface points to be assigned a height
    fprintf(f, '%d\n', nR);
    
    r= linspace(0, range_box/1000, nR); % !!! range of surface points
    y= 1+sin(200*2*pi*r); % !!! height of surface points (surface pattern)
    y=y-min(y); 
    
    for k=1:nR
        fprintf(f, '%f ', r(k));
        fprintf(f, '%f \n', y(k));
    end
    fclose(f);
end


%% run bellhop
bellhop(title);
%plotray('example_Bellhop');
arr_info= read_arrivals_asc([title, '.arr']);
%plotarr([title, '.arr'],1,1,1);
P= arr_info.Narr;
ns= arr_info.NumTopBnc(1:P);
nb= arr_info.NumBotBnc(1:P);
[ref_0, ind_0]= min(ns+nb);

tbell= arr_info.delay(1:P); 
taumean= tbell-tbell(ind_0);
hp= arr_info.A(1:P); hp=abs(hp).*sign(real(hp)); hp=hp/max((hp)); 
lmean=tbell(1:P)*c; [lmin, lmin_ind]=min(lmean);
theta= -arr_info.SrcDeclAngle(1:P)*pi/180; 
% plotssp
