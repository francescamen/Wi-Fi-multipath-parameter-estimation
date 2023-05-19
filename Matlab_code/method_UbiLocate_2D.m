
%     Copyright (C) 2023 Alejandro Blanco, Francesca Meneghello
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <https://www.gnu.org/licenses/>.

close all
clc
clear

% add some parameters to load the file
addpath("functions/")
% Bandwidth
BW = 80;
% Number of spatial stream
M = 1;

% index to the active subcarriers
switch BW
   case 20
      grid_toa = [-26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, -12, ...
-11, -10, -9, -8, -7 -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, ...
16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26];

   case 40
       
 case 80
      grid_toa = [-122, -121, -120, -119, -118, -117, -116, -115, -114, -113, -112, -111, -110, ...
-109, -108, -107, -106, -105, -104, -102, -101, -100, -99, -98, -97, -96, -95, ...
-94, -93, -92, -91, -90, -89, -88, -87, -86, -85, -84, -83, -82, -81, -80, -79, ...
-78, -77, -76, -74, -73, -72, -71, -70, -69, -68, -67, -66, -65, -64, -63, -62, ...
-61, -60, -59, -58, -57, -56, -55, -54, -53, -52, -51, -50, -49, -48, -47, -46, ...
-45, -44, -43, -42, -41, -40, -38, -37, -36, -35, -34, -33, -32, -31, -30, -29, ...
-28, -27, -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14, -13, ...
-12, -10, -9, -8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, ...
17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, ...
40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, ...
62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, ...
85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 104, 105, ...
106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, ...
122];
    otherwise
      K = 64;
end

%%
delay_los = 1e-8;
aoa_grid = -90:1:90;
n_paths = 2;  % number of paths

for aoa_obst=aoa_grid
    for prop_delay_diff=2e-10:2e-10:1e-8      
        delay_obst = delay_los + prop_delay_diff;

        name_base = strcat('simulation_artificial_delayobst_', num2str(delay_obst), '_aoaobst_', num2str(aoa_obst));
        save_dir = '../results/ubilocate/';
        save_name = strcat(save_dir, 'opr_sim_', name_base, '.mat');
        if isfile(save_name)
            continue
        end

        % take the CSI data
        exp_dir = '../simulation_files/change_delay_aoa/';
        csi_data = load(strcat(exp_dir, 'CFR_', name_base, '.mat')).CFR;
        propagation_delays = load(strcat(exp_dir, 'delay_', name_base, '.mat')).propagation_delays;
        propagation_aoa = load(strcat(exp_dir, 'aoa_', name_base, '.mat')).propagation_aoa;

        % get the # of snapshots, subcarriers, number of antennas to received and
        % everything
        [snapshots, K, N, M] = size(csi_data);
        
        ToA = {};
        AoA = {};
        power = {};
        optimization_times = {};
        
        for snaptshot = 1:snapshots
            channel_toa = (csi_data(snaptshot,:,:,1));
            tic;
            [AoA_aux_2, power_aux_2, ToA_aux_2] = Decompose_2D(channel_toa, n_paths);
            optimization_times{snaptshot, 1} = toc;
            
            ToA{snaptshot,1} = (((ToA_aux_2)*256)*(1/80e6)); % in seconds
            angle_res = real(asin(AoA_aux_2/pi)*180/pi);
            power{snaptshot,1} = power_aux_2;
            AoA{snaptshot,1} = angle_res;

        end
        
        save(save_name, 'optimization_times');
        name_file = strcat(save_dir, 'paths_amplitude_list_', name_base, '.mat');
        save(name_file, 'power');
        name_file = strcat(save_dir, 'paths_aoa_list_', name_base, '.mat');
        save(name_file, 'AoA');
        name_file = strcat(save_dir, 'paths_toa_list_', name_base, '.mat');
        save(name_file, 'ToA');
    end
end
