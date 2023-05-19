
%     Copyright (C) 2023 Alejandro Blanco
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

%% path to functions
addpath("functions/")
mkdir("../simulation_files")
mkdir("../simulation_files/change_delay_aoa")

%% Channel simulator
% antennas
N = 4;

% subcarriers
K = 256;
delete_idxs_len = 22;
delete_idxs = zeros(delete_idxs_len, 1, "int16");
delete_idxs(:) = [0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, ...
    203, 231, 251, 252, 253, 254, 255] + 1;

% number of paths   
L = 2;

% bw
BW = 80;
% signal to noise ratio
SNR = 15;

% delay of the direct path
delay_los = 1e-8;

%% Propagation environment simulation
aoa_grid = -90:1:90;
num_pos = size(aoa_grid, 2);
CFR = zeros(num_pos,K-delete_idxs_len,N);
%% Link Simulation
for aoa_obst=aoa_grid
    for prop_delay_diff=2e-10:2e-10:1e-8      
        delay_obst = delay_los + prop_delay_diff;

        name_save_cfr = strcat('../simulation_files/change_delay_aoa/CFR_simulation_artificial_delayobst_', num2str(delay_obst), '_aoaobst_', num2str(aoa_obst), '.mat');
%         if isfile(name_save_cfr)
%             continue
%         end

        propagation_delays = {};
        propagation_aoa = {};
%         propagation_path_loss = {};

        for fr=1:num_pos
            aoa_los = aoa_grid(fr);

            % AoA vector 
            AoA = [aoa_los, aoa_obst];
            propagation_aoa{fr} = AoA;
            AoA = deg2rad(AoA);
            
            % ToF vector [ns]
            ToF = [delay_los, delay_obst ];
            propagation_delays{fr} = ToF;
            ToF = ToF * 1e9; % move to ns
            
            % alpha
            alpha = randn(1,L) + 1i*rand(1,L);
            % normalize alpha to 1
            alpha = alpha./abs(alpha);
            power = alpha.*conj(alpha);
            % simulate the channel 
            [channel_noiseless] = Channel_Simulator_Power_Freq(BW, N, L, K, AoA, ToF, alpha);
            % [channel_final] = Channel_Simulator_Power_Freq(BW, N, L, K, AoA, ToA, alpha)
%             index_delay_channel = 0:1:(K-1);
%             index_time_channel = index_delay_channel*(1/BW)*1e3;
            
            % Adding noise
            
            power_channel = mean(abs((channel_noiseless(:)).').^2);
            SNR_nat = 10^(SNR/10);
            power_noise = power_channel/SNR_nat;
            noise = sqrt(power_noise/2)*(randn(N,K)+randn(N,K)*1i);
            CFR_sample = channel_noiseless + noise;
            CFR_sample(:, delete_idxs) = [];
            CFR(fr,:,:) = CFR_sample.';
        end

        save(name_save_cfr, 'CFR') %, '-v7.3')
        
        name_save = strcat('../simulation_files/change_delay_aoa/delay_simulation_artificial_delayobst_', num2str(delay_obst), '_aoaobst_', num2str(aoa_obst), '.mat');
        save(name_save, 'propagation_delays') %, '-v7.3')
        
        name_save = strcat('../simulation_files/change_delay_aoa/aoa_simulation_artificial_delayobst_', num2str(delay_obst), '_aoaobst_', num2str(aoa_obst), '.mat');
        save(name_save, 'propagation_aoa') %, '-v7.3')
        
    end
end
