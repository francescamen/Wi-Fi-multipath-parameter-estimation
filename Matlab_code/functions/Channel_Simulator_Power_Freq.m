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

function [channel_final] = Channel_Simulator_Power_Freq(BW, N, L, K, AoA, ToA, alpha)
    
    % distance among antennas. Lambda/2;
    d = 1/2;

    % AoA
    steering_matrix_AoA = exp(-1i*sin(AoA.')*pi*2*d.*(0:(N-1)));


    % ToA
    ToA_samples = (ToA*1e-9)/(1/(BW*1e6));

    steering_matrix_ToA = exp((-1i*(0:(K-1)).*(ToA_samples.')*2*pi)/K);


    % Channel

    channel = zeros(L,N,K);

    for i = 1:L
        channel_freq = repmat(steering_matrix_ToA(i,:),N,1);
%         channel_time = ifft(channel_freq,K,2);
        channel_freq = channel_freq * alpha(i);
%         figure, plot(abs(channel_time_alpha.'));
        channel(i,:,:) = (channel_freq .* steering_matrix_AoA(i,:).');
        
    end
    channel_final = squeeze(sum(channel,1));
end