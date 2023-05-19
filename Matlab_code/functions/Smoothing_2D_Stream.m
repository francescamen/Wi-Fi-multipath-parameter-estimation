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

function [R] = Smoothing_2D_Stream(channel,sm_f_aoa, sm_f_toa, isFB)
    
    % take the size
    [K,N,M]=size(channel);
    % aux variable for the loop
    index = 1;
    % number of the iterations for the loop
    iterations_aoa = N-sm_f_aoa+1;
    iterations_toa = K-sm_f_toa+1;

    % initialization
    channels = zeros(sm_f_toa*sm_f_aoa,iterations_aoa*iterations_toa,M);
    
    % loop to take the subarrays channel
    for K_aoa = 1:iterations_aoa
        for K_toa = 1:iterations_toa
            for tx_id = 1:M
                aux = channel(K_toa:(K_toa+sm_f_toa-1),K_aoa:(K_aoa+sm_f_aoa-1), tx_id);
                channels(:,index,tx_id) = aux(:);
            end
             index = index + 1;
        end
    end
    
    % smoothed channel correlation initialization
    R = zeros(sm_f_toa*sm_f_aoa,sm_f_toa*sm_f_aoa);
    
    % loop to estimate the correlation matrices and sum them
    for iter = 1:(iterations_aoa*iterations_toa)
       aux = channels(:,iter,:);
       aux = squeeze(aux);
       R_aux = (aux*aux');
       if (isFB)
          J = flipud(eye(length(R_aux)));
           R_m = J*conj(R_aux)*J;
           R = R + (R_aux + R_m)/2;
       else
            R = R + R_aux;
       end
       
    end
    % normalized the correlation matrix
    R = R/(sm_f_toa*sm_f_aoa);

end

