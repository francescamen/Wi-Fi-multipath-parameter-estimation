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

function [ps_db, D] = MUSIC_opt_TH(C, S, n_e, enableGPu)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    % The number of samples
    
    if nargin < 3
        enableGPu = false;
        n_e = 0;
    end


    % Estimate eigen values and vectors by eigen decomposition
    % The diag(S) represents the eigen values and the columns of U the vectors.
    [U,D] = eig(C);
    D = diag(D);
    [D,ind] = sort(D, 'descend');
    U = U(:,ind); 
%     [V,D] = eig(C);

    % estimate the number of signals
%     TH_power = 1e-6;
    if n_e == 0
        D(5:end) = [];
        X = rdivide(D, circshift(D,-1));
        TH_rate = 1.5;
        X_index = X>TH_rate;
        X_index(end) = [];
        n_e = find(X_index,1,'last') + 1;
    end
    % It is needed to estimate the noise space, taking into account that we
    % have D signals. The noise space should be a matrix of Nx(N-D)
    Un = U(:,n_e:end);

    % Now, it is needed to find the angle of arrival. To do so, we compute the
    % steering vector depending on the each angle. The step of the angle will
    % be 0.5 degree. We based on that the distance between antennas y half
    % wavelenght. We compute the steering matrix, it is faster.

%     % step of the angle
%     steps = 1/360;
%     theta = 0:(180*steps):180-(1*steps);
%     % pass to radian
%     theta = deg2rad(theta);
% 
%     % Calculate the steering matrix
%     S = ones(n,length(theta));
% 
%     for i = 1:n
%         S(i,:) = exp(-1i*(i-1)*pi*cos(theta));
%     end

    try
    canUseGPU = parallel.gpu.GPUDevice.isAvailable;
    catch ME
    canUseGPU = false;
    end
    if(canUseGPU && enableGPu)
        S_gpu = gpuArray(S);
        Un_gpu = gpuArray(Un);
        ps = sum(abs((S_gpu')*Un_gpu).^2,2);
        ps = ps.^-1;
        ps = gather(ps);
    else
        % Pseudo-spectrum
%         tic
        ps = sum(abs((S')*Un).^2,2);
%         ps = real(ps);
        ps = ps.^-1;
%         toc
    end

    % tic
    % ps2 = (S')*Un*(Un')*S;
    % ps2 = real(diag(ps2));
    % toc

    % figure, plot(theta,ps)
    ps_db = 10*log10(ps);
%     pd_db_norm = ps_db-max(ps_db);
%     figure, plot(theta,pd_db_norm)
%     toc

%     tic;
%     Sgpu=gpuArray(S);
%     Ungpu=gpuArray(Un);
%     A=(Sgpu')*Ungpu*(Ungpu')*Sgpu;
%     A_diag = real(diag(A));
%     A_diag = A_diag.^-1;
%     toc
%     A_diag = 10*log10(A_diag);
%     toc
%     [~, index] = max(ps);

end

