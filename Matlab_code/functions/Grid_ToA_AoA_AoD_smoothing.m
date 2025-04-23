function [S_toa, S_aoa, S_aod, index_time, theta_aoa, theta_aod] = Grid_ToA_AoA_AoD_smoothing(step_toa, step_aoa, step_aod, K, N, M, BW, N_toa)
%UNTITLED9 Summary of this function goes here
%   Detailed explanation goes here

    %%%%%% ToA %%%%%%
    % estimate the patterns (S) to look in MUSIC
%     step_toa = 0.02;
    index_carrier = 0:(N_toa-1);
    index_delay = 0:step_toa:((K)-step_toa);
    index_time = index_delay*(1/BW)*1e3;

    grid_carrier = repmat(index_carrier, length(index_delay),1);
    grid = (grid_carrier).' .* index_delay;

    S_toa = exp((-1i*grid*2*pi)/K);


    %%%%%% AoA %%%%%%
    % step of the angle
%     step_angle = 1/720;
    theta_aoa = -90:(180*step_aoa):(90-step_aoa);
    % pass to radian
    theta_aoa = deg2rad(theta_aoa);

    % Calculate the steering matrix
    S_aoa = ones(N,length(theta_aoa));

    for i = 1:N
        S_aoa(i,:) = exp(-1i*(i-1)*pi*sin(theta_aoa));
    end
    
    %%%%%% AoD %%%%%%
    % step of the angle
%     step_angle = 1/720;
    theta_aod = -90:(180*step_aod):(90-step_aod);
    % pass to radian
    theta_aod = deg2rad(theta_aod);

    % Calculate the steering matrix
    S_aod = ones(M,length(theta_aod));

    for i = 1:M
        S_aod(i,:) = exp(-1i*(i-1)*pi*sin(theta_aod));
    end
end

