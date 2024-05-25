function [ToA_estimated, AoA_estimated, max_values, neig, fh] = JADE_Reduced_Spotfi_WiFi_Four(channel, step_toa, step_aoa, sm_f_aoa, K, BW, sm_f_toa, print_spectrum, grid_toa, isFB, n_e)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here

    if nargin < 11
        n_e = 3;
    end
    
    [~,N, M] = size(channel);
    %% interpolate values that are missing in the CSI

    % subcarrier position to interpolate
    xq = [-103,-75,-39,-11,-1,0,1,11,39,75,103];

    % initialize the variable to save the channel +  interpolated values
    channel_interp = zeros(length(grid_toa)+length(xq), N,M);

    % indexes
    index_channel = grid_toa + (K-12)/2 + 1;
    index_channel_interp = xq + (K-12)/2 + 1;

    % loop to iterate over the antennas
    for tx_id = 1:M   
        for rx_id = 1:N
            % spline interpolation
            vq = spline(grid_toa, channel(:,rx_id, tx_id), xq);
            % take the values
            channel_interp(index_channel,rx_id, tx_id) = channel(:,rx_id, tx_id);
            channel_interp(index_channel_interp,rx_id, tx_id) = vq;
        end
    end

    %% Estimate the steering vectors (AoA and ToA)
    [S_toa, S_aoa, index_time, theta] = Grid_AoA_ToA_smoothing(step_toa, step_aoa, sm_f_aoa, K, BW, sm_f_toa);

    [~,len_toa] = size(S_toa);
    [~,len_aoa] = size(S_aoa);

    %% First estimation of ToA
    channel_toa = channel_interp(:,:,1);



    [C_toa] = SS(channel_toa,sm_f_toa,1);


    % Initial ToA estimation
    [ps_db_toa, ~] = MUSIC_opt_TH(C_toa, S_toa, 2, false);
%     [ps_db_toa, ~] = MVDR(C_toa, S_toa);

%     figure, plot(index_time, ps_db_toa);
    % apply a TH
    ps_db_toa(ps_db_toa < min(ps_db_toa) + 1) = min(ps_db_toa);
    [~, LOCKS] = max(ps_db_toa);
    index_toa = zeros(16/step_toa,1);
    %         for iter_peak = 1: length(LOCKS)
    index_toa(:,1) = ((LOCKS-1)- (8/(step_toa))):((LOCKS-2) + (8/(step_toa)));
    index_toa(:,1) = mod(index_toa, len_toa)+1;
    %         end



    %% Joint AoA and ToA estimation

    S_toa_reduced = S_toa(:,index_toa(:));
    S_aoa_toa = kron(S_aoa, S_toa_reduced);
    
    
    [len_R,~] = size(S_aoa_toa);
    
%     R = zeros(len_R);
    
    R = Spotfi_2D_Stream(channel_interp,sm_f_aoa, sm_f_toa, isFB);

    % sp = rmusic_1d(R, 3)

    % apply MUSIC
%     tic
    [ps_db, D] = MUSIC_opt_TH(R, S_aoa_toa, n_e ,0);
%     toc
%     tic
%     [ps_db] = MVDR(R, S_aoa_toa);
%     toc

    % take the lengths of the steering vectors
    [~,len_toa_reduced] = size(S_toa_reduced);


    % reshape the MUSIC spectrum
    ps_db_reshape = reshape(ps_db,len_toa_reduced, len_aoa);
    % normalized it
    ps_db_reshape = ps_db_reshape - max(max(ps_db_reshape));
    % take the min outside the non looked steering vectors
    ps_db_reshape_total = ones(len_toa,len_aoa)*min(ps_db_reshape(:));
    ps_db_reshape_total(index_toa(:),:) = ps_db_reshape;

    change_Data = 0;
    lim_down = index_time(index_toa(1));
    lim_up = index_time(index_toa(end));
    n_circ = 0;
    if(lim_down > lim_up)
        change_Data = 1;
        n_circ = length(index_toa);
        lim_down = (lim_down - index_time(end))+index_time(n_circ) ;
        lim_up = lim_up + index_time(n_circ);
        ps_db_reshape_total = circshift(ps_db_reshape_total,n_circ);

    end
    
    fh = []; 
    
    if (print_spectrum)
        % plot the spectrum

        fh = figure("visible", "on");
        h = pcolor(rad2deg(theta),index_time.', ps_db_reshape_total);
        colormap jet
        set(h, 'EdgeColor', 'none');
        colorbar
        xlabel('AoA [deg]')
        ylabel('ToA [ns]')
        title("Jade Smoothing")
        ylim([lim_down lim_up])

    end

    % take the peaks of the matrix
    BW = imregionalmax(ps_db_reshape_total,8);
    max_values = ps_db_reshape_total(BW);
    % put a th to remove false peaks
    max_values(max_values < max(ps_db_reshape_total(:) + min(ps_db_reshape_total(:))+2)) = [];

    % if length(max_values) > 2
    %     max_values(2:(end-1)) = [];
    % end

    % find the max values
    ToA_estimated = zeros(1,length(max_values));
    AoA_estimated = zeros(1,length(max_values));

    rows = zeros(1,length(max_values));
    cols = zeros(1,length(max_values));

    neig = zeros(3,3,length(max_values));

    
    for ii = 1:length(max_values)
        [rows(ii),cols(ii)] = find(ps_db_reshape_total == max_values(ii));
    end
        
    index_col = cols == 1 | cols == len_aoa;
    index_rows = rows == 1 | rows == len_toa;

    aux = [-1,0,1];

    index_corner = index_col & index_rows;

    if sum(index_corner) > 0
        ToA_estimated(index_corner) = [];
        AoA_estimated(index_corner) = [];
        max_values(index_corner) = [];
        index_col(index_corner) = [];
        index_rows(index_corner) = [];
        cols(index_corner) = [];
        rows(index_corner) = [];
    end

    for ii = 1:length(index_col)
        if(index_col(ii) == 1)
            if(cols(ii) == 1)
                max_value_other_border = max(max(ps_db_reshape_total(rows(ii)+[-1,0,1],len_aoa:-1:(len_aoa-5))));
                if(max_value_other_border > max_values(ii))
                    ToA_estimated(ii) = nan;
                    AoA_estimated(ii) = nan;
                    max_values(ii) = nan;
                end
            else
                max_value_other_border = max(max(ps_db_reshape_total(rows(ii)+[-1,0,1],1:5)));
                if(max_value_other_border > max_values(ii))
                    ToA_estimated(ii) = nan;
                    AoA_estimated(ii) = nan;
                    max_values(ii) = nan;
                end
            end
        end
        if(index_rows(ii) == 1)
            if(rows(ii) == 1)
                max_value_other_border = max(max(ps_db_reshape_total(len_toa:-1:(len_toa-5),cols(ii)+[-1,0,1])));
                if(max_value_other_border > max_values(ii))
                    ToA_estimated(ii) = nan;
                    AoA_estimated(ii) = nan;
                    max_values(ii) = nan;
                end
            else
                max_value_other_border = max(max(ps_db_reshape_total(1:5,cols(ii)+[-1,0,1])));
                if(max_value_other_border > max_values(ii))
                    ToA_estimated(ii) = nan;
                    AoA_estimated(ii) = nan;
                    max_values(ii) = nan;
                end
            end
        end
    end
    
    index_nans = isnan(ToA_estimated);
    if(~isempty(index_nans))
        if(sum(index_nans) > 0)
            ToA_estimated(index_nans) = [];
            AoA_estimated(index_nans) = [];
            max_values(index_nans) = [];
        end
    end

    for ii = 1:length(max_values)
        [rows(ii),cols(ii)] = find(ps_db_reshape_total == max_values(ii));
        ToA_estimated(ii) = index_time(rows(ii));
%         if (change_Data)
%             ToA_estimated(ii) = index_time(rows(ii)+n_circ);
%         end
        AoA_estimated(ii) = rad2deg(theta(cols(ii)));
        
        index_rows = rows(ii)+[-1,0,1];
        if(rows(ii) == 1)
            index_rows(1) = len_toa;
        elseif (rows(ii) == len_toa)
            index_rows(end) = 1;
        end
        
        index_cols = cols(ii)+[-1,0,1];
        if(cols(ii) == 1)
            index_cols(1) = len_aoa;
        elseif (rows(ii) == len_aoa)
            index_cols(end) = 1;
        end
        
%         neig(:,:,ii) = ps_db_reshape_total(index_rows,index_cols);
        

        
    end

    % 
%     fprintf("%%%%%%%%%% Parameters to estimate %%%%%%%%%%")
%     fprintf("\n")
%     ToA_estimated;
%     AoA_estimated;



end

