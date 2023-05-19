
"""
    Copyright (C) 2023 Francesca Meneghello, Antonio Cusano
    contact: meneghello@dei.unipd.it
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from sklearn.linear_model import OrthogonalMatchingPursuit
from utilityfunct_optimization import *
from utilityfunct_aoa_toa_doppler import build_toa_matrix, build_aoa_matrix, build_toa_aoa_matrix
import matplotlib.pyplot as plt
import time
from matplotlib import rcParams
rcParams['font.size'] = 12


def convert_optimization_options(method, options):

    "Takes the list of string and return it as a tuple"

    if method == 'omp':

        if options[0] == 'None':
            opt_0 = None  # n_nonzero_coefs
        else:
            opt_0 = int(options[0])

        if options[1] == 'None':
            opt_1 = None  # tol
        else:
            opt_1 = float(options[1])
            opt_0 = None

        args = (opt_0, opt_1)

    else:
        args = ()

    return args


def optimize_aoa_toa_lasso1(signal_considered, frequency_vector, t_min, t_max, delta_t, delta_t_refined,
                            range_refined_down, range_refined_up, num_angles, n_tot, path_loss_sim=None,
                            delays_sim=None, aoa_sim=None):

    num_time_steps = signal_considered.shape[0]

    paths_amplitude_list = []
    paths_toa_list = []
    paths_aoa_list = []

    r_optim = []
    r_end_points = np.zeros((num_time_steps, 4))  # t_min, t_max, dt, n_angles of the saved r_optim
    optimization_times = np.zeros(num_time_steps)

    # Dictionary initialization
    T_matrix, time_vector = build_toa_matrix(frequency_vector, delta_t, t_min, t_max)
    T_matrix = np.conj(T_matrix.T)

    # Optimizer initialization
    subcarriers_space = 1
    start_subcarrier = 0
    end_subcarrier = frequency_vector.shape[0]
    select_subcarriers_single = np.arange(start_subcarrier, end_subcarrier, subcarriers_space)
    select_sub_number = select_subcarriers_single.shape[0]
    select_subcarriers = np.zeros(select_sub_number * n_tot, dtype=int)
    for ant_i in range(n_tot):
        select_subcarriers[ant_i*select_sub_number:(ant_i+1)*select_sub_number] = ant_i*select_sub_number + \
                                                                                  select_subcarriers_single

    row_T = int(T_matrix.shape[0] / subcarriers_space)
    col_T = T_matrix.shape[1]
    m = 2 * row_T
    n = 2 * col_T
    In = scipy.sparse.eye(n)
    Im = scipy.sparse.eye(m)
    On = scipy.sparse.csc_matrix((n, n))
    Onm = scipy.sparse.csc_matrix((n, m))
    P = scipy.sparse.block_diag([On, Im, On], format='csc')
    q = np.zeros(2 * n + m)
    A2 = scipy.sparse.hstack([In, Onm, -In])
    A3 = scipy.sparse.hstack([In, Onm, In])
    ones_n_matr = np.ones(n)
    zeros_n_matr = np.zeros(n)
    zeros_nm_matr = np.zeros(n + m)

    # Time steps optimization
    for time_step in range(0, num_time_steps):
        time_start = time.time()
        signal_time = signal_considered[time_step, :, :]

        complex_opt_r = lasso_regression_osqp_fast(signal_time[:, 0], T_matrix, select_subcarriers_single,
                                                   row_T, col_T, Im, Onm, P, q, A2, A3, ones_n_matr,
                                                   zeros_n_matr, zeros_nm_matr, delta_t_refined, num_angles,
                                                   threshold=False)

        position_max_r = np.argmax(abs(complex_opt_r))
        time_max_r = time_vector[position_max_r]

        t_min_step = max(time_max_r - range_refined_down, t_min)
        t_max_step = min(time_max_r + range_refined_up, t_max)

        T_matrix_refined, time_vector_refined, angle_vector_refined = build_toa_aoa_matrix(
            frequency_vector, delta_t_refined, t_min_step, t_max_step, num_angles, n_tot)
        T_matrix_refined = np.conj(T_matrix_refined)

        # Auxiliary data for second step
        row_T_refined = int(T_matrix_refined.shape[0] / subcarriers_space)
        col_T_refined = T_matrix_refined.shape[1]
        m_refined = 2 * row_T_refined
        n_refined = 2 * col_T_refined
        In_refined = scipy.sparse.eye(n_refined)
        Im_refined = scipy.sparse.eye(m_refined)
        On_refined = scipy.sparse.csc_matrix((n_refined, n_refined))
        Onm_refined = scipy.sparse.csc_matrix((n_refined, m_refined))
        P_refined = scipy.sparse.block_diag([On_refined, Im_refined, On_refined], format='csc')
        q_refined = np.zeros(2 * n_refined + m_refined)
        A2_refined = scipy.sparse.hstack([In_refined, Onm_refined, -In_refined])
        A3_refined = scipy.sparse.hstack([In_refined, Onm_refined, In_refined])
        ones_n_matr_refined = np.ones(n_refined)
        zeros_n_matr_refined = np.zeros(n_refined)
        zeros_nm_matr_refined = np.zeros(n_refined + m_refined)

        signal_time = signal_time.flatten(order='F')  # first antenna, second antenna ...
        complex_opt_r_refined = lasso_regression_osqp_fast(signal_time, T_matrix_refined, select_subcarriers,
                                                           row_T_refined, col_T_refined, Im_refined,
                                                           Onm_refined, P_refined, q_refined, A2_refined,
                                                           A3_refined, ones_n_matr_refined,
                                                           zeros_n_matr_refined, zeros_nm_matr_refined,
                                                           delta_t_refined, num_angles)

        # fills opt_r columnwise ((t_min->t_max, ang_min)=1st column, etc)
        complex_opt_r_refined_reshape = np.reshape(complex_opt_r_refined, (-1, num_angles), order='F')
        r_optim.append(complex_opt_r_refined_reshape)

        r_optim_t_reshape = np.reshape(complex_opt_r_refined_reshape, (-1,),
                                       order='F')  # fills opt_r columnwise ((t_min->t_max, ang_min)=1st column, etc)
        sort_idxs = np.flip(np.argsort(abs(r_optim_t_reshape)))
        sort_amplitude = abs(r_optim_t_reshape[sort_idxs])

        amplitude_threshold = sort_amplitude[0] * 1E-3
        paths_refined_amplitude = sort_amplitude[sort_amplitude > amplitude_threshold]
        sort_idx_end = np.sum(sort_amplitude > amplitude_threshold)
        sort_idxs = sort_idxs[:sort_idx_end]
        idx_toa = sort_idxs % time_vector_refined.shape[0]
        idx_aoa = sort_idxs // time_vector_refined.shape[0]
        paths_refined_toa = time_vector_refined[idx_toa]
        paths_refined_aoa = angle_vector_refined[idx_aoa] * 180 / mt.pi

        paths_refined_amplitude_array = np.asarray(paths_refined_amplitude[:sort_idx_end])
        paths_refined_aoa_array = np.asarray(paths_refined_aoa)
        paths_refined_toa_array = np.asarray(paths_refined_toa)

        paths_amplitude_list.append(paths_refined_amplitude_array)
        paths_aoa_list.append(paths_refined_aoa_array)
        paths_toa_list.append(paths_refined_toa_array)

        r_end_points[time_step, :] = np.array([t_min_step, t_max_step, delta_t_refined, num_angles])

        time_end = time.time()
        optimization_times[time_step] = time_end-time_start

    r_optim = np.asarray(r_optim)

    # plot_combined(complex_opt_r_refined_reshape, time_vector_refined, angle_vector_refined,
    #              path_loss_sim, delays_sim, aoa_sim, time_step)

    return r_optim, None, r_end_points, optimization_times, paths_amplitude_list, paths_toa_list, paths_aoa_list


def optimize_aoa_toa_omp(signal_considered, frequency_vector, t_min, t_max, delta_t, delta_t_refined,
                         range_refined_down, range_refined_up, num_angles, n_tot):

    num_time_steps = signal_considered.shape[0]

    paths_amplitude_list = []
    paths_toa_list = []
    paths_aoa_list = []

    r_optim = []
    r_end_points = np.zeros((num_time_steps, 4))  # t_min, t_max, dt, n_angles of the saved r_optim
    optimization_times = np.zeros(num_time_steps)

    # Dictionary initialization
    T_matrix, time_vector = build_toa_matrix(frequency_vector, delta_t, t_min, t_max)
    T_matrix = np.conj(T_matrix.T)

    # Optimizer initialization
    omp_mod = OrthogonalMatchingPursuit()

    # Time steps optimization
    for time_step in range(num_time_steps):

        time_start = time.time()

        signal_time = signal_considered[time_step, :, :]

        T_matrix_real, signal_time_real = convert_TH_real(T_matrix, signal_time[:, 0])
        omp_mod.fit(T_matrix_real, signal_time_real)
        real_opt_r_coarse = omp_mod.coef_

        complex_opt_r = convert_to_complex_r(real_opt_r_coarse)

        position_max_r = np.argmax(abs(complex_opt_r))
        time_max_r = time_vector[position_max_r]

        t_min_step = max(time_max_r - range_refined_down, t_min)
        t_max_step = min(time_max_r + range_refined_up, t_max)

        T_matrix_refined, time_vector_refined, angle_vector_refined = build_toa_aoa_matrix(
            frequency_vector, delta_t_refined, t_min_step, t_max_step, num_angles, n_tot)
        T_matrix_refined = np.conj(T_matrix_refined)

        signal_time = signal_time.flatten(order='F')  # first antenna, second antenna ...

        T_matrix_refined_real, signal_time_real = convert_TH_real(T_matrix_refined, signal_time)
        omp_mod.fit(T_matrix_refined_real, signal_time_real)
        real_opt_r_refined = omp_mod.coef_

        complex_opt_r_refined = convert_to_complex_r(real_opt_r_refined)

        # fills opt_r columnwise ((t_min->t_max, ang_min)=1st column, etc)
        complex_opt_r_refined_reshape = np.reshape(complex_opt_r_refined, (-1, num_angles), order='F')
        r_optim.append(complex_opt_r_refined_reshape)

        r_optim_t_reshape = np.reshape(complex_opt_r_refined_reshape, (-1,),
                                       order='F')  # fills opt_r columnwise ((t_min->t_max, ang_min)=1st column, etc)
        sort_idxs = np.flip(np.argsort(abs(r_optim_t_reshape)))
        sort_amplitude = abs(r_optim_t_reshape[sort_idxs])

        amplitude_threshold = sort_amplitude[0] * 1E-3
        paths_refined_amplitude = sort_amplitude[sort_amplitude > amplitude_threshold]
        sort_idx_end = np.sum(sort_amplitude > amplitude_threshold)
        sort_idxs = sort_idxs[:sort_idx_end]
        idx_toa = sort_idxs % time_vector_refined.shape[0]
        idx_aoa = sort_idxs // time_vector_refined.shape[0]
        paths_refined_toa = time_vector_refined[idx_toa]
        paths_refined_aoa = angle_vector_refined[idx_aoa] * 180 / mt.pi

        paths_refined_amplitude_array = np.asarray(paths_refined_amplitude[:sort_idx_end])
        paths_refined_aoa_array = np.asarray(paths_refined_aoa)
        paths_refined_toa_array = np.asarray(paths_refined_toa)

        paths_amplitude_list.append(paths_refined_amplitude_array)
        paths_aoa_list.append(paths_refined_aoa_array)
        paths_toa_list.append(paths_refined_toa_array)

        r_end_points[time_step, :] = np.array([t_min_step, t_max_step, delta_t_refined, num_angles])

        time_end = time.time()
        optimization_times[time_step] = time_end-time_start

    r_optim = np.asarray(r_optim)

    return r_optim, None, r_end_points, optimization_times, paths_amplitude_list, paths_toa_list, paths_aoa_list


def optimize_aoa_toa_iht(signal_considered, frequency_vector, t_min, t_max, delta_t, delta_t_refined,
                         range_refined_down, range_refined_up, num_angles, n_tot, s=2,  refinement=True,
                         path_loss_sim=None, delays_sim=None, aoa_sim=None, use_prior=False):

    num_time_steps = signal_considered.shape[0]

    paths_amplitude_list = []
    paths_toa_list = []
    paths_aoa_list = []

    r_optim = []
    r_optim_refined = []
    r_end_points = [- range_refined_down, range_refined_up, delta_t_refined, num_angles]
    optimization_times = np.zeros(num_time_steps)

    # Dictionary initialization
    T_matrix_times, time_vector = build_toa_matrix(frequency_vector, delta_t, t_min, t_max)
    T_matrix_angles, angle_vector, _ = build_aoa_matrix(num_angles, n_tot, frequency_vector)
    T_matrix_angles = np.conj(T_matrix_angles.T)

    # Time steps optimization
    for time_step in range(0, num_time_steps):

        time_start = time.time()

        signal_time = signal_considered[time_step, :, :]

        # coarse estimation
        matrix_cfr_toa = np.dot(T_matrix_times, signal_time)
        power_matrix_cfr_toa = np.sum(np.abs(matrix_cfr_toa), 1)
        time_idx_max = np.argmax(power_matrix_cfr_toa)

        time_max_r = time_vector[time_idx_max]

        exp_time = np.exp(1j * 2 * mt.pi * frequency_vector * time_max_r)
        exp_time = np.expand_dims(exp_time, axis=1)
        signal_time = np.multiply(signal_time, exp_time)

        range_ref_down = -range_refined_down if time_max_r-range_refined_down > 0 else 0
        T_matrix_refined, time_vector_refined, angle_vector_refined = build_toa_aoa_matrix(
            frequency_vector, delta_t_refined, range_ref_down, range_refined_up, num_angles, n_tot)
        T_matrix_refined = np.conj(T_matrix_refined)
        T_matrix_times_refined, time_vector_refined = build_toa_matrix(frequency_vector, delta_t_refined,
                                                                       range_ref_down, range_refined_up)
        T_matrix_times_refined = np.conj(T_matrix_times_refined.T)

        if use_prior and path_loss_sim is not None:
            # GROUND TRUTH SIMULATION
            sorted_idx_sim = np.argsort(abs(path_loss_sim[0, time_step]))[0, :]

            elevation_sorted_sim = (aoa_sim[0, time_step][1, sorted_idx_sim])
            azimuth_sorted_sim = (aoa_sim[0, time_step][0, sorted_idx_sim])

            azimuth_sorted_sim_2 = np.arcsin(np.sin(azimuth_sorted_sim / 180 * mt.pi)
                                             * np.cos(elevation_sorted_sim / 180 * mt.pi)) * 180 / mt.pi

            az_positive = azimuth_sorted_sim_2 > 0
            az_negative = azimuth_sorted_sim_2 < 0
            azimuth_sorted_sim_2[az_positive] -= 180
            azimuth_sorted_sim_2[az_negative] += 180

            swap_idx_pos = azimuth_sorted_sim_2 > 90
            swap_idx_neg = azimuth_sorted_sim_2 < -90
            azimuth_sorted_sim_2[swap_idx_pos] = 180 - azimuth_sorted_sim_2[swap_idx_pos]
            azimuth_sorted_sim_2[swap_idx_neg] = - 180 - azimuth_sorted_sim_2[swap_idx_neg]

            times_sorted_sim = delays_sim[0, time_step][:, sorted_idx_sim]

            priors_aoa = list(azimuth_sorted_sim_2)
            priors_aoa = [(priors_aoa[i] / (180 / num_angles) + (num_angles // 2)) for i in range(len(priors_aoa))]
            priors_toa = list(times_sorted_sim[0, :])
            priors_toa = priors_toa - priors_toa[0]
            priors_toa = [priors_toa[i] / delta_t_refined for i in range(len(priors_aoa))]
            search_range_aoa = int(5 // (180 / num_angles))
            search_range_toa = int(5e-9 // delta_t_refined)

            complex_opt_r_not_refined, complex_opt_r_refined = iht_algorithm_complete(T_matrix_refined,
                                                                                      T_matrix_times_refined,
                                                                                      T_matrix_angles, signal_time,
                                                                                      T_matrix_times_refined.shape[1],
                                                                                      delta_t_refined,
                                                                                      num_angles, s, refinement,
                                                                                      priors_aoa, priors_toa,
                                                                                      search_range_aoa,
                                                                                      search_range_toa,
                                                                                      )
        else:
            complex_opt_r_not_refined, complex_opt_r_refined = iht_algorithm_complete(T_matrix_refined,
                                                                                      T_matrix_times_refined,
                                                                                      T_matrix_angles, signal_time,
                                                                                      T_matrix_times_refined.shape[1],
                                                                                      delta_t_refined,
                                                                                      num_angles, s, refinement)

        # fills columnwise ((t_min->t_max, ang_min)=1st column, etc)
        complex_opt_r_reshape = np.reshape(complex_opt_r_not_refined, (-1, num_angles), order='F')
        r_optim.append(complex_opt_r_reshape)

        if complex_opt_r_refined is not None:
            complex_opt_r_refined_reshape = np.reshape(complex_opt_r_refined, (-1, num_angles), order='F')
            r_optim_refined.append(complex_opt_r_refined_reshape)
            complex_opt_r_reshape = complex_opt_r_refined_reshape

        r_optim_t_reshape = np.reshape(complex_opt_r_reshape, (-1,),
                                       order='F')  # fills columnwise ((t_min->t_max, ang_min)=1st column, etc)
        sort_idxs = np.flip(np.argsort(abs(r_optim_t_reshape)))
        sort_amplitude = abs(r_optim_t_reshape[sort_idxs])

        amplitude_threshold = sort_amplitude[0] * 1E-3
        paths_refined_amplitude = sort_amplitude[sort_amplitude > amplitude_threshold]
        sort_idx_end = np.sum(sort_amplitude > amplitude_threshold)
        sort_idxs = sort_idxs[:sort_idx_end]
        idx_toa = sort_idxs % time_vector_refined.shape[0]
        idx_aoa = sort_idxs // time_vector_refined.shape[0]
        paths_refined_toa = time_vector_refined[idx_toa] + time_max_r
        paths_refined_aoa = angle_vector_refined[idx_aoa] * 180 / mt.pi

        paths_refined_amplitude_array = np.asarray(paths_refined_amplitude[:sort_idx_end])
        paths_refined_aoa_array = - np.asarray(paths_refined_aoa)
        paths_refined_toa_array = np.asarray(paths_refined_toa)

        paths_amplitude_list.append(paths_refined_amplitude_array)
        paths_aoa_list.append(paths_refined_aoa_array)
        paths_toa_list.append(paths_refined_toa_array)

        time_end = time.time()
        optimization_times[time_step] = time_end-time_start

    # plot_r_scatter(complex_opt_r_reshape, time_vector_refined, angle_vector_refined)
    # plot_r_scatter(complex_opt_r_refined_reshape, time_vector_refined, angle_vector_refined)
    # plot_combined(complex_opt_r_refined_reshape, time_vector_refined + time_max_r, angle_vector_refined,
    #              path_loss_sim, delays_sim, aoa_sim, time_step)

    r_optim = np.asarray(r_optim)
    r_optim_refined = np.asarray(r_optim_refined)

    return r_optim, r_optim_refined, r_end_points, optimization_times, paths_amplitude_list, paths_toa_list, \
        paths_aoa_list


def plot_r_scatter(complex_opt_r_refined_reshape, time_vector_refined, angle_vector_refined):
    complex_opt_r_refined_reshape = np.nan_to_num(complex_opt_r_refined_reshape)
    paths_power = np.power(np.abs(complex_opt_r_refined_reshape), 2)
    paths_power = 10 * np.log10(paths_power / np.amax(np.nan_to_num(paths_power)))  # dB
    non_zero_idx = np.nonzero(complex_opt_r_refined_reshape)
    paths_power = paths_power[non_zero_idx]
    vmin = -60  # min(np.reshape(paths_power, -1))
    vmax = 0  # max(np.reshape(paths_power, -1))

    plt.figure(figsize=(5, 4))
    toa_array = time_vector_refined[non_zero_idx[0]]
    aoa_array = angle_vector_refined[non_zero_idx[1]] * 180 / mt.pi
    plt.scatter(toa_array * 1E9, aoa_array,
                c=paths_power,
                marker='o', cmap='Blues', s=20,
                vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('power [dB]', rotation=90)
    plt.xlabel('ToA [ns]')
    plt.ylabel('AoA [deg]')
    plt.xlim([-1, 13])  # range_considered + 100 * delta_t])
    plt.ylim([-90, 90])
    plt.yticks(np.arange(-90, 91, 20))
    plt.grid()
    plt.title('IHT')
    plt.tight_layout()
    plt.show()


def plot_groundtruth(path_loss_sim, delays_sim, aoa_sim, time_step):
    # GROUND TRUTH SIMULATION
    sorted_idx_sim = np.argsort(abs(path_loss_sim[0, time_step]))[0, :]

    elevation_sorted_sim = (aoa_sim[0, time_step][1, sorted_idx_sim])
    azimuth_sorted_sim = (aoa_sim[0, time_step][0, sorted_idx_sim])

    azimuth_sorted_sim_2 = np.arcsin(np.sin(azimuth_sorted_sim / 180 * mt.pi)
                                     * np.cos(elevation_sorted_sim / 180 * mt.pi)) * 180 / mt.pi

    az_positive = azimuth_sorted_sim_2 > 0
    az_negative = azimuth_sorted_sim_2 < 0
    azimuth_sorted_sim_2[az_positive] -= 180
    azimuth_sorted_sim_2[az_negative] += 180

    swap_idx_pos = azimuth_sorted_sim_2 > 90
    swap_idx_neg = azimuth_sorted_sim_2 < -90
    azimuth_sorted_sim_2[swap_idx_pos] = 180 - azimuth_sorted_sim_2[swap_idx_pos]
    azimuth_sorted_sim_2[swap_idx_neg] = - 180 - azimuth_sorted_sim_2[swap_idx_neg]

    times_sorted_sim = delays_sim[0, time_step][:, sorted_idx_sim]
    path_loss_sorted_sim = path_loss_sim[0, time_step][:, sorted_idx_sim]
    times_sorted_sim = times_sorted_sim - times_sorted_sim[:, 0]

    # plot ground truth
    paths_power = - path_loss_sorted_sim + path_loss_sorted_sim[:, 0]  # dB
    # paths_power = path_loss_sorted_sim / np.amax(np.nan_to_num(path_loss_sorted_sim))  # dB
    paths_power = paths_power[0, :]
    vmin = -40  # min(np.reshape(paths_power, -1))
    vmax = 0  # max(np.reshape(paths_power, -1))
    plt.figure(figsize=(5, 4))
    toa_array = times_sorted_sim #- times_sorted_sim[:, 0]
    # azimuth_sorted_sim_2 = np.arcsin(np.sin(angle_sim / 180 * mt.pi)
    #                                  * np.cos(elevation_sorted_sim / 180 * mt.pi)) * 180 / mt.pi

    plt.scatter(toa_array * 1E9, azimuth_sorted_sim_2,
                c=paths_power,
                marker='o', cmap='Blues', s=20,
                vmin=vmin, vmax=vmax)
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('power [dB]', rotation=90)
    plt.xlabel('ToA [ns]')
    plt.ylabel('AoA [deg]')
    plt.xlim([-1, 13])  # range_considered + 100 * delta_t])
    plt.ylim([-90, 90])
    plt.yticks(np.arange(-90, 91, 20))
    plt.grid()
    plt.tight_layout()
    plt.show()


def plot_combined(complex_opt_r_refined_reshape, time_vector_refined, angle_vector_refined,
                  path_loss_sim, delays_sim, aoa_sim, time_step):
    # GROUND TRUTH SIMULATION
    if path_loss_sim is not None:
        sorted_idx_sim = np.argsort(abs(path_loss_sim[0, time_step]))[0, :]
    else:
        sorted_idx_sim = np.arange(aoa_sim[0, time_step].shape[1])

    azimuth_sorted_sim = (aoa_sim[0, time_step][0, sorted_idx_sim])

    if aoa_sim.shape[0] > 1:
        elevation_sorted_sim = (aoa_sim[0, time_step][1, sorted_idx_sim])
        azimuth_sorted_sim_2 = np.arcsin(np.sin(azimuth_sorted_sim / 180 * mt.pi)
                                         * np.cos(elevation_sorted_sim / 180 * mt.pi)) * 180 / mt.pi
    else:
        azimuth_sorted_sim_2 = azimuth_sorted_sim

    az_positive = azimuth_sorted_sim_2 > 0
    az_negative = azimuth_sorted_sim_2 < 0
    azimuth_sorted_sim_2[az_positive] -= 180
    azimuth_sorted_sim_2[az_negative] += 180

    swap_idx_pos = azimuth_sorted_sim_2 > 90
    swap_idx_neg = azimuth_sorted_sim_2 < -90
    azimuth_sorted_sim_2[swap_idx_pos] = 180 - azimuth_sorted_sim_2[swap_idx_pos]
    azimuth_sorted_sim_2[swap_idx_neg] = - 180 - azimuth_sorted_sim_2[swap_idx_neg]

    times_sorted_sim = delays_sim[0, time_step][:, sorted_idx_sim]
    if path_loss_sim is not None:
        path_loss_sorted_sim = path_loss_sim[0, time_step][:, sorted_idx_sim]
    else:
        path_loss_sorted_sim = np.ones((aoa_sim[0, time_step].shape[1], 1))
    times_sorted_sim = times_sorted_sim

    vmin = -40
    vmax = 0
    plt.figure(figsize=(5, 4))

    # plot ground truth
    paths_power = - path_loss_sorted_sim + path_loss_sorted_sim[:, 0]  # dB
    paths_power = paths_power[0, :]
    toa_array = times_sorted_sim
    plt.scatter(toa_array * 1E9, azimuth_sorted_sim_2,
                c=paths_power,
                marker='o', cmap='Blues', s=20,
                vmin=vmin, vmax=vmax, label='ground')

    cbar = plt.colorbar()

    # plot sim
    complex_opt_r_refined_reshape = np.nan_to_num(complex_opt_r_refined_reshape)
    paths_power = np.power(np.abs(complex_opt_r_refined_reshape), 2)
    paths_power = 10 * np.log10(paths_power / np.amax(np.nan_to_num(paths_power)))  # dB
    non_zero_idx = np.nonzero(complex_opt_r_refined_reshape)
    paths_power = paths_power[non_zero_idx]
    toa_array = time_vector_refined[non_zero_idx[0]]
    aoa_array = angle_vector_refined[non_zero_idx[1]] * 180 / mt.pi
    plt.scatter(toa_array * 1E9, aoa_array,
                c=paths_power,
                marker='x', cmap='Reds', s=20,
                vmin=vmin, vmax=vmax, label='IHT')

    cbar.ax.set_ylabel('power [dB]', rotation=90)
    plt.xlabel('ToA [ns]')
    plt.ylabel('AoA [deg]')
    plt.xlim([-1, 13])  # range_considered + 100 * delta_t])
    plt.ylim([-90, 90])
    plt.yticks(np.arange(-90, 91, 20))
    plt.grid()
    plt.legend(prop={'size': 10})
    plt.tight_layout()
    plt.show()


def plot_r(complex_opt_r_refined_reshape, time_vector_refined, angle_vector_refined):
    complex_opt_r_refined_reshape = np.nan_to_num(complex_opt_r_refined_reshape)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    plt1 = ax1.pcolormesh(np.abs(complex_opt_r_refined_reshape).T, shading='gouraud', cmap='viridis', linewidth=0,
                          rasterized=True)
    plt1.set_edgecolor('face')
    cbar1 = fig.colorbar(plt1)
    ax1.set_xlabel(r'time [ns]')
    ax1.set_ylabel(r'angle')
    ax1.set_xticks(np.arange(0, time_vector_refined.shape[0], 50))
    ax1.set_xticklabels(np.round(time_vector_refined[0::50] * 1e9, 1))
    ax1.set_yticks(np.arange(0, 181, 30))
    ax1.set_yticklabels(np.arange(-90, 91, 30))
    plt.show()


def find_max(complex_opt_r_refined_reshape, time_vector_refined, angle_vector_refined):
    r_optim_t_reshape = np.reshape(complex_opt_r_refined_reshape, (-1,), order='F')
    sort_idxs = np.flip(np.argsort(abs(r_optim_t_reshape)))
    sort_amplitude = abs(r_optim_t_reshape[sort_idxs])
    amplitude_threshold = sort_amplitude[0] * 1E-3
    sort_amplitude = sort_amplitude[sort_amplitude > amplitude_threshold]
    sort_idx_end = np.sum(sort_amplitude > amplitude_threshold)
    sort_idxs = sort_idxs[:sort_idx_end]
    idx_toa = sort_idxs % time_vector_refined.shape[0]
    idx_aoa = sort_idxs // time_vector_refined.shape[0]
    times_sorted = time_vector_refined[idx_toa]
    angles_sorted = (angle_vector_refined[idx_aoa] * 180 / mt.pi) #+ 90
    # swap_idx = angles_sorted > 180
    # angles_sorted[swap_idx] = angles_sorted[swap_idx] - 360
    # swap_idx = angles_sorted < -180
    # angles_sorted[swap_idx] = angles_sorted[swap_idx] + 360
    print('times_sorted: ', times_sorted)
    print('angles_sorted: ', angles_sorted)


opt_methods_dict = {'lasso': optimize_aoa_toa_lasso1,
                    'omp': optimize_aoa_toa_omp,
                    'iht': optimize_aoa_toa_iht,
                    'iht_noref': optimize_aoa_toa_iht
                    }


def optimize_aoa_toa_omp_ext(signal_considered, frequency_vector, t_min, t_max, delta_t, delta_t_refined,
                             range_refined_down, range_refined_up, num_angles, n_tot, n_nonzero_coefs=None, tol=None,
                             path_loss_sim=None, delays_sim=None, aoa_sim=None):

    num_time_steps = signal_considered.shape[0]

    paths_amplitude_list = []
    paths_toa_list = []
    paths_aoa_list = []

    r_optim = []
    r_end_points = np.zeros((num_time_steps, 4))  # t_min, t_max, dt, n_angles of the saved r_optim
    optimization_times = np.zeros(num_time_steps)

    # Dictionary initialization
    T_matrix, time_vector = build_toa_matrix(frequency_vector, delta_t, t_min, t_max)
    T_matrix = np.conj(T_matrix.T)

    # Optimizer initialization
    omp_mod = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs, tol=tol)

    # Time steps optimization
    for time_step in range(num_time_steps):

        time_start = time.time()

        signal_time = signal_considered[time_step, :, :]

        T_matrix_real, signal_time_real = convert_TH_real(T_matrix, signal_time[:, 0])
        omp_mod.fit(T_matrix_real, signal_time_real)
        real_opt_r_coarse = omp_mod.coef_

        complex_opt_r = convert_to_complex_r(real_opt_r_coarse)

        position_max_r = np.argmax(abs(complex_opt_r))
        time_max_r = time_vector[position_max_r]

        t_min_step = max(time_max_r - range_refined_down, t_min)
        t_max_step = min(time_max_r + range_refined_up, t_max)

        T_matrix_refined, time_vector_refined, angle_vector_refined = build_toa_aoa_matrix(
            frequency_vector, delta_t_refined, t_min_step, t_max_step, num_angles, n_tot)
        T_matrix_refined = np.conj(T_matrix_refined)

        signal_time = signal_time.flatten(order='F')  # first antenna, second antenna ...

        T_matrix_refined_real, signal_time_real = convert_TH_real(T_matrix_refined, signal_time)
        omp_mod.fit(T_matrix_refined_real, signal_time_real)
        real_opt_r_refined = omp_mod.coef_

        complex_opt_r_refined = convert_to_complex_r(real_opt_r_refined)

        # fills opt_r columnwise ((t_min->t_max, ang_min)=1st column, etc)
        complex_opt_r_refined_reshape = np.reshape(complex_opt_r_refined, (-1, num_angles), order='F')
        r_optim.append(complex_opt_r_refined_reshape)

        r_optim_t_reshape = np.reshape(complex_opt_r_refined_reshape, (-1,),
                                       order='F')  # fills opt_r columnwise ((t_min->t_max, ang_min)=1st column, etc)
        sort_idxs = np.flip(np.argsort(abs(r_optim_t_reshape)))
        sort_amplitude = abs(r_optim_t_reshape[sort_idxs])

        amplitude_threshold = sort_amplitude[0] * 1E-3
        paths_refined_amplitude = sort_amplitude[sort_amplitude > amplitude_threshold]
        sort_idx_end = np.sum(sort_amplitude > amplitude_threshold)
        sort_idxs = sort_idxs[:sort_idx_end]
        idx_toa = sort_idxs % time_vector_refined.shape[0]
        idx_aoa = sort_idxs // time_vector_refined.shape[0]
        paths_refined_toa = time_vector_refined[idx_toa]
        paths_refined_aoa = angle_vector_refined[idx_aoa] * 180 / mt.pi

        paths_refined_amplitude_array = np.asarray(paths_refined_amplitude[:sort_idx_end])
        paths_refined_aoa_array = - np.asarray(paths_refined_aoa)
        paths_refined_toa_array = np.asarray(paths_refined_toa)

        paths_amplitude_list.append(paths_refined_amplitude_array)
        paths_aoa_list.append(paths_refined_aoa_array)
        paths_toa_list.append(paths_refined_toa_array)

        r_end_points[time_step, :] = np.array([t_min_step, t_max_step, delta_t_refined, num_angles])

        time_end = time.time()
        optimization_times[time_step] = time_end-time_start

    r_optim = np.asarray(r_optim)

    # plot_combined(complex_opt_r_refined_reshape, time_vector_refined, angle_vector_refined,
    #               path_loss_sim, delays_sim, aoa_sim, time_step)

    return r_optim, None, r_end_points, optimization_times, paths_amplitude_list, paths_toa_list, paths_aoa_list


opt_methods_dict_ext = {'omp': optimize_aoa_toa_omp_ext}
