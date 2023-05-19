
"""
    Copyright (C) 2023 Francesca Meneghello
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

import pandas as pd
from utilityfunct_aoa_toa_doppler import *


def joint_aoa_tof_estimation(cfr_sample, aoa_matrix, toa_matrix, num_ant, num_subc, num_angles):
    matrix_cfr_aoa = np.dot(cfr_sample, aoa_matrix)
    matrix_cfr_aoa_toa = np.dot(toa_matrix, matrix_cfr_aoa) / (num_ant * num_subc)

    power_matrix_cfr_aoa_toa = np.power(np.abs(matrix_cfr_aoa_toa), 2)
    index_max = np.argmax(power_matrix_cfr_aoa_toa)
    time_idx_max = int(index_max / num_angles)
    angle_idx_max = int(index_max % num_angles)

    amplitudes_time_max = matrix_cfr_aoa_toa[time_idx_max, angle_idx_max]

    return amplitudes_time_max, power_matrix_cfr_aoa_toa[time_idx_max, angle_idx_max], time_idx_max, angle_idx_max


def joint_aoa_tof_estimation_priors(cfr_sample, aoa_matrix, toa_matrix, num_ant, num_subc, num_angles, pr_aoa, pr_toa,
                                    search_range_aoa, search_range_toa):
    aoa_matrix_masked = np.zeros_like(aoa_matrix)
    start_aoa = max(0, int(pr_aoa - search_range_aoa))
    end_aoa = min(num_angles, int(pr_aoa + search_range_aoa))
    aoa_matrix_masked[:, start_aoa:end_aoa] = aoa_matrix[:, start_aoa:end_aoa]

    matrix_cfr_aoa = np.dot(cfr_sample, aoa_matrix_masked)

    toa_matrix_masked = np.zeros_like(toa_matrix)
    start_toa = max(0, int(pr_toa - search_range_toa))
    end_toa = min(num_angles, int(pr_toa + search_range_toa))
    toa_matrix_masked[start_toa:end_toa, :] = toa_matrix[start_toa:end_toa, :]

    matrix_cfr_aoa_toa = np.dot(toa_matrix_masked, matrix_cfr_aoa) / (num_ant * num_subc)

    power_matrix_cfr_aoa_toa = np.power(np.abs(matrix_cfr_aoa_toa), 2)
    index_max = np.argmax(power_matrix_cfr_aoa_toa)
    time_idx_max = int(index_max / num_angles)
    angle_idx_max = int(index_max % num_angles)

    amplitudes_time_max = matrix_cfr_aoa_toa[time_idx_max, angle_idx_max]

    return amplitudes_time_max, power_matrix_cfr_aoa_toa[time_idx_max, angle_idx_max], time_idx_max, angle_idx_max


def individual_aoa_tof_estimation(cfr_sample, aoa_matrix, toa_matrix, num_ant, num_subc, index_toa):
    cfr_sample_toa = np.dot(toa_matrix[index_toa, :], cfr_sample)

    matrix_cfr_aoa = np.dot(cfr_sample_toa, aoa_matrix)
    angle_idx_max = np.argmax(np.power(np.abs(matrix_cfr_aoa), 2))
    cfr_sample_aoa = np.dot(cfr_sample, aoa_matrix[:, angle_idx_max])

    matrix_cfr_toa = np.dot(toa_matrix, cfr_sample_aoa) / (num_ant * num_subc)
    power_matrix_cfr_toa = np.power(np.abs(matrix_cfr_toa), 2)
    time_idx_max = np.argmax(power_matrix_cfr_toa)

    amplitudes_time_max = matrix_cfr_toa[time_idx_max]

    return amplitudes_time_max, power_matrix_cfr_toa[time_idx_max], time_idx_max, angle_idx_max


def md_track_2d_prior(cfr_sample, aoa_matrix, toa_matrix, num_ant, num_subc, num_angles, num_iteration_refinement,
                      threshold, priors_aoa, priors_toa, search_range_aoa, search_range_toa):
    # start = time.time()
    paths = []

    # mD-Track INITIAL ESTIMATION
    paths_amplitude = []
    paths_power = []
    paths_toa = []
    paths_aoa = []
    cfr_sample_residual = cfr_sample
    num_priors = len(priors_aoa)
    pr_idx = 0
    while True:
        if pr_idx < num_priors:
            path_amplitude, path_power, path_toa, path_aoa = joint_aoa_tof_estimation_priors(cfr_sample_residual,
                                                                                             aoa_matrix,
                                                                                             toa_matrix, num_ant,
                                                                                             num_subc, num_angles,
                                                                                             priors_aoa[pr_idx],
                                                                                             priors_toa[pr_idx],
                                                                                             search_range_aoa,
                                                                                             search_range_toa)
            pr_idx += 1
        else:
            break
            path_amplitude, path_power, path_toa, path_aoa = joint_aoa_tof_estimation(cfr_sample_residual,
                                                                                      aoa_matrix, toa_matrix, num_ant,
                                                                                      num_subc, num_angles)

            try:
                ref = max(paths_power)
                if path_power <= ref*10**threshold:
                    break
            except IndexError:
                True  # it is the first path, keep it and go ahead

        paths_amplitude.append(path_amplitude)
        paths_power.append(path_power)
        paths_toa.append(path_toa)
        paths_aoa.append(path_aoa)

        signal_path = path_amplitude * \
                      np.conj(aoa_matrix[:, path_aoa]) * \
                      np.tile(np.expand_dims(np.conj(toa_matrix[path_toa, :]), -1), num_ant)
        paths.append(signal_path)

        cfr_sample_residual = cfr_sample_residual - signal_path

    num_estimated_paths = len(paths)

    # mD-Track ITERATIVE REFINEMENT
    paths_refined_amplitude = []
    paths_refined_toa = []
    paths_refined_aoa = []
    for iteration in range(num_iteration_refinement):
        for path_idx in range(num_estimated_paths):
            cfr_single_path = paths[path_idx] + cfr_sample_residual
            path_amplitude, path_power, path_toa, path_aoa = individual_aoa_tof_estimation(cfr_single_path,
                                                                                           aoa_matrix, toa_matrix,
                                                                                           num_ant, num_subc,
                                                                                           paths_toa[path_idx])

            if iteration == num_iteration_refinement-1:
                paths_refined_amplitude.append(path_amplitude)
                paths_refined_toa.append(path_toa)
                paths_refined_aoa.append(path_aoa)

            signal_path_refined = path_amplitude * \
                                  np.conj(aoa_matrix[:, path_aoa]) * \
                                  np.tile(np.expand_dims(np.conj(toa_matrix[path_toa, :]), -1), num_ant)

            paths[path_idx] = signal_path_refined  # update the path with the refinement

            cfr_cumulative_paths = sum(paths)

            cfr_sample_residual = cfr_sample - cfr_cumulative_paths

    # end = time.time()
    # print(end - start)

    return paths, paths_amplitude, paths_toa, paths_aoa


def md_track_2d(cfr_sample, aoa_matrix, toa_matrix, num_ant, num_subc, num_angles, num_iteration_refinement, threshold):
    # start = time.time()
    paths = []

    # mD-Track INITIAL ESTIMATION
    paths_amplitude = []
    paths_power = []
    paths_toa = []
    paths_aoa = []
    cfr_sample_residual = cfr_sample
    while True:
        path_amplitude, path_power, path_toa, path_aoa = joint_aoa_tof_estimation(cfr_sample_residual, aoa_matrix,
                                                                                  toa_matrix, num_ant, num_subc,
                                                                                  num_angles)

        try:
            ref = paths_power[0]
            if path_power <= ref*10**threshold:
                break
        except IndexError:
            True  # it is the first path, keep it and go ahead

        paths_amplitude.append(path_amplitude)
        paths_power.append(path_power)
        paths_toa.append(path_toa)
        paths_aoa.append(path_aoa)

        signal_path = path_amplitude * \
                      np.conj(aoa_matrix[:, path_aoa]) * \
                      np.tile(np.expand_dims(np.conj(toa_matrix[path_toa, :]), -1), num_ant)
        paths.append(signal_path)

        cfr_sample_residual = cfr_sample_residual - signal_path

    num_estimated_paths = len(paths)

    # mD-Track ITERATIVE REFINEMENT
    paths_refined_amplitude = []
    paths_refined_toa = []
    paths_refined_aoa = []
    for iteration in range(num_iteration_refinement):
        for path_idx in range(num_estimated_paths):
            cfr_single_path = paths[path_idx] + cfr_sample_residual
            path_amplitude, path_power, path_toa, path_aoa = individual_aoa_tof_estimation(cfr_single_path,
                                                                                           aoa_matrix, toa_matrix,
                                                                                           num_ant, num_subc,
                                                                                           paths_toa[path_idx])

            if iteration == num_iteration_refinement-1:
                paths_refined_amplitude.append(path_amplitude)
                paths_refined_toa.append(path_toa)
                paths_refined_aoa.append(path_aoa)

            signal_path_refined = path_amplitude * \
                                  np.conj(aoa_matrix[:, path_aoa]) * \
                                  np.tile(np.expand_dims(np.conj(toa_matrix[path_toa, :]), -1), num_ant)

            paths[path_idx] = signal_path_refined  # update the path with the refinement

            cfr_cumulative_paths = sum(paths)

            cfr_sample_residual = cfr_sample - cfr_cumulative_paths

    # end = time.time()
    # print(end - start)

    return paths, paths_refined_amplitude, paths_refined_toa, paths_refined_aoa


def md_track_2d_elegant(cfr_sample, aoa_matrix, toa_matrix, num_ant, num_subc, num_angles):
    # start = time.time()
    paths = pd.Series()

    # mD-Track INITIAL ESTIMATION
    paths_parameters = pd.DataFrame(columns=['amplitude', 'power', 'ToA_idx', 'AoA_idx'])
    cfr_sample_residual = cfr_sample
    while True:
        path_param = joint_aoa_tof_estimation(cfr_sample_residual, aoa_matrix, toa_matrix, num_ant, num_subc,
                                              num_angles)
        try:
            ref = paths_parameters.loc[0]['power']
            if path_param['power'] < ref*10**(-1.5):
                break
        except KeyError:
            True  # it is the first path, keep it and go ahead

        paths_parameters = paths_parameters.append([path_param], ignore_index=True)

        signal_path = path_param['amplitude'] * \
                      np.transpose(aoa_matrix[path_param['AoA_idx'], ...]) * \
                      np.tile(np.expand_dims(toa_matrix[:, path_param['ToA_idx']], -1), num_ant)
        paths = paths.append(pd.Series([signal_path]))

        cfr_sample_residual = cfr_sample_residual - signal_path

    num_estimated_paths = len(paths)

    # mD-Track ITERATIVE REFINEMENT
    paths_parameters_refined = pd.DataFrame(columns=['amplitude', 'power', 'ToA_idx', 'AoA_idx'])
    num_iterations = 10
    for iteration in range(num_iterations):
        for path_idx in range(num_estimated_paths):
            cfr_single_path = paths.iloc[path_idx] + cfr_sample_residual
            path_param_refined = individual_aoa_tof_estimation(cfr_single_path, aoa_matrix, toa_matrix,
                                                               num_ant, num_subc,
                                                               paths_parameters.loc[path_idx]['ToA_idx'])

            if iteration == num_iterations-1:
                paths_parameters_refined = paths_parameters_refined.append([path_param_refined], ignore_index=True)

            signal_path_refined = path_param_refined['amplitude'] * \
                                  np.transpose(aoa_matrix[path_param_refined['AoA_idx'], ...]) * \
                                  np.tile(np.expand_dims(toa_matrix[:, path_param_refined['ToA_idx']], -1), num_ant)

            paths.iloc[path_idx] = signal_path_refined  # update the path with the refinement

            cfr_cumulative_paths = paths.sum()

            cfr_sample_residual = cfr_sample - cfr_cumulative_paths

    # end = time.time()
    # time_elapsed = end - start
    return paths


def joint_aoa_tof_dop_estimation(cfr_window, aoa_matrix, toa_matrix, dop_matrix, num_ant, num_subc, num_pkts,
                                  num_angles, num_times, num_dop):

    # time_s = time.time()
    matrix_cfr_aoa_toa_dop = np.zeros((num_times, num_dop, num_angles), dtype=complex)
    for index_aoa in range(num_angles):
        vector_aoa = np.reshape(aoa_matrix[:, index_aoa], (num_ant, 1, 1))
        matrix_cfr_aoa = np.multiply(cfr_window, vector_aoa)
        matrix_cfr_aoa_sum = np.sum(matrix_cfr_aoa, 0)

        matrix_cfr_aoa_toa = np.dot(toa_matrix, matrix_cfr_aoa_sum)

        matrix_cfr_aoa_toa_dop[:, :, index_aoa] = np.dot(matrix_cfr_aoa_toa, dop_matrix) / \
                                                  (num_pkts * num_subc * num_ant)
    # time_e = time.time()
    # print(time_e-time_s)

    power_matrix_cfr_aoa_toa_dop = np.power(np.abs(matrix_cfr_aoa_toa_dop), 2)
    index_max = np.argmax(power_matrix_cfr_aoa_toa_dop)
    time_idx_max = int(index_max / (num_angles * num_dop))
    angle_dop_idx_max = int(index_max % (num_angles * num_dop))
    dop_idx_max = int(angle_dop_idx_max / num_angles)
    angle_idx_max = int(angle_dop_idx_max % num_angles)

    amplitudes_time_max = matrix_cfr_aoa_toa_dop[time_idx_max, dop_idx_max, angle_idx_max]

    return amplitudes_time_max, power_matrix_cfr_aoa_toa_dop[time_idx_max, dop_idx_max, angle_idx_max], \
           dop_idx_max, time_idx_max, angle_idx_max


def individual_aoa_tof_dop_estimation(cfr_window, aoa_matrix, toa_matrix, dop_matrix, num_ant, num_subc, num_pkts,
                                      index_toa, index_dop):
    toa_matrix_select = np.reshape(toa_matrix[index_toa, :], (1, -1, 1))
    cfr_window_toa = np.multiply(toa_matrix_select, cfr_window)
    cfr_window_toa = np.sum(cfr_window_toa, axis=1)
    cfr_window_toa_dop = np.dot(cfr_window_toa, dop_matrix[:, index_dop])
    matrix_cfr_aoa = np.dot(cfr_window_toa_dop, (aoa_matrix)) / (num_ant * num_subc * num_pkts)
    angle_idx_max = np.argmax(np.power(np.abs(matrix_cfr_aoa), 2))

    aoa_matrix_select = np.reshape(aoa_matrix[:, angle_idx_max], (-1, 1, 1))
    cfr_window_aoa = np.sum(np.multiply(cfr_window, aoa_matrix_select), 0)
    cfr_window_aoa_dop = np.dot(cfr_window_aoa, dop_matrix[:, index_dop])
    matrix_cfr_toa = np.dot(toa_matrix, cfr_window_aoa_dop) / (num_ant * num_subc * num_pkts)
    time_idx_max = np.argmax(np.power(np.abs(matrix_cfr_toa), 2))

    aoa_matrix_select = np.reshape(aoa_matrix[:, angle_idx_max], (-1, 1, 1))
    cfr_window_aoa = np.sum(np.multiply(cfr_window, aoa_matrix_select), 0)
    cfr_window_aoa_toa = np.dot(toa_matrix[index_toa, :], cfr_window_aoa)
    matrix_cfr_dop = np.dot(cfr_window_aoa_toa, dop_matrix) / (num_ant * num_subc * num_pkts)
    power_matrix_cfr_dop = np.power(np.abs(matrix_cfr_dop), 2)
    dop_idx_max = np.argmax(power_matrix_cfr_dop)

    amplitudes_time_max = matrix_cfr_dop[time_idx_max]

    return amplitudes_time_max, power_matrix_cfr_dop[dop_idx_max], dop_idx_max, time_idx_max, angle_idx_max


def md_track_3d(cfr_window, aoa_matrix, toa_matrix, dop_matrix, num_ant, num_subc, num_angles, num_pkts,
                num_times, num_freq, num_iteration_refinement, threshold):
    # start = time.time()
    paths = []

    # mD-Track INITIAL ESTIMATION
    paths_amplitude = []
    paths_power = []
    paths_dop = []
    paths_toa = []
    paths_aoa = []
    cfr_window_residual = cfr_window
    while True:
        path_amplitude, path_power, path_dop, path_toa, path_aoa = joint_aoa_tof_dop_estimation(cfr_window_residual,
                                                                                                aoa_matrix, toa_matrix,
                                                                                                dop_matrix, num_ant,
                                                                                                num_subc, num_pkts,
                                                                                                num_angles, num_times,
                                                                                                num_freq)
        try:
            ref = paths_power[0]
            if path_power < ref * 10 ** (threshold):
                break
        except IndexError:
            True  # it is the first path, keep it and go ahead

        paths_amplitude.append(path_amplitude)
        paths_power.append(path_power)
        paths_dop.append(path_dop)
        paths_toa.append(path_toa)
        paths_aoa.append(path_aoa)

        dopl_sign = np.reshape(np.conj(dop_matrix[:, path_dop]), (1, 1, -1))
        time_aoa_signal = np.expand_dims(np.expand_dims(np.conj(aoa_matrix[:, path_aoa]), -1) * \
                          np.repeat(np.expand_dims(np.conj(toa_matrix[path_toa, :]), 0), num_ant, axis=0), -1)
        signal_path = path_amplitude * dopl_sign * time_aoa_signal

        paths.append(signal_path)

        cfr_window_residual = cfr_window_residual - signal_path

    num_estimated_paths = len(paths)

    # mD-Track ITERATIVE REFINEMENT
    paths_refined_amplitude = []
    paths_refined_dop = []
    paths_refined_toa = []
    paths_refined_aoa = []
    for iteration in range(num_iteration_refinement):
        for path_idx in range(num_estimated_paths):
            cfr_single_path = paths[path_idx] + cfr_window_residual
            path_amplitude, path_power, path_dop, path_toa, path_aoa = individual_aoa_tof_dop_estimation(
                cfr_single_path, aoa_matrix, toa_matrix, dop_matrix, num_ant, num_subc, num_pkts, paths_toa[path_idx],
                paths_dop[path_idx])

            if iteration == num_iteration_refinement - 1:
                paths_refined_amplitude.append(path_amplitude)
                paths_refined_dop.append(path_dop)
                paths_refined_toa.append(path_toa)
                paths_refined_aoa.append(path_aoa)

            dopl_sign = np.reshape(np.conj(dop_matrix[:, path_dop]), (1, 1, -1))
            time_aoa_signal = np.expand_dims(np.expand_dims(np.conj(aoa_matrix[:, path_aoa]), -1) * \
                                             np.repeat(np.expand_dims(np.conj(toa_matrix[path_toa, :]), 0), num_ant,
                                                       axis=0), -1)
            signal_path_refined = path_amplitude * dopl_sign * time_aoa_signal

            paths[path_idx] = signal_path_refined  # update the path with the refinement

            cfr_cumulative_paths = sum(paths)

            cfr_window_residual = cfr_window - cfr_cumulative_paths

    # end = time.time()
    # print(end - start)

    return paths, paths_refined_amplitude, paths_refined_dop, paths_refined_toa, paths_refined_aoa
