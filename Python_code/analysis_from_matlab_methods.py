
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

import argparse
import os
import pickle
import numpy as np
import math as mt
import scipy.io as sio


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('step_aoa_first', help='Step for the AoA of first path', type=int)
    parser.add_argument('step_aoa_second', help='Step for the AoA of second path', type=int)
    parser.add_argument('max_num_paths', help='Maximum number of paths detected', type=int)
    parser.add_argument('opt_method', help='Optimization routine')
    parser.add_argument('exp_dir', help='Name base of the directory')
    parser.add_argument('name_base', help='Name base of the simulation ToA')
    parser.add_argument('--dB', help='Whether the amplitudes are in dB (default 0, i.e., not dB)',
                        default=0, required=False, type=int)
    args = parser.parse_args()

    exp_dir = args.exp_dir
    name_base = args.name_base  # simulation
    print(name_base)

    step_aoa_first = args.step_aoa_first
    aoa_fist_array = np.arange(-90, 90, step_aoa_first)
    num_aoa_first = aoa_fist_array.shape[0]

    path_find = np.zeros((180, args.max_num_paths + 1))
    # in the last column we have the number of times all the paths are identified
    estim_toa_error = np.zeros((180, args.max_num_paths))
    estim_aoa_error = np.zeros((180, args.max_num_paths))
    counter = np.zeros((180, 1))  # to then average the results
    execution_times = np.zeros((180, 1))

    toa_sep_threshold = 5e-9
    aoa_sep_threshold = 40

    for aoa_second in range(-90, 91, args.step_aoa_second):

        name_file = exp_dir + 'CFR_' + name_base + '_aoaobst_' + str(aoa_second) + '.mat'
        csi_buff = sio.loadmat(name_file)
        signal_complete = (csi_buff['CFR'])

        name_file = exp_dir + 'delay_' + name_base + '_aoaobst_' + str(aoa_second) + '.mat'
        csi_buff = sio.loadmat(name_file)
        delays_sim = (csi_buff['propagation_delays'])

        name_file = exp_dir + 'aoa_' + name_base + '_aoaobst_' + str(aoa_second) + '.mat'
        csi_buff = sio.loadmat(name_file)
        aoa_sim = (csi_buff['propagation_aoa'])

        # open results optimization
        save_dir = '../results/' + args.opt_method + '/'
        name_file = save_dir + 'paths_amplitude_list_' + name_base + '_aoaobst_' + str(aoa_second) + '.mat'
        try:
            csi_buff = sio.loadmat(name_file)
            paths_amplitude_list = list((csi_buff['power'])[:, 0])
        except FileNotFoundError:
            print('file not found', name_file)
            continue
        except KeyError:
            paths_amplitude_list = list((csi_buff['power_dict'])[:, 0])
        name_file = save_dir + 'paths_aoa_list_' + name_base + '_aoaobst_' + str(aoa_second) + '.mat'
        csi_buff = sio.loadmat(name_file)
        paths_aoa_list = list((csi_buff['AoA'])[:, 0])

        name_file = save_dir + 'paths_toa_list_' + name_base + '_aoaobst_' + str(aoa_second) + '.mat'
        csi_buff = sio.loadmat(name_file)
        paths_toa_list = list((csi_buff['ToA'])[:, 0])

        name_file = save_dir + 'opr_sim_' + name_base + '_aoaobst_' + str(aoa_second) + '.mat'
        csi_buff = sio.loadmat(name_file)
        execution_time = csi_buff['optimization_times']

        # parameters
        n_tot = 4
        F_frequency = 256
        delta_f = 312.5E3
        delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, 203, 231, 251, 252, 253,
                                  254, 255], dtype=int)
        frequency_vector_idx = np.arange(F_frequency)
        frequency_vector_complete = delta_f * (frequency_vector_idx - F_frequency / 2)
        frequency_vector_idx = np.delete(frequency_vector_idx, delete_idxs, axis=0)
        frequency_vector = np.delete(frequency_vector_complete, delete_idxs, axis=0)
        T = 1 / delta_f

        for aoa_idx_first in range(num_aoa_first):
            aoa_first = aoa_fist_array[aoa_idx_first]

            # RESULTS OF OPTIMIZATION
            paths_refined_amplitude_array = paths_amplitude_list[aoa_idx_first][:, 0]
            paths_refined_aoa_array = np.squeeze(paths_aoa_list[aoa_idx_first])
            paths_refined_toa_array = np.squeeze(paths_toa_list[aoa_idx_first])

            if args.dB:
                paths_refined_amplitude_array = np.power(10, paths_refined_amplitude_array/20)
                sorted_idx = np.argsort(paths_refined_toa_array)
                if len(paths_refined_amplitude_array) > 1:
                    paths_refined_aoa_array_sorted = paths_refined_aoa_array[sorted_idx]
                    paths_refined_toa_array = paths_refined_toa_array[sorted_idx]
                    paths_refined_amplitude_array = paths_refined_amplitude_array[sorted_idx]

            # paths_refined_toa_array = paths_refined_toa_array - paths_refined_toa_array[0]
            if args.opt_method == "ubilocate":
                paths_refined_toa_array = paths_refined_toa_array - T/2

            if args.opt_method == "spotfi":
                paths_refined_toa_array = paths_refined_toa_array - 2e-7

            if len(paths_refined_amplitude_array) > 1:
                paths_refined_aoa_array_sorted = paths_refined_aoa_array  # [sorted_idx]
                paths_refined_toa_array_sorted = paths_refined_toa_array  # [sorted_idx]
            else:
                paths_refined_aoa_array_sorted = list([paths_refined_aoa_array])
                paths_refined_toa_array_sorted = list([paths_refined_toa_array])

            # GROUND TRUTH SIMULATION
            sorted_idx_sim = np.arange(2)

            azimuth_sorted_sim = (aoa_sim[0, aoa_idx_first][0, sorted_idx_sim])
            azimuth_sorted_sim_2 = np.copy(np.asarray(-180 + azimuth_sorted_sim, dtype='int16'))

            az_positive = azimuth_sorted_sim_2 > 0
            az_negative = azimuth_sorted_sim_2 < 0
            azimuth_sorted_sim_2[az_positive] -= 180
            azimuth_sorted_sim_2[az_negative] += 180

            swap_idx_pos = azimuth_sorted_sim_2 > 90
            swap_idx_neg = azimuth_sorted_sim_2 < -90
            azimuth_sorted_sim_2[swap_idx_pos] = 180 - azimuth_sorted_sim_2[swap_idx_pos]
            azimuth_sorted_sim_2[swap_idx_neg] = - 180 - azimuth_sorted_sim_2[swap_idx_neg]

            times_sorted_sim = delays_sim[0, aoa_idx_first][:, sorted_idx_sim]
            path_loss_sorted_sim = None

            aoa_first_true = int(np.round(azimuth_sorted_sim_2[0], 1))
            aoa_second_true = int(np.round(azimuth_sorted_sim_2[1]))
            diff_aoa_first_second = mt.floor(aoa_first_true - azimuth_sorted_sim_2[1])
            aoa_idx_diff = diff_aoa_first_second
            if aoa_first_true * aoa_second_true > 0:
                pass
            elif aoa_first_true >= 0:  # and consequently aoa_second < 0
                if diff_aoa_first_second >= 90:
                    aoa_idx_diff = - (90 - aoa_first_true + 90 + aoa_second_true)
            else:  # consequently aoa_first < 0 and aoa_second > 0
                if diff_aoa_first_second < - 90:
                    aoa_idx_diff = 90 - aoa_second_true + 90 + aoa_first_true

            # print(aoa_idx_diff)
            aoa_idx_diff += 90

            # Check if paths have been separated and if yes the error of the identification
            paths_found = 0
            for path_idx in range(times_sorted_sim.shape[1]):
                toa_diff = abs(paths_refined_toa_array_sorted - times_sorted_sim[0, path_idx])
                aoa_diff = abs(paths_refined_aoa_array_sorted - azimuth_sorted_sim_2[path_idx])
                aoa_diff = np.minimum(aoa_diff, 180-aoa_diff)
                toa_below_threshold = set(np.argwhere(toa_diff < toa_sep_threshold)[:, 0])
                aoa_below_threshold = set(np.argwhere(aoa_diff < aoa_sep_threshold)[:, 0])
                path_set_estim = toa_below_threshold.intersection(aoa_below_threshold)

                path_set_estim = list(path_set_estim)

                if path_set_estim:  # path_set_estim.shape[0] != 0:
                    paths_found += 1
                    path_idx_estim = min(path_set_estim)  # select the strongest path among the compatible ones
                    estim_toa_error[aoa_idx_diff, path_idx] += toa_diff[path_idx_estim]
                    estim_aoa_error[aoa_idx_diff, path_idx] += aoa_diff[path_idx_estim]
                    paths_refined_toa_array_sorted = np.delete(paths_refined_toa_array_sorted, path_idx_estim)
                    paths_refined_aoa_array_sorted = np.delete(paths_refined_aoa_array_sorted, path_idx_estim)
                    path_find[aoa_idx_diff, path_idx] += 1
            if paths_found == times_sorted_sim.shape[1]:
                path_find[aoa_idx_diff, path_idx + 1] += 1
            counter[aoa_idx_diff] += 1

            if execution_time[aoa_idx_first, 0].shape[1] > 0:
                execution_times[aoa_idx_diff] += execution_time[aoa_idx_first, 0][0, 0]

    print(counter[:, 0])
    path_find_avg = np.divide(path_find, counter)
    execution_times_avg = np.divide(execution_times, counter)
    estim_toa_error_avg = np.divide(estim_toa_error, path_find[:, :-1])
    estim_aoa_error_avg = np.divide(estim_aoa_error, path_find[:, :-1])

    save_dir = '../results/processed_' + args.opt_method + '/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    name_file = save_dir + 'path_find_avg_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(path_find_avg, fp)
    name_file = save_dir + 'computing_time_avg_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(execution_times_avg, fp)

    name_file = save_dir + 'estim_toa_error_avg_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(estim_toa_error_avg, fp)
    name_file = save_dir + 'estim_aoa_error_avg_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(estim_aoa_error_avg, fp)
