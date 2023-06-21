
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
import scipy.io as sio
import numpy as np
import time
import pickle
import math as mt
import matplotlib.pyplot as plt
import os

from utilityfunct_aoa_toa_doppler import build_aoa_matrix, build_toa_matrix
from utilityfunct_md_track import md_track_2d


def plot_combined(paths_refined_amplitude_array, paths_refined_toa_array, paths_refined_aoa_array,
                  path_loss_sorted_sim, times_sorted_sim, azimuth_sorted_sim_2):
    vmin = -40
    vmax = 0
    plt.figure(figsize=(5, 4))

    # plot ground truth
    if path_loss_sorted_sim is not None:
        paths_power = - path_loss_sorted_sim + path_loss_sorted_sim[:, 0]  # dB
        paths_power = paths_power[0, :]
    else:
        paths_power = np.ones_like(azimuth_sorted_sim_2)
    toa_array = times_sorted_sim
    plt.scatter(toa_array * 1E9, azimuth_sorted_sim_2,
                c=paths_power,
                marker='o', cmap='Blues', s=20,
                vmin=vmin, vmax=vmax, label='ground')

    cbar = plt.colorbar()

    # plot sim
    paths_power = np.power(np.abs(paths_refined_amplitude_array), 2)
    paths_power = 10 * np.log10(paths_power / np.amax(np.nan_to_num(paths_power)))  # dB
    toa_array = paths_refined_toa_array  # - paths_refined_toa_array[0]
    plt.scatter(toa_array * 1E9, paths_refined_aoa_array,
                c=paths_power,
                marker='x', cmap='Reds', s=20,
                vmin=vmin, vmax=vmax, label='mdTrack')

    cbar.ax.set_ylabel('power [dB]', rotation=90)
    plt.xlabel('ToA [ns]')
    plt.ylabel('AoA [deg]')
    plt.ylim([-90, 90])
    plt.yticks(np.arange(-90, 91, 20))
    plt.grid()
    plt.legend(prop={'size': 10})
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('nss', help='Number of spatial streams', type=int)
    parser.add_argument('ncore', help='Number of cores', type=int)
    parser.add_argument('exp_dir', help='Name base of the directory')
    parser.add_argument('name_base', help='Name base of the simulation')
    parser.add_argument('--delta_t', help='Delta ToA for grid search in multiples of 10^-11', default=50, type=int, required=False)
    args = parser.parse_args()

    exp_dir = args.exp_dir
    name_base = args.name_base  # simulation

    delta_t = np.round(args.delta_t * 1e-11, 11)  # 5E-10  # 1.25e-11
    save_dir = '../results/mdTrack' + str(delta_t) + '/'
    if os.path.exists(save_dir + 'paths_list_' + name_base + '.txt'):
        print('Already processed')
        exit()

    num_ant = args.nss * args.ncore

    name_file = exp_dir + 'CFR_' + name_base + '.mat'
    csi_buff = sio.loadmat(name_file)
    signal_complete = (csi_buff['CFR'])

    name_file = exp_dir + 'delay_' + name_base + '.mat'
    csi_buff = sio.loadmat(name_file)
    delays_sim = (csi_buff['propagation_delays'])

    name_file = exp_dir + 'aoa_' + name_base + '.mat'
    csi_buff = sio.loadmat(name_file)
    aoa_sim = (csi_buff['propagation_aoa'])

    name_file = exp_dir + 'path_loss_' + name_base + '.mat'
    try:
        csi_buff = sio.loadmat(name_file)
        path_loss_sim = (csi_buff['propagation_path_loss'])
    except FileNotFoundError:
        path_loss_sim = None

    F_frequency = 256
    delta_f = 312.5E3
    frequency_vector_idx = np.arange(F_frequency)
    frequency_vector_hz = delta_f * (frequency_vector_idx - F_frequency / 2)
    # control_subcarriers = [0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255]
    # pilot_subcarriers = [25, 53, 89, 117, 139, 167, 203, 231]
    delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, 203, 231, 251, 252, 253,
                              254, 255], dtype=int)
    frequency_vector_idx = np.delete(frequency_vector_idx, delete_idxs, axis=0)
    frequency_vector_hz = np.delete(frequency_vector_hz, delete_idxs, axis=0)

    frequency_vector_hz = frequency_vector_hz

    H_complete_valid = signal_complete

    T = 3.2e-6
    range_considered = 1e-8
    idxs_range_considered = int(range_considered/delta_t + 1)
    t_min = 0
    t_max = T / 3  # T/2

    num_angles = 360
    num_paths = 100
    num_subc = frequency_vector_idx.shape[0]
    ToA_matrix, time_vector = build_toa_matrix(frequency_vector_hz, delta_t, t_min, t_max)
    AoA_matrix, angles_vector, cos_ant_vector = build_aoa_matrix(num_angles, num_ant)
    AoA_matrix_reshaped = np.reshape(AoA_matrix, (AoA_matrix.shape[0], -1))

    num_time_steps = aoa_sim.shape[1]

    # mD-track 2D: remove offsets CFO, PDD, SFO
    paths_list = []
    paths_amplitude_list = []
    paths_toa_list = []
    paths_aoa_list = []
    optimization_times = np.zeros(num_time_steps)

    num_iteration_refinement = 10
    threshold = -2.5

    for time_idx in range(0, num_time_steps):
        # time_start = time.time()
        cfr_sample = H_complete_valid[time_idx, :, :]

        # coarse estimation
        matrix_cfr_toa = np.dot(ToA_matrix, cfr_sample)
        power_matrix_cfr_toa = np.sum(np.abs(matrix_cfr_toa), 1)
        time_idx_max = np.argmax(power_matrix_cfr_toa)
        time_max = time_vector[time_idx_max]
        index_start_toa = int(max(0, time_idx_max - idxs_range_considered))
        index_end_toa = int(min(time_vector.shape[0], time_idx_max + idxs_range_considered))
        ToA_matrix_considered = ToA_matrix[index_start_toa:index_end_toa, :]
        time_vector_considered = time_vector[index_start_toa:index_end_toa]

        #####
        # GROUND TRUTH SIMULATION
        if path_loss_sim is not None:
            sorted_idx_sim = np.argsort(abs(path_loss_sim[0, time_idx]))[0, :]
        else:
            sorted_idx_sim = np.arange(2)

        azimuth_sorted_sim = (aoa_sim[0, time_idx][0, sorted_idx_sim])
        if aoa_sim.shape[0] > 1:
            elevation_sorted_sim = (aoa_sim[0, time_idx][1, sorted_idx_sim])
            azimuth_sorted_sim_2 = np.arcsin(np.sin(azimuth_sorted_sim / 180 * mt.pi)
                                             * np.cos(elevation_sorted_sim / 180 * mt.pi)) * 180 / mt.pi
        else:
            azimuth_sorted_sim_2 = np.copy(np.asarray(azimuth_sorted_sim, dtype='int16'))

        az_positive = azimuth_sorted_sim_2 > 0
        az_negative = azimuth_sorted_sim_2 < 0
        azimuth_sorted_sim_2[az_positive] -= 180
        azimuth_sorted_sim_2[az_negative] += 180

        swap_idx_pos = azimuth_sorted_sim_2 > 90
        swap_idx_neg = azimuth_sorted_sim_2 < -90
        azimuth_sorted_sim_2[swap_idx_pos] = 180 - azimuth_sorted_sim_2[swap_idx_pos]
        azimuth_sorted_sim_2[swap_idx_neg] = - 180 - azimuth_sorted_sim_2[swap_idx_neg]

        times_sorted_sim = delays_sim[0, time_idx][:, sorted_idx_sim]
        if path_loss_sim is not None:
            path_loss_sorted_sim = path_loss_sim[0, time_idx][:, sorted_idx_sim]
        else:
            path_loss_sorted_sim = None

        cfr_simulation_time = np.exp(1j * 2 * mt.pi * np.expand_dims(frequency_vector_hz, axis=-1) * times_sorted_sim)
        cfr_simulation_aoa = np.exp(1j * mt.pi * np.dot(np.expand_dims(np.linspace(0, 3, 4), axis=-1),
                                                        np.expand_dims(azimuth_sorted_sim_2, axis=0)))
        cfr_simulation_time = np.stack([cfr_simulation_time]*num_ant, axis=0)
        cfr_simulation_time = np.moveaxis(cfr_simulation_time, [0, 1, 2], [1, 0, 2])
        cfr_simulation_aoa = np.stack([cfr_simulation_aoa]*num_subc, axis=0)
        if path_loss_sim is not None:
            cfr_simulation_ampl = np.stack([np.stack([1/np.power(20, path_loss_sorted_sim[0, :]/10)]*num_ant, axis=0)]*num_subc,
                                           axis=0)
        else:
            cfr_simulation_ampl = 1
        cfr_simulation = np.sum(cfr_simulation_ampl*cfr_simulation_aoa*cfr_simulation_time, axis=2)
        #####

        #####
        # MULTI-PATH PARAMETERS ESTIMATION
        start = time.time()

        paths, paths_refined_amplitude, paths_refined_toa_idx, paths_refined_aoa_idx = md_track_2d(
            cfr_sample, AoA_matrix, ToA_matrix_considered, num_ant, num_subc, num_angles, num_iteration_refinement,
            threshold)
        end = time.time()
        optimization_times[time_idx] = end-start

        paths_refined_aoa = angles_vector[paths_refined_aoa_idx] * 180 / mt.pi
        paths_refined_toa = time_vector_considered[paths_refined_toa_idx]
        paths_refined_amplitude_array = np.asarray(paths_refined_amplitude)
        paths_refined_aoa_array = np.asarray(paths_refined_aoa)
        paths_refined_toa_array = np.asarray(paths_refined_toa)

        paths_list.append(paths)
        paths_amplitude_list.append(paths_refined_amplitude_array)
        paths_aoa_list.append(paths_refined_aoa_array)
        paths_toa_list.append(paths_refined_toa_array)
        #####

    # Saving results
    save_name = save_dir + 'opr_sim_' + name_base + '.txt'  # + '.npz'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with open(save_name, "wb") as fp:  # Pickling
        pickle.dump(optimization_times, fp)

    name_file = save_dir + 'paths_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_list, fp)
    name_file = save_dir + 'paths_amplitude_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_amplitude_list, fp)
    name_file = save_dir + 'paths_aoa_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_aoa_list, fp)
    name_file = save_dir + 'paths_toa_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_toa_list, fp)

    # plot_combined(paths_refined_amplitude_array, paths_refined_toa_array, paths_refined_aoa_array,
    #               path_loss_sorted_sim, times_sorted_sim, azimuth_sorted_sim_2)
