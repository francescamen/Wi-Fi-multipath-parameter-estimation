
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

import argparse
from utilityfunct_optimization import *
import pickle
import scipy.io as sio
import utilityfunct_optimization_routines as opr
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('start_r', help='Start processing', type=int)
    parser.add_argument('end_r', help='End processing', type=int)
    parser.add_argument('step_r', help='Step of sample processing', type=int)
    parser.add_argument('opt_method', help='Optimization routine: lasso, omp, iht, iht_noref')
    parser.add_argument('exp_dir', help='Name base of the directory')
    parser.add_argument('name_base', help='Name base of the simulation')
    args = parser.parse_args()

    exp_dir = args.exp_dir
    name_base = args.name_base  # simulation

    save_dir = '../results/' + args.opt_method + '/'
    save_name = save_dir + 'paths_aoa_list_' + name_base + '.txt'

    if os.path.exists(save_name):
        print('Already processed')
        exit()

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

    start_r = args.start_r
    if args.end_r != -1:
        end_r = args.end_r
    else:
        end_r = signal_complete.shape[0]
        step_r = args.step_r
    step_r = args.step_r

    method = args.opt_method
    if method == 'iht' or method == 'iht_noref' or method == 'lasso':
        optimize_aoa_toa = opr.opt_methods_dict[method]
    elif method == 'omp':
        optimize_aoa_toa = opr.opt_methods_dict_ext[method]

    # Optimization parameters
    delta_t = 1E-9
    range_refined_up = 1E-8
    range_refined_down = 1E-8

    n_tot = 4

    F_frequency = 256
    delta_f = 312.5E3

    T = 1 / delta_f  # 3.2E-6
    t_min = 0  # - T / 3  # + delta_t
    t_max = T / 3  # - delta_t

    # Signal selection
    delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 25, 53, 89, 117, 127, 128, 129, 139, 167, 203, 231, 251, 252, 253,
                              254, 255], dtype=int)
    frequency_vector_idx = np.arange(F_frequency)
    frequency_vector_complete = delta_f * (frequency_vector_idx - F_frequency / 2)

    frequency_vector_idx = np.delete(frequency_vector_idx, delete_idxs, axis=0)
    frequency_vector = np.delete(frequency_vector_complete, delete_idxs, axis=0)

    frequency_vector = frequency_vector

    sample_selector = range(start_r, end_r, step_r)
    signal_considered = signal_complete[sample_selector, :, :]
    if path_loss_sim is not None:
        path_loss_sim_considered = path_loss_sim[:, sample_selector]
    else:
        path_loss_sim_considered = None
    delays_sim_considered = delays_sim[:, sample_selector]
    aoa_sim_considered = aoa_sim[:, sample_selector]

    # Optimization run
    if method == 'iht':
        num_paths = 2
        delta_t_refined = 5E-10
        num_angles = 360
        r_optim, r_optim_refined, r_end_points, optimization_times, paths_amplitude_list, paths_toa_list, \
            paths_aoa_list = \
            optimize_aoa_toa(signal_considered, frequency_vector, t_min, t_max, delta_t, delta_t_refined,
                             range_refined_down, range_refined_up, num_angles, n_tot, num_paths, refinement=True,
                             use_prior=False, path_loss_sim=path_loss_sim_considered, delays_sim=delays_sim_considered,
                             aoa_sim=aoa_sim_considered)
    elif method == 'iht_noref':
        num_paths = 2
        delta_t_refined = 5E-10
        num_angles = 360
        r_optim, r_optim_refined, r_end_points, optimization_times, paths_amplitude_list, paths_toa_list, \
            paths_aoa_list = \
            optimize_aoa_toa(signal_considered, frequency_vector, t_min, t_max, delta_t, delta_t_refined,
                             range_refined_down, range_refined_up, num_angles, n_tot, num_paths, refinement=False,
                             use_prior=False, path_loss_sim=path_loss_sim_considered, delays_sim=delays_sim_considered,
                             aoa_sim=aoa_sim_considered)
    elif method == 'lasso':
        delta_t_refined = 1E-9
        num_angles = 180
        r_optim, r_optim_refined, r_end_points, optimization_times, paths_amplitude_list, paths_toa_list, \
            paths_aoa_list = \
            optimize_aoa_toa(signal_considered, frequency_vector, t_min, t_max, delta_t, delta_t_refined,
                             range_refined_down, range_refined_up, num_angles, n_tot,
                             path_loss_sim=path_loss_sim_considered, delays_sim=delays_sim_considered,
                             aoa_sim=aoa_sim_considered)
    elif method == 'omp':
        delta_t_refined = 5E-10
        num_angles = 360
        r_optim, r_optim_refined, r_end_points, optimization_times, paths_amplitude_list, paths_toa_list, \
            paths_aoa_list = \
            optimize_aoa_toa(signal_considered, frequency_vector, t_min, t_max, delta_t, delta_t_refined,
                             range_refined_down, range_refined_up, num_angles, n_tot, n_nonzero_coefs=2,
                             path_loss_sim=path_loss_sim_considered, delays_sim=delays_sim_considered,
                             aoa_sim=aoa_sim_considered)

    # Saving results
    save_name = save_dir + 'opr_sim_' + name_base + '.npz'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_dct = dict(r_optim=r_optim, end_points=r_end_points, opt_times=optimization_times)
    np.savez(save_name, **save_dct)

    name_file = save_dir + 'paths_amplitude_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_amplitude_list, fp)
    name_file = save_dir + 'paths_aoa_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_aoa_list, fp)
    name_file = save_dir + 'paths_toa_list_' + name_base + '.txt'
    with open(name_file, "wb") as fp:  # Pickling
        pickle.dump(paths_toa_list, fp)
