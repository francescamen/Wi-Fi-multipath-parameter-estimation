
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
import pickle
from matplotlib.ticker import AutoMinorLocator
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import math as mt

rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['text.usetex'] = 'true'
rcParams['text.latex.preamble'] = [r'\usepackage{newtxmath}']
rcParams['axes.linewidth'] = 0.5

cmap = 'viridis'
alpha = 0.9


def plot_labels(axi, angles, space, ylabel=True, fontsize=14):
    if ylabel:
        axi.set_ylabel(r'AoA diff. [deg]', fontsize=fontsize)
    axi.set_xlabel(r'ToA diff. [ns]', fontsize=fontsize)
    num_angles = angles.shape[0]
    range_angles = angles[-1] - angles[0]

    axi.set_yticks(np.arange(0, range_angles+1, range_angles/(num_angles-1)))
    axi.set_yticklabels(angles, fontsize=fontsize)
    # axi.set_ylim([1, 182])
    axi.set_xticks(np.arange(0, no_points_ToA, space) + 0.5)
    axi.set_xticklabels(np.around(np.arange((start_ToA-ToA_first_path)*1e9,
                        (end_ToA-ToA_first_path)*1e9, space*step_ToA*1e9), 1), fontsize=fontsize)
    axi.set_title(opt_method_name, fontsize=fontsize-1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('start_ToA', help='Start for ToA', type=float)
    parser.add_argument('end_ToA', help='End for Toa', type=float)
    parser.add_argument('step_ToA', help='Step for ToA', type=float)
    parser.add_argument('step_AoA', help='Step for AoA', type=float)
    parser.add_argument('--num_AoA', help='Number of AoA', default=180, type=int, required=False)
    args = parser.parse_args()

    ToA_first_path = 1e-8
    start_ToA = args.start_ToA
    end_ToA = args.end_ToA
    step_ToA = args.step_ToA
    step_AoA = args.step_AoA
    num_AoA = args.num_AoA

    ToA_array = np.around(np.arange(start_ToA, end_ToA+step_ToA, step_ToA), 10)
    no_points_ToA = ToA_array.shape[0]
    no_points_AoA = mt.floor(num_AoA / step_AoA)

    num_rows = 3
    num_columns = 2

    fig_dist, ax_dist = plt.subplots(num_rows, num_columns, constrained_layout=True)
    fig_dist.set_size_inches(6, 7)
    axs_dist = []

    fig_toa_first, ax_toa_first = plt.subplots(num_rows, num_columns, constrained_layout=True)
    fig_toa_first.set_size_inches(3, 5)
    axs_toa_first = []
    toa_min_first = 0
    toa_max_first = 3.5

    fig_toa_second, ax_toa_second = plt.subplots(num_rows, num_columns, constrained_layout=True)
    fig_toa_second.set_size_inches(3, 5)
    axs_toa_second = []
    toa_min_second = 0
    toa_max_second = 4.5

    fig_aoa_first, ax_aoa_first = plt.subplots(num_rows, num_columns, constrained_layout=True)
    fig_aoa_first.set_size_inches(3, 5)
    axs_aoa_first = []
    aoa_min_first = 0
    aoa_max_first = 16

    fig_aoa_second, ax_aoa_second = plt.subplots(num_rows, num_columns, constrained_layout=True)
    fig_aoa_second.set_size_inches(3, 5)
    axs_aoa_second = []
    aoa_min_second = 0
    aoa_max_second = 30

    fig_time, ax_time = plt.subplots(num_rows, num_columns, constrained_layout=True)
    fig_time.set_size_inches(6, 7)
    axs_time = []
    min_time = 1
    max_time = 2

    methods = ['mdTrack', 'spotfi', 'ubilocate', 'omp', 'iht_noref', 'iht']
    method_names = [r'\textbf{mD-Track}', r'\textbf{SpotFi}', r'\textbf{UbiLocate}', r'\textbf{OMP}',
                    r'\textbf{IHT}', r'\textbf{IHT enhanced}']
    for i_method, opt_method in enumerate(methods):
        opt_method_name = method_names[i_method]
        row_idx = i_method // num_columns
        col_idx = i_method % num_columns
        save_dir = '../results/processed_ok_paper/processed_' + opt_method + '/'
        execution_times_avg = np.zeros((no_points_ToA, no_points_AoA))
        matrix_plot_find_first = np.zeros((no_points_ToA, no_points_AoA))
        matrix_plot_find_second = np.zeros((no_points_ToA, no_points_AoA))
        matrix_plot_find_distinguish = np.zeros((no_points_ToA, no_points_AoA))
        matrix_plot_toa_err_first = np.zeros((no_points_ToA, no_points_AoA))
        matrix_plot_toa_err_second = np.zeros((no_points_ToA, no_points_AoA))
        matrix_plot_aoa_err_first = np.zeros((no_points_ToA, no_points_AoA))
        matrix_plot_aoa_err_second = np.zeros((no_points_ToA, no_points_AoA))

        for ToA_idx in range(no_points_ToA):
            ToA_value = ToA_array[ToA_idx]

            name_file = save_dir + 'path_find_avg_simulation_artificial_delayobst_' + str(ToA_value) + '.txt'
            with open(name_file, "rb") as fp:  # Pickling
                path_find_avg = pickle.load(fp)
            name_file = save_dir + 'computing_time_avg_simulation_artificial_delayobst_' + str(ToA_value) + '.txt'
            with open(name_file, "rb") as fp:  # Pickling
                execution_times_avg[ToA_idx, :] = np.squeeze(pickle.load(fp))

            name_file = save_dir + 'estim_toa_error_avg_simulation_artificial_delayobst_' + str(ToA_value) + '.txt'
            with open(name_file, "rb") as fp:  # Pickling
                estim_toa_error_avg = pickle.load(fp)
            name_file = save_dir + 'estim_aoa_error_avg_simulation_artificial_delayobst_' + str(ToA_value) + '.txt'
            with open(name_file, "rb") as fp:  # Pickling
                estim_aoa_error_avg = pickle.load(fp)

            path_idx = 0
            matrix_plot_find_first[ToA_idx, :] = path_find_avg[:, path_idx]
            matrix_plot_toa_err_first[ToA_idx, :] = np.nan_to_num(estim_toa_error_avg[:, path_idx], nan=5e-9)
            matrix_plot_aoa_err_first[ToA_idx, :] = np.nan_to_num(estim_aoa_error_avg[:, path_idx], nan=5e-9)

            path_idx = 1
            matrix_plot_find_second[ToA_idx, :] = path_find_avg[:, path_idx]
            matrix_plot_toa_err_second[ToA_idx, :] = np.nan_to_num(estim_toa_error_avg[:, path_idx], nan=20)
            matrix_plot_aoa_err_second[ToA_idx, :] = np.nan_to_num(estim_aoa_error_avg[:, path_idx], nan=20)

            matrix_plot_find_distinguish[ToA_idx, :] = path_find_avg[:, -1]

            np.argwhere(np.sum(path_find_avg, axis=1) > 1)[:, 0]

        fontsize = 14
        plt.rcParams.update({'font.size': fontsize})
        space = 8
        angles = np.arange(-90, 91, 30)

        # distinguish
        matrix_plot_find_distinguish_compact = (matrix_plot_find_distinguish[:, 90:] +
                                                np.flip(matrix_plot_find_distinguish[:, :90], axis=1))/2
        axi = ax_dist[row_idx, col_idx]
        plt1_dist = axi.pcolormesh(matrix_plot_find_distinguish.T, cmap=cmap+'_r', linewidth=0, rasterized=True, alpha=alpha)
        plot_labels(axi, angles, space, fontsize=fontsize)
        axi.yaxis.set_minor_locator(AutoMinorLocator(3))
        axi.xaxis.set_minor_locator(AutoMinorLocator(3))
        axs_dist.append(axi)

        fontsize = 13
        plt.rcParams.update({'font.size': fontsize})
        space = 16
        angles = np.arange(-90, 91, 45)

        # toa first
        matrix_plot_toa_err_first_compact = (matrix_plot_toa_err_first[:, 90:] +
                                             np.flip(matrix_plot_toa_err_first[:, :90], axis=1))/2
        axi = ax_toa_first[row_idx, col_idx]
        plt1_toa_first = axi.pcolormesh(matrix_plot_toa_err_first.T*1e9, cmap=cmap, linewidth=0, rasterized=True,
                                        vmin=toa_min_first, vmax=toa_max_first, alpha=alpha)
        plot_labels(axi, angles, space, ylabel=True, fontsize=fontsize)
        axi.yaxis.set_minor_locator(AutoMinorLocator(2))
        axi.xaxis.set_minor_locator(AutoMinorLocator(3))
        axs_toa_first.append(axi)

        # toa second
        matrix_plot_toa_err_second_compact = (matrix_plot_toa_err_second[:, 90:] +
                                              np.flip(matrix_plot_toa_err_second[:, :90], axis=1))/2
        axi = ax_toa_second[row_idx, col_idx]
        plt1_toa_second = axi.pcolormesh(matrix_plot_toa_err_second.T*1e9, cmap=cmap, linewidth=0, rasterized=True,
                                         vmin=toa_min_second, vmax=toa_max_second, alpha=alpha)
        plot_labels(axi, angles, space, ylabel=False, fontsize=fontsize)
        axi.yaxis.set_minor_locator(AutoMinorLocator(2))
        axi.xaxis.set_minor_locator(AutoMinorLocator(3))
        axs_toa_second.append(axi)

        # aoa first
        matrix_plot_aoa_err_first_compact = (matrix_plot_aoa_err_first[:, 90:] +
                                             np.flip(matrix_plot_aoa_err_first[:, :90], axis=1))/2
        axi = ax_aoa_first[row_idx, col_idx]
        plt1_aoa_first = axi.pcolormesh(matrix_plot_aoa_err_first.T, cmap=cmap, linewidth=0, rasterized=True,
                                        vmin=aoa_min_first, vmax=aoa_max_first, alpha=alpha)
        plot_labels(axi, angles, space, ylabel=True, fontsize=fontsize)
        axi.yaxis.set_minor_locator(AutoMinorLocator(2))
        axi.xaxis.set_minor_locator(AutoMinorLocator(3))
        axs_aoa_first.append(axi)

        # aoa second
        matrix_plot_aoa_err_second_compact = (matrix_plot_aoa_err_second[:, 90:] +
                                              np.flip(matrix_plot_aoa_err_second[:, :90], axis=1))/2
        axi = ax_aoa_second[row_idx, col_idx]
        plt1_aoa_second = axi.pcolormesh(matrix_plot_aoa_err_second.T, cmap=cmap, linewidth=0, rasterized=True,
                                         vmin=aoa_min_second, vmax=aoa_max_second, alpha=alpha)
        plot_labels(axi, angles, space, ylabel=False, fontsize=fontsize)
        axi.yaxis.set_minor_locator(AutoMinorLocator(2))
        axi.xaxis.set_minor_locator(AutoMinorLocator(3))
        axs_aoa_second.append(axi)

        # processing time
        axi = ax_time[row_idx, col_idx]
        plt1_time = axi.pcolormesh(execution_times_avg.T, cmap=cmap, linewidth=0, rasterized=True,
                                   vmin=min_time, vmax=max_time, alpha=alpha)
        plot_labels(axi, angles, space)
        axs_time.append(axi)
        print(opt_method, np.mean(execution_times_avg))

    # distinguish
    tick_font_size = 13
    cbar_dist = fig_dist.colorbar(plt1_dist, ax=ax_dist, fraction=.05, aspect=40)
    cbar_dist.ax.tick_params(labelsize=tick_font_size)
    for axi in axs_dist:
        axi.label_outer()
    # fig_dist.show()
    name_file = './plots/distinguish.png'
    fig_dist.savefig(name_file, bbox_inches='tight')
    plt.close()

    tick_font_size = 12
    # toa first
    cbar_toa_first = fig_toa_first.colorbar(plt1_toa_first, ax=ax_toa_first, fraction=.1, aspect=20, location="bottom")
    cbar_toa_first.ax.tick_params(labelsize=tick_font_size)
    for axi in axs_toa_first:
        axi.label_outer()
    # fig_toa_first.show()
    name_file = './plots/error_toa_first.pdf'
    fig_toa_first.savefig(name_file, bbox_inches='tight')
    plt.close()

    # toa second
    cbar_toa_first = fig_toa_second.colorbar(plt1_toa_second, ax=ax_toa_second, fraction=.1, aspect=20,
                                             location="bottom")
    cbar_toa_first.ax.tick_params(labelsize=tick_font_size)
    for axi in axs_toa_second:
        axi.label_outer()
    # fig_toa_second.show()
    name_file = './plots/error_toa_second.pdf'
    fig_toa_second.savefig(name_file, bbox_inches='tight')
    plt.close()

    # aoa first
    cbar_aoa_first = fig_aoa_first.colorbar(plt1_aoa_first, ax=ax_aoa_first, fraction=.1, aspect=20,
                                            location="bottom")
    cbar_aoa_first.ax.tick_params(labelsize=tick_font_size)
    for axi in axs_aoa_first:
        axi.label_outer()
    # fig_toa_first.show()
    name_file = './plots/error_aoa_first.pdf'
    fig_aoa_first.savefig(name_file, bbox_inches='tight')
    plt.close()

    # aoa second
    cbar_aoa_first = fig_aoa_second.colorbar(plt1_aoa_second, ax=ax_aoa_second, fraction=.1, aspect=20,
                                             location="bottom")
    cbar_aoa_first.ax.tick_params(labelsize=tick_font_size)
    for axi in axs_aoa_second:
        axi.label_outer()
    # fig_toa_second.show()
    name_file = './plots/error_aoa_second.pdf'
    fig_aoa_second.savefig(name_file, bbox_inches='tight')
    plt.close()

    # time
    cbar_time = fig_time.colorbar(plt1_time, ax=ax_time, fraction=.05, aspect=40)
    for axi in axs_time:
        axi.label_outer()
    # fig_toa_second.show()
    # name_file = './plots/processing_time.pdf'
    # fig_time.savefig(name_file, bbox_inches='tight')
    # plt.close()
