
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

import numpy as np
import math as mt


def build_aoa_matrix(num_angles_, num_antennas):
    angles_vector_ = np.linspace(-mt.pi/2 * np.ones(num_antennas), mt.pi/2 * np.ones(num_antennas), num_angles_)
    sin_vector = np.sin(angles_vector_)
    sin_ant_vector_ = sin_vector * np.arange(0, num_antennas, 1)
    vector_exp = np.transpose(sin_ant_vector_)
    aoa_matrix_ = np.exp(1j * mt.pi * vector_exp)
    return aoa_matrix_, angles_vector_[:, 0], sin_ant_vector_


def build_toa_matrix(frequency_vector_hz, delta_t_, t_min_, t_max_):
    l_paths = int(np.round((t_max_ - t_min_) / delta_t_))
    time_vector_ = np.linspace(t_min_, t_max_, l_paths)
    time_freq_matrix = np.expand_dims(time_vector_, -1) * np.expand_dims(frequency_vector_hz, 0)
    toa_matrix_ = np.exp(1j * 2 * mt.pi * time_freq_matrix)
    return toa_matrix_, time_vector_


def build_toa_aoa_matrix(frequency_vector_hz, delta_t_, t_min_, t_max_, num_angles_, num_antennas):
    l_paths = int(np.round((t_max_ - t_min_) / delta_t_))
    time_vector_ = np.linspace(t_min_, t_max_, l_paths)
    angles_vector_ = np.linspace(-mt.pi/2, mt.pi/2, num_angles_)
    toa_coord, aoa_coord = np.meshgrid(time_vector_, angles_vector_)

    toa_coord_vector = np.reshape(toa_coord, (-1, 1))
    time_freq_matrix = toa_coord_vector * np.expand_dims(frequency_vector_hz, 0)
    toa_matrix_ = 2 * time_freq_matrix

    aoa_coord_vector = np.reshape(aoa_coord, (-1, 1))
    sin_vector = np.sin(aoa_coord_vector)
    sin_ant_vector_ = sin_vector * np.arange(0, num_antennas, 1)
    aoa_matrix_ = np.transpose(sin_ant_vector_)

    toa_aoa_combinations = aoa_matrix_.shape[1]
    num_freq = frequency_vector_hz.shape[0]
    Tmatrix = np.zeros((num_freq * num_antennas, toa_aoa_combinations), dtype=complex)
    for a_i in range(num_antennas):
        start_idx = a_i * num_freq
        end_idx = (a_i + 1) * num_freq
        Tmatrix[start_idx:end_idx, :] = aoa_matrix_[a_i:a_i + 1, :] + toa_matrix_.T
    Tmatrix = np.exp(1j * mt.pi * Tmatrix)

    return Tmatrix, time_vector_, angles_vector_


def build_dop_matrix(n_pkt, tc, step):
    freq_max = 1 / tc
    num_freq = int(n_pkt / step)
    grid_freq = np.expand_dims(np.linspace(- freq_max / 2, freq_max / 2, num_freq), -1)
    grid_packets = np.tile(np.expand_dims(np.arange(n_pkt), -1), grid_freq.shape[0])
    dop_matrix = np.exp(1j * 2 * mt.pi * tc * grid_packets * grid_freq.T)
    return dop_matrix, grid_freq
