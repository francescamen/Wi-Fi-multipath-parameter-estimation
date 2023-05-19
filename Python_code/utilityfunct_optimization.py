
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
import osqp
import scipy
from scipy.signal import find_peaks


def convert_to_complex_r(real_im_n):
    len_vect = real_im_n.shape[0] // 2
    complex_n = real_im_n[:len_vect] + 1j * real_im_n[len_vect:]
    return complex_n


def lasso_regression_osqp_fast(H_matrix_, T_matrix_, selected_subcarriers, row_T, col_T, Im, Onm, P, q, A2, A3,
                               ones_n_matr, zeros_n_matr, zeros_nm_matr, delta_t_refined, num_angles, threshold=True):
    # time_start = time.time()
    T_matrix_selected = T_matrix_[selected_subcarriers, :]
    H_matrix_selected = H_matrix_[selected_subcarriers]

    T_matrix_real, H_matrix_real = convert_TH_real(T_matrix_selected, H_matrix_selected)

    n = col_T*2

    # OSQP data
    A = scipy.sparse.vstack([scipy.sparse.hstack([T_matrix_real, -Im, Onm.T]),
                             A2,
                             A3], format='csc')
    l = np.hstack([H_matrix_real, - np.inf * ones_n_matr, zeros_n_matr])
    u = np.hstack([H_matrix_real, zeros_n_matr, np.inf * ones_n_matr])

    # Create an OSQP object
    prob = osqp.OSQP()

    # Setup workspace
    prob.setup(P, q, A, l, u, warm_start=True, verbose=False)

    # Update linear cost
    lambd = 1E-1
    q_new = np.hstack([zeros_nm_matr, lambd * ones_n_matr])
    prob.update(q=q_new)

    # Solve
    res = prob.solve()
    x_out = res.x
    x_out_cut = x_out[:n]

    r_opt = convert_to_complex_r(x_out_cut)
    # plt.figure()
    # plt.stem(x_out_cut)
    # plt.show()
    # plt.figure()
    # plt.stem(abs(r_opt))
    # plt.show()

    # time_end = time.time()
    # print(-time_start+time_end)

    time_diff_thresh = int(1E-9 / delta_t_refined)

    num_rows = mt.floor(r_opt.shape[0] / num_angles)
    if threshold:
        r_tilde, new_s = thresholding_lasso(r_opt, 2, num_rows, time_diff_thresh=time_diff_thresh,
                                            angle_diff_thresh=5//(180/num_angles),
                                            peaks_det=True, power_factor=1000)
        r_opt = r_tilde
    return r_opt


def thresholding_lasso(vector_, s, num_rows, time_diff_thresh, angle_diff_thresh, power_thresh=0,
                       peaks_det=True, power_factor=1000):
    vector_ = np.nan_to_num(vector_)
    if peaks_det:
        peaks = find_peaks(np.abs(vector_), distance=200)[0]

        if peaks.shape[0] > 0:
            sort_idxs = np.flip(np.argsort(np.abs(vector_[peaks])))
            sort_idxs = peaks[sort_idxs]
            end_idx = peaks.shape[0]
        else:
            sort_idxs = np.flip(np.argsort(np.abs(vector_)))
            end_idx = 10
    else:
        sort_idxs = np.flip(np.argsort(np.abs(vector_)))
        end_idx = 200

    first_s_idxs = sort_idxs  # [:s]
    sorted_pow = np.abs(vector_[first_s_idxs]) ** 2

    first_s_idxs_times = first_s_idxs % num_rows
    first_s_idxs_angles = first_s_idxs // num_rows
    list_retain_idxs = [0]

    # vect = np.reshape(vector_, (-1, num_angles), order='F')
    # plt.figure()
    # plt.pcolormesh(np.abs(vect.T))

    # plt.figure()
    # plt.scatter(peaks, np.abs(vector_[peaks]))
    # plt.show()
    # plt.figure()
    # plt.scatter(first_s_idxs_times, first_s_idxs_angles)
    # plt.show()

    pow_max = max(power_thresh, sorted_pow[0])
    for idx in range(1, end_idx):
        if len(list_retain_idxs) > s:
            break
        insert = True
        pow_idx = sorted_pow[idx]
        if pow_idx < pow_max / power_factor:
            break
        time_idx = first_s_idxs_times[idx]
        angle_idx = first_s_idxs_angles[idx]
        for idx_ret in list_retain_idxs:
            time_diff = time_idx - first_s_idxs_times[idx_ret]
            angle_diff = abs(angle_idx - first_s_idxs_angles[idx_ret])
            angle_diff = np.minimum(angle_diff, 180 - angle_diff)
            if (time_diff < time_diff_thresh and angle_diff < angle_diff_thresh) or time_diff == 0:
                insert = False
                break
        if insert:
            list_retain_idxs.append(idx)
    first_s_idxs = first_s_idxs[list_retain_idxs]

    vector_thresh = np.zeros_like(vector_)
    vector_thresh[first_s_idxs] = vector_[first_s_idxs]
    new_s = first_s_idxs.shape[0]
    return vector_thresh, new_s


def convert_TH_real(T_matrix_comp, H_comp):

    row_T, col_T = np.shape(T_matrix_comp)

    T_matrix_real = np.zeros((2*row_T, 2*col_T))
    T_matrix_real[:row_T, :col_T] = np.real(T_matrix_comp)
    T_matrix_real[row_T:, col_T:] = np.real(T_matrix_comp)
    T_matrix_real[row_T:, :col_T] = np.imag(T_matrix_comp)
    T_matrix_real[:row_T, col_T:] = - np.imag(T_matrix_comp)

    H_real = np.zeros((2*row_T))
    H_real[:row_T] = np.real(H_comp)
    H_real[row_T:] = np.imag(H_comp)

    return T_matrix_real, H_real


def hard_thresholding_operator_base(vector_, s):
    sort_idxs = np.flip(np.argsort(np.abs(vector_)))
    zeros_idxs = sort_idxs[s:]
    vector_thresh = np.copy(vector_)
    vector_thresh[zeros_idxs] = 0
    return vector_thresh, s


def hard_thresholding_operator(vector_, s, num_rows, time_diff_thresh, angle_diff_thresh, power_thresh=0,
                               first_iteration=False, peaks_det=False, power_factor=400):
    vector_ = np.nan_to_num(vector_)
    if peaks_det:
        peaks = find_peaks(np.abs(vector_), distance=200)[0]
        # peaks_ref = find_peaks(np.abs(vector_[peaks]))[0]
        # peaks = peaks[peaks_ref]
        if peaks.shape[0] > 0:
            sort_idxs = np.flip(np.argsort(np.abs(vector_[peaks])))
            sort_idxs = peaks[sort_idxs]
            end_idx = peaks.shape[0]
        else:
            sort_idxs = np.flip(np.argsort(np.abs(vector_)))
            end_idx = 10
    else:
        sort_idxs = np.flip(np.argsort(np.abs(vector_)))
        end_idx = 200

    first_s_idxs = sort_idxs  # [:s]
    sorted_pow = np.abs(vector_[first_s_idxs])**2

    first_s_idxs_times = first_s_idxs % num_rows
    first_s_idxs_angles = first_s_idxs // num_rows
    list_retain_idxs = [0]

    # plt.figure()
    # plt.scatter(peaks, np.abs(vector_[peaks]))
    # plt.show()
    # plt.figure()
    # plt.scatter(first_s_idxs_times, first_s_idxs_angles)
    # plt.show()

    if not first_iteration:
        pow_max = max(power_thresh, sorted_pow[0])
        for idx in range(1, end_idx):
            if len(list_retain_idxs) > s:
                break
            insert = True
            pow_idx = sorted_pow[idx]
            if pow_idx < pow_max/power_factor:
                break
            time_idx = first_s_idxs_times[idx]
            angle_idx = first_s_idxs_angles[idx]
            for idx_ret in list_retain_idxs:
                time_diff = time_idx - first_s_idxs_times[idx_ret]
                angle_diff = abs(angle_idx - first_s_idxs_angles[idx_ret])
                angle_diff = np.minimum(angle_diff, 180 - angle_diff)
                if (time_diff < time_diff_thresh and angle_diff < angle_diff_thresh) or time_diff == 0:
                    insert = False
                    break
            if insert:
                list_retain_idxs.append(idx)
    first_s_idxs = first_s_idxs[list_retain_idxs]

    # plt.figure()
    # plt.scatter(first_s_idxs % num_rows, first_s_idxs // num_rows)
    # plt.show()

    # plt.figure()
    # plt.plot(np.abs(vector_))
    # plt.scatter(first_s_idxs, np.abs(vector_[first_s_idxs]), c='r', s=50)
    # plt.show()

    vector_thresh = np.zeros_like(vector_)
    vector_thresh[first_s_idxs] = vector_[first_s_idxs]
    new_s = first_s_idxs.shape[0]

    return vector_thresh, new_s


def support_computation(vector):
    support_elem = np.sort(np.nonzero(vector), axis=0)[0, :]
    vect_support = vector[support_elem]
    return vect_support, support_elem


def norm2_square(vector):
    norm2_vect = np.abs(np.dot(vector.conj().T, vector))
    return norm2_vect


def norm2(vector):
    norm2_vect = np.sqrt(norm2_square(vector))
    return norm2_vect


def compute_step_condition(r_tilde, r_vector, T_matrix, c):
    diff_r = abs(r_tilde - r_vector)
    num_condition = norm2_square(diff_r)
    den_condition = norm2_square(np.dot(T_matrix, diff_r))
    step_condition = (1 - c) * num_condition / den_condition
    return step_condition


def iht_algorithm_complete(T_matrix, T_matrix_times, T_matrix_angles, H_matrix, num_rows, step_time, num_angles, s,
                           refinement=True, priors_aoa=None, priors_toa=None, search_range_aoa=None,
                           search_range_toa=None):
    if priors_aoa is not None:
        T_matrix_angles_refinement = np.zeros_like(T_matrix_angles)
        T_matrix_times_refinement = np.zeros_like(T_matrix_times)
        T_matrix_refinement = np.zeros_like(T_matrix)
        for i_ref in range(len(priors_toa)):
            time_i = priors_toa[i_ref]
            time_idxs_start = max(0, int(time_i - search_range_toa))
            time_idxs_end = min(int(time_i + search_range_toa), num_rows)
            T_matrix_times_refinement[:, time_idxs_start:time_idxs_end] = T_matrix_times[:,
                                                                          time_idxs_start:time_idxs_end]
            angle_i = priors_aoa[i_ref]
            angle_idxs_start = max(0, int(angle_i - search_range_aoa))
            angle_idxs_end = min(int(angle_i + search_range_aoa), num_angles)
            T_matrix_angles_refinement[angle_idxs_start:angle_idxs_end, :] = \
                T_matrix_angles[angle_idxs_start:angle_idxs_end, :]
            mask = np.zeros((num_rows, num_angles), dtype=bool)
            mask[time_idxs_start:time_idxs_end, angle_idxs_start:angle_idxs_end] = 1
            mask = mask.flatten(order='F')
            T_matrix_refinement[:, mask > 0] = T_matrix[:, mask > 0]
        T_matrix_angles = T_matrix_angles_refinement
        T_matrix_times = T_matrix_times_refinement
        T_matrix = T_matrix_refinement

    r_vector = np.zeros((T_matrix.shape[1], ), dtype=complex)
    factor1 = H_matrix
    product1_time = np.dot(T_matrix_times.conj().T, H_matrix)
    product1 = np.dot(T_matrix.conj().T, H_matrix.flatten(order='F'))

    time_diff_thresh = int(1E-9 / step_time)
    first_iteration = True
    hard_thresh, new_s = hard_thresholding_operator(product1, s, num_rows, time_diff_thresh=time_diff_thresh,
                                                    angle_diff_thresh=1//(180/num_angles),
                                                    first_iteration=first_iteration)
    _, support_elem = support_computation(hard_thresh)

    c = 1E-3
    kappa = 1 / (1 - c) + 5E-1
    upd_den = kappa * (1 - c)

    stop = False
    i = 0
    while not stop:
        i += 1
        r_vector_old = np.copy(r_vector)
        r_vector, factor1, support_elem, Tr_reshape = iht_inner_loop(H_matrix, T_matrix, T_matrix_times,
                                                                     T_matrix_angles, r_vector, factor1,
                                                                     support_elem, s, c, num_rows, step_time,
                                                                     num_angles, upd_den,
                                                                     time_diff_thresh=time_diff_thresh,
                                                                     first_iteration=first_iteration)
        stop_condition = norm2(r_vector_old - r_vector)
        # plt.figure()
        # plt.plot(abs(factor1[:, 2]))
        # plt.show()
        if stop_condition < 1E-5 or i >= 100:
            stop = True

        first_iteration = False

    r_vector_not_ref = np.copy(r_vector)

    # REFINEMENT
    if refinement:

        vect_support = np.abs(r_vector[support_elem]) ** 2
        if vect_support.shape[0] > 0:
            pow_max = np.amax(vect_support)
        else:
            pow_max = 0
        new_s = s + 1
        r_vector_not_ref = np.copy(r_vector)
        r_vector_add = np.zeros_like(r_vector)
        while True:
            new_s -= 1
            if new_s < 1:
                break
            r_support = r_vector[support_elem]

            idxs_times = support_elem % num_rows
            sort_idx_r = np.argsort(idxs_times)

            try:
                time_min = idxs_times[sort_idx_r[0]]
            except IndexError:
                break

            idx_s = np.argwhere(idxs_times[sort_idx_r] == time_min)[:, 0]
            idx_s_sort = np.flip(np.argsort(np.abs(r_support[sort_idx_r[idx_s]])))
            idx_s = idx_s[idx_s_sort[0]]
            abs_r_path = abs(r_support[sort_idx_r[idx_s]]) ** 2
            idx_r = support_elem[sort_idx_r[idx_s]]
            if (abs_r_path < pow_max/400).all():
                break

            r_vector_path = r_vector[idx_r]
            T_matrix_path = T_matrix[:, idx_r]
            Tr = np.dot(T_matrix_path, r_vector_path)
            Tr_reshape_path = np.reshape(Tr, H_matrix.shape, order='F')
            H_matrix = H_matrix - Tr_reshape_path
            r_vector_cut = np.copy(r_vector)
            r_vector_cut[idx_r] = 0
            r_vector_add[idx_r] = r_vector[idx_r]

            factor1 = H_matrix
            product1_time = np.dot(T_matrix_times.conj().T, H_matrix)
            product1_angles = np.dot(product1_time, T_matrix_angles.conj().T)
            product1 = product1_angles.flatten(order='F')
            # product1_control = np.dot(T_matrix.conj().T, H_matrix.flatten(order='F'))
            hard_thresh, _ = hard_thresholding_operator(product1, new_s, num_rows,
                                                        time_diff_thresh=time_diff_thresh,
                                                        angle_diff_thresh=2//(180/num_angles),
                                                        power_thresh=pow_max, peaks_det=True)
            _, support_elem = support_computation(hard_thresh)

            i = 0
            stop = False
            first_iteration = True
            r_vector = np.copy(r_vector_cut)
            while not stop:
                i += 1
                r_vector_old = np.copy(r_vector)
                r_vector, factor1, support_elem, Tr_reshape = iht_inner_loop\
                    (H_matrix, T_matrix, T_matrix_times, T_matrix_angles, r_vector, factor1, support_elem,
                     new_s, c, num_rows, step_time, num_angles, upd_den, time_diff_thresh=time_diff_thresh,
                     pow_max=pow_max, first_iteration=first_iteration, peaks_det=True)
                stop_condition = norm2(r_vector_old - r_vector)  # np.sum(norm2(factor1))
                # plt.figure()
                # plt.plot(abs(factor1[:, 2]))
                # plt.show()
                first_iteration = False
                if stop_condition < 1E-5 or i >= 10:
                    stop = True
        r_vector = r_vector + r_vector_add

    return r_vector_not_ref, r_vector


def iht_inner_loop(H_matrix, T_matrix, T_matrix_times, T_matrix_angles, r_vector, factor1, support_elem, s, c,
                   num_rows, step_time, num_angles, upd_den, advanced_thresholding=True,
                   time_diff_thresh=20, pow_max=0, first_iteration=False, peaks_det=True):
    g_i_time = np.dot(T_matrix_times.conj().T, factor1)
    g_i_angles = np.dot(g_i_time, T_matrix_angles.conj().T)
    g_i = g_i_angles.flatten(order='F')
    T_matrix_support = T_matrix[:, support_elem]
    g_i_support = g_i[support_elem]
    factor_den_support = np.dot(T_matrix_support, g_i_support)
    step = norm2_square(g_i_support) / norm2_square(factor_den_support)

    if step == mt.inf:
        r_vector_non_zero = r_vector[support_elem]
        T_matrix_non_zero = T_matrix[:, support_elem]
        Tr = np.dot(T_matrix_non_zero, r_vector_non_zero)
        Tr_reshape = np.reshape(Tr, H_matrix.shape, order='F')
        factor1 = H_matrix - Tr_reshape
        return r_vector, factor1, support_elem, Tr_reshape
    factor = r_vector + step * g_i
    if advanced_thresholding:
        r_tilde, new_s = hard_thresholding_operator(factor, s, num_rows, time_diff_thresh=time_diff_thresh,
                                                    angle_diff_thresh=2//(180/num_angles), power_thresh=pow_max,
                                                    first_iteration=first_iteration, peaks_det=peaks_det)
    else:
        r_tilde, new_s = hard_thresholding_operator_base(factor, s)

    # plt.figure()
    # plt.plot(20*np.log10(np.abs(g_i)))
    # plt.show()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.stem(abs(r_tilde))
    # ax2.plot(abs(step * g_i))
    # plt.show()

    # plt.figure()
    # plot_r = np.log10(abs(np.reshape(r_vector, (-1, num_angles), order='F')).T)
    # plot_r[plot_r == -mt.inf] = -5
    # plt.pcolormesh(plot_r)
    # plt.yticks(np.arange(0, num_angles + 1, num_angles/10), np.arange(-90, 91, 180//10))
    # plt.xticks(np.arange(0, num_rows+1, num_rows/8), np.round(np.arange(0, num_rows+1, num_rows/8)*step_time*1E9, 2))
    # plt.xlabel('ToA [ns]')
    # plt.ylabel('AoA [ns]')
    # plt.colorbar()
    # plt.show()
    # plt.figure()
    # plt.pcolormesh(abs(np.reshape(step * g_i, (-1, num_angles), order='F')).T)
    # plt.yticks(np.arange(0, num_angles + 1, num_angles/10), np.arange(-90, 91, 180//10))
    # plt.xticks(np.arange(0, num_rows+1, num_rows/8), np.round(np.arange(0, num_rows+1, num_rows/8)*step_time*1E9, 2))
    # plt.xlabel('ToA [ns]')
    # plt.ylabel('AoA [ns]')
    # plt.colorbar()
    # plt.show()
    # plt.figure()
    # plt.pcolormesh(abs(np.reshape(factor, (-1, num_angles), order='F')).T)
    # plt.yticks(np.arange(0, num_angles + 1, num_angles/10), np.arange(-90, 91, 180//10))
    # plt.xticks(np.arange(0, num_rows+1, num_rows/8), np.round(np.arange(0, num_rows+1, num_rows/8)*step_time*1E9, 2))
    # plt.xlabel('ToA [ns]')
    # plt.ylabel('AoA [ns]')
    # plt.colorbar()
    # plt.show()

    support_elem_prev = np.copy(support_elem)
    _, support_elem = support_computation(r_tilde)
    if support_elem_prev.shape[0] == support_elem.shape[0] and np.sum(np.equal(support_elem_prev, support_elem)):
        r_vector = np.copy(r_tilde)
    else:
        step_condition = 1 * compute_step_condition(r_tilde, r_vector, T_matrix, c)
        if step <= step_condition:
            r_vector = np.copy(r_tilde)
        else:
            while step > step_condition:
                step = step / upd_den
                factor = r_vector + 1 * step * g_i
                if advanced_thresholding:
                    r_tilde, new_s = hard_thresholding_operator(factor, s, num_rows,
                                                                time_diff_thresh=time_diff_thresh,
                                                                angle_diff_thresh=2//(180/num_angles), power_thresh=pow_max,
                                                                peaks_det=peaks_det)
                else:
                    r_tilde, new_s = hard_thresholding_operator_base(factor, s)

                step_condition = 1 * compute_step_condition(r_tilde, r_vector, T_matrix, c)
            _, support_elem = support_computation(r_tilde)
            r_vector = np.copy(r_tilde)

    r_vector_non_zero = r_vector[support_elem]
    T_matrix_non_zero = T_matrix[:, support_elem]
    Tr = np.dot(T_matrix_non_zero, r_vector_non_zero)
    Tr_reshape = np.reshape(Tr, H_matrix.shape, order='F')
    factor1 = H_matrix - Tr_reshape

    return r_vector, factor1, support_elem, Tr_reshape
