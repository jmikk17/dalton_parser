"""Dispersion analysis from Dalton output."""

import re
import sys

from dalton_parser.utils.helpers import print_non_zero_matrix_elements

import numpy as np
from scipy.interpolate import pade

FREQ_SQ_LIST = [
    0.0000000e00,
    -4.3700000e-05,
    -1.3086000e-03,
    -9.1102000e-03,
    -3.9063200e-02,
    -1.3720890e-01,
    -4.5550980e-01,
    -1.5999700e00,
    -6.8604430e00,
    -4.7760340e01,
    -1.4306370e03,
]

WEIGHT_LIST = [
    0.000000000,
    0.002723378,
    0.006838052,
    0.012362954,
    0.020857905,
    0.035634293,
    0.064927079,
    0.133488123,
    0.339261344,
    1.306351932,
    15.5845986,
]


def integrate_c6(full_response: np.ndarray, operator_to_idx: dict, coord_dict: list[dict], atomic_moment_order: int):
    """Numerically integrate C6 data from full response matrix using set weights from WEIGHT_LIST.

    C6 coeficients are defined as C_ABCD= int(0 to infty) alpha_AB(omega)*alpha_CD(omega) domega

    Currently setup to use the same response matrix for both alphas
    """
    n_atoms = len(coord_dict)
    n_labels = len(operator_to_idx)
    integrated_data = np.zeros((n_labels, n_labels, n_labels, n_labels))

    print("Order of labels:", list(operator_to_idx.keys()))

    print_non_zero_matrix_elements(full_response[:, :, 0], "0 freq. response")

    n_freq = len(WEIGHT_LIST)

    for i in range(n_labels):
        for j in range(n_labels):
            for k in range(n_labels):
                for l in range(n_labels):
                    for freq in range(n_freq):
                        integrated_data[i, j, k, l] += (
                            full_response[i, j, freq] * full_response[k, l, freq] * WEIGHT_LIST[freq]
                        )

    print("Example of 0,0,0,x:")
    print("For water this is oxygen charge on 3 first indices, and all on the last:")
    print(integrated_data[0, 0, 0, :])

    # Pick out charge values for testing, i.e. the first three indices:

    # assumes charge terms are the first indices

    charge_C6 = integrated_data[0:n_atoms, 0:n_atoms, 0:n_atoms, 0:n_atoms]

    if atomic_moment_order >= 1:
        dipole_C6 = integrated_data[n_atoms:, n_atoms:, n_atoms:, n_atoms:]
        print(np.shape(dipole_C6))

    # print("Charge C6 values:")
    # print(charge_C6)

    # Collapse indices

    charge_collapsed = np.reshape(charge_C6, [n_atoms * n_atoms, n_atoms * n_atoms])
    if atomic_moment_order >= 1:
        dipole_collapsed = np.reshape(dipole_C6, [3 * n_atoms * 3 * n_atoms, 3 * n_atoms * 3 * n_atoms])

    # print("Charge C6 collapsed:")
    # print(charge_collapsed)

    # Get svd approximation

    u, s, vh = np.linalg.svd(charge_collapsed)
    print("SVD singular values for charge:")
    print(s)
    print("First singular value in % of total:", s[0] / np.sum(s) * 100)

    # print("SVD U matrix:")
    # print(u)
    # print("SVD Vh matrix:")
    # print(vh)

    # Test if U and V are the same. Test each element difference to 10^-5
    # for i in range(np.shape(u)[0]):
    #    for j in range(np.shape(u)[1]):
    #        assert abs(u[i, j] - vh[j, i]) < 1e-5

    # svd for dipole

    if atomic_moment_order >= 1:
        u_d, s_d, vh_d = np.linalg.svd(dipole_collapsed)
        print("SVD singular values for dipole:")
        print(s_d)
        print("First singular value in % of total:", s_d[0] / np.sum(s_d) * 100)

    # pick out first values

    u1 = u[:, 0]
    s1 = s[0]
    vh1 = vh[0, :]

    if atomic_moment_order >= 1:
        u1_d = u_d[:, 0]
        u2_d = u_d[:, 1]
        u3_d = u_d[:, 2]
        u4_d = u_d[:, 3]
        s1_d = s_d[0]
        vh1_d = vh_d[0, :]

    # approximation

    u1 = np.reshape(u1, [n_atoms * n_atoms, 1])
    vh1 = np.reshape(vh1, [1, n_atoms * n_atoms])
    print(np.shape(u1), np.shape(vh1))

    approx = s1 * np.outer(u1, vh1)

    if atomic_moment_order >= 1:
        u1_d = np.reshape(u1_d, [3 * n_atoms * 3 * n_atoms, 1])
        vh1_d = np.reshape(vh1_d, [1, 3 * n_atoms * 3 * n_atoms])
        print(np.shape(u1_d), np.shape(vh1_d))
        approx_d = s1_d * np.outer(u1_d, vh1_d)
        print("Approximation (dipole):")
        print(approx_d)

    # evaluate approximation

    diff_matrix = charge_collapsed - approx
    # print("Difference matrix (charge):")
    # print(diff_matrix)
    print("Max difference from charge svd:", np.max(np.abs(diff_matrix)))

    if atomic_moment_order >= 1:
        diff_matrix_d = dipole_collapsed - approx_d
        print("Difference matrix (dipole):")
        print(diff_matrix_d)
        print("Max difference from dipole svd:", np.max(np.abs(diff_matrix_d)))
        # max difference in % value:

    print(np.reshape(vh1, [n_atoms, n_atoms]))

    u1 = np.reshape(u1, [n_atoms, n_atoms])
    vh1 = np.reshape(vh1, [n_atoms, n_atoms])

    if atomic_moment_order >= 1:
        u1_d = np.reshape(u1_d, [3 * n_atoms, 3 * n_atoms])
        vh1_d = np.reshape(vh1_d, [3 * n_atoms, 3 * n_atoms])
        print(np.shape(u1_d), np.shape(vh1_d))

        # test for including more SVD values
        u2_d = np.reshape(u2_d, [3 * n_atoms, 3 * n_atoms])
        u3_d = np.reshape(u3_d, [3 * n_atoms, 3 * n_atoms])
        u4_d = np.reshape(u4_d, [3 * n_atoms, 3 * n_atoms])
        print(np.shape(u2_d), np.shape(u3_d), np.shape(u4_d))

    # Calculate real distance using coord dict:
    R_AB = np.zeros((n_atoms, n_atoms))
    coords = np.zeros((n_atoms, 3))
    for i, dict_i in enumerate(coord_dict):
        for j, dict_j in enumerate(coord_dict):
            coords[i, :] = np.array([dict_i["x"], dict_i["y"], dict_i["z"]])
            coords[j, :] = np.array([dict_j["x"], dict_j["y"], dict_j["z"]])
            R_AB[i, j] = np.linalg.norm(coords[i, :] - coords[j, :])

    print("Interatomic distance matrix R_AB:")
    print(R_AB)

    e_disp = 0.0
    e_disp_d = 0.0

    # Calculate dispersion energy using full 4-body terms for reference
    # Current "screening" is 1 for a=b and 0 otherwise

    for i in range(n_atoms):
        for j in range(n_atoms):
            for k in range(n_atoms):
                for l in range(n_atoms):
                    if R_AB[i, j] == 0 or R_AB[k, l] == 0:
                        continue
                    else:
                        e_disp += charge_C6[i, j, k, l] / (R_AB[i, j] * R_AB[k, l])

    # Same for dipole interaction
    if atomic_moment_order >= 1:
        T_ij = np.zeros((3, 3, n_atoms, n_atoms))
        for i in range(n_atoms):
            for j in range(n_atoms):
                if i == j:
                    continue
                for alpha in range(3):
                    for beta in range(3):
                        delta_term = (1.0 if alpha == beta else 0.0) / R_AB[i, j] ** 3
                        aniso_term = (
                            -3.0
                            * (coords[i, alpha] - coords[j, alpha])
                            * (coords[i, beta] - coords[j, beta])
                            / R_AB[i, j] ** 5
                        )
                        T_ij[alpha, beta, i, j] = delta_term + aniso_term

        for i in range(3 * n_atoms):
            for j in range(3 * n_atoms):
                for k in range(3 * n_atoms):
                    for l in range(3 * n_atoms):
                        atom_i = i // 3
                        atom_j = j // 3
                        atom_k = k // 3
                        atom_l = l // 3
                        if R_AB[atom_i, atom_j] == 0 or R_AB[atom_k, atom_l] == 0:
                            continue
                        else:
                            e_disp_d += (
                                dipole_C6[i, j, k, l]
                                * T_ij[i % 3, j % 3, atom_i, atom_j]
                                * T_ij[k % 3, l % 3, atom_k, atom_l]
                            )

    # Calculate the AB sum
    sum_AB = 0.0
    for i in range(n_atoms):
        for j in range(n_atoms):
            if R_AB[i, j] != 0:
                sum_AB += u1[i, j] / R_AB[i, j]
    # Calculate the CD sum
    sum_CD = 0.0
    for k in range(n_atoms):
        for l in range(n_atoms):
            if R_AB[k, l] != 0:
                sum_CD += vh1[k, l] / R_AB[k, l]

    e_disp_approx1 = s1 * sum_AB * sum_CD

    # extra approximation
    e_disp_approx2 = 0.0
    for i in range(n_atoms):
        for j in range(n_atoms):
            if R_AB[i, j] != 0:
                e_disp_approx2 += np.sqrt(s1) * u1[i, j] / R_AB[i, j]

    print("E_disp:", e_disp)
    print("E_disp_approx:", e_disp_approx1)
    print("E_disp_approx2:", e_disp_approx2**2)

    if atomic_moment_order >= 1:
        sum_AB = 0.0
        for i in range(3 * n_atoms):
            for j in range(3 * n_atoms):
                atom_i = i // 3
                atom_j = j // 3
                if R_AB[atom_i, atom_j] == 0:
                    continue
                sum_AB += T_ij[i % 3, j % 3, atom_i, atom_j] * u1_d[i, j]

        sum_CD = 0.0
        for k in range(3 * n_atoms):
            for l in range(3 * n_atoms):
                atom_k = k // 3
                atom_l = l // 3
                if R_AB[atom_k, atom_l] == 0:
                    continue
                sum_CD += T_ij[k % 3, l % 3, atom_k, atom_l] * vh1_d[k, l]

        e_disp_approx1_d = s1_d * sum_AB * sum_CD
        e_disp_approx2_d = np.zeros((4))
        for i in range(3 * n_atoms):
            for j in range(3 * n_atoms):
                atom_i = i // 3
                atom_j = j // 3
                if R_AB[atom_i, atom_j] == 0:
                    continue
                # e_disp_approx2_d += np.sqrt(s1_d) * T_ij[i % 3, j % 3, atom_i, atom_j] * u1_d[i, j]
                e_disp_approx2_d[0] += np.sqrt(s_d[0]) * T_ij[i % 3, j % 3, atom_i, atom_j] * u1_d[i, j]
                e_disp_approx2_d[1] += np.sqrt(s_d[1]) * T_ij[i % 3, j % 3, atom_i, atom_j] * u2_d[i, j]
                e_disp_approx2_d[2] += np.sqrt(s_d[2]) * T_ij[i % 3, j % 3, atom_i, atom_j] * u3_d[i, j]
                e_disp_approx2_d[3] += np.sqrt(s_d[3]) * T_ij[i % 3, j % 3, atom_i, atom_j] * u4_d[i, j]

        for i in range(4):
            e_disp_approx2_d[i] = e_disp_approx2_d[i] * e_disp_approx2_d[i]

        print("E_disp (dipole):", e_disp_d)
        print("E_disp_approx (dipole):", e_disp_approx1_d)
        print("E_disp_approx2 (dipole):", e_disp_approx2_d.sum())

        for i in range(4):
            tmp = 0.0
            for j in range(i + 1):
                tmp += e_disp_approx2_d[j]
            print("E_disp_approx2_d rank", i, ":", tmp)

    return integrated_data


def pade_approx(content: str) -> dict:
    """Extract Cauchy moments from the Dalton output, calculate alpha(i omega) using Pade approximations."""
    block_pattern = (
        r"\s*(AM\w+)\s+(AM\w+)\s+(-?\d+)\s+([-\d.Ee+]+)\s*\n"
        r"((?:\s+-?\d+\s+[-\d.Ee+]+\s*\n)+)"
    )

    for match in re.finditer(block_pattern, content):
        # The first line contains both labels and values, the rest is only data for Cauchy number n and value D_AB
        # Assumes coefficients starts from D(-4)=S(+2), and we need to start from D(0)=S(-2), D(n) = S(-n-2)
        operator1 = match.group(1)
        operator2 = match.group(2)
        _ = int(match.group(3))
        _ = float(match.group(4))
        data_block = match.group(5)

        n_list = []
        d_ab_list = []
        first = True

        for line in data_block.strip().splitlines():
            if first:
                first = False
                continue
            n, d_ab = line.split()
            n_list.append(int(n))
            d_ab_list.append(float(d_ab))

        print("Operator 1:", operator1, "Operator 2:", operator2)
        print(f"n_list: {n_list}")
        print(f"d_ab_list: {d_ab_list}")

        k = 4

        scale = max(abs(x) for x in d_ab_list) or 1.0
        d_ab_scaled = [x / scale for x in d_ab_list]

        if len(n_list) < (k * 2 + 1):
            sys.exit(f"Not enough moments for Pade, need {k * 2 + 1} moments, got {len(n_list)}")
        p_low, q_low = pade(d_ab_scaled, n=k, m=(k - 1))
        p_high, q_high = pade(d_ab_scaled, n=k, m=k)

        for i in range(10):
            z_value = FREQ_SQ_LIST[i]

            pade_result_low = scale * p_low(z_value) / q_low(z_value)
            pade_result_high = scale * p_high(z_value) / q_high(z_value)

            normal_expansion = (
                d_ab_list[0] * z_value**0
                + d_ab_list[1] * z_value**1
                + d_ab_list[2] * z_value**2
                + d_ab_list[3] * z_value**3
                + d_ab_list[4] * z_value**4
            )
            print(
                "Omega sq:",
                FREQ_SQ_LIST[i],
                "Normal power series:",
                normal_expansion,
                "Pade approximation:",
                pade_result_low,
                pade_result_high,
            )

    sys.exit("Pade approximation not implemented yet")
