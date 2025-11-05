"""Dispersion analysis from Dalton output. All code here is currently work in progress."""

import re
import sys

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

# Weights include 1/2pi or hbar/2pi, so should be multiplied by 6 for getting isotropic?
# See SI of https://doi.org/10.1039/C7CP02399E for factor of 6
# Looks like it's dipole-dipole specific?

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

DEBUG = True


def dispersion_testing(
    full_response: np.ndarray,
    operator_to_idx: dict,
    coord_dict: list[dict],
    atomic_moment_order: int,
) -> None:
    """Perform dispersion testing on the full response matrix.

    Args:
        full_response (np.ndarray): Response matrix of n_labels by n_labels by n_freq
        operator_to_idx (dict): Mapping of operator labels to their indices
        coord_dict (list[dict]): List of coordinate dictionaries for each atom
        atomic_moment_order (int): Order of atomic moments to consider
    """
    # Define dimensions
    n_tot_labels = len(operator_to_idx)
    n_freq = len(WEIGHT_LIST)

    if DEBUG:
        print("Dimensions of response input:", full_response.shape)

    # Integrate C coefficients, currently for one molecule
    integrated_data = integrate_c6(full_response, n_tot_labels, n_freq)

    if DEBUG:
        print("Dimension of 4-index integrated data:", integrated_data.shape)

    # Reshape index
    # "old" is the "wrong" way, which gives good results, but doesn't simplify disp. calculations
    # "new" is the "right" way, which gives a worse description
    reshaped_data_old = integrated_data.transpose(0, 1, 2, 3).reshape(n_tot_labels**2, n_tot_labels**2)
    reshaped_data_new = integrated_data.transpose(0, 2, 1, 3).reshape(n_tot_labels**2, n_tot_labels**2)

    if DEBUG:
        print("Dimension of reshaped data (old):", reshaped_data_old.shape)
        print("Dimension of reshaped data (new):", reshaped_data_new.shape)

    u_old, s_old, vh_old = np.linalg.svd(reshaped_data_old)

    if DEBUG:
        print("SVD U matrix dimensions:", u_old.shape)
        print("SVD Vh matrix dimensions:", vh_old.shape)
        print("SVD U and Vh max difference:", np.max(np.abs(u_old - vh_old.transpose())))
        print("SVD singular values:", s_old)
        print("First singular value in % of total:", s_old[0] / np.sum(s_old) * 100)

    # Test if we can further separate the components as C_ab = C_a * C_b
    # If we can, we can still apply the old way to simplify the description

    approx_data = u_old[:, 0].reshape(n_tot_labels, n_tot_labels)

    if DEBUG:
        print("Dimension of approximated data:", approx_data.shape)

    u_old2, s_old2, vh_old2 = np.linalg.svd(approx_data)

    if DEBUG:
        print("SVD U matrix dimensions (2nd):", u_old2.shape)
        print("SVD Vh matrix dimensions (2nd):", vh_old2.shape)
        print("SVD U and Vh max difference (2nd):", np.max(np.abs(u_old2 - vh_old2.transpose())))
        print("SVD singular values (2nd):", s_old2)
        print("First singular value in % of total (2nd):", s_old2[0] / np.sum(s_old2) * 100)

    u_new, s_new, vh_new = np.linalg.svd(reshaped_data_new)

    if DEBUG:
        print("SVD U matrix dimensions (new):", u_new.shape)
        print("SVD Vh matrix dimensions (new):", vh_new.shape)
        print("SVD U and Vh max difference (new):", np.max(np.abs(u_new - vh_new.transpose())))
        print("SVD singular values (new):", s_new)
        print("First singular value in % of total (new):", s_new[0] / np.sum(s_new) * 100)

    # Create the N by N approximation
    u_old = np.sqrt(s_old[0]) * u_old[:, 0].reshape(n_tot_labels, n_tot_labels)
    above = 0
    below = 0
    thresh = 1e-3
    for i in range(u_old.shape[0]):
        for j in range(u_old.shape[1]):
            if abs(u_old[i, j]) < thresh:
                below += 1
            else:
                above += 1
    if DEBUG:
        print("Total number of values:", above + below)
        print(f"Number of values above {thresh}:", above)
        print("In procentage:", above / (above + below) * 100)
        print(f"Number of values below {thresh}:", below)
        print("In procentage:", below / (above + below) * 100)


def integrate_c6(full_response: np.ndarray, n_tot_labels: int, n_freq: int) -> np.ndarray:
    integrated_data = np.zeros((n_tot_labels, n_tot_labels, n_tot_labels, n_tot_labels))

    for i in range(n_tot_labels):
        for j in range(n_tot_labels):
            for k in range(n_tot_labels):
                for l in range(n_tot_labels):
                    for freq in range(n_freq):
                        integrated_data[i, j, k, l] += (
                            full_response[i, j, freq] * full_response[k, l, freq] * WEIGHT_LIST[freq]
                        )
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
