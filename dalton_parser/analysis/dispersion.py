"""Dispersion analysis from Dalton output."""

import re
import sys

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
