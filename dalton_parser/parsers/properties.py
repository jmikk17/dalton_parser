"""Parser for properties from Dalton output."""

import re
import sys

import numpy as np
import pandas as pd

from dalton_parser.utils.helpers import get_label

# C6 xyz labels following the Orient format
XYZ_TO_SPHERICAL = {
    "00": 0,  # 00
    "0z": 1,  # 10
    "0x": 2,  # 11c
    "0y": 3,  # 11s
}

ORIENT_ORDER = ["00", "0z", "0x", "0y"]


def extract_2nd_order_prop(content: str, wave_function: str, atomic_moment_order: int) -> dict:
    """Extract second order properties from Dalton output.

    Args:
        content (str): Content of the Dalton output file
        wave_function (str): The type of wave function used in the calculation, used for regex pattern selection
        atomic_moment_order (int): The atomic moment order of the calculation

    Returns:
        dict: Dictionary of the format {label: value} for each property

    """
    properties_00 = {}
    properties_0b = {}
    properties_ab = {}
    properties_a0 = {}

    if wave_function == "CC":
        pattern = (
            r"(\w+)\s+\(unrel\.\)\s+[-+]?[\d\.]+\s+(\w+)\s+\(unrel\.\)\s+[-+]?[\d\.]+\s+([-+]?\d+\.\d+(?:E[-+]?\d+)?)"
        )
    else:
        pattern = r"@\s*-<<\s*(\w+)\s*;\s*(\w+)\s*>>\s*=\s*([-+]?\d+\.\d+[Ee][-+]?\d+)"

    for match in re.finditer(pattern, content):
        operator1 = match.group(1)
        operator2 = match.group(2)
        value = match.group(3)

        index1, _, xyz_comp1 = get_label(operator1)
        index2, _, xyz_comp2 = get_label(operator2)

        if xyz_comp1 == "00" and xyz_comp2 == "00":
            key = f"{index1}_{index2}"
            properties_00[key] = -float(value)
        if xyz_comp1 != "00" and xyz_comp2 == "00":
            key = f"{index1}:{xyz_comp1}_{index2}"
            properties_a0[key] = -float(value)
        if xyz_comp1 == "00" and xyz_comp2 != "00":
            key = f"{index1}_{index2}:{xyz_comp2}"
            properties_0b[key] = -float(value)
        if xyz_comp1 != "00" and xyz_comp2 != "00":
            key = f"{index1}:{xyz_comp1}_{index2}:{xyz_comp2}"
            properties_ab[key] = -float(value)

    if atomic_moment_order == 0:
        return {"00": properties_00}
    if wave_function == "CC":
        return {"00": properties_00, "0b": properties_0b, "a0": properties_a0, "ab": properties_ab}
    return {"00": properties_00, "0b": properties_0b, "ab": properties_ab}


def extract_1st_order_prop(content: str) -> list:
    """Extract MBIS charges from the Dalton output.

    Args:
        content (str): Content of the Dalton output file

    Returns:
        list: MBIS charges

    """
    mbis_pattern = r"MBIS converged!.*?Final converged results\s+Qatom(.*?)(?:\n\s*\n|\Z)"
    mbis_match = re.search(mbis_pattern, content, re.DOTALL)

    charge_list = []
    if mbis_match:
        charge_section = mbis_match.group(1)
        charge_pattern = r"\s+(\d+)\s+([-+]?\d+\.\d+)"
        charge_list.extend(
            [float(charge_match.group(2)) for charge_match in re.finditer(charge_pattern, charge_section)],
        )
    return charge_list


def read_2nd_order_prop(content: dict) -> pd.DataFrame:
    """Read the second order properties from dictionary into DataFrame with info from labels.

    Args:
        content (dict): Dictionary of parsed content from Dalton output file.

    Returns:
        pd.DataFrame: Dataframe with atm index, charge, xyz comp. and value.

    """
    parsed_data = []
    method = content["wave_function"]
    property_dict = content["2nd_order_properties"]
    for dict_type, inner_dict in property_dict.items():
        # loop over 00, 0b, a0, ab dicts
        for key, value in inner_dict.items():
            label_list = key.split("_")
            if dict_type == "00":
                index1 = label_list[0]
                index2 = label_list[1]
                xyz_comp1 = "00"
                xyz_comp2 = "00"
            if dict_type == "0b":
                index1 = label_list[0]
                xyz_comp1 = "00"
                idx_split = label_list[1].split(":")
                index2 = idx_split[0]
                xyz_comp2 = idx_split[1]
            if dict_type == "a0":
                index2 = label_list[1]
                xyz_comp2 = "00"
                idx_split = label_list[0].split(":")
                index1 = idx_split[0]
                xyz_comp1 = idx_split[1]
            if dict_type == "ab":
                idx_split1 = label_list[0].split(":")
                index1 = idx_split1[0]
                xyz_comp1 = idx_split1[1]
                idx_split2 = label_list[1].split(":")
                index2 = idx_split2[0]
                xyz_comp2 = idx_split2[1]

            parsed_data.append(
                {
                    "index1": int(index1),
                    "xyz1": xyz_comp1,
                    "index2": int(index2),
                    "xyz2": xyz_comp2,
                    "value": value,
                },
            )
            if method != "CC" and label_list[0] != label_list[1]:
                # CC prints all values, for other WF we manually add the other triangle of the matrix
                parsed_data.append(
                    {
                        "index2": int(index1),
                        "xyz2": xyz_comp1,
                        "index1": int(index2),
                        "xyz1": xyz_comp2,
                        "value": value,
                    },
                )

    return pd.DataFrame(parsed_data)


def extract_imaginary(content: str, atomic_moment_order: int, atoms: int, n_freq: int) -> dict:
    """Extract alpha(i omega) from the Dalton output.

    Args:
        content (str): Content of the Dalton output file
        atomic_moment_order (int): Atomic moment order
        atoms (int): Number of atoms
        n_freq (int): Number of frequencies to extract

    Returns:
        dict: Dictionary of alpha(i omega) values. Outer keys are the labels of the atom pairs, inner keys are the
              frequencies, and values are the corresponding alpha(i omega) values.

    Todo:
        - Logic for parsing needs to be double checked.
        - The current implementation assumes 4x4 matrices for each pair+frequency,
          but this should be generalized using atomic_moment_order
        - Clean up, function way too long

    """
    results = {}

    imaginary_pattern = (
        r"(AM\w+)\s+(AM\w+)\s+([-]?\d+\.\d+)\n\s+GRIDSQ\s+ALPHA\n((?:\s+[-]?\d+\.\d+\s+[-]?\d+\.\d+\n){11})"
    )

    labels_per_atom = 0
    for i in range(atomic_moment_order + 1):
        labels_per_atom += 2 * i + 1

    tot_labels = labels_per_atom * atoms
    full_response = np.zeros((tot_labels, tot_labels, n_freq), dtype=float)

    operator_to_idx = {}
    current_idx = 0

    # Collect all unique operators from Dalton
    for match in re.finditer(imaginary_pattern, content):
        op1, op2 = match.group(1), match.group(2)
        if op1 not in operator_to_idx:
            operator_to_idx[op1] = current_idx
            current_idx += 1
        if op2 not in operator_to_idx:
            operator_to_idx[op2] = current_idx
            current_idx += 1

    if len(operator_to_idx) != tot_labels:
        sys.exit(f"Error: Expected {tot_labels} unique operators, found {len(operator_to_idx)}.")

    # Fill the triangular matrix following Dalton output structure
    frequencies = []
    for match in re.finditer(imaginary_pattern, content):
        operator1 = match.group(1)
        operator2 = match.group(2)
        data_block = match.group(4)

        full_idx1 = operator_to_idx[operator1]
        full_idx2 = operator_to_idx[operator2]

        data_lines = data_block.strip().split("\n")
        for freq_idx, line in enumerate(data_lines):
            parts = line.strip().split()
            gridsq = -float(parts[0])
            alpha = float(parts[1])

            # Store frequency value on first pass
            if len(frequencies) <= freq_idx:
                frequencies.append(gridsq)

            full_response[full_idx1, full_idx2, freq_idx] = alpha

    # Fill out other triangle of the matrix
    for freq_idx in range(n_freq):
        for i in range(tot_labels):
            for j in range(tot_labels):
                if full_response[i, j, freq_idx] != 0 and full_response[j, i, freq_idx] == 0:
                    full_response[j, i, freq_idx] = full_response[i, j, freq_idx]

    # Create atom-index to dalton-index mapping based on Dalton ordering
    # This is used to pick out 4x4 submatrices for specific atom pairs
    atom_to_indices = {}
    for op, dalton_idx in operator_to_idx.items():
        atom_idx, _, component = get_label(op)
        if atom_idx not in atom_to_indices:
            atom_to_indices[atom_idx] = []
        atom_to_indices[atom_idx].append((dalton_idx, component))

    # Sort each atom's indices by spherical harmonic order used in Orient
    for atom_idx in atom_to_indices:
        atom_to_indices[atom_idx].sort(key=lambda x: ORIENT_ORDER.index(x[1]))

    # Extract submatrices for each atom pair
    results = {}

    for atom1 in range(1, atoms + 1):
        for atom2 in range(atom1, atoms + 1):
            pair_key = f"{atom1}_{atom2}"
            results[pair_key] = {}

            # Get Dalton indices for each atom (in component order)
            atom1_indices = [idx for idx, _ in atom_to_indices[atom1]]
            atom2_indices = [idx for idx, _ in atom_to_indices[atom2]]

            # Extract submatrix for each frequency
            for freq_idx in range(n_freq):
                freq = frequencies[freq_idx]

                # Build 4x4 submatrix by picking specific indices from full matrix
                submatrix = np.zeros((labels_per_atom, labels_per_atom))
                for i, dalton_idx1 in enumerate(atom1_indices):
                    for j, dalton_idx2 in enumerate(atom2_indices):
                        submatrix[i, j] = full_response[dalton_idx1, dalton_idx2, freq_idx]

                results[pair_key][freq] = submatrix

    return results
