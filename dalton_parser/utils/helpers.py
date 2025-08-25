"""Helper utility functions."""

import numpy as np


def get_label(label: str) -> tuple[int, int, str]:
    """Get the atom info from the property label.

    Args:
        label (str): Property label

    Returns:
        list: Atom info in the format [index, nuc_charge, component]

    """
    index = int(label[2:4])
    nuc_charge = int(label[4:6])
    component = label[6:]
    return index, nuc_charge, component


def print_non_zero_matrix_elements(matrix: np.ndarray, name: str = "") -> None:
    """Sketch non-zero elements in a 2D matrix.

    Args:
        matrix (np.ndarray): Input 2D matrix

    """
    print("Non-zero matrix elements of " + name + " (X = non-zero, 0 = zero):")
    for i in range(matrix.shape[0]):
        row = ""
        for j in range(matrix.shape[1]):
            if matrix[i, j] != 0:
                row += "X "
            else:
                row += "0 "
        print(row)
