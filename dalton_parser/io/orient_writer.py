"""Writer for Orient format C6 data."""

from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

import numpy as np

# Orient format template
ORIENT_HEADER_TEMPLATE = (
    "POL  SITE-LABELS  {site1}  {site2}  SITE-INDICES     {idx1}     {idx2}  "
    "RANK  0 :   {rank}   BY     0 :   {rank}   FREQ2  {freq}  CARTSPHER S"
)

# Matrix dimensions for different atomic moment orders
# Each side runs from charge (0) to quadrupole (2), giving 8x8 total
MATRIX_SIZE = 8

# Active region sizes for different ranks
ACTIVE_REGION_SIZE = {
    0: 1,  # Only [0,0] element
    1: 4,  # Inner 4x4
    2: 8,  # Full 8x8
}

# Threshold for floating point precision issues
FREQENCY_THRESHOLD = 1e-10


def orient_header(site1: str, site2: str, idx1: int, idx2: int, freq_formatted: str, rank: int) -> str:
    """Generate Orient format header for C6 data.

    Args:
        site1 (str): Atom 1 label
        site2 (str): Atom 2 label
        idx1 (int): Atom 1 index
        idx2 (int): Atom 2 index
        freq_formatted (str): Formatted frequency string
        rank (int): Rank value (0, 1, 2, ...)

    Returns:
        str: Header for C6 data in Orient format

    """
    return ORIENT_HEADER_TEMPLATE.format(
        site1=site1,
        site2=site2,
        idx1=idx1,
        idx2=idx2,
        freq=freq_formatted,
        rank=rank,
    )


def get_all_frequencies(c6_dict: dict) -> list[float]:
    """Extract and sort all frequencies in C6 data.

    Args:
        c6_dict (dict): Dictionary containing C6 data

    Returns:
        list[float]: Sorted frequencies in descending order

    """
    all_frequencies = set()
    for freq_data in c6_dict.values():
        all_frequencies.update([float(f) for f in freq_data])
    return sorted(all_frequencies, reverse=True)


def write_matrix_data(file_handle: TextIO, matrix: np.ndarray, rank: int) -> None:
    """Write 8x8 matrix data with active region based on rank.

    Always writes an 8x8 matrix, but only certain regions contain real data:
    - rank 0: Only [0,0] element is non-zero
    - rank 1: Inner 4x4 region is non-zero
    - rank 2: Full 8x8 matrix is non-zero

    Args:
        file_handle: File handle to write to
        matrix: 8x8 matrix data to write
        rank (int): Determines active region size

    """
    active_size = ACTIVE_REGION_SIZE.get(rank, MATRIX_SIZE)

    for i in range(MATRIX_SIZE):
        row = ""
        for j in range(MATRIX_SIZE):
            value_str = f"{matrix[i, j]:.7E}" if i < active_size and j < active_size else "0.0000000E+00"
            # Adjust spacing based on minus sign in the string
            if value_str.startswith("-"):
                row += value_str + "  "
            else:
                row += value_str + "   "

        file_handle.write(f"{row}\n")

    file_handle.write("END\n")


def create_output_files(output_file: str, max_rank: int) -> dict[int, Path]:
    """Create output file paths for different ranks.

    Args:
        output_file (str): Base output file name
        max_rank (int): Maximum rank needed (atomic_moment_order)

    Returns:
        dict[int, Path]: Mapping of rank to file path

    """
    files = {}
    for rank in range(max_rank + 1):
        file_path = Path(str(output_file).replace(".orient", f".orient{rank}"))
        files[rank] = file_path
    return files


def write_c6(c6_dict: dict, labels: list, atomic_moment_order: int, output_file: str = "c6_tmp.txt") -> None:
    """Write the C6 data in Orient format with fixed 8x8 matrices.

    Creates files based on atomic_moment_order with progressive active regions:
    - .orient0: 8x8 matrix with only [0,0] non-zero (charge only)
    - .orient1: 8x8 matrix with 4x4 active region (charge + dipole)
    - .orient2: 8x8 matrix fully active (charge + dipole + quadrupole)

    Args:
        c6_dict (dict): Dictionary containing C6 data
        labels (list): List of atom labels
        atomic_moment_order (int): Maximum atomic moment order (0=charge, 1=dipole, 2=quadrupole)
        output_file (str): Base output file path

    """
    output_files = create_output_files(output_file, atomic_moment_order)

    sorted_frequencies = get_all_frequencies(c6_dict)

    with ExitStack() as stack:
        file_handles = {}
        for rank, file_path in output_files.items():
            file_handles[rank] = stack.enter_context(file_path.open("w"))

        for freq in sorted_frequencies:
            for key, freq_data in c6_dict.items():
                idx1, idx2 = key.split("_")
                site1 = labels[int(idx1) - 1]
                site2 = labels[int(idx2) - 1]

                matrix = freq_data[freq]
                freq_formatted = "0.0000000E+00" if freq == 0.0 else f"{freq:.7E}"

                # Write seperate files from 0 up to atomic_moment_order
                for rank in range(atomic_moment_order + 1):
                    header = orient_header(site1, site2, idx1, idx2, freq_formatted, rank)
                    file_handle = file_handles[rank]

                    file_handle.write(header + "\n")
                    write_matrix_data(file_handle, matrix, rank)
