"""Core processing functions for Dalton output parsing."""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING

import numpy as np

from .analysis.alpha import alpha_calc
from .analysis.dispersion import integrate_c6, pade_approx
from .config import get_file_names
from .io.file_operations import read_file
from .io.orient_writer import write_c6
from .parsers.calculation import extract_calculation_info, extract_wave_function_type
from .parsers.coordinates import combine_coords_with_charges, extract_coordinates, read_coords
from .parsers.properties import (
    extract_1st_order_prop,
    extract_2nd_order_prop,
    extract_imaginary,
    read_2nd_order_prop,
)

if TYPE_CHECKING:
    import argparse


def process_files(args: argparse.Namespace) -> dict | None:
    """Process input files based on command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        dict: Processing results or None for C6 processing

    """
    input_file, output_file = get_file_names(args)

    if args.mode in ("parse", "all"):
        content = read_file(input_file, ".out")
        result = parse_dalton_output(content)
        if args.mode == "all":
            result.update(alpha_analysis(result))
        return result, output_file
    if args.mode == "alpha":
        content = read_file(input_file, ".json")
        result = alpha_analysis(content)
        return result, output_file
    if args.mode == "c6":
        content = read_file(input_file, ".out")
        alpha_imaginary_analysis(content, output_file)
        return None, None

    return None, None


def parse_dalton_output(content: str) -> None | dict:
    """Parse a Dalton output file and extract coordinates.

    Args:
        content (str): Content of Dalton output file

    Returns:
        dict: Dictionary containing parsed coordinates

    """
    main_dict = extract_calculation_info(content)
    if not main_dict:
        sys.exit("Error: No calculation details found")

    coord_dict = extract_coordinates(content)
    if not coord_dict:
        sys.exit("Error: No coordinates found")

    charge_list = extract_1st_order_prop(content)
    if not charge_list:
        warnings.warn("No charges found", stacklevel=1)
        main_dict.update({"atoms": coord_dict})
    else:
        main_dict.update({"atoms": combine_coords_with_charges(coord_dict, charge_list)})

    property_dict = {
        "2nd_order_properties": extract_2nd_order_prop(
            content,
            main_dict["wave_function"],
            main_dict["atomic_moment_order"],
        ),
    }
    if not property_dict:
        sys.exit("Error: No second order properties found")
    main_dict.update(property_dict)

    return main_dict


def alpha_analysis(content: dict) -> dict:
    """Calculate alpha contributions from second order properties.

    Args:
        content (dict): content of parsed JSON file

    Returns:
        dict: Dictionary containing alpha contributions

    """
    property_df = read_2nd_order_prop(content)

    geometry = read_coords(content)
    atmmom = content["atomic_moment_order"]

    alpha_dict = alpha_calc(property_df, geometry, atmmom)
    if alpha_dict is None:
        sys.exit("Error: Alpha analysis failed, dict is empty")

    return alpha_calc(property_df, geometry, atmmom)


def alpha_imaginary_analysis(content: str, output_file: str) -> dict:
    """Extract alpha(i omega) data from Dalton output file, and write as orient fmtB polarizability file.

    Args:
        content (str): Content of Dalton output file
        output_file (str): Output file name

    Todo:
        - Move write_c6 out to main, pass out results instead
        - n_freq is currently hardcoded

    """
    imaginary_dict = {}
    wave_function = extract_wave_function_type(content)
    calc_info = extract_calculation_info(content)
    atmmom = calc_info["atomic_moment_order"]
    coord_list = extract_coordinates(content)

    if not wave_function:
        sys.exit("Error: No wave function type found")
    if wave_function["wave_function"] != "CC":
        # TODO This requires a min print level of 5 in the Dalton input file
        imaginary_dict, full_response, operator_to_idx = extract_imaginary(content, atmmom, len(coord_list), n_freq=11)
    elif wave_function["wave_function"] == "CC":
        sys.exit("Error: C6 data extraction is not supported for CC wave functions yet")
        # Also add a seperate reading function for CC Padé moments
        imaginary_dict = pade_approx(content)
    if not imaginary_dict:
        sys.exit("Error: No C6 data found")

    integrate_c6(full_response, operator_to_idx, coord_list, atmmom)

    labels = extract_coordinates(content, label_only=True)

    write_c6(imaginary_dict, labels, calc_info["atomic_moment_order"], output_file)
