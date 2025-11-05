"""Configuration and command-line argument parsing."""

import argparse
import sys
from pathlib import Path

from .io.file_operations import check_file_extension


def setup_parser() -> argparse.Namespace:
    """Set up the argument parser for the script.

    Returns:
        argparse.Namespace: Parsed arguments

    """
    parser = argparse.ArgumentParser(description="Process dalton output files into JSON format")
    parser.add_argument(
        "--mode",
        choices=["parse", "alpha", "c6", "all"],
        default="all",
        help="Processing mode: parse=parse only, alpha=analyze existing JSON, c6=extract C6 data, all=parse+analyze (default: %(default)s)",
    )
    parser.add_argument("input_file", help="Input file to process")
    parser.add_argument("-o", "--output", help="Output file (default: input_file.json)")

    return parser.parse_args()


def get_file_names(args: argparse.Namespace) -> tuple[str, str]:
    """Get the input and output file names from the arguments.

    Args:
        args (argparse.Namespace): Parsed arguments

    Returns:
        tuple: Input and output file names

    """
    if not args.output:
        path = Path(args.input_file)
        if args.mode in ("parse", "all", "c6"):
            args.output = path.with_suffix(".orient" if args.mode == "c6" else ".json")
        elif args.mode == "alpha":
            args.output = path.with_suffix(".alpha.json")

    if args.mode in ("parse", "all", "c6") and not check_file_extension(args.input_file, ".out"):
        sys.exit("Error: Input file must be a Dalton output file (.out)")
    elif args.mode == "alpha" and not check_file_extension(args.input_file, ".json"):
        sys.exit("Error: Input file must be a JSON file (.json)")

    return str(args.input_file), str(args.output)
