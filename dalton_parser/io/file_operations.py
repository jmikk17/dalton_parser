"""File I/O operations for Dalton parser."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal, overload

import numpy as np


def check_file_extension(file_path: str, ext: str) -> bool:
    """Check if the given path exists and has a valid extension.

    Args:
        file_path (str): Path to the file to check
        ext (str): Expected extension of the file

    Returns:
        bool: True if it's a valid extension and file exists, False otherwise

    """
    if not Path(file_path).is_file():
        return False

    return file_path.lower().endswith(ext)


@overload
def read_file(file_path: str, ext: Literal[".out"]) -> str: ...


@overload
def read_file(file_path: str, ext: Literal[".json"]) -> dict: ...


def read_file(file_path: str, ext: str) -> str | dict:
    """Read the content of the file.

    Args:
        file_path (str): Path to the file to read
        ext (str): Extension of the file (".out" or ".json")

    Returns:
        str: Content of the file when ext is ".out"
        dict: Content of the file when ext is ".json"

    """
    try:
        with Path(file_path).open("r") as file:
            if ext == ".out":
                return file.read()
            if ext == ".json":
                return json.load(file)
            sys.exit("Error: Invalid file extension")
    except FileNotFoundError:
        sys.exit(f"Error: File '{file_path}' not found")
    except OSError as e:
        sys.exit(f"Error reading file: {e}")


def write_file(file_path: str, content: dict) -> None:
    """Write the content to the file.

    Args:
        file_path (str): Path to the file to write
        content (dict): Content to dump to the file

    """
    try:
        with Path(file_path).open("w") as file:
            json.dump(content, file, indent=2, cls=NumpyEncoder)
    except OSError as e:
        sys.exit(f"Error writing file: {e}")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types."""

    def default(self, obj: object) -> object:
        """Convert numpy data types to standard Python types.

        Args:
            obj (object): Object to be converted

        Returns:
            object: Converted object

        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)
