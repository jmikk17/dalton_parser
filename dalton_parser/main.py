"""Main entry point for the Dalton parser."""

from .config import setup_parser
from .core import process_files
from .io.file_operations import write_file


def main() -> None:
    """Select between parsing, alpha analysis, or C6 processing.

    Default option is to parse the Dalton output file and perform alpha analysis, but both can be selected separately.

    """
    args = setup_parser()

    result, output_file = process_files(args)

    if result is not None and output_file is not None:
        write_file(output_file, result)


if __name__ == "__main__":
    main()
