# Dalton Parser

A Python package for parsing and analyzing Dalton quantum chemistry output files.

## Installation

```bash
pip install -e .
```

## Usage

### Command Line Interface

The package provides a command-line interface for common operations:

```bash
# Parse a Dalton output file and perform alpha analysis (default)
dalton-parser input.out

# Only parse the output file
dalton-parser -p input.out

# Only perform alpha analysis on existing JSON
dalton-parser -a input.json

# Extract C6 data for Orient format
dalton-parser -c input.out
```

### Python API

You can also use the package programmatically:

```python
import dalton_parser

# Parse Dalton output
with open('dalton_output.out', 'r') as f:
    content = f.read()
    result = dalton_parser.parse_dalton_output(content)

# Perform alpha analysis
alpha_result = dalton_parser.alpha_analysis(result)
```

## Project Structure

```
dalton_parser/
├── __init__.py           # Package initialization
├── main.py              # CLI entry point
├── config.py            # Configuration and argument parsing
├── core.py              # Core processing functions
├── analysis/            # Analysis modules
│   ├── __init__.py
│   └── alpha.py         # Alpha polarizability analysis
├── parsers/             # Parser modules
│   ├── __init__.py
│   ├── calculation.py   # Calculation info parser
│   ├── coordinates.py   # Coordinate parser
│   └── properties.py    # Properties parser
├── io/                  # Input/Output modules
│   ├── __init__.py
│   ├── file_operations.py # File I/O operations
│   └── orient_writer.py   # Orient format writer
└── utils/               # Utility modules
    ├── __init__.py
    └── helpers.py       # Helper functions
```
