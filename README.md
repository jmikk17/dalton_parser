# Dalton parcer for bond capacity calculations

A Python tool for extracting and analyzing data from output files from the crk-oslo feature branch of the Dalton fork (https://github.com/jmikk17/dalton_bc).

## Description

This tool parses Dalton output files to extract various properties including:
- Calculation information (wave function type, basis set, coordinates)
- Population analysis (MBIS)
- Distributed polarizabilities (bond capacities)

## Installation
Requires NumPy, SciPy, and Pandas.
```bash
git clone https://github.com/jmikk17/dalton_parser.git
```

## Usage
The following command preforms the parsing and analysis and saves it to "dalton_output.json":
```bash
python main.py dalton_output.out
```