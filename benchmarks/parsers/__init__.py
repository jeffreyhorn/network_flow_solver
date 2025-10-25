"""Parsers for benchmark problem formats.

This module provides parsers for various standard network flow problem formats
used in benchmark repositories (DIMACS, OR-Library, LEMON, etc.).

These parsers are separate from the production code in src/network_solver/io.py
to avoid mixing benchmark infrastructure with core solver functionality.
"""

from .dimacs import parse_dimacs_file, parse_dimacs_string

__all__ = [
    "parse_dimacs_file",
    "parse_dimacs_string",
]
