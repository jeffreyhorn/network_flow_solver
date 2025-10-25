"""DIMACS minimum cost flow format parser.

This module implements a parser for the DIMACS minimum cost network flow format,
which is the standard format used by the DIMACS Implementation Challenges and
many benchmark repositories.

DIMACS Format Specification:
    c <comment lines - ignored>
    p min <num_nodes> <num_arcs>
    n <node_id> <supply>
    a <tail_id> <head_id> <lower_bound> <capacity> <cost>

Reference:
    http://lpsolve.sourceforge.net/5.5/DIMACS_mcf.htm
    DIMACS Implementation Challenge - Network Flows and Matching (1991)

Example:
    c Sample minimum cost flow problem
    c 4 nodes, 5 arcs
    p min 4 5
    n 1 10
    n 4 -10
    a 1 2 0 15 2
    a 1 3 0 8 2
    a 2 3 0 20 1
    a 2 4 0 4 3
    a 3 4 0 15 1

Notes:
    - Node IDs in DIMACS are typically 1-indexed integers
    - This parser converts them to strings for NetworkProblem compatibility
    - Comment lines (starting with 'c') are ignored
    - Nodes with unspecified supply default to 0.0 (transshipment nodes)
    - Lower bounds are optional in some DIMACS variants (defaults to 0)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.network_solver.data import NetworkProblem, build_problem
from src.network_solver.exceptions import InvalidProblemError


def parse_dimacs_string(dimacs_content: str) -> NetworkProblem:
    """Parse a DIMACS minimum cost flow problem from a string.

    Args:
        dimacs_content: String containing DIMACS format problem specification.

    Returns:
        NetworkProblem instance representing the parsed problem.

    Raises:
        InvalidProblemError: If the DIMACS format is invalid or incomplete.

    Example:
        >>> dimacs = '''
        ... c Simple problem
        ... p min 3 2
        ... n 1 10
        ... n 3 -10
        ... a 1 2 0 20 1
        ... a 2 3 0 20 1
        ... '''
        >>> problem = parse_dimacs_string(dimacs)
        >>> len(problem.nodes)
        3
        >>> len(problem.arcs)
        2
    """
    lines = dimacs_content.strip().split("\n")
    return _parse_dimacs_lines(lines)


def parse_dimacs_file(file_path: str | Path) -> NetworkProblem:
    """Parse a DIMACS minimum cost flow problem from a file.

    Args:
        file_path: Path to DIMACS format file.

    Returns:
        NetworkProblem instance representing the parsed problem.

    Raises:
        InvalidProblemError: If the DIMACS format is invalid or incomplete.
        FileNotFoundError: If the file does not exist.

    Example:
        >>> problem = parse_dimacs_file('benchmarks/problems/dimacs/sample.min')
        >>> problem.directed
        True
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DIMACS file not found: {file_path}")

    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n\r") for line in f]

    return _parse_dimacs_lines(lines)


def _parse_dimacs_lines(lines: list[str]) -> NetworkProblem:
    """Internal parser for DIMACS format lines.

    Args:
        lines: List of DIMACS format lines (without trailing newlines).

    Returns:
        NetworkProblem instance.

    Raises:
        InvalidProblemError: If format is invalid.
    """
    num_nodes: int | None = None
    num_arcs: int | None = None
    node_supplies: dict[str, float] = {}
    arcs: list[dict[str, Any]] = []

    problem_line_seen = False

    for line_num, line in enumerate(lines, start=1):
        line = line.strip()

        # Skip empty lines and comments
        if not line or line.startswith("c"):
            continue

        tokens = line.split()
        if not tokens:
            continue

        line_type = tokens[0]

        try:
            if line_type == "p":
                # Problem descriptor: p min <nodes> <arcs>
                if problem_line_seen:
                    raise InvalidProblemError(
                        f"Line {line_num}: Multiple problem descriptor lines found. "
                        "Only one 'p min' line is allowed."
                    )
                if len(tokens) != 4:
                    raise InvalidProblemError(
                        f"Line {line_num}: Invalid problem descriptor format. "
                        f"Expected 'p min <nodes> <arcs>', got: {line}"
                    )
                if tokens[1] != "min":
                    raise InvalidProblemError(
                        f"Line {line_num}: Only 'min' (minimum cost flow) problems supported. "
                        f"Got: {tokens[1]}"
                    )
                num_nodes = int(tokens[2])
                num_arcs = int(tokens[3])
                problem_line_seen = True

                if num_nodes <= 0:
                    raise InvalidProblemError(
                        f"Line {line_num}: Number of nodes must be positive, got {num_nodes}"
                    )
                if num_arcs < 0:
                    raise InvalidProblemError(
                        f"Line {line_num}: Number of arcs cannot be negative, got {num_arcs}"
                    )

            elif line_type == "n":
                # Node descriptor: n <node_id> <supply>
                if not problem_line_seen:
                    raise InvalidProblemError(
                        f"Line {line_num}: Node descriptor before problem descriptor. "
                        "The 'p min' line must come first."
                    )
                if len(tokens) != 3:
                    raise InvalidProblemError(
                        f"Line {line_num}: Invalid node descriptor format. "
                        f"Expected 'n <node_id> <supply>', got: {line}"
                    )
                node_id = tokens[1]
                supply = float(tokens[2])
                node_supplies[node_id] = supply

            elif line_type == "a":
                # Arc descriptor: a <tail> <head> <lower> <capacity> <cost>
                # Note: Some DIMACS variants omit lower bound (assume 0)
                if not problem_line_seen:
                    raise InvalidProblemError(
                        f"Line {line_num}: Arc descriptor before problem descriptor. "
                        "The 'p min' line must come first."
                    )

                # Support both 5-field (with lower) and 4-field (without lower) formats
                if len(tokens) == 6:
                    # Standard format: a <tail> <head> <lower> <capacity> <cost>
                    tail = tokens[1]
                    head = tokens[2]
                    lower = float(tokens[3])
                    capacity_str = tokens[4]
                    cost = float(tokens[5])
                elif len(tokens) == 5:
                    # Variant without lower bound: a <tail> <head> <capacity> <cost>
                    tail = tokens[1]
                    head = tokens[2]
                    lower = 0.0
                    capacity_str = tokens[3]
                    cost = float(tokens[4])
                else:
                    raise InvalidProblemError(
                        f"Line {line_num}: Invalid arc descriptor format. "
                        f"Expected 'a <tail> <head> <lower> <capacity> <cost>' "
                        f"or 'a <tail> <head> <capacity> <cost>', got: {line}"
                    )

                # Handle infinite capacity (sometimes encoded as -1 or very large number)
                if capacity_str == "-1" or capacity_str.lower() == "inf":
                    capacity = None
                else:
                    capacity_val = float(capacity_str)
                    # Treat very large capacities as infinite
                    capacity = None if capacity_val >= 1e15 else capacity_val

                arcs.append(
                    {
                        "tail": tail,
                        "head": head,
                        "lower": lower,
                        "capacity": capacity,
                        "cost": cost,
                    }
                )

            else:
                raise InvalidProblemError(
                    f"Line {line_num}: Unknown line type '{line_type}'. "
                    "Expected 'c' (comment), 'p' (problem), 'n' (node), or 'a' (arc)."
                )

        except (ValueError, IndexError) as e:
            raise InvalidProblemError(
                f"Line {line_num}: Failed to parse line: {line}. Error: {e}"
            ) from e

    # Validate that we saw a problem descriptor
    if not problem_line_seen:
        raise InvalidProblemError(
            "No problem descriptor found. DIMACS file must contain a 'p min <nodes> <arcs>' line."
        )

    # Validate counts match
    if num_arcs != len(arcs):
        raise InvalidProblemError(
            f"Arc count mismatch: problem descriptor specifies {num_arcs} arcs, "
            f"but {len(arcs)} arc descriptors found."
        )

    # Build node list with all nodes (including transshipment nodes with supply=0)
    # DIMACS node IDs are typically 1 to num_nodes
    nodes = []
    for i in range(1, num_nodes + 1):
        node_id = str(i)
        supply = node_supplies.get(node_id, 0.0)
        nodes.append({"id": node_id, "supply": supply})

    # Also check if any node IDs in arcs are outside the expected range
    all_node_ids = {str(i) for i in range(1, num_nodes + 1)}
    arc_node_ids = set()
    for arc in arcs:
        arc_node_ids.add(arc["tail"])
        arc_node_ids.add(arc["head"])

    unexpected_nodes = arc_node_ids - all_node_ids
    if unexpected_nodes:
        raise InvalidProblemError(
            f"Arc references node IDs outside the expected range [1, {num_nodes}]: "
            f"{sorted(unexpected_nodes)}"
        )

    # Build NetworkProblem using the standard builder
    # DIMACS problems are always directed
    return build_problem(
        nodes=nodes,
        arcs=arcs,
        directed=True,
        tolerance=1e-6,  # Standard numerical tolerance
    )
