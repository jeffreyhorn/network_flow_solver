"""File I/O helpers for network programming problems."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

from .data import FlowResult, NetworkProblem, build_problem
from .exceptions import InvalidProblemError


def _normalize_edges(raw: Iterable[Mapping[str, Any]]) -> Sequence[dict[str, Any]]:
    # Normalize incoming edge dictionaries so downstream dataclasses receive a uniform schema.
    edges = []
    for edge in raw:
        if "tail" not in edge or "head" not in edge:
            raise InvalidProblemError(
                f"Invalid edge specification: {edge}. Each edge must have 'tail' and 'head' fields."
            )
        normalized = {
            "tail": edge["tail"],
            "head": edge["head"],
            "capacity": edge.get("capacity"),
            "cost": edge.get("cost", 0.0),
            "lower": edge.get("lower", 0.0),
        }
        edges.append(normalized)
    return edges


def load_problem(path: str | Path) -> NetworkProblem:
    """Load a network programming instance from a JSON file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        payload: MutableMapping[str, Any] = json.load(fh)
    directed = bool(payload.get("directed", True))
    tolerance = float(payload.get("tolerance", 1e-3))
    nodes = payload.get("nodes")
    edges = payload.get("edges") or payload.get("arcs")
    if not isinstance(nodes, list) or not isinstance(edges, list):
        raise InvalidProblemError(
            "Invalid problem format: JSON must include 'nodes' and 'edges' (or 'arcs') arrays. "
            f"Got nodes type: {type(nodes).__name__}, edges type: {type(edges).__name__ if edges else 'None'}"
        )
    # Defer to the core builder so validation rules remain centralized in one place.
    return build_problem(
        nodes=nodes,
        arcs=_normalize_edges(edges),
        directed=directed,
        tolerance=tolerance,
    )


def save_result(path: str | Path, result: FlowResult) -> None:
    """Persist a solver result to JSON."""
    # Sort flow entries for deterministic output that is easy to diff in fixtures.
    data = {
        "status": result.status,
        "objective": result.objective,
        "iterations": result.iterations,
        "flows": [
            {"tail": tail, "head": head, "flow": flow}
            for (tail, head), flow in sorted(result.flows.items())
        ],
        "duals": {node_id: dual for node_id, dual in sorted(result.duals.items())},
    }
    with Path(path).open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=False)
