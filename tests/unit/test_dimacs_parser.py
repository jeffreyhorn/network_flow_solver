"""Unit tests for DIMACS format parser."""

import pytest
from pathlib import Path

from benchmarks.parsers.dimacs import parse_dimacs_string, parse_dimacs_file
from src.network_solver.exceptions import InvalidProblemError


class TestDIMACSParser:
    """Tests for DIMACS minimum cost flow format parser."""

    def test_parse_simple_problem(self):
        """Test parsing a simple DIMACS problem."""
        dimacs = """
        c Simple problem
        p min 3 2
        n 1 10
        n 3 -10
        a 1 2 0 20 1
        a 2 3 0 20 1
        """
        problem = parse_dimacs_string(dimacs)

        assert len(problem.nodes) == 3
        assert len(problem.arcs) == 2
        assert problem.directed is True

        # Check nodes
        assert problem.nodes["1"].supply == 10.0
        assert problem.nodes["2"].supply == 0.0  # Transshipment
        assert problem.nodes["3"].supply == -10.0

        # Check arcs
        assert problem.arcs[0].tail == "1"
        assert problem.arcs[0].head == "2"
        assert problem.arcs[0].lower == 0.0
        assert problem.arcs[0].capacity == 20.0
        assert problem.arcs[0].cost == 1.0

    def test_parse_transportation_problem(self):
        """Test parsing a transportation problem."""
        dimacs = """
        c Transportation problem
        p min 4 4
        n 1 10
        n 2 15
        n 3 -12
        n 4 -13
        a 1 3 0 10 4
        a 1 4 0 10 6
        a 2 3 0 12 3
        a 2 4 0 15 5
        """
        problem = parse_dimacs_string(dimacs)

        assert len(problem.nodes) == 4
        assert len(problem.arcs) == 4

        # Verify supply/demand balance
        total_supply = sum(node.supply for node in problem.nodes.values())
        assert abs(total_supply) < 1e-9  # Should be balanced

    def test_parse_with_lower_bounds(self):
        """Test parsing arcs with non-zero lower bounds."""
        dimacs = """
        c Problem with lower bounds
        p min 3 2
        n 1 100
        n 3 -100
        a 1 2 10 50 2
        a 2 3 20 80 3
        """
        problem = parse_dimacs_string(dimacs)

        assert problem.arcs[0].lower == 10.0
        assert problem.arcs[0].capacity == 50.0
        assert problem.arcs[1].lower == 20.0
        assert problem.arcs[1].capacity == 80.0

    def test_parse_without_lower_bounds(self):
        """Test parsing 4-field arc format (without lower bounds)."""
        dimacs = """
        c Arc format without lower bounds
        p min 3 2
        n 1 50
        n 3 -50
        a 1 2 100 5
        a 2 3 100 3
        """
        problem = parse_dimacs_string(dimacs)

        assert problem.arcs[0].lower == 0.0  # Default lower bound
        assert problem.arcs[0].capacity == 100.0
        assert problem.arcs[0].cost == 5.0

    def test_parse_infinite_capacity(self):
        """Test parsing arcs with infinite capacity."""
        dimacs = """
        c Infinite capacity arcs
        p min 3 2
        n 1 10
        n 3 -10
        a 1 2 0 -1 1
        a 2 3 0 inf 2
        """
        problem = parse_dimacs_string(dimacs)

        assert problem.arcs[0].capacity is None  # -1 -> infinite
        assert problem.arcs[1].capacity is None  # inf -> infinite

    def test_parse_very_large_capacity_as_infinite(self):
        """Test that very large capacities are treated as infinite."""
        dimacs = """
        c Very large capacity
        p min 2 1
        n 1 10
        n 2 -10
        a 1 2 0 9999999999999999 1
        """
        problem = parse_dimacs_string(dimacs)

        assert problem.arcs[0].capacity is None  # Very large -> infinite

    def test_parse_comments_and_whitespace(self):
        """Test that comments and extra whitespace are handled correctly."""
        dimacs = """
        c This is a comment
        c Another comment line

        p min 2 1

        c Node descriptors
        n 1 5
        n 2 -5

        c Arc descriptor
        a 1 2 0 10 2

        c End of file
        """
        problem = parse_dimacs_string(dimacs)

        assert len(problem.nodes) == 2
        assert len(problem.arcs) == 1

    def test_parse_assignment_problem(self):
        """Test parsing a small assignment problem."""
        dimacs = """
        c 2x2 assignment problem
        p min 4 4
        n 1 1
        n 2 1
        n 3 -1
        n 4 -1
        a 1 3 0 1 5
        a 1 4 0 1 7
        a 2 3 0 1 6
        a 2 4 0 1 4
        """
        problem = parse_dimacs_string(dimacs)

        assert len(problem.nodes) == 4
        assert len(problem.arcs) == 4

        # All capacities should be 1 for assignment
        for arc in problem.arcs:
            assert arc.capacity == 1.0

    def test_parse_file(self):
        """Test parsing from a file."""
        # Use one of the generated test instances
        file_path = Path("benchmarks/problems/generated/tiny_transportation.min")
        if file_path.exists():
            problem = parse_dimacs_file(file_path)
            assert len(problem.nodes) == 4
            assert len(problem.arcs) == 4

    def test_file_not_found(self):
        """Test error handling for missing files."""
        with pytest.raises(FileNotFoundError):
            parse_dimacs_file("nonexistent_file.min")

    def test_missing_problem_descriptor(self):
        """Test error when problem descriptor is missing."""
        dimacs = """
        c No problem descriptor
        n 1 10
        a 1 2 0 10 1
        """
        with pytest.raises(InvalidProblemError, match="before problem descriptor"):
            parse_dimacs_string(dimacs)

    def test_multiple_problem_descriptors(self):
        """Test error when multiple problem descriptors exist."""
        dimacs = """
        p min 2 1
        p min 3 2
        n 1 10
        a 1 2 0 10 1
        """
        with pytest.raises(InvalidProblemError, match="Multiple problem descriptor"):
            parse_dimacs_string(dimacs)

    def test_wrong_problem_type(self):
        """Test error for non-minimum-cost-flow problems."""
        dimacs = """
        p max 3 2
        n 1 10
        a 1 2 0 10 1
        """
        with pytest.raises(InvalidProblemError, match="Only 'min'.*supported"):
            parse_dimacs_string(dimacs)

    def test_node_before_problem_descriptor(self):
        """Test error when node appears before problem descriptor."""
        dimacs = """
        n 1 10
        p min 2 1
        a 1 2 0 10 1
        """
        with pytest.raises(InvalidProblemError, match="before problem descriptor"):
            parse_dimacs_string(dimacs)

    def test_arc_before_problem_descriptor(self):
        """Test error when arc appears before problem descriptor."""
        dimacs = """
        a 1 2 0 10 1
        p min 2 1
        """
        with pytest.raises(InvalidProblemError, match="before problem descriptor"):
            parse_dimacs_string(dimacs)

    def test_arc_count_mismatch(self):
        """Test error when arc count doesn't match problem descriptor."""
        dimacs = """
        p min 3 3
        n 1 10
        n 3 -10
        a 1 2 0 10 1
        a 2 3 0 10 1
        """
        with pytest.raises(InvalidProblemError, match="Arc count mismatch"):
            parse_dimacs_string(dimacs)

    def test_invalid_problem_descriptor_format(self):
        """Test error for malformed problem descriptor."""
        dimacs = """
        p min 3
        """
        with pytest.raises(InvalidProblemError, match="Invalid problem descriptor"):
            parse_dimacs_string(dimacs)

    def test_invalid_node_descriptor_format(self):
        """Test error for malformed node descriptor."""
        dimacs = """
        p min 2 1
        n 1
        a 1 2 0 10 1
        """
        with pytest.raises(InvalidProblemError, match="Invalid node descriptor"):
            parse_dimacs_string(dimacs)

    def test_invalid_arc_descriptor_format(self):
        """Test error for malformed arc descriptor."""
        dimacs = """
        p min 2 1
        n 1 10
        n 2 -10
        a 1 2 10
        """
        with pytest.raises(InvalidProblemError, match="Invalid arc descriptor"):
            parse_dimacs_string(dimacs)

    def test_unknown_line_type(self):
        """Test error for unknown line types."""
        dimacs = """
        p min 2 1
        x 1 2
        a 1 2 0 10 1
        """
        with pytest.raises(InvalidProblemError, match="Unknown line type"):
            parse_dimacs_string(dimacs)

    def test_negative_node_count(self):
        """Test error for negative node count."""
        dimacs = """
        p min -1 0
        """
        with pytest.raises(InvalidProblemError, match="must be positive"):
            parse_dimacs_string(dimacs)

    def test_negative_arc_count(self):
        """Test error for negative arc count."""
        dimacs = """
        p min 2 -1
        """
        with pytest.raises(InvalidProblemError, match="cannot be negative"):
            parse_dimacs_string(dimacs)

    def test_arc_with_node_outside_range(self):
        """Test error when arc references node ID outside declared range."""
        dimacs = """
        p min 3 1
        n 1 10
        n 2 -10
        a 1 5 0 10 1
        """
        with pytest.raises(InvalidProblemError, match="outside the expected range"):
            parse_dimacs_string(dimacs)

    def test_parse_all_generated_instances(self):
        """Test parsing all generated test instances."""
        generated_dir = Path("benchmarks/problems/generated")
        if not generated_dir.exists():
            pytest.skip("Generated instances directory not found")

        dimacs_files = list(generated_dir.glob("*.min"))
        if not dimacs_files:
            pytest.skip("No generated DIMACS instances found")

        for dimacs_file in dimacs_files:
            problem = parse_dimacs_file(dimacs_file)
            assert problem is not None
            assert len(problem.nodes) > 0
            assert len(problem.arcs) >= 0
            assert problem.directed is True

            # Verify supply/demand balance
            total_supply = sum(node.supply for node in problem.nodes.values())
            assert abs(total_supply) < 1e-6, f"Unbalanced problem in {dimacs_file.name}"
