"""Unit tests for __init__.py module public API."""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class TestVisualizationStubs:
    """Tests for visualization stub functions when dependencies not installed."""

    def test_visualization_stubs_raise_import_error(self):
        """Test that visualization stubs raise helpful ImportError when called."""
        # Import and check if visualization is available
        import network_solver

        # If visualization dependencies are NOT installed, the stub functions should raise
        if not network_solver._has_visualization:
            with pytest.raises(ImportError, match="Visualization requires optional dependencies"):
                network_solver.visualize_network({}, {})

            with pytest.raises(ImportError, match="Visualization requires optional dependencies"):
                network_solver.visualize_flows({}, {}, {})

            with pytest.raises(ImportError, match="Visualization requires optional dependencies"):
                network_solver.visualize_bottlenecks({}, {}, {})
        else:
            # If dependencies ARE installed, functions should exist and be callable
            # (we won't test full functionality here, just that they exist)
            assert callable(network_solver.visualize_network)
            assert callable(network_solver.visualize_flows)
            assert callable(network_solver.visualize_bottlenecks)

    def test_version_is_defined(self):
        """Test that __version__ is defined."""
        import network_solver

        assert hasattr(network_solver, "__version__")
        assert isinstance(network_solver.__version__, str)
        assert len(network_solver.__version__) > 0

    def test_has_visualization_flag_is_boolean(self):
        """Test that _has_visualization flag is a boolean."""
        import network_solver

        assert hasattr(network_solver, "_has_visualization")
        assert isinstance(network_solver._has_visualization, bool)

    def test_public_api_exports(self):
        """Test that key public API functions are exported."""
        import network_solver

        # Core solving functions
        assert hasattr(network_solver, "solve_min_cost_flow")
        assert hasattr(network_solver, "build_problem")

        # IO functions
        assert hasattr(network_solver, "load_problem")
        assert hasattr(network_solver, "save_result")

        # Visualization functions (either real or stubs)
        assert hasattr(network_solver, "visualize_network")
        assert hasattr(network_solver, "visualize_flows")
        assert hasattr(network_solver, "visualize_bottlenecks")
