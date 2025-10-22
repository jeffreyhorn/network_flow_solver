"""Visualization utilities for network flow problems and solutions.

This module provides functions to visualize network structures, flow solutions,
and bottleneck analysis using matplotlib and networkx.

Example:
    >>> from network_solver import visualize_network, visualize_flows
    >>>
    >>> # Visualize problem structure
    >>> fig = visualize_network(problem)
    >>> fig.savefig("network.png")
    >>>
    >>> # Visualize flow solution with bottlenecks
    >>> fig = visualize_flows(problem, result, highlight_bottlenecks=True)
    >>> fig.savefig("flows.png")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .data import FlowResult, NetworkProblem

logger = logging.getLogger(__name__)

# Check for optional dependencies
try:
    import matplotlib.cm  # type: ignore[import-untyped,unused-ignore]
    import matplotlib.pyplot as plt
    import networkx as nx  # type: ignore[import-untyped,unused-ignore]
    from matplotlib.colors import Normalize  # type: ignore[import-untyped,unused-ignore]
    from matplotlib.figure import Figure

    _HAS_VISUALIZATION_DEPS = True
except ImportError:
    _HAS_VISUALIZATION_DEPS = False
    Figure = Any  # type: ignore[misc, assignment]


def _check_dependencies() -> None:
    """Check if visualization dependencies are installed."""
    if not _HAS_VISUALIZATION_DEPS:
        msg = (
            "Visualization requires optional dependencies. "
            "Install with: pip install 'network_solver[visualization]'"
        )
        raise ImportError(msg)


def visualize_network(
    problem: NetworkProblem,
    layout: str = "spring",
    figsize: tuple[float, float] = (12, 8),
    node_size: int = 1000,
    font_size: int = 10,
    show_arc_labels: bool = True,
    title: str | None = None,
) -> Figure:
    """Visualize network structure showing nodes, arcs, supplies, and costs.

    Creates a network graph visualization with:
    - Source nodes (supply > 0) in green
    - Sink nodes (supply < 0) in red
    - Transshipment nodes (supply = 0) in lightblue
    - Arcs with costs and capacities labeled
    - Node supplies displayed

    Args:
        problem: Network flow problem to visualize
        layout: Graph layout algorithm ("spring", "circular", "kamada_kawai", "planar")
        figsize: Figure size (width, height) in inches
        node_size: Size of node markers
        font_size: Font size for labels
        show_arc_labels: Whether to show cost/capacity labels on arcs
        title: Custom title for the plot (default: "Network Structure")

    Returns:
        matplotlib Figure object

    Raises:
        ImportError: If matplotlib or networkx are not installed

    Example:
        >>> from network_solver import build_problem, visualize_network
        >>>
        >>> nodes = [
        ...     {"id": "factory", "supply": 100.0},
        ...     {"id": "warehouse", "supply": -100.0},
        ... ]
        >>> arcs = [
        ...     {"tail": "factory", "head": "warehouse", "capacity": 150.0, "cost": 2.5},
        ... ]
        >>> problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        >>>
        >>> fig = visualize_network(problem)
        >>> fig.savefig("network.png")

    See Also:
        visualize_flows() - Visualize flow solution
        visualize_bottlenecks() - Visualize bottleneck analysis
    """
    _check_dependencies()

    # Create directed graph
    G = nx.DiGraph() if problem.directed else nx.Graph()

    # Add nodes with attributes
    for node_id, node in problem.nodes.items():
        G.add_node(node_id, supply=node.supply)

    # Add edges with attributes
    for arc in problem.arcs:
        G.add_edge(
            arc.tail,
            arc.head,
            cost=arc.cost,
            capacity=arc.capacity,
            lower=arc.lower,
        )

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Compute layout
    layout_funcs = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "planar": nx.planar_layout,
    }
    if layout not in layout_funcs:
        logger.warning(f"Unknown layout '{layout}', using 'spring'")
        layout = "spring"

    try:
        pos = layout_funcs[layout](G)
    except Exception as e:
        logger.warning(f"Layout '{layout}' failed: {e}, using 'spring'")
        pos = nx.spring_layout(G)

    # Categorize nodes by supply
    source_nodes = [n for n, d in G.nodes(data=True) if d["supply"] > problem.tolerance]
    sink_nodes = [n for n, d in G.nodes(data=True) if d["supply"] < -problem.tolerance]
    transship_nodes = [n for n, d in G.nodes(data=True) if abs(d["supply"]) <= problem.tolerance]

    # Draw nodes by category
    if source_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=source_nodes,
            node_color="lightgreen",
            node_size=node_size,
            ax=ax,
            label="Sources",
        )
    if sink_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=sink_nodes,
            node_color="lightcoral",
            node_size=node_size,
            ax=ax,
            label="Sinks",
        )
    if transship_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=transship_nodes,
            node_color="lightblue",
            node_size=node_size,
            ax=ax,
            label="Transshipment",
        )

    # Draw edges
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color="gray",
        arrows=True,
        arrowsize=20,
        ax=ax,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw node labels with supply
    node_labels = {
        node_id: f"{node_id}\n({node.supply:+.0f})" for node_id, node in problem.nodes.items()
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size, ax=ax)

    # Draw edge labels with cost/capacity
    if show_arc_labels:
        edge_labels = {}
        for arc in problem.arcs:
            capacity_str = f"{arc.capacity:.0f}" if arc.capacity is not None else "∞"
            edge_labels[(arc.tail, arc.head)] = f"${arc.cost:.1f}\n[{capacity_str}]"
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_size=font_size - 2, ax=ax
        )

    # Set title and legend
    ax.set_title(title or "Network Structure", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=font_size)
    ax.axis("off")

    plt.tight_layout()
    return fig


def visualize_flows(
    problem: NetworkProblem,
    result: FlowResult,
    layout: str = "spring",
    figsize: tuple[float, float] = (14, 10),
    node_size: int = 1200,
    font_size: int = 10,
    highlight_bottlenecks: bool = True,
    bottleneck_threshold: float = 0.9,
    show_zero_flows: bool = False,
    title: str | None = None,
) -> Figure:
    """Visualize flow solution with optional bottleneck highlighting.

    Creates a network visualization showing:
    - Flow values on each arc
    - Arc thickness proportional to flow magnitude
    - Bottleneck arcs highlighted in red (utilization ≥ threshold)
    - Node supplies and flow conservation
    - Arc costs and utilization percentages

    Args:
        problem: Network flow problem
        result: Flow solution from solve_min_cost_flow()
        layout: Graph layout algorithm ("spring", "circular", "kamada_kawai", "planar")
        figsize: Figure size (width, height) in inches
        node_size: Size of node markers
        font_size: Font size for labels
        highlight_bottlenecks: Whether to highlight high-utilization arcs
        bottleneck_threshold: Utilization threshold for bottleneck (default: 0.9 = 90%)
        show_zero_flows: Whether to show arcs with zero flow
        title: Custom title for the plot (default: "Flow Solution")

    Returns:
        matplotlib Figure object

    Raises:
        ImportError: If matplotlib or networkx are not installed

    Example:
        >>> from network_solver import build_problem, solve_min_cost_flow, visualize_flows
        >>>
        >>> problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        >>> result = solve_min_cost_flow(problem)
        >>>
        >>> # Visualize flows with bottlenecks highlighted
        >>> fig = visualize_flows(problem, result, highlight_bottlenecks=True)
        >>> fig.savefig("flows.png")
        >>>
        >>> # Hide zero flows for cleaner visualization
        >>> fig = visualize_flows(problem, result, show_zero_flows=False)
        >>> fig.savefig("flows_nonzero.png")

    See Also:
        visualize_network() - Visualize problem structure
        visualize_bottlenecks() - Focused bottleneck visualization
        compute_bottleneck_arcs() - Identify bottlenecks programmatically
    """
    _check_dependencies()

    # Create directed graph with flows
    G = nx.DiGraph() if problem.directed else nx.Graph()

    # Add nodes
    for node_id, node in problem.nodes.items():
        G.add_node(node_id, supply=node.supply)

    # Add edges with flow data
    arc_map = {(arc.tail, arc.head): arc for arc in problem.arcs}
    flows_to_show = []

    for (tail, head), flow in result.flows.items():
        if not show_zero_flows and abs(flow) < problem.tolerance:
            continue

        arc = arc_map.get((tail, head))
        if arc is None:
            continue

        flows_to_show.append((tail, head, flow, arc))
        G.add_edge(tail, head, flow=flow, arc=arc)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Compute layout
    layout_funcs = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "planar": nx.planar_layout,
    }
    if layout not in layout_funcs:
        logger.warning(f"Unknown layout '{layout}', using 'spring'")
        layout = "spring"

    try:
        pos = layout_funcs[layout](G)
    except Exception as e:
        logger.warning(f"Layout '{layout}' failed: {e}, using 'spring'")
        pos = nx.spring_layout(G)

    # Categorize nodes by supply
    source_nodes = [n for n, d in G.nodes(data=True) if d["supply"] > problem.tolerance]
    sink_nodes = [n for n, d in G.nodes(data=True) if d["supply"] < -problem.tolerance]
    transship_nodes = [n for n, d in G.nodes(data=True) if abs(d["supply"]) <= problem.tolerance]

    # Draw nodes by category
    if source_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=source_nodes,
            node_color="lightgreen",
            node_size=node_size,
            ax=ax,
            label="Sources",
        )
    if sink_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=sink_nodes,
            node_color="lightcoral",
            node_size=node_size,
            ax=ax,
            label="Sinks",
        )
    if transship_nodes:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=transship_nodes,
            node_color="lightblue",
            node_size=node_size,
            ax=ax,
            label="Transshipment",
        )

    # Separate bottleneck and regular arcs
    bottleneck_edges = []
    regular_edges = []
    bottleneck_widths = []
    regular_widths = []

    # Maximum flow for normalization
    max_flow = max((abs(flow) for _, _, flow, _ in flows_to_show), default=1.0)

    for tail, head, flow, arc in flows_to_show:
        # Calculate utilization
        utilization = 0.0
        if arc.capacity is not None and arc.capacity > problem.tolerance:
            utilization = abs(flow) / arc.capacity

        # Calculate edge width (proportional to flow)
        width = 1.0 + 5.0 * (abs(flow) / max_flow) if max_flow > 0 else 1.0

        is_bottleneck = highlight_bottlenecks and utilization >= bottleneck_threshold

        if is_bottleneck:
            bottleneck_edges.append((tail, head))
            bottleneck_widths.append(width)
        else:
            regular_edges.append((tail, head))
            regular_widths.append(width)

    # Draw regular edges
    if regular_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=regular_edges,
            edge_color="gray",
            width=regular_widths,
            arrows=True,
            arrowsize=20,
            ax=ax,
            connectionstyle="arc3,rad=0.1",
            alpha=0.6,
        )

    # Draw bottleneck edges
    if bottleneck_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=bottleneck_edges,
            edge_color="red",
            width=bottleneck_widths,
            arrows=True,
            arrowsize=20,
            ax=ax,
            connectionstyle="arc3,rad=0.1",
            alpha=0.8,
            label="Bottlenecks",
        )

    # Draw node labels
    node_labels = {
        node_id: f"{node_id}\n({node.supply:+.0f})" for node_id, node in problem.nodes.items()
    }
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size, ax=ax)

    # Draw edge labels with flow and utilization
    edge_labels = {}
    for tail, head, flow, arc in flows_to_show:
        utilization = 0.0
        capacity_str = ""
        if arc.capacity is not None:
            utilization = abs(flow) / arc.capacity if arc.capacity > 0 else 0.0
            capacity_str = f"/{arc.capacity:.0f}"

        if highlight_bottlenecks and utilization >= bottleneck_threshold:
            edge_labels[(tail, head)] = f"{flow:.0f}{capacity_str}\n({utilization * 100:.0f}%)"
        else:
            edge_labels[(tail, head)] = f"{flow:.0f}{capacity_str}"

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size - 1, ax=ax)

    # Set title and legend
    title_text = title or f"Flow Solution (Cost: ${result.objective:.2f})"
    ax.set_title(title_text, fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=font_size)
    ax.axis("off")

    # Add statistics text box
    stats_text = (
        f"Status: {result.status}\n"
        f"Iterations: {result.iterations}\n"
        f"Objective: ${result.objective:.2f}"
    )
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=font_size - 1,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    return fig


def visualize_bottlenecks(
    problem: NetworkProblem,
    result: FlowResult,
    threshold: float = 0.8,
    layout: str = "spring",
    figsize: tuple[float, float] = (14, 10),
    node_size: int = 1200,
    font_size: int = 10,
    title: str | None = None,
) -> Figure:
    """Visualize bottleneck analysis with utilization heatmap.

    Creates a focused visualization highlighting capacity constraints:
    - Arcs colored by utilization percentage (gradient from green to red)
    - Only shows arcs above threshold utilization
    - Displays utilization percentages and slack capacity
    - Color bar showing utilization scale

    Args:
        problem: Network flow problem
        result: Flow solution from solve_min_cost_flow()
        threshold: Minimum utilization to display (default: 0.8 = 80%)
        layout: Graph layout algorithm ("spring", "circular", "kamada_kawai", "planar")
        figsize: Figure size (width, height) in inches
        node_size: Size of node markers
        font_size: Font size for labels
        title: Custom title for the plot (default: "Bottleneck Analysis")

    Returns:
        matplotlib Figure object

    Raises:
        ImportError: If matplotlib or networkx are not installed

    Example:
        >>> from network_solver import build_problem, solve_min_cost_flow, visualize_bottlenecks
        >>>
        >>> problem = build_problem(nodes, arcs, directed=True, tolerance=1e-6)
        >>> result = solve_min_cost_flow(problem)
        >>>
        >>> # Show arcs with ≥80% utilization
        >>> fig = visualize_bottlenecks(problem, result, threshold=0.8)
        >>> fig.savefig("bottlenecks.png")

    See Also:
        visualize_flows() - Full flow visualization
        compute_bottleneck_arcs() - Identify bottlenecks programmatically
    """
    _check_dependencies()

    # Create directed graph
    G = nx.DiGraph() if problem.directed else nx.Graph()

    # Add nodes
    for node_id, node in problem.nodes.items():
        G.add_node(node_id, supply=node.supply)

    # Find bottleneck arcs
    arc_map = {(arc.tail, arc.head): arc for arc in problem.arcs}
    bottleneck_data = []

    for (tail, head), flow in result.flows.items():
        arc = arc_map.get((tail, head))
        if arc is None or arc.capacity is None:
            continue

        if arc.capacity <= problem.tolerance:
            continue

        utilization = abs(flow) / arc.capacity
        if utilization >= threshold:
            slack = arc.capacity - abs(flow)
            bottleneck_data.append((tail, head, flow, arc, utilization, slack))
            G.add_edge(tail, head, flow=flow, arc=arc, utilization=utilization)

    if not bottleneck_data:
        # No bottlenecks found - create empty plot with message
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5,
            0.5,
            f"No bottlenecks found\n(threshold: {threshold * 100:.0f}%)",
            ha="center",
            va="center",
            fontsize=16,
            transform=ax.transAxes,
        )
        ax.set_title(
            title or f"Bottleneck Analysis (≥{threshold * 100:.0f}% utilization)",
            fontsize=14,
            fontweight="bold",
        )
        ax.axis("off")
        return fig

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Compute layout
    layout_funcs = {
        "spring": nx.spring_layout,
        "circular": nx.circular_layout,
        "kamada_kawai": nx.kamada_kawai_layout,
        "planar": nx.planar_layout,
    }
    if layout not in layout_funcs:
        logger.warning(f"Unknown layout '{layout}', using 'spring'")
        layout = "spring"

    try:
        pos = layout_funcs[layout](G)
    except Exception as e:
        logger.warning(f"Layout '{layout}' failed: {e}, using 'spring'")
        pos = nx.spring_layout(G)

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color="lightgray",
        node_size=node_size,
        ax=ax,
    )

    # Draw edges with color gradient based on utilization
    edges = [(tail, head) for tail, head, _, _, _, _ in bottleneck_data]
    utilizations = [util for _, _, _, _, util, _ in bottleneck_data]

    # Use colormap for utilization gradient
    cmap = matplotlib.cm.RdYlGn_r  # type: ignore[attr-defined] # Red (high) to Yellow to Green (low)
    edge_colors = [cmap(util) for util in utilizations]

    # Calculate edge widths
    max_flow = max((abs(flow) for _, _, flow, _, _, _ in bottleneck_data), default=1.0)
    widths = [
        2.0 + 6.0 * (abs(flow) / max_flow) if max_flow > 0 else 2.0
        for _, _, flow, _, _, _ in bottleneck_data
    ]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        edge_color=edge_colors,
        width=widths,
        arrows=True,
        arrowsize=20,
        ax=ax,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw node labels
    node_labels = {node_id: node_id for node_id in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=font_size, ax=ax)

    # Draw edge labels with utilization and slack
    edge_labels = {}
    for tail, head, flow, arc, utilization, slack in bottleneck_data:
        edge_labels[(tail, head)] = (
            f"{abs(flow):.0f}/{arc.capacity:.0f}\n{utilization * 100:.0f}%\nslack:{slack:.0f}"
        )

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=font_size - 2, ax=ax)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=threshold, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Utilization", rotation=270, labelpad=20, fontsize=font_size)

    # Set title
    title_text = title or f"Bottleneck Analysis (≥{threshold * 100:.0f}% utilization)"
    ax.set_title(title_text, fontsize=14, fontweight="bold")
    ax.axis("off")

    # Add statistics
    num_bottlenecks = len(bottleneck_data)
    avg_utilization = sum(utilizations) / len(utilizations)
    stats_text = (
        f"Bottleneck arcs: {num_bottlenecks}\n"
        f"Avg utilization: {avg_utilization * 100:.1f}%\n"
        f"Threshold: {threshold * 100:.0f}%"
    )
    ax.text(
        0.02,
        0.02,
        stats_text,
        transform=ax.transAxes,
        fontsize=font_size - 1,
        verticalalignment="bottom",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    return fig
