# Downloading Benchmark Instances

This guide explains how to download DIMACS minimum cost flow benchmark instances for testing and comparison.

## Quick Start

```bash
# Download small instances (recommended for initial testing)
python benchmarks/scripts/download_dimacs.py --small

# List all available instances
python benchmarks/scripts/download_dimacs.py --list

# Download all instances
python benchmarks/scripts/download_dimacs.py --all
```

## Available Instance Families

The download script provides access to the **LEMON Benchmark Suite**, which is a well-maintained collection of DIMACS minimum cost flow problem instances.

### NETGEN Instances
- **Family**: `netgen_small`
- **Description**: Generated using the NETGEN random network generator
- **Size**: ~8,000 nodes, ~20,000 arcs per instance
- **Count**: 3 instances
- **Problem Type**: General minimum cost flow with varying density

### GRIDGEN Instances
- **Family**: `gridgen_small`
- **Description**: Grid-based network structures
- **Size**: ~8,000 nodes, ~30,000 arcs per instance
- **Count**: 2 instances
- **Problem Type**: Grid networks with regular structure

### GOTO Instances
- **Family**: `goto_small`
- **Description**: Grid-on-torus networks (wrapped grid)
- **Size**: ~8,000 nodes, ~32,000 arcs per instance
- **Count**: 2 instances
- **Problem Type**: Toroidal grid networks (no boundary effects)

## Command-Line Options

```bash
python benchmarks/scripts/download_dimacs.py [OPTIONS]

Options:
  --list          List all available instances without downloading
  --all           Download all available instances
  --small         Download small instances only (recommended)
  --medium        Download medium instances only (future)
  --large         Download large instances only (future)
  --force         Force re-download even if files exist
```

## Download Location

Downloaded instances are stored in:
```
benchmarks/problems/lemon/
├── netgen/
│   ├── netgen-8-1.dmx
│   ├── netgen-8-2.dmx
│   └── netgen-8-3.dmx
├── gridgen/
│   ├── gridgen-8-1.dmx
│   └── gridgen-8-2.dmx
└── goto/
    ├── goto-8-1.dmx
    └── goto-8-2.dmx
```

**Note**: These files are excluded from git via `.gitignore` to avoid repository bloat. Each user downloads their own copies.

## Using Downloaded Instances

Once downloaded, you can parse and solve the instances using the DIMACS parser:

```python
from benchmarks.parsers.dimacs import parse_dimacs_file
from src.network_solver.solver import solve_min_cost_flow

# Parse a downloaded instance
problem = parse_dimacs_file('benchmarks/problems/lemon/netgen/netgen-8-1.dmx')

# Solve it
result = solve_min_cost_flow(problem)

print(f"Status: {result.status}")
print(f"Optimal cost: {result.objective}")
print(f"Iterations: {result.iterations}")
```

## Source and License

**Source**: LEMON Benchmark Suite
- URL: https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData
- Maintained by: ELTE University, Budapest

**Citation**:
```
Péter Kovács. Minimum-cost flow algorithms: an experimental evaluation.
Optimization Methods and Software, 30:94-127, 2015.
```

**License**:
- **LEMON library code**: Boost Software License 1.0 (very permissive)
- **Generated instances** (NETGEN, GRIDGEN, GOTO): Public Domain
- **Attribution**: Required for academic/research use

## Troubleshooting

### Download fails with connection error

The LEMON benchmark server may occasionally be unavailable. If downloads fail:

1. **Wait and retry**: The server may be temporarily down
2. **Check your internet connection**: Ensure you can reach http://lime.cs.elte.hu
3. **Manual download**: Visit https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData in your browser

### File already exists

By default, the script skips files that already exist. Use `--force` to re-download:

```bash
python benchmarks/scripts/download_dimacs.py --small --force
```

## Extending the Instance Collection

To add more instance families:

1. Edit `benchmarks/scripts/download_dimacs.py`
2. Add new entries to the `DIMACS_INSTANCES` dictionary
3. Follow the existing pattern with `url_base`, `files`, and `local_dir`
4. Test with `--list` to verify the new family appears

See `docs/benchmarks/BENCHMARK_SOURCES.md` for other potential sources:
- OR-Library instances
- CommaLAB multicommodity instances
- Network Repository datasets

## Phase 4 Status

**Current**: MVP implementation with 7 small instances (~8K nodes each)

**Future Phases**:
- Phase 5: Add medium instances (10K-50K nodes)
- Phase 6: Add large instances (>50K nodes)
- Phase 7: Real-world road network instances
- Phase 8: Computer vision maximum flow instances

---

**Last Updated**: 2025-10-25  
**Phase**: Phase 4 - Problem Acquisition
