# Downloading Benchmark Instances

This guide explains how to download DIMACS minimum cost flow benchmark instances for testing and comparison.

## Quick Start

```bash
# Download small instances (recommended - downloads 11 instances, ~700 KB total)
python benchmarks/scripts/download_dimacs.py --small

# Download medium instances (24 instances, ~10 MB compressed)
python benchmarks/scripts/download_dimacs.py --medium

# List all available instances
python benchmarks/scripts/download_dimacs.py --list

# Download all instances
python benchmarks/scripts/download_dimacs.py --all
```

**Status**: ✅ **Automated downloads working!** Files are automatically downloaded from LEMON server, decompressed, and ready to parse.

## Available Instance Families

The download script provides access to the **LEMON Benchmark Suite**, which is a well-maintained collection of DIMACS minimum cost flow problem instances.

### NETGEN Instances
- **Family**: `netgen_small`
- **Description**: Generated using the NETGEN random network generator
- **Size**: 256-512 nodes, 2K-4K arcs per instance (varying density)
- **Count**: 5 instances (netgen_8_08a/b, 09a, 10a, 11a)
- **Problem Type**: General minimum cost flow with varying density
- **Download size**: ~250 KB compressed → ~650 KB decompressed

### GRIDGEN Instances
- **Family**: `gridgen_small`
- **Description**: Grid-based network structures
- **Size**: 256-512 nodes, 2K-4K arcs per instance
- **Count**: 3 instances (gridgen_8_08a/b, 09a)
- **Problem Type**: Grid networks with regular structure
- **Download size**: ~65 KB compressed → ~160 KB decompressed

### GOTO Instances
- **Family**: `goto_small`
- **Description**: Grid-on-torus networks (wrapped grid)
- **Size**: 256-512 nodes, 2K-4K arcs per instance
- **Count**: 3 instances (goto_8_08a/b, 09a)
- **Problem Type**: Toroidal grid networks (no boundary effects)
- **Download size**: ~65 KB compressed → ~160 KB decompressed

### Medium Instances (Phase 5)

**NETGEN Medium** (`netgen_medium`):
- **Size**: 4K-16K nodes, 32K-128K arcs per instance
- **Count**: 8 instances (netgen_8_12a/b/c, 13a/b/c, 14a/b)
- **Download size**: ~3 MB compressed → ~10 MB decompressed

**GRIDGEN Medium** (`gridgen_medium`):
- **Size**: 4K-16K nodes, 32K-128K arcs per instance
- **Count**: 8 instances (gridgen_8_12a/b/c, 13a/b/c, 14a/b)
- **Download size**: ~3 MB compressed → ~10 MB decompressed

**GOTO Medium** (`goto_medium`):
- **Size**: 4K-16K nodes, 32K-128K arcs per instance
- **Count**: 8 instances (goto_8_12a/b/c, 13a/b/c, 14a/b)
- **Download size**: ~3 MB compressed → ~10 MB decompressed

## Command-Line Options

```bash
python benchmarks/scripts/download_dimacs.py [OPTIONS]

Options:
  --list             List all available instances without downloading
  --all              Download all available instances
  --small            Download small instances only (11 instances, ~700 KB)
  --medium           Download medium instances only (24 instances, ~10 MB compressed)
  --large            Download large instances only (future)
  --force            Force re-download even if files exist
  --max-size KB      Only download files ≤KB (compressed size)
```

### Size-Based Filtering

You can limit downloads to files under a certain size threshold using `--max-size`:

```bash
# Download only files ≤20KB (compressed)
python benchmarks/scripts/download_dimacs.py --small --max-size 20

# This allows you to start small and expand capacity later
python benchmarks/scripts/download_dimacs.py --all --max-size 50

# No size limit (downloads everything in selected category)
python benchmarks/scripts/download_dimacs.py --small
```

**Size limiting is based on compressed file size** (the `.min.gz` files on the server). This allows you to:
- Start with tiny instances when disk space or bandwidth is limited
- Gradually increase `--max-size` as capacity allows
- Control exactly how much disk space the benchmark suite uses

**Example sizes** (compressed):
- netgen_8_08a.min.gz: ~15.6 KB → 41.4 KB decompressed
- gridgen_8_08a.min.gz: ~12.3 KB → 35.2 KB decompressed
- goto_8_08a.min.gz: ~10.8 KB → 30.5 KB decompressed

## Download Location

Downloaded instances are stored in:
```
benchmarks/problems/lemon/
├── netgen/
│   ├── netgen_8_08a.min
│   ├── netgen_8_08b.min
│   ├── netgen_8_09a.min
│   ├── netgen_8_10a.min
│   └── netgen_8_11a.min
├── gridgen/
│   ├── gridgen_8_08a.min
│   ├── gridgen_8_08b.min
│   └── gridgen_8_09a.min
└── goto/
    ├── goto_8_08a.min
    ├── goto_8_08b.min
    └── goto_8_09a.min
```

**Note**: 
- Files are automatically decompressed from `.min.gz` to `.min` format
- These files are excluded from git via `.gitignore` to avoid repository bloat
- Each user downloads their own copies (~700 KB total for small instances)

## Using Downloaded Instances

Once downloaded, you can parse and solve the instances using the DIMACS parser:

```python
from benchmarks.parsers.dimacs import parse_dimacs_file
from src.network_solver.solver import solve_min_cost_flow

# Parse a downloaded instance
problem = parse_dimacs_file('benchmarks/problems/lemon/netgen/netgen_8_08a.min')

# Solve it
result = solve_min_cost_flow(problem)

print(f"Status: {result.status}")
print(f"Optimal cost: {result.objective}")
print(f"Iterations: {result.iterations}")
print(f"Nodes: {len(problem.nodes)}")
print(f"Arcs: {len(problem.arcs)}")

# Example output:
# Status: optimal
# Optimal cost: 3934294.0
# Iterations: 287
# Nodes: 256
# Arcs: 2048
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
3. **Manual download**: Visit the specific family URL in your browser:
   - NETGEN: http://lime.cs.elte.hu/~kpeter/data/mcf/netgen/
   - GRIDGEN: http://lime.cs.elte.hu/~kpeter/data/mcf/gridgen/
   - GOTO: http://lime.cs.elte.hu/~kpeter/data/mcf/goto/
4. **Decompress manually**: Use `gunzip` on downloaded `.min.gz` files

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

## Status

**Current (Phase 5)**: 
- ✅ 11 small instances (256-512 nodes)
- ✅ 24 medium instances (4K-16K nodes)
- ✅ Automated download with gzip decompression
- ✅ Size-based filtering support

**Future Phases**:
- Phase 6: Add large instances (>16K nodes) and performance comparison tools
- Phase 7: Real-world road network instances
- Phase 8: Computer vision maximum flow instances

---

**Last Updated**: 2025-10-25  
**Phase**: Phase 5 - Medium Instances & Benchmark Runner
