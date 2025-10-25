# LEMON Benchmark Instances

This directory is for storing DIMACS minimum cost flow benchmark instances from the LEMON project.

## Phase 4 MVP Status

**Note**: The automated download script framework is in place (`benchmarks/scripts/download_dimacs.py`), but specific instance URLs need verification as the LEMON server structure may have changed.

## Manual Download (Current Recommended Method)

1. **Visit the LEMON Benchmark Data page**:
   https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData

2. **Download instance families**:
   - NETGEN: http://lime.cs.elte.hu/~kpeter/data/mcf/netgen/
   - GRIDGEN: http://lime.cs.elte.hu/~kpeter/data/mcf/gridgen/
   - GOTO: http://lime.cs.elte.hu/~kpeter/data/mcf/goto/
   - GRIDGRAPH: http://lime.cs.elte.hu/~kpeter/data/mcf/gridgraph/
   - ROAD: http://lime.cs.elte.hu/~kpeter/data/mcf/road/
   - VISION: http://lime.cs.elte.hu/~kpeter/data/mcf/vision/

3. **Save files here** following this structure:
   ```
   benchmarks/problems/lemon/
   ├── netgen/         # NETGEN instances
   ├── gridgen/        # GRIDGEN instances  
   ├── goto/           # GOTO instances
   ├── gridgraph/      # GRIDGRAPH instances
   ├── road/           # Road network instances
   └── vision/         # Computer vision instances
   ```

4. **Files are not committed to git** (excluded via `.gitignore`)

## Using Downloaded Instances

Once you have downloaded `.dmx` files (DIMACS format), you can parse and solve them:

```python
from benchmarks.parsers.dimacs import parse_dimacs_file
from src.network_solver.solver import solve_min_cost_flow

# Parse instance
problem = parse_dimacs_file('benchmarks/problems/lemon/netgen/some-instance.dmx')

# Solve
result = solve_min_cost_flow(problem)
print(f"Optimal cost: {result.objective}")
```

## Citation

When using LEMON benchmark instances, please cite:

```
Péter Kovács. Minimum-cost flow algorithms: an experimental evaluation.
Optimization Methods and Software, 30:94-127, 2015.
```

## License

- **LEMON library code**: Boost Software License 1.0 (very permissive)
- **Generated instances** (NETGEN, GRIDGEN, GOTO, GRIDGRAPH): Public Domain
- **Real-world instances** (ROAD, VISION): Verify original source licenses

See `benchmarks/metadata/licenses.json` for complete license information.

## Future Work

- Verify exact file names and URLs on LEMON server
- Update `benchmarks/scripts/download_dimacs.py` with working URLs
- Automate the download process
- Add checksum verification

---

**Last Updated**: 2025-10-25  
**Phase**: Phase 4 - Problem Acquisition (MVP)
