# Future Option: Architectural Refactor for NumPy-Native Design

**Status**: On hold - Explore other optimization approaches first  
**Effort**: 2-4 weeks  
**Risk**: High (breaking changes)  
**Potential Benefit**: 2-5x speedup via JIT compilation

---

## Background

Phase 5 JIT attempts failed due to **architectural mismatch**:
- Current solver uses Python-native data structures (dataclasses, list-of-lists, dicts)
- JIT requires NumPy arrays
- Conversion overhead exceeded any JIT speedup

**Key finding**: JIT *would* work if the solver used NumPy arrays throughout, eliminating all conversion overhead.

---

## Proposed Refactor

### Current Architecture (Python-Native)

```python
@dataclass
class ArcState:
    tail: int
    head: int
    cost: float
    flow: float
    in_tree: bool
    # ... 9 total fields

# Solver stores arcs as list of objects
self.arcs: list[ArcState] = [...]

# Tree adjacency as list-of-lists
self.tree_adj: list[list[int]] = [
    [arc1, arc2],      # node 0's adjacent arcs
    [arc3, arc4, arc5], # node 1's adjacent arcs
    # ...
]
```

**Pros**:
- Easy to read and debug
- Pythonic and intuitive
- Type-safe (dataclasses provide structure)
- Good for small-medium problems

**Cons**:
- Can't use Numba JIT (requires NumPy arrays)
- Conversion overhead if we try to JIT
- Memory overhead from Python objects

---

### Proposed Architecture (NumPy-Native)

```python
# Store arc data as parallel NumPy arrays
self.arc_tails: NDArray[np.int32]      # shape: (num_arcs,)
self.arc_heads: NDArray[np.int32]      # shape: (num_arcs,)
self.arc_costs: NDArray[np.float64]    # shape: (num_arcs,)
self.arc_flows: NDArray[np.float64]    # shape: (num_arcs,)
self.arc_in_tree: NDArray[np.bool_]    # shape: (num_arcs,)
self.arc_lowers: NDArray[np.float64]   # shape: (num_arcs,)
self.arc_uppers: NDArray[np.float64]   # shape: (num_arcs,)
self.arc_artificial: NDArray[np.bool_] # shape: (num_arcs,)
# ... etc for all arc fields

# Tree adjacency as CSR (Compressed Sparse Row) format
self.tree_adj_indices: NDArray[np.int32]  # Flattened arc indices
self.tree_adj_offsets: NDArray[np.int32]  # Offset for each node (length: num_nodes+1)

# Access pattern:
def get_adjacent_arcs(node: int) -> NDArray[np.int32]:
    start = self.tree_adj_offsets[node]
    end = self.tree_adj_offsets[node + 1]
    return self.tree_adj_indices[start:end]
```

**Pros**:
- ✅ Enables Numba JIT compilation (no conversion overhead!)
- ✅ Better memory locality (cache-friendly)
- ✅ Vectorized operations possible
- ✅ 2-5x speedup potential for hot loops

**Cons**:
- ❌ Less intuitive (indices instead of objects)
- ❌ Harder to debug (can't inspect "an arc" easily)
- ❌ Breaking change (API changes)
- ❌ Type safety reduced (arrays don't have field names)
- ❌ Large refactor (100s of lines changed)

---

## Migration Strategy

### Phase 1: Internal Refactor (Week 1-2)

**Keep existing API, change internals**:

```python
class SimplexSolver:
    def __init__(self, problem: NetworkProblem):
        # User-facing: keep Arc/Node objects for API compatibility
        arcs_list = [Arc(tail=..., head=..., cost=...) for ...]
        
        # Internal: convert to NumPy arrays once
        self._arc_tails = np.array([arc.tail for arc in arcs_list], dtype=np.int32)
        self._arc_heads = np.array([arc.head for arc in arcs_list], dtype=np.int32)
        # ... etc
        
        # Maintain sync (tricky!)
        self._sync_needed = True
```

**Challenges**:
- Two sources of truth (objects vs arrays)
- When to sync? (performance vs correctness trade-off)
- Complexity increases

**Benefit**: API remains unchanged, internal optimizations possible

---

### Phase 2: JIT Hot Loops (Week 2-3)

Once arrays exist, JIT-compile hot loops:

```python
@njit(cache=True)
def _find_entering_arc_jit(
    arc_costs: NDArray[np.float64],
    arc_in_tree: NDArray[np.bool_],
    arc_lowers: NDArray[np.float64],
    arc_uppers: NDArray[np.float64],
    node_potentials: NDArray[np.float64],
    arc_tails: NDArray[np.int32],
    arc_heads: NDArray[np.int32],
) -> tuple[int, float]:
    """JIT-compiled arc pricing."""
    best_arc = -1
    best_rc = 0.0
    
    for i in range(len(arc_costs)):
        if arc_in_tree[i]:
            continue
        
        # Compute reduced cost
        rc = arc_costs[i] - node_potentials[arc_heads[i]] + node_potentials[arc_tails[i]]
        
        # Check bounds and update best
        if arc_lowers[i] == 0.0 and rc < best_rc:
            best_rc = rc
            best_arc = i
        # ... etc
    
    return best_arc, best_rc
```

**Expected speedup**: 2-3x on hot loops
**No conversion overhead**: Arrays already exist!

---

### Phase 3: Refactor tree_adj to CSR (Week 3-4)

**Current usage patterns**:
```python
# Iterate over adjacent arcs
for arc_idx in self.tree_adj[node]:
    process(arc_idx)

# Append to adjacency list  
self.tree_adj[node].append(arc_idx)
```

**CSR equivalent**:
```python
# Iterate over adjacent arcs (read-only)
start = self.tree_adj_offsets[node]
end = self.tree_adj_offsets[node + 1]
for i in range(start, end):
    arc_idx = self.tree_adj_indices[i]
    process(arc_idx)

# Building adjacency (done once per pivot)
# Use _build_tree_adj_jit() which is already implemented
```

**Challenges**:
- CSR is read-only (can't append)
- Must rebuild on changes (already doing this!)
- Code changes needed in ~15 places

**Benefit**: Eliminates 186s conversion overhead from Phase 5 attempt

---

## Expected Performance Impact

### Conservative Estimate: 2x overall speedup

| Component | Current | With JIT | Speedup |
|-----------|---------|----------|---------|
| rebuild() | 133s | 70s | 1.9x |
| _update_tree_sets() | 123s | 40s | 3.1x |
| collect_cycle() | 65s | 25s | 2.6x |
| pricing | 40s | 20s | 2.0x |
| Other | 270s | 220s | 1.2x |
| **Total** | **631s** | **375s** | **1.7x** |

### Optimistic Estimate: 3-5x overall speedup

With additional optimizations:
- Vectorized reduced cost computation
- Parallel pricing (multi-core)
- Better memory layout
- Could reach 150-200s (3-4x speedup)

---

## Risks and Mitigations

### Risk 1: Breaking API Changes

**Impact**: High  
**Mitigation**: 
- Keep old API as wrapper around new implementation
- Gradual migration path
- Comprehensive test suite (already have 576 tests)

### Risk 2: Bugs from Refactor

**Impact**: High  
**Mitigation**:
- Refactor incrementally (one component at a time)
- Test after each change
- Maintain 100% agreement with OR-Tools (correctness validation)

### Risk 3: Time Investment vs Payoff

**Impact**: Medium  
**Mitigation**:
- Only pursue if Phase 6-7 optimizations don't work
- Prototype first (1 week) to validate approach
- Have clear go/no-go criteria (must achieve 2x speedup minimum)

### Risk 4: Code Readability Loss

**Impact**: Medium  
**Mitigation**:
- Excellent documentation
- Helper functions to access "arc i" data
- Keep high-level algorithms clear

---

## Decision Criteria

### When to pursue this refactor:

✅ **YES, proceed if**:
1. Phase 6-8 optimizations don't achieve target (50x vs OR-Tools)
2. 5-10x speedup is critical for project goals
3. Have 2-4 weeks dedicated to this work
4. Team comfortable with NumPy/Numba programming

❌ **NO, skip if**:
1. Current performance (150-300x slower than OR-Tools) is acceptable
2. Code clarity and maintainability are priorities
3. Limited development time available
4. Other optimization approaches (algorithmic) still unexplored

### Go/No-Go Criteria After Phase 8:

**Proceed with refactor if**:
- Current: Still >100x slower than OR-Tools
- All algorithmic optimizations attempted (Phases 6-8)
- Prototype shows >2x speedup is achievable
- Budget allows 2-4 weeks of work

**Skip refactor if**:
- Achieved <50x slower than OR-Tools through other means
- Educational/research value matters more than speed
- Team prefers code clarity over performance

---

## Recommendation

**Current stance**: Hold off on this refactor

**Rationale**:
1. Phase 5 taught us JIT *can* work with right data layout
2. But refactor is expensive (2-4 weeks, high risk)
3. Should explore other optimization avenues first:
   - Phase 6: Better pricing strategies (1.2-1.8x potential)
   - Phase 7: Memory optimization (1.1-1.3x potential)
   - Phase 8: Parallel pricing (1.2-1.5x potential)
4. If Phases 6-8 collectively achieve 2x speedup, we're at ~75x slower than OR-Tools
5. That may be acceptable for a pure Python educational solver

**Re-evaluate after Phase 8**:
- If still >100x slower and speed is critical → pursue refactor
- If <75x slower or speed not critical → accept current architecture

---

## Reference Implementation

Phase 5 already created the JIT infrastructure:
- `jit_tree_ops.py` has working JIT functions
- `_build_tree_adj_jit()` shows how CSR format works
- `collect_cycle_jit()` shows how BFS can be JIT-compiled

These can serve as templates for the full refactor.

---

## Conclusion

This architectural refactor is a **viable path to 2-5x speedup**, but should be **last resort after exploring other options**.

The decision should be made after Phase 8, based on:
1. Total speedup achieved from Phases 1-8
2. Whether 50x vs OR-Tools goal was met
3. Project priorities (speed vs clarity)
4. Available development time

**Document owner**: User (Jeff)  
**Next review**: After Phase 8 completion  
**Status**: On hold pending Phase 6-8 results

---

**End of Future Architectural Refactor Plan**
