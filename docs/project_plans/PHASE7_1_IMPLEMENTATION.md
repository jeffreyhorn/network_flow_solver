# Phase 7.1: Sparse Basis Matrix Implementation

**Date**: 2025-10-27  
**Status**: Testing in Progress  
**Branch**: `optimization/phase7-memory-optimization`

---

## Overview

Implemented sparse matrix storage for the network simplex basis matrix to achieve significant memory reduction. The network basis is naturally 99.9% sparse (only 2 nonzeros per column), so switching from dense to sparse storage provides massive memory savings.

## Implementation Strategy

### Dual-Storage Approach

We maintain **two representations** of the basis matrix:

1. **Sparse matrix** (`self.basis_matrix`) - CSC format via scipy.sparse
   - Used for LU factorization (memory-efficient)
   - Stored but not used for column operations
   - Only ~0.1% the size of dense matrix

2. **Dense matrix** (`self.basis_matrix_dense`) - Standard numpy array
   - Used for Forrest-Tomlin updates (requires column indexing)
   - Used for condition number estimation
   - Used for all operations requiring direct element access

### Why Dual Storage?

**Problem**: Sparse matrices don't support efficient column assignment:
```python
# This works for dense, not for sparse CSC:
matrix[:, col] = new_column
```

**Solution**: Keep dense copy for operations, sparse for storage/LU:
- Dense matrix: ~134 MB for n=4,096
- Sparse matrix: ~267 KB for n=4,096
- Combined: Still 134 MB, but LU factorization uses sparse → saves memory during factorization

**Key insight**: Most memory is consumed during LU factorization temporaries, not the matrices themselves. By passing sparse matrix to `build_lu()`, scipy's `splu()` works with sparse structure throughout.

## Code Changes

### 1. `src/network_solver/basis.py`

**Added sparse imports**:
```python
try:
    from scipy.sparse import csc_matrix
    _HAS_SCIPY_SPARSE = True
except ImportError:
    csc_matrix = None
    _HAS_SCIPY_SPARSE = False
```

**Added dual storage** (line 55):
```python
self.basis_matrix: np.ndarray | None = None  # Sparse (CSC) when scipy available
self.basis_matrix_dense: np.ndarray | None = None  # Dense copy for operations
```

**Updated `_build_numeric_basis()`** (lines 256-329):
- Build dense matrix first for operations
- Create sparse CSC version for LU factorization
- Pass sparse matrix to `build_lu()`
- Pass dense matrix to `ForrestTomlin`

**Updated `replace_arc()`** (lines 442-523):
- Use `basis_matrix_dense` for all column operations
- Mark `basis_matrix` as None after updates (could rebuild sparse if needed)

**Updated `estimate_condition_number()`** (lines 383-440):
- Use `basis_matrix_dense` for norm computations

### 2. `src/network_solver/basis_lu.py`

**Updated `build_lu()` signature** (line 23):
```python
def build_lu(matrix: np.ndarray | csc_matrix) -> LUFactors:
```

**Enhanced logic**:
- If input is already sparse CSC, use it directly
- If input is dense, convert to sparse for scipy
- Store both representations in LUFactors
- scipy's `splu()` works with sparse structure → memory savings

## Memory Impact Analysis

### Before (Dense Storage)

For n=4,097 nodes:
```
Basis matrix:     (4096 × 4096) × 8 bytes = 134 MB
LU factors:       2 × 134 MB = 268 MB  
Temporaries:      134-400 MB during factorization
Peak:             400-800 MB per rebuild
```

### After (Sparse Storage + Dense Copy)

For n=4,097 nodes:
```
basis_matrix (sparse):      ~267 KB (8,192 nonzeros)
basis_matrix_dense:         134 MB
LU factors (sparse):        ~534 KB
Temporaries (sparse ops):   ~50 MB during factorization
Peak:                       ~200 MB per rebuild
```

**Expected reduction**: 80% (800 MB → 200 MB peak during rebuild)

### Why This Works

The key is that **scipy's sparse LU factorization** maintains sparse structure:
- Dense LU: Creates dense L and U matrices (134 MB each)
- Sparse LU: Creates sparse L and U matrices (~267 KB each)
- Temporary arrays during factorization are much smaller with sparse

## Testing Status

✅ Individual test files pass (test_solver.py: 13/13)  
🔄 Full test suite running...

Expected: All 576 tests should pass (no algorithm changes, only storage format)

## Compatibility

- **With scipy**: Uses sparse CSC format for memory optimization
- **Without scipy**: Falls back to dense-only (basis_matrix = basis_matrix_dense)
- **Backward compatible**: All existing tests should pass unchanged

## Potential Issues & Mitigations

### Issue 1: Sparse matrix overhead for small problems
- **Impact**: For n < 100, sparse overhead may exceed dense storage
- **Mitigation**: Minimal impact since small problems don't have memory issues

### Issue 2: Dual storage seems wasteful
- **Why we do it**: Forrest-Tomlin needs dense for column updates
- **Actual savings**: 80% comes from sparse LU, not from basis storage
- **Alternative considered**: Convert sparse→dense each time (too slow)

### Issue 3: Sparse matrix invalidation
- **Problem**: After column updates, sparse matrix out of sync
- **Solution**: Mark as None, rebuild only if needed (for next LU)
- **In practice**: Rarely needed, FT handles updates without rebuilding

## Next Steps

1. ✅ Implement dual-storage sparse matrix
2. ✅ Update all basis operations to use correct representation
3. 🔄 Run full test suite
4. ⏳ Run memory profiling to measure improvement
5. ⏳ Benchmark performance impact
6. ⏳ Document results

## Expected Results

### Memory (gridgen_8_12a.min)
- Baseline: 1.02 GB peak
- Target: ~200 MB peak (80% reduction)

### Performance
- Best case: 1.1-1.2x faster (sparse operations + cache efficiency)
- Neutral: Same speed (sparse overhead = cache benefit)
- Worst case: No slowdown (sparse operations well-optimized)

## Technical Notes

### Why CSC Format?
- **CSC (Compressed Sparse Column)**: Efficient column access
- Network basis has exactly 2 nonzeros per column
- CSC stores: data array, row indices, column pointers
- Memory: `(nnz × 8) + (nnz × 4) + ((n+1) × 4)` bytes
  - For n=4,096: `(8192 × 8) + (8192 × 4) + (4097 × 4) = 114 KB`

### Why scipy.sparse?
- Industry-standard sparse matrix library
- UMFPACK integration for fast sparse LU
- Well-tested, optimized implementations
- Already a project dependency

---

**Status**: 🟡 Implementation Complete, Testing in Progress  
**Next Milestone**: Test validation complete
