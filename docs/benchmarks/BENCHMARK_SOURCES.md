# Benchmark Sources for Network Flow Problems

This document catalogs publicly available benchmark sources for minimum cost network flow problems, including licensing information, problem characteristics, and access methods.

**Last Updated**: 2025-10-25  
**Phase**: Phase 1 - Research and Cataloging

---

## Table of Contents

1. [Benchmark Repositories](#benchmark-repositories)
2. [Problem Generators](#problem-generators)
3. [License Summary](#license-summary)
4. [Recommendations](#recommendations)
5. [References](#references)

---

## Benchmark Repositories

### 1. DIMACS Implementation Challenge

**Name**: First DIMACS Implementation Challenge - Network Flows and Matching (1990-1991)

**URLs**:
- Main archive: http://archive.dimacs.rutgers.edu/Challenges/
- Published volume: http://archive.dimacs.rutgers.edu/Volumes/Vol12.html
- Network Repository mirror: https://networkrepository.com/dimacs.php

**Description**:
The inaugural DIMACS Implementation Challenge focused on network flow and matching algorithms. Participants from the US, Europe, and Japan conducted experiments from November 1990 to August 1991, culminating in a workshop at Rutgers University in October 1991.

**Problem Types**:
- Minimum cost flow
- Maximum flow
- Matching (maximum weight, maximum cardinality)
- Assignment problems

**Problem Characteristics**:
- Sizes: Small to very large (varies by instance family)
- Formats: DIMACS standard format
- Known solutions: Many instances have published optimal solutions

**Data Sources**:
- Standard generators: NETGEN, GRIDGEN, GOTO, GRIDGRAPH
- Real-world instances: Transportation networks
- Hand-crafted instances: Designed to challenge specific algorithms

**Access**:
- FTP: `ftp://dimacs.rutgers.edu/pub/netflow/`
- HTTP archive: http://archive.dimacs.rutgers.edu/

**License**:
- **Type**: Public Domain / Academic Use
- **Terms**: Benchmark data appears to be in the public domain and freely available for educational and scientific use
- **Restrictions**: Published volumes may have additional AMS (American Mathematical Society) copyright restrictions
- **Redistribution**: Allowed for educational and scientific purposes with proper acknowledgment
- **Citation**: Required - cite the DIMACS volume and original problem sources

**Citation**:
```
Johnson, D.S. and McGeoch, C.C. (Eds.). Network Flows and Matching: 
First DIMACS Implementation Challenge. DIMACS Series in Discrete 
Mathematics and Theoretical Computer Science, Volume 12, 
American Mathematical Society, Providence, RI, 1993.
```

**Notes**:
- Users are requested to notify DIMACS of usage to help document research impact
- License restrictions may apply to published materials through AMS
- Benchmark instances themselves appear freely available

---

### 2. OR-Library

**Name**: OR-Library (J.E. Beasley Collection)

**URLs**:
- Main site: https://people.brunel.ac.uk/~mastjjb/jeb/info.html
- Legal page: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/legal.html
- Network flow problems: https://people.brunel.ac.uk/~mastjjb/jeb/orlib/ (various subdirectories)

**Description**:
Comprehensive collection of test problems for operations research, originally described in Beasley (1990). Includes various network flow problem types.

**Problem Types**:
- Single commodity network flow
- Concave cost network flow
- Single source network flow
- Uncapacitated network flow
- Various other OR problems (not network flow related)

**Problem Characteristics**:
- Sizes: Varies by problem family
- Formats: Various formats, some use DIMACS, others use custom formats
- Known solutions: Most problems include known optimal solutions

**Access**:
- Direct download from Brunel University website
- Problems organized by problem type in subdirectories

**License**:
- **Type**: Academic Use with Attribution
- **Terms**: Data sets can be used by students and teachers for OR purposes
- **Restrictions**: Proper acknowledgment and identification of J.E. Beasley as author required
- **Redistribution**: Allowed with attribution
- **Citation**: Required

**Citation**:
```
Beasley, J.E. OR-Library: distributing test problems by electronic mail. 
Journal of the Operational Research Society 41(11) (1990) pp1069-1072.
```

**Notes**:
- Consult legal page for specific restrictions
- Contact Professor Beasley through Brunel University for detailed terms
- Many problem families have specific formats - NOT a single standard format
- **Scoping recommendation**: Start with OR-Library instances that use DIMACS format

---

### 3. LEMON Benchmark Data

**Name**: LEMON (Library for Efficient Modeling and Optimization in Networks)

**URLs**:
- Main library: https://lemon.cs.elte.hu/
- Benchmark data: https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData
- License page: https://lemon.cs.elte.hu/trac/lemon/wiki/License

**Description**:
Benchmark test suite used in experimental evaluation of minimum-cost flow algorithms. Part of the COIN-OR open-source optimization project.

**Problem Types**:
- Minimum cost flow (primary focus)
- Various network structures (grids, random, real-world)

**Problem Characteristics**:
- Sizes: Small to very large
- Formats: Primarily DIMACS format
- Known solutions: Optimal solutions included for validation

**Data Sources**:
- Standard generators: NETGEN, GRIDGEN, GOTO, GRIDGRAPH
- Real-world networks: Road networks (TIGER/Line data from 9th DIMACS Challenge)
- Computer vision: Maximum flow instances from University of Western Ontario

**Access**:
- Download from LEMON wiki page
- Instances available at FTP site referenced in publications

**License**:
- **Type**: Boost Software License 1.0
- **Terms**: Free of charge for any person or organization, commercial or non-commercial
- **Restrictions**: Minimal - must include copyright notices in source code distributions
- **Redistribution**: Freely allowed
- **Citation**: Recommended (see below)

**Citation**:
```
Péter Kovács. Minimum-cost flow algorithms: an experimental evaluation. 
Optimization Methods and Software, 30:94-127, 2015.
```

**Notes**:
- Part of COIN-OR initiative
- Very permissive license - excellent choice for benchmarking
- Library implementations available in C++
- Benchmark data is separate from library code

---

### 4. CommaLAB

**Name**: CommaLAB - Computational Mathematics Laboratory, University of Pisa

**URLs**:
- Main site: https://commalab.di.unipi.it/
- Single-commodity MCF: https://commalab.di.unipi.it/datasets/mcf/
- Multi-commodity MCF: https://commalab.di.unipi.it/datasets/mmcf/

**Description**:
Collection of test instances and generators for various network flow problem types, hosted by the Computational Mathematics Laboratory at University of Pisa.

**Problem Types**:
- Linear multicommodity flow
- Nonlinear multicommodity flow
- Multicommodity network design
- Unsplittable multicommodity network flow
- Single-commodity minimum cost flow

**Problem Characteristics**:
- Sizes: Varies widely by family
- Formats: Various formats (DIMACS, custom formats)
- Known solutions: Some instances have known optima

**Data Families**:
- Mnetgen, PDS, JLF, hydrothermal (linear)
- Vance, AerTranspo, Planar/Grid, Jump (linear)
- genflot, 3DTables (nonlinear)
- Canad, Reserve, Gunluk (network design)
- FNSS-based: GARR, SNDLib, Waxman (unsplittable)
- Mingozzi (unsplittable design)

**Access**:
- Direct download from CommaLAB website
- Organized by problem type

**License**:
- **Type**: Unspecified / Academic Use Implied
- **Terms**: Site described as "service to the optimization community"
- **Restrictions**: Various datasets attributed to specific researchers
- **Redistribution**: **Unclear** - recommend contacting CommaLAB team
- **Citation**: Required - varies by dataset, cite original researchers

**Notes**:
- **No explicit license terms stated** - verification needed before redistribution
- Some instances derive from real-world networks (Italian GARR, SNDLib)
- MMCFBlock component is part of open-source SMS++ project
- **Recommendation**: Contact CommaLAB for explicit redistribution permissions
- Attribution appears necessary for researcher-contributed datasets

**Attribution Examples**:
- Frangioni, Gendron, Castro, Vance (various families)
- See individual dataset pages for specific citations

---

### 5. Network Repository

**Name**: Network Repository (Interactive Network Data Repository)

**URLs**:
- Main site: https://networkrepository.com/
- DIMACS networks: https://networkrepository.com/dimacs.php
- Policy page: https://networkrepository.com/policy.php

**Description**:
The first interactive network data repository with real-time visual analytics. Hosts various network datasets including DIMACS benchmark instances.

**Problem Types**:
- DIMACS challenge instances (various challenges)
- General network data
- Graph datasets from various domains

**Problem Characteristics**:
- Sizes: Wide range
- Formats: Various formats including DIMACS
- Known solutions: Depends on specific dataset

**Access**:
- Web download interface with interactive exploration
- Direct file downloads

**License**:
- **Type**: Creative Commons Attribution-ShareAlike (CC BY-SA style)
- **Terms**: Attribution required for any public use
- **Restrictions**: 
  - Must provide link to license and indicate changes
  - ShareAlike: derivatives must use same license
  - No additional legal/technological restrictions allowed
  - Dataset-specific citations must be honored
- **Redistribution**: Allowed with attribution and ShareAlike
- **Citation**: Required - varies by dataset

**Notes**:
- Public domain elements exempt from license requirements
- No warranties given
- Each dataset page may have additional citation requests
- More restrictive than some other sources due to ShareAlike requirement

---

### 6. Computer Vision Research Group, University of Western Ontario

**Name**: Computer Vision Research Group Benchmark Instances

**URLs**:
- Original site: http://vision.csd.uwo.ca/data/maxflow/
- (May be mirrored elsewhere - verify current availability)

**Description**:
Maximum flow problem instances arising from computer vision applications. Used as basis for some LEMON benchmark conversions.

**Problem Types**:
- Maximum flow (computer vision applications)
- Can be converted to minimum cost flow

**Problem Characteristics**:
- Sizes: Large-scale problems from image processing
- Formats: Custom formats
- Known solutions: Optimal flow values typically included

**Access**:
- Check University of Western Ontario website
- May be mirrored by LEMON or other repositories

**License**:
- **Type**: Unspecified / Academic Use Implied
- **Terms**: No explicit license found
- **Restrictions**: Unknown
- **Redistribution**: **Unclear** - recommend contacting research group
- **Citation**: Required - cite original research papers

**Notes**:
- **Verify current availability** - website may have moved
- Some instances used in LEMON benchmark suite
- Contact University of Western Ontario Computer Vision group for current terms
- **Recommendation**: May be easier to use LEMON-hosted versions with clear license

---

## Problem Generators

### 1. NETGEN

**Name**: NETGEN - Network Flow Problem Generator

**Original Paper**:
```
Klingman, D., A. Napier, and J. Stutz. NETGEN: A Program for 
Generating Large Scale Capacitated Assignment, Transportation, 
and Minimum Cost Flow Network Problems. Management Science 
20.5 (1974): 814-821.
```

**Languages**: Fortran 77 (original), C (equivalent port)

**Capabilities**:
- Generates capacitated and uncapacitated problems
- Minimum cost flow / transshipment problems
- Transportation problems
- Assignment problems
- Fully connected or sparse network structures

**Algorithm**:
1. Defines source and sink nodes with random supply distribution
2. Generates "skeleton arcs" to ensure feasibility (paths from sources to sinks with adequate capacity)
3. Randomly generates additional arcs until desired density reached

**Output Format**: DIMACS standard format

**Source Code**:
- **Fortran 77**: https://www.netlib.org/lp/generators/netgen
- **C version**: 
  - http://archive.dimacs.rutgers.edu/pub/netflow/generators/network/netgen/netgen.c
  - http://elib.zib.de/pub/mp-testdata/generators/netgen/netgen.c

**Modern Implementations**:
- **Python**: https://pypi.org/project/pynetgen/ (PyNETGEN)
- **GitHub library**: https://github.com/emmanuj/netgen

**License**: Public Domain / Netlib (freely available)

**Notes**:
- **No maintained Python ports** of original - implementations are independent rewrites
- Original Fortran/C versions are standard reference
- Widely used in OR literature
- **For MVP**: Use existing NETGEN-generated instances or wrap C version with subprocess calls

---

### 2. GRIDGEN

**Name**: GRIDGEN - Grid Network Generator

**Languages**: C

**Capabilities**:
- Generates grid-like network structures
- Rectangular array of nodes with specified rows and columns
- Master source and sink on opposite sides
- Produces structured networks for testing

**Output Format**: DIMACS standard format

**Source Code**:
- http://archive.dimacs.rutgers.edu/pub/netflow/generators/network/gridgen/gridgen.c

**License**: Public Domain / DIMACS

**Notes**:
- Available from DIMACS netflow archive
- Often used with NETGEN for problem variety
- Grid structures useful for testing specific algorithms

---

### 3. GOTO

**Name**: GOTO - Grid On Torus Generator

**Author**: Andrew V. Goldberg, Stanford University (1991)

**Languages**: C

**Capabilities**:
- Generates grid on torus network structures
- Wraps grid edges to create toroidal topology
- Provides unique network characteristics

**Output Format**: DIMACS standard format

**Source Code**:
- http://archive.dimacs.rutgers.edu/pub/netflow/generators/network/grid-on-torus/goto.c

**License**: Public Domain / DIMACS

**Notes**:
- Created for DIMACS 1st Implementation Challenge
- Toroidal structure eliminates boundary effects
- Available from DIMACS archive

---

### 4. GRIDGRAPH

**Name**: GRIDGRAPH - Grid-based Graph Generator

**Languages**: C

**Capabilities**:
- Generates grid-based graph structures
- Part of DIMACS generator suite
- Produces problems in standard format

**Output Format**: DIMACS standard format

**Source Code**:
- Available from DIMACS archive: http://archive.dimacs.rutgers.edu/
- (Specific download link to be determined during implementation)

**License**: Public Domain / DIMACS

**Notes**:
- One of six standard DIMACS generators
- Used alongside NETGEN, GRIDGEN, GOTO, Complete, RmfGen

---

## License Summary

### Can Redistribute with Acknowledgment
- **DIMACS** - Public domain / academic use, cite challenge and sources
- **LEMON** - Boost Software License 1.0, very permissive
- **Network Repository** - CC BY-SA, attribution and ShareAlike required

### Requires Verification / Contact
- **OR-Library** - Academic use with attribution, verify specific terms with J.E. Beasley
- **CommaLAB** - No explicit license, contact for redistribution permissions
- **CV Western Ontario** - No explicit license, verify availability and terms

### General Requirements

**All Sources**:
- ✅ **Citation required** for academic/research use
- ✅ **Acknowledgment** of original authors/sources
- ⚠️ **Verify current terms** before redistribution or commercial use

**Problem Generators**:
- ✅ NETGEN, GRIDGEN, GOTO, GRIDGRAPH - Public domain / freely available
- ✅ Can be downloaded, compiled, and used
- ✅ Generated instances can be redistributed (cite generator)

---

## Recommendations

### For MVP (Phase 1-5)

1. **Primary Source: DIMACS**
   - Well-documented standard format
   - Clear (permissive) license terms
   - Large collection with known solutions
   - Use download scripts, not file commits

2. **Secondary Source: LEMON**
   - Excellent license (Boost 1.0)
   - High-quality benchmark suite
   - Known optimal solutions
   - Subset overlaps with DIMACS

3. **Defer to Later**:
   - OR-Library: Multiple formats, need format research
   - CommaLAB: License unclear, contact needed
   - CV Western Ontario: Availability uncertain

### Distribution Strategy

**Recommended Approach**:
- **Download scripts** rather than committing benchmark files
- Store in `benchmarks/problems/` (excluded from git via `.gitignore`)
- Commit `problem_catalog.json` with metadata
- Include license information in catalog
- Provide clear attribution in documentation

**For a Few Small Instances**:
- Can commit 2-3 small representative problems (<10KB each)
- Include license header in file comments
- Useful for testing without downloads

### Citation Best Practices

1. **In Code/Documentation**:
   ```
   # Problem source: DIMACS 1st Implementation Challenge
   # Generated by: NETGEN
   # Citation: Johnson & McGeoch (1993), DIMACS Volume 12
   ```

2. **In Publications**:
   - Cite the benchmark source collection
   - Cite specific generator if used
   - Acknowledge original authors

3. **In Repository**:
   - `benchmarks/metadata/licenses.json` - Full license information
   - `benchmarks/README.md` - Quick reference and attributions
   - Individual problem files - Headers with source information

---

## References

### Primary Publications

1. **DIMACS 1st Challenge**:
   Johnson, D.S. and McGeoch, C.C. (Eds.). Network Flows and Matching: First DIMACS Implementation Challenge. DIMACS Series in Discrete Mathematics and Theoretical Computer Science, Volume 12, American Mathematical Society, Providence, RI, 1993.

2. **OR-Library**:
   Beasley, J.E. OR-Library: distributing test problems by electronic mail. Journal of the Operational Research Society 41(11) (1990) pp1069-1072.

3. **LEMON Benchmarks**:
   Péter Kovács. Minimum-cost flow algorithms: an experimental evaluation. Optimization Methods and Software, 30:94-127, 2015.

4. **NETGEN**:
   Klingman, D., A. Napier, and J. Stutz. NETGEN: A Program for Generating Large Scale Capacitated Assignment, Transportation, and Minimum Cost Flow Network Problems. Management Science 20.5 (1974): 814-821.

### Web Resources

- DIMACS Archive: http://archive.dimacs.rutgers.edu/Challenges/
- OR-Library: https://people.brunel.ac.uk/~mastjjb/jeb/info.html
- LEMON: https://lemon.cs.elte.hu/
- LEMON Benchmarks: https://lemon.cs.elte.hu/trac/lemon/wiki/MinCostFlowData
- CommaLAB: https://commalab.di.unipi.it/
- Network Repository: https://networkrepository.com/
- Netlib: https://www.netlib.org/lp/generators/

---

## Appendix: Problem Characteristics

### Problem Size Categories

**Small**: 10-100 nodes, 50-500 arcs
- Good for: Unit testing, parser validation, algorithm debugging
- Examples: Hand-crafted instances, small NETGEN problems

**Medium**: 100-1,000 nodes, 500-10,000 arcs
- Good for: Performance testing, solver comparison
- Examples: Standard NETGEN/GRIDGEN instances

**Large**: 1,000-10,000 nodes, 10,000-100,000 arcs
- Good for: Scalability testing, optimization benchmarking
- Examples: Large NETGEN, real-world networks

**Very Large**: >10,000 nodes, >100,000 arcs
- Good for: Stress testing, parallel algorithm evaluation
- Examples: Road networks, very large generated instances

### Problem Type Classification

1. **Transportation**: Bipartite, sources → sinks only
2. **Assignment**: n×n bipartite, unit values
3. **General MCF**: Arbitrary structure with transshipment nodes
4. **Max Flow**: Single source, single sink, uniform costs
5. **Multicommodity**: Multiple commodities sharing network capacity

### Format Notes

**DIMACS Standard Format**:
```
c Comment lines
p min <nodes> <arcs>
n <node_id> <supply>
a <tail> <head> <lower> <upper> <cost>
```

**Variations**:
- Some generators omit lower bounds (assumed 0)
- Node IDs: typically 1-indexed integers
- Supply balance: Σ supply = 0 required
- Comments: ignored by parsers

---

## Version History

- **2025-10-25**: Initial Phase 1 research and cataloging
  - Comprehensive license audit
  - Problem generator documentation
  - Benchmark source identification

---

**Disclaimer**: This document represents research findings as of the last update date. License terms may change. Always verify current terms with source organizations before redistribution or commercial use. When in doubt, contact the data providers directly.
