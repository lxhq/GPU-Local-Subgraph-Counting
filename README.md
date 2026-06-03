# SCOPE-MT

## Background

We study local subgraph counting queries, Q = (p, o), to count how many times a given k-node pattern graph p appears around every node v in a data graph G when the given node orbit o in p maps to v.

This repository contains the multi-threaded CPU implementation, SCOPE-MT. It parallelizes independent prefix ranges and uses per-thread intermediate tables.

## Compile

1. Compile and link to the [nauty](https://pallini.di.uniroma1.it) library.  The nauty library is used to compute automorphisms and symmetry-breaking rules. We include a copy of the nauty library in /utility/automorphisms and show the steps.

```shell
cd utility/automorphism/
./configure
make
mv nauty.a libnauty.a
```

If it complains, "relocation R_X86_64_32 against `.rodata.str1.1' can not be used when making a shared object; recompile with the "-fPIC" option. 

```shell
cd utility/automorphism/
vim makefile
# add -fPIC to the end of line 6.
make
mv nauty.a libnauty.a
```

2. Compile and link to the [GLPK](https://www.gnu.org/software/glpk/) library. The GLPK library is used to compute fractional edge covers. Edit the paths in CMakeLists.txt accordingly.

3. Install or provide [oneTBB](https://github.com/uxlfoundation/oneTBB). SCOPE-MT uses TBB for multi-threaded execution, and CMake locates it through `find_package(TBB REQUIRED)`.

4. Build the project in Release mode.

```shell
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

## Input format

The data graph should start with 'n, m' where n is the number of nodes and m is the number of undirected edges, followed by the edge list. The node id should be consecutive and should start from 0.

Example:

```
3 2
0 1
1 2
```

The query graph file has an additional line, '1 id', where 'id' is this pattern's orbit(representative).

Example:

```
3 2
0 1
1 2
1 0
```

The queries are in the ./exp/pattern_graph directory, and the data graphs can be downloaded from [SNAP](https://snap.stanford.edu/data/index.html) or the [Network Repository](https://networkrepository.com).

## Preprocessing

Usage:

```shell
./preprocess.out <data graph input path> <reordered graph output path> <intersection cache output path>
```

We preprocessed the data graphs in our experiments by indexing triangles as a simple intersection cache. It is static, and there is no swap. This can make SCOPE faster. Note that EVOKE also index triangles., and DISC has a more advanced intersection cache with swapping.

## Execution and output

### scope.out:

| Option | Description                                                  |
| ------ | ------------------------------------------------------------ |
| -q     | the query graph path (single query) or directory (batch query) |
| -d     | the data graph path                                          |
| -t     | the intersection cache path, optional                        |
| -r     | the result path (single query) or directory (batch query), optional |
| -b     | with -b: batch query, without -b: single query               |
| -n     | number of worker threads. The default is 20 |
| -match-only | run subgraph enumeration only, without table-reset and join-aggregation operations |
| -profile-reset | run normal execution and report table-reset timing |

Example:

```
./build/executable/scope.out -q ./exp/pattern_graph/5voc -d ./exp/data_graph/web-spam.txt -r ./result/5voc/web-spam -n 20 -b
./build/executable/scope.out -q ./exp/pattern_graph/5voc/62.txt -d ./exp/data_graph/web-spam.txt -r ./result/5voc/web-spam/62.txt -n 20
```

These commands run SCOPE-MT with 20 threads, which is also the default thread count.

In the output, the $i$-th line shows the local subgraph count of the data node $i-1$.

### Time-Breakdown Profiling

The following options are intended for reproducing the SCOPE-MT component-level time-breakdown experiments. They are disabled by default for normal runs and are intended for the parallel, non-sharing execution path without triangle acceleration.

| Option | Description |
| --- | --- |
| `-profile-reset` | Run normal execution and report CPU table-reset time, calls, and bytes in stdout. |
| `-match-only` | Run without any table-reset and join-aggregation operations. |

For the reported time breakdown, we use three timing sources.

First, run the normal computation without profiling flags. This gives the clean per-query runtime:

```shell
./build/executable/scope.out \
  -q ./exp/pattern_graph/5voc \
  -d ./exp/data_graph/web-spam.txt \
  -n 20 \
  -b
```

Second, run the normal computation with reset profiling. This run is used to measure per-query table-reset time; its total runtime is not used as the clean total runtime in the reported breakdown:

```shell
./build/executable/scope.out \
  -q ./exp/pattern_graph/5voc \
  -d ./exp/data_graph/web-spam.txt \
  -n 20 \
  -b \
  -profile-reset
```

Third, run match-only mode to measure subgraph enumeration without table reset or join aggregation:

```shell
./build/executable/scope.out \
  -q ./exp/pattern_graph/5voc \
  -d ./exp/data_graph/web-spam.txt \
  -n 20 \
  -b \
  -match-only
```

The component times are computed as:

```text
Total = sum of per-query execution times from the clean normal run
SM    = sum of per-query execution times from the match-only run
Reset = sum of per-query reset wall-time estimates from the profile-reset run
JA    = Total - SM - Reset
```

Unlike GPU-SCOPE-LF, SCOPE-MT does not use adaptive GPU batch schedules, so no batch-schedule record/replay step is needed for this CPU time-breakdown measurement.

### batch.out

This version uses the precomputed plan for queries. We use it for GNN datasets, where there are thousands of relatively small data graphs to count. You need to specify the query graph path and intersection cache path. We provided generated plans in ./exp/plan for all 5-node and 6-node queries. To precompute plans for other query sets, you need to modify the function 'generatePlan' in 'batch.cpp'.

### 5voc.out

We further optimized the code for orbit counting for 5-vertex queries by hand, following the generated plans. This is not included in the paper. It is about 2-3 times faster than EVOKE. You need to specify the data graph path and intersection cache path.
