# GPU-Accelerated SCOPE

## Background

This project is forked from [SCOPE](https://github.com/magic62442/subgraph-counting)

We study local subgraph counting queries, Q = (p, o), to count how many times a given k-node pattern graph p appears around every node v in a data graph G when the given node orbit o in p maps to v. 

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

3. Build the project.

Please find your GPU's compute capability at [here](https://developer.nvidia.com/cuda/gpus).

For example, if RTX 5090 is used, it owns a compute capability of 12.0. We build the project like:

```shell
mkdir build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=120 ..
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

The queries are in the ./exp/pattern_graph directory, and the data graphs can be downloaded from [SNAP](https://snap.stanford.edu/data/index.html) or the [Network Repository](https://networkrepository.com). After download, please make sure the data graph and query graph are connected and node ids are consecutive from 0.

## Execution and output

### scope.out:

| Option | Description                                                  |
| ------ | ------------------------------------------------------------ |
| -q     | the query graph path (single query) or directory (batch query) |
| -d     | the data graph path                                          |
| -r     | the result path (single query) or directory (batch query), optional |
| -m     | the available device memory in GB                   |
| -b     | with -b: batch query, without -b: single query               |

Example:

```
./build/executable/scope.out -q ./exp/pattern_graph/5voc/62.txt -d ./exp/data_graph/web-spam.txt -r ./result/5voc/web-spam/62.txt -m 10
./build/executable/scope.out -q ./exp/pattern_graph/5voc/ -d ./exp/data_graph/web-spam.txt -r ./result/5voc/web-spam/ -b -m 10
```

In the output, the $i$-th line shows the local subgraph count of the data node $i-1$.