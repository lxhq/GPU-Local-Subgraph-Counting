# GPU-Accelerated Local Subgraph Counting

## Background

We start this project by forking from [SCOPE](https://github.com/magic62442/subgraph-counting).

We study **local subgraph counting queries**, denoted as $Q=(q,o)$. The goal is to count how many times a given $k$-node pattern graph $q$ appears around every node $v$ in a data graph $G$, such that the node orbit $o$ in $q$ maps to $v$.

## System Configuration
### 1. Compilers and Build Tools
Please ensure that ```gcc```, ```g++```, ```cmake```, and ```make``` are installed. You can verify the installations and check their versions using:
```shell
gcc --version
g++ --version
cmake --version
make --version
```
> **Note:** Please ensure the versions of gcc and g++ are aligned (i.e., they are the same version). Mismatched versions may cause the compiler to fail when linking C++ standard libraries.

**Reference**: In our experiments, we used GCC/G++ 11.4.0.

### 2. CUDA Environment
Please ensure that the NVIDIA GPU driver and the CUDA compiler (```nvcc```) are both installed. You can check their status with:
```shell
# Check GPU driver status
nvidia-smi
# Check CUDA compiler version
nvcc --version
```
If these tools are missing, please download and install the CUDA Toolkit from the [official NVIDIA website](https://developer.nvidia.com/cuda/toolkit).

**Reference**: In our experiments, we used CUDA Toolkit 12.8.

## Prerequisites

Before building the project, please verify or compile the required dependencies.

### 1. hpc_helper and kiss_rng

This project uses the GPU open-addressing hash table [warpcore](https://github.com/sleeepyjack/warpcore). It depends on [hpc_helper](https://gitlab.rlp.net/pararch/hpc_helpers) and [kiss_rng](https://github.com/sleeepyjack/kiss_rng).

> **Note:** These dependencies will be downloaded **automatically** by CMake when you build the project. Please ensure you have an active Internet connection during the build process.

### 2. Nauty Library

The [nauty](https://pallini.di.uniroma1.it) library is used to compute automorphisms and symmetry-breaking rules. A copy is included in `utility/automorphism`.

```shell
cd utility/automorphism/
./configure

# Edit the makefile to add -fPIC to the CFLAGS
# 1. Open the makefile
vim makefile
# 2. Append '-fPIC' to the end of line 6 (CFLAGS=...)

make
mv nauty.a libnauty.a
```

### 3. GLPK Library

The [GLPK](https://www.gnu.org/software/glpk/) library is used to compute fractional edge covers. You must compile and install it.

**Example Installation (v4.35):**
You can download GLPK from the [GNU FTP](https://ftp.gnu.org/gnu/glpk/) and install it as follows:

```shell
tar -xzf glpk-4.35.tar.gz
cd glpk-4.35
./configure
make
make check
sudo make install
```

**Linking Verification:**
Pay attention to the output of `sudo make install`. In our experiment, the headers and libraries are installed in:

* `/usr/local/include`
* `/usr/local/lib`

If your installation path differs, please update the paths in `CMakeLists.txt` (around line 70) accordingly.

## Build Instructions

### 1. Determine Compute Capability

Find your GPU's compute capability on the [NVIDIA Developer website](https://developer.nvidia.com/cuda/gpus).

For example, if an **RTX 5090** is used, it has a compute capability of **12.0**.

### 2. Compile the Project

Build the project using CMake. Replace `120` below with your GPU's specific architecture code.

```shell
mkdir build
cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=120 ..
make
```

### 3. Optional Build Arguments

We provide a `HASH_TABLE_TYPE` option to select the backend implementation.

| Value | Description | Paper Reference |
| --- | --- | --- |
| **0** | Use the original `warpcore` library. |  |
| **1** | **(Default)** Use a lightweight `warpcore` (unused functionalities removed). | **GPU-SCOPE-LF** |
| **2** | Use lock-based GPU hash tables. | **GPU-SCOPE-LOCK** |
| **3** | Use dense arrays. | **GPU-SCOPE** |

**Example:**

```shell
cmake -DCMAKE_CUDA_ARCHITECTURES=120 -DHASH_TABLE_TYPE=3 ..
```
Build **GPU-SCOPE** variant

## Input Format

### Data Graph

The file must start with `n m` (vertices, undirected edges), followed by the edge list. Vertex IDs must be consecutive integers starting from 0.

```text
3 2
0 1
1 2
```

### Query Graph

The format is identical to the data graph, but with an additional footer line `1 id`, where `id` is the pattern's representative orbit.

```text
3 2
0 1
1 2
1 0
```

### Datasets

The queries are provided in `./exp/pattern_graph`. Public data graphs can be downloaded from [SNAP](https://snap.stanford.edu/data/index.html) or [Network Repository](https://networkrepository.com).

**Our Processed Dataset:**
We have made the dataset used in our experiments available on Google Drive:

* **[Download Dataset Here](https://drive.google.com/file/d/1pQoCaGwohyY22HehHkIhjm7SLf0aKhCS/view?usp=drive_link)**

## Execution and Output

The executable is located at `./build/executable/scope.out`.

| Option | Required? | Description |
| --- | --- | --- |
| `-q` | Yes | Query graph path (single file) OR directory (batch mode). |
| `-d` | Yes | Data graph path. |
| `-r` | Optional | Result path (single file) OR directory (batch mode). |
| `-b` | No | Batch mode flag (required if `-q` is a directory). |

### Examples

**Running a Single Query:**

```shell
./build/executable/scope.out \
  -q ./exp/pattern_graph/5voc/62.txt \
  -d ./exp/data_graph/web-spam.txt \
  -r ./result/web-spam/5voc/62.txt
```

**Running a Batch of Queries:**

```shell
./build/executable/scope.out \
  -q ./exp/pattern_graph/5voc/ \
  -d ./exp/data_graph/web-spam.txt \
  -r ./result/web-spam/5voc/ \
  -b
```
> **Note:** Please make sure the result directory exist. Otherwise, the result file won't be written. For example, please make sure '''./result/web-spam/5voc/''' exist before executing above commands.
### Output

The output file contains the local subgraph counts. The $i$-th line corresponds to the count for **Node ID $i-1$** in the data graph.

---

## Advanced Configuration (Optional)

The following arguments are available for fine-tuning performance. **We simply used the default value in our experiment**

| Option | Description |
| --- | --- |
| `-prob` | The probing budget for the open-addressing hash table. **Default: 64**. |
| `-mem` | Total device memory budget (in GB). **Default: 90% of available memory**. |
| `-ratio` | Ratio between memory for Subgraph Enumeration (SE) and Hash Table (HT). **Default: 1**. |
