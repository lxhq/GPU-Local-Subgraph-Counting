# GPU-Accelerated Local Subgraph Counting

## System Configuration
This project is intended to be built and run on Linux. We run our experiments on Ubuntu 22.04 LTS.

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

Build the project using CMake. Replace `120` below with your GPU's specific architecture code. For example, compute capability 12.0 corresponds to `120`.

```shell
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 ..
make
```

### 3. Optional Build Arguments

We provide the following optional CMake arguments.

`HASH_TABLE_TYPE` selects the backend implementation:

| Value | Description | Paper Reference |
| --- | --- | --- |
| **0** | Use the original `warpcore` library. |  |
| **1** | **(Default)** Use a lightweight `warpcore` (unused functionalities removed). | **GPU-SCOPE-LF** |
| **2** | Use lock-based GPU hash tables. | **GPU-SCOPE-LOCK** |
| **3** | Use dense arrays. | **GPU-SCOPE** |

`ENABLE_OCCUPANCY_PROFILE` controls whether hash-table occupancy instrumentation is compiled:

| Value | Description |
| --- | --- |
| **OFF** | **(Default)** Build the normal runtime path without occupancy instrumentation. |
| **ON** | Enable the `-occupancy-profile` runtime option for reproducing hash-table occupancy measurements. This is intended for profiling only and currently supports `HASH_TABLE_TYPE=1`. |

**Example:**

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DHASH_TABLE_TYPE=3 ..
```
Build the **GPU-SCOPE** variant.

```shell
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120 -DHASH_TABLE_TYPE=1 -DENABLE_OCCUPANCY_PROFILE=ON ..
```
Build the **GPU-SCOPE-LF** variant with hash-table occupancy profiling enabled.

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

* **[Download Dataset Here](https://drive.google.com/file/d/1EFw2_urKWxES21-OPCIcwzFn3IdcXAuL/view?usp=drive_link)**

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

> **Note:** Please make sure the result directory exists. Otherwise, the result files will not be written. For example, please make sure `./result/web-spam/5voc/` exists before executing the batch command above.

### Output

The output file contains the local subgraph counts. The $i$-th line corresponds to the count for **Vertex ID $i-1$** in the data graph.

For example, consider the output below (**the bracketed line numbers are added for readability and are not part of the output file**):
```text
[1] 0
[2] 0
... ...
[4766] 2569147
[4767] 2390996
```
As the vertex IDs always start from 0, **line 1 represents** the local subgraph count for **vertex 0**, and similarly **line 4767 represents** the local subgraph count for vertex 4766.

## Advanced Configuration (Optional)

The following arguments are available for fine-tuning performance.

| Option | Description |
| --- | --- |
| `-prob` | The probing budget for the open-addressing hash table. **Default: 64**. |
| `-mem` | Total device memory budget (in GB). **Default: 90% of available memory**. |
| `-exec-mem` | Execution workspace memory budget (in GB). This limits the memory used inside each tree execution, including subgraph-enumeration buffers and aggregation tables. **Default: 0**, meaning no separate execution cap beyond `-mem`. |
| `-ratio` | Ratio between memory for Subgraph Enumeration (SE) and Hash Table (HT). **Default: 1**. |
| `-occupancy-profile` | Print average hash-table occupancy in stdout. This requires building with `-DENABLE_OCCUPANCY_PROFILE=ON` and is currently supported for the lock-free hash-table build (`HASH_TABLE_TYPE=1`) only. |

### Query-Structure Profiling

Use `-query-structure-profile` to print the decomposition properties used in the query-structure analysis and exit without running the data-graph computation. This mode only requires `-q` and optional `-b`.

| Option | Description |
| --- | --- |
| `-query-structure-profile` | Print query-decomposition properties and exit before data-graph execution. Requires `-q`; supports `-b` for a query directory. |

```shell
./build/executable/scope.out \
  -q ./exp/pattern_graph/6voc/ \
  -b \
  -query-structure-profile
```

The output columns are `query`, `partition_number`, `shared_vertex_set_size`, and `tree_width`. These values are computed as the maximum over all generated decomposition trees of each query; `tree_width` is reported as the graph-theoretic tree width, i.e., the internal code width minus one.

### Time-Breakdown Profiling

The time-breakdown experiment uses three runs to isolate the main components without changing the normal execution path. Directly inserting timers around every internal operation is undesirable because the components are coupled through the batch schedule, and fine-grained timer instrumentation can perturb GPU execution. Since batch boundaries may vary across executions, we record and replay the batch schedule so that all component measurements use the same batch boundaries.

In the first run, the program executes normally. This gives the clean total runtime.

In the second run, the program executes normally with reset profiling enabled and records the batch schedule. This run provides the table-reset time and the batch boundaries used for replay. Its total runtime is not used as the clean total runtime in the reported breakdown.

In the third run, the program replays the same workload using the recorded batch schedule with `-match-only`, which disables table reset and join aggregation. Therefore, the replay measures the subgraph-enumeration time under the same batch boundaries.

The component times are computed as follows:

```text
Total runtime             = clean normal runtime
Subgraph enumeration time = match-only replay runtime in the third run
Table reset time          = table-reset time reported in the second run
Join-aggregation time     = Total runtime
                            - Subgraph enumeration time
                            - Table reset time
```

The following options are intended for reproducing the component-level time-breakdown experiments. They are disabled by default for normal runs.

| Option | Description |
| --- | --- |
| `-profile-reset` | Measure GPU table reset time, calls, and bytes in stdout. |
| `-record-batch-schedule <path>` | Write the adaptive GPU batch schedule to a binary file or directory. In batch-query mode, one `.schedule.bin` file is written per query. |
| `-replay-batch-schedule <path>` | Replay a previously recorded adaptive GPU batch schedule from a binary file or directory. |
| `-match-only` | Run without any table-reset and join-aggregation operations. This option must be used together with `-replay-batch-schedule`. |

First, run the normal execution:

```shell
./build/executable/scope.out \
  -q ./exp/pattern_graph/5voc/ \
  -d ./exp/data_graph/web-spam.txt \
  -r ./result/web-spam/5voc/ \
  -b
```

Second, run the normal execution with reset profiling and schedule recording enabled:

```shell
./build/executable/scope.out \
  -q ./exp/pattern_graph/5voc/ \
  -d ./exp/data_graph/web-spam.txt \
  -r ./result/web-spam/5voc/ \
  -b \
  -profile-reset \
  -record-batch-schedule ./result/web-spam/5voc_schedule/
```

Third, replay the recorded schedule with match-only mode enabled:

```shell
./build/executable/scope.out \
  -q ./exp/pattern_graph/5voc/ \
  -d ./exp/data_graph/web-spam.txt \
  -b \
  -replay-batch-schedule ./result/web-spam/5voc_schedule/ \
  -match-only
```

### Restart-Cost Profiling

The restart-cost experiment uses two main runs. In the first run, the program executes normally and records the prefix ranges that finish successfully before each hash-table insertion failure. In the second run, the program replays the same workload using these recorded prefix ranges as fixed batch boundaries. Therefore, the replay skips the trial ranges that previously triggered insertion failures. Since each restart is caused by such an insertion failure, this replay reduces the restart count and the refined replay is used for the restart-free timing. We estimate the restart discovery cost as:

```text
Restart cost = record-run runtime - restart-free replay runtime
Cost fraction = Restart cost / record-run runtime
```

| Option | Description |
| --- | --- |
| `-record-restart-schedule <path>` | Write the completed prefix ranges that finish before hash-table insertion failures to a binary file or directory. In batch-query mode, one `.schedule.bin` file is written per query. |
| `-replay-restart-schedule <path>` | Replay previously recorded completed prefix ranges as fixed batch boundaries. |

No extra option is required for restart-frequency profiling; normal execution always prints the restart counters. A restart is counted when a planned prefix batch is truncated by a hash-table insertion failure. The printed fields include the number of restart profile batch iterations, restart count, attempted prefixes, completed prefixes, truncated prefixes, truncated fraction, and safeguard count.

First, run the normal execution with schedule recording enabled:

```shell
./build/executable/scope.out \
  -q ./exp/pattern_graph/6voc/103.txt \
  -d ./exp/data_graph/web-spam.txt \
  -r ./result/web-spam/6voc_103.txt \
  -record-restart-schedule ./result/web-spam/6voc_103_restart.schedule.bin
```

Then replay the recorded ranges:

```shell
./build/executable/scope.out \
  -q ./exp/pattern_graph/6voc/103.txt \
  -d ./exp/data_graph/web-spam.txt \
  -r ./result/web-spam/6voc_103_replay.txt \
  -replay-restart-schedule ./result/web-spam/6voc_103_restart.schedule.bin
```

If a replay run still reports nonzero restarts, record a refined schedule by using `-replay-restart-schedule` and `-record-restart-schedule` together with different paths, then replay the refined schedule.

### Nsight Compute Metrics

We profile these GPU implementations using the three metrics below, which are
available on RTX 5090:

| Figure metric | Nsight Compute metric | How it is used |
| --- | --- | --- |
| Instructions per cycle | `sm__inst_executed.avg.per_cycle_active` | Averaged across profiled kernels, weighted by kernel duration. |
| Global writes (GB) | `dram__bytes_write.sum` | Summed across profiled kernel invocations and converted to GB. |
| Atomic active cycles (M) | `lts__d_atomic_input_cycles_active.avg` | Multiplied by the number of invocations, summed across profiled kernels, and converted to millions of cycles. |

## Comparison

In the paper, we report the speedup over the original CPU SCOPE implementation. Our goal is to process queries **without any pre-built index**. We found that the original [SCOPE](https://github.com/magic62442/subgraph-counting) code may encounter runtime errors when running without a pre-built triangle index (```-t``` option).

For reproducibility, we include our bug-fixed version of SCOPE in the `SCOPE` branch of this repository. We also include our multi-threaded CPU baseline, SCOPE-MT, in the `SCOPE-MT` branch. The build process for both branches is similar to the build process for this GPU version (see above), except that SCOPE-MT additionally requires oneTBB.

For the additional GPU baselines used in our comparison, please refer to:

* [lxhq/VDMC-LSC](https://github.com/lxhq/VDMC-LSC)
* [lxhq/G2Miner-LSC](https://github.com/lxhq/G2Miner-LSC)
