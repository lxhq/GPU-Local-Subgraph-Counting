#pragma once

#include <unordered_set>
#include <warpcore/counting_hash_table.cuh>

#include "graph.h"
#include "tree.h"
#include "triangle.h"
#include "execution.h"
#include "memory_manager.h"
#include "cuda_helpers.h"
#include "globals.h"
#include "gpu_config.h"
#include "nodeGPU.h"
#include "counting_kernels.h"
#include "wrapper_variables.h"
#include "subgraph_matching.h"

void executeTreeGPU(
        const Tree &t,
        const DataGraph &dun,
        HashTable *H,
        MemoryManager &memory_manager,
        const uint32_t prob_limit,
        float ratio
);

void multiJoinTreeGPU(
        const Tree &t,
        const DataGraph &dun,
        HashTable *H,
        MemoryManager &memory_manager,
        const uint32_t prob_limit,
        float ratio
);

void copyMetaToGPU(
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const EdgeID *outID,
        const EdgeID *unID,
        const EdgeID *reverseID,
        MemoryManager &memory_manager
);