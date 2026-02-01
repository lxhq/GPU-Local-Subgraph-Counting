#pragma once

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdint>
#include <warpcore/counting_hash_table.cuh>

#include "globals.h"
#include "nodeGPU.h"
#include "gpu_config.h"
#include "lock_based_composite_hashtable.h"
#include "lock_free_hashtable.h"

__global__ void firstLevelKernel(uint32_t index_len, uint32_t next_start_index, uint32_t count, uint32_t* target_);

__global__ void countKernel(uint32_t index_len, uint32_t depth, uint32_t* embedding, uint64_t embedding_size, uint32_t* matching_count, NodeGPU* d_node);

__global__ void writeKernel(uint32_t index_len, uint32_t depth, uint32_t* embedding, uint64_t embedding_size, uint64_t* exclusive_sum, uint32_t* new_embedding,
                            uint32_t next_level_max_embedding_count_, uint32_t* actual_start_index, NodeGPU* d_node);


__global__ void indexPrefixKernel(
        uint32_t index_len,
        uint32_t level,
        uint32_t* target,
        uint32_t next_level_count,
        uint32_t* nodesNeedsIndexing,
        uint32_t nodesCount
);

__global__ void indexPrefixWithinBatchKernel(
        uint32_t index_len,
        uint32_t level,
        uint32_t* target,
        uint32_t next_level_count,
        uint32_t* nodesNeedsIndexing,
        uint32_t nodesCount,
        uint32_t batchSize
);

__global__ void copyPrefixKernel(
        uint32_t* source,
        uint32_t source_count,
        uint32_t* target,
        uint32_t index_len,
        NodeGPU* d_node,
        uint32_t level
);

__global__ void finalLevelWithLocalCacheKernel(
        uint32_t index_len,
        uint32_t depth,
        uint32_t* embedding,
        uint64_t embedding_size,
        NodeGPU* d_node,
        uint64_t* failed_write,
        uint64_t** d_d_H,
        void** d_d_partition_H,
        uint64_t hashtable_size,
        bool isRoot,
        uint32_t numTreeNodes,
        uint32_t nID,
        uint32_t prob_limit
);

__global__ void finalLevelWithEdgeWithLocalCacheKernel(
        uint32_t index_len,
        uint32_t depth,
        uint32_t* embedding,
        uint64_t embedding_size,
        NodeGPU* d_node,
        uint64_t* failed_write,
        uint64_t** d_d_H,
        void** d_d_partition_H,
        uint64_t hashtable_size,
        bool isRoot,
        uint32_t numTreeNodes,
        uint32_t nID,
        uint32_t prob_limit
);

__global__ void finalLevelKernel(
        uint32_t index_len,
        uint32_t depth,
        uint32_t* embedding,
        uint64_t embedding_size,
        NodeGPU* d_node,
        uint64_t* failed_write,
        uint64_t** d_d_H,
        void** d_d_partition_H,
        uint64_t hashtable_size,
        bool isRoot,
        uint32_t numTreeNodes,
        uint32_t nID,
        uint32_t prob_limit
);

__global__ void finalLevelWithEdgeKernel(
        uint32_t index_len,
        uint32_t depth,
        uint32_t* embedding,
        uint64_t embedding_size,
        NodeGPU* d_node,
        uint64_t* failed_write,
        uint64_t** d_d_H,
        void** d_d_partition_H,
        uint64_t hashtable_size,
        bool isRoot,
        uint32_t numTreeNodes,
        uint32_t nID,
        uint32_t prob_limit
);

__global__ void writeToHashTableKernel(
        uint32_t index_len,
        uint32_t depth,
        uint32_t* embedding,
        uint64_t embedding_size,
        NodeGPU* d_node,
        uint64_t* failed_write,
        uint64_t** d_d_H,
        void** d_d_partition_H,
        uint64_t hashtable_size,
        bool isRoot,
        uint32_t numTreeNodes,
        uint32_t nID,
        uint32_t prob_limit
);

__global__ void writeToHashTableWithEdgeKeyKernel(
        uint32_t index_len,
        uint32_t depth,
        uint32_t* embedding,
        uint64_t embedding_size,
        NodeGPU* d_node,
        uint64_t* failed_write,
        uint64_t** d_d_H,
        void** d_d_partition_H,
        uint64_t hashtable_size,
        bool isRoot,
        uint32_t numTreeNodes,
        uint32_t nID,
        uint32_t prob_limit
);

