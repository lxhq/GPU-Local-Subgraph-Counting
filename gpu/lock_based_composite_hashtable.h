#pragma once

#include <cstdint>
#include <stdio.h>
#include <cooperative_groups.h>
#include "nodeGPU.h"

__device__ bool insertLockHashTable(uint64_t* hashtable, uint64_t hashtable_size,
                                uint32_t* partial_embeddings, NodeGPU& local_node,
                                uint32_t key, uint32_t prob_limit, uint64_t value, uint32_t index_len, uint32_t insertNodeID);

__device__ uint64_t lookupLockHashTable(uint64_t* hashtable, uint64_t hashtable_size,
                                uint32_t* partial_embeddings, NodeGPU& local_node,
                                uint32_t key, uint32_t prob_limit, uint32_t index_len, uint32_t lookupNodeID);