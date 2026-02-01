#pragma once

#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include "graph.h"
#include "tree.h"
#include "memory_manager.h"
#include "cuda_helpers.h"
#include "globals.h"
#include "gpu_config.h"
#include "nodeGPU.h"
#include "counting_kernels.h"
#include "wrapper_variables.h"


class SubgraphMatching {
private:
    MemoryManager& memory_manager_;             // memory manager reference
    uint32_t* source_;                          // memory for maintain the source embeddings
    uint32_t source_embedding_count_;           // number of source embeddings
    NodeGPU* d_node_;                           // node information on device memory
    uint64_t target_memory_size_;               // memory size for target_ memory location in bytes
    uint32_t level_;                            // current level
    bool is_last_level_;                        // whether it is the last level, no need to leave space for exclusive sum for the last level
    uint32_t index_len_;                        // the index length of each embedding   
    uint32_t* target_;                          // memory for maintain the target embeddings
    uint64_t* exclusive_sum_;                   // exclusive sum for the next level embeddings
    uint32_t next_start_index_;                 // the next start index of the source level
    uint64_t next_level_total_embedding_count_; // number of embeddings in the next level
    uint32_t next_level_max_embedding_count_;   // number of embeddings that can be stored in the target_ memory location

public:
    // compute exclusive sum for the next level if current level is not the first level
    SubgraphMatching(MemoryManager& memory_manager, uint32_t *source, uint32_t source_embedding_count_,
                    NodeGPU* d_node, uint64_t target_memory_size, uint32_t level,
                    bool is_last_level, uint32_t index_len);
    
    // release the allocated memory
    ~SubgraphMatching();

    // return true if next_start_index_ < source_embedding_count_ otherwise return false
    bool hashNext() {
        return next_start_index_ < source_embedding_count_ && (next_level_total_embedding_count_ != 0);
    }         

    // write down the next level embeddings, move the next_start_index, return the number of next level embeddings
    uint32_t* next(uint32_t& count);
};