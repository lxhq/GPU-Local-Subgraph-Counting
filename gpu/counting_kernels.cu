#include "counting_kernels.h"


__forceinline__ __device__ uint64_t lookup(
    uint64_t* h,
    NodeGPU& local_node,
    uint32_t index_len,
    void* ht,
    uint64_t hashtable_size,
    uint32_t* local_partial_embedding,
    uint32_t lookupPos,
    uint32_t key,
    const cooperative_groups::thread_block_tile<1>& ht_group,
    uint32_t prob_limit,
    uint32_t lookUpNodeID
) {
    if (ht == nullptr) {
        return h[key];
    }
#if HASH_TABLE_TYPE == 0
    uint64_t retrieve_key = ((uint64_t)local_partial_embedding[lookupPos] << 32) | key;
    uint64_t cnt = 0;
    ((warpcore::CountingHashTable<>*)ht)->retrieve(retrieve_key, cnt, ht_group);
    return cnt;
#elif HASH_TABLE_TYPE == 1
    return lookupHashTable(
            (uint64_t*)ht,
            hashtable_size,
            local_partial_embedding[lookupPos],
            key,
            prob_limit
        );
#elif HASH_TABLE_TYPE == 2
    return lookupLockHashTable(
        (uint64_t*)ht, hashtable_size, local_partial_embedding,
        local_node, key, prob_limit, index_len, lookUpNodeID);
#elif HASH_TABLE_TYPE == 3
    return ((uint64_t*)ht)[local_partial_embedding[lookupPos] * C_BASELINE_TABLE_SIZE + key];
#endif
}

__forceinline__ __device__ bool insert(
    unsigned long long* h,
    NodeGPU& local_node,
    uint32_t index_len,
    void* ht,
    uint64_t hashtable_size,
    uint32_t* local_partial_embedding,
    uint32_t storePos,
    uint32_t key,
    const cooperative_groups::thread_block_tile<1>& ht_group,
    uint64_t cnt,
    uint32_t prob_limit,
    uint32_t insertNodeID
) {
    if (ht == nullptr) {
        atomicAdd(&h[key], cnt);
        return false;
    }
#if HASH_TABLE_TYPE == 0
    const auto status = ((warpcore::CountingHashTable<>*)ht)->insert(((uint64_t)local_partial_embedding[storePos] << 32) | key, cnt, ht_group);
    return status != warpcore::Status::none() && status != warpcore::Status::duplicate_key();
#elif HASH_TABLE_TYPE == 1
    return !insertHashTable(
            (uint64_t*)ht,
            hashtable_size,
            local_partial_embedding[storePos],
            key,
            prob_limit,
            cnt
    );
#elif HASH_TABLE_TYPE == 2
    return !insertLockHashTable(
        (uint64_t*)ht, hashtable_size, local_partial_embedding,
        local_node, key, prob_limit, cnt, index_len, insertNodeID
    );
#elif HASH_TABLE_TYPE == 3
    atomicAdd(&((uint64_t*)ht)[local_partial_embedding[storePos] * C_BASELINE_TABLE_SIZE + key], cnt);
    return false;
#endif
}

// performs galloping binary search to find the index of the smallest value in the array that is greater than or equal to the target.
__forceinline__ __device__ uint32_t GallopingBinarySearch(uint32_t* array, uint32_t end, uint32_t target) {
    // Check if the target is less than or equal to the first element
    if (array[0] >= target)
        return 0;

    // Initialize the bound for exponential search
    uint32_t bound = 1;

    // Exponential search to find the range where the target may exist
    while (bound < end && array[bound] < target) {
        bound *= 2;
    }

    // Calculate the left and right bounds for binary search
    uint32_t left = bound / 2;
    uint32_t right = min(bound, end - 1);

    // Perform binary search within the identified range
    while (left <= right) {
        uint32_t mid = left + ((right - left) >> 1);
        uint32_t mid_value = array[mid];
        if (mid_value == target) {
            return mid;
        } else if (mid_value < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    // 'left' is the index of the smallest value >= target
    // If all elements are less than the target, the last index is returned
    return left < end ? left : end - 1;
}

__forceinline__ __device__ bool gpuIntersection(uint32_t prob_vertex, uint32_t shorest_neighbor,
                                                uint32_t all_pred_neighbor_count,
                                                uint32_t* all_pred_neighbor_size,
                                                uint32_t** all_pred_neighbor) {
    for (uint32_t i = 0; i < all_pred_neighbor_count; i++) {
        if (i == shorest_neighbor) {
            continue;
        }
        uint32_t* array = all_pred_neighbor[i];
        uint32_t array_size = all_pred_neighbor_size[i];
        uint32_t idx = GallopingBinarySearch(array, array_size, prob_vertex);
        if (array[idx] != prob_vertex) {
            return false;
        }
    }
    return true;
}

__forceinline__ __device__ uint32_t findDoutEdgeId(uint32_t v, uint32_t w) {
    if (v > w) {
        uint32_t temp = v;
        v = w;
        w = temp;
    }
    uint32_t idx =  GallopingBinarySearch(C_NEIGHBORS[OUT_GRAPH] + C_OFFSETS[OUT_GRAPH][v],
                                            C_OFFSETS[OUT_GRAPH][v + 1] - C_OFFSETS[OUT_GRAPH][v],
                                            w);
    return C_OFFSETS[OUT_GRAPH][v] + idx;
}

__forceinline__ __device__ uint32_t findDunEdgeId(uint32_t v, uint32_t w) {
    uint32_t idx =  GallopingBinarySearch(C_NEIGHBORS[UN_GRAPH] + C_OFFSETS[UN_GRAPH][v],
                                            C_OFFSETS[UN_GRAPH][v + 1] - C_OFFSETS[UN_GRAPH][v],
                                            w);
    return C_OFFSETS[UN_GRAPH][v] + idx;
}

__forceinline__ __device__ uint32_t computeEdgeKey(
        EdgeID startOffset,
        ui pos,
        int edgeType,
        VertexID src,
        VertexID dst
) {
    ui edgeID = 0;
    switch (edgeType) {
        case 1:
            edgeID = startOffset + pos;
            break;
        case 2:
            edgeID = C_OUT_ID[startOffset + pos];
            break;
        case 3: {
            edgeID = findDoutEdgeId(src, dst);
            break;
        }
        case 4: {
            edgeID = findDunEdgeId(src, dst);
            break;
        }
        case 5:
            edgeID = C_UN_ID[startOffset + pos];
            break;
        case 6:
            edgeID = C_REVERSE_ID[startOffset + pos];
            break;
    }
    return edgeID;
}

__global__ void firstLevelKernel(uint32_t index_len, uint32_t next_start_index, uint32_t count, uint32_t* target_) {
    uint64_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_id >= count) {
        return;
    }
    for (uint32_t i = 0; i < index_len; i++) {
        target_[global_thread_id * (index_len + 1) + i] = 0;
    }
    target_[global_thread_id * (index_len + 1) + index_len] = next_start_index + global_thread_id;
}


__global__ void countKernel(uint32_t index_len, uint32_t depth, uint32_t* embedding,
                            uint64_t embedding_size, uint32_t* matching_count, NodeGPU* d_node) {
    __shared__ uint32_t local_partial_embedding[WARP_PER_BLOCK][MAX_PATTERN_SIZE + 2];
    __shared__ uint32_t local_count[WARP_PER_BLOCK];
    __shared__ uint32_t* local_all_pred_neighbor[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_all_pred_neighbor_size[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_max_min_target[WARP_PER_BLOCK][2];
    __shared__ uint32_t local_shortest_neighbor[WARP_PER_BLOCK];
    __shared__ NodeGPU local_node;

    if (threadIdx.x == 0) {
        local_node = *d_node;
        matching_count[embedding_size] = 0;
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint64_t global_warp_id = blockIdx.x * WARP_PER_BLOCK + warp_id;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;

    if (global_warp_id >= embedding_size) {
        return;
    }
    if (lane_id < depth + index_len) {
        local_partial_embedding[warp_id][lane_id] = embedding[global_warp_id * (depth + index_len) + lane_id];
    }
    __syncwarp();

    if (lane_id < local_node.pre_nbrs_count[depth]) {
        uint32_t v = local_partial_embedding[warp_id][local_node.pre_nbrs_pos[depth][lane_id] + index_len];
        uint32_t type = local_node.pre_nbr_graph_type[depth][lane_id];
        local_all_pred_neighbor_size[warp_id][lane_id] = C_OFFSETS[type][v + 1] - C_OFFSETS[type][v];
        local_all_pred_neighbor[warp_id][lane_id] = C_NEIGHBORS[type] + C_OFFSETS[type][v];
    }
    __syncwarp();

    if (lane_id == 0) {
        local_count[warp_id] = 0;
        local_max_min_target[warp_id][0] = 0;
        local_max_min_target[warp_id][1] = UINT32_MAX;

        uint32_t shortest_neighbor = UINT32_MAX;
        uint32_t neighbor_size = UINT32_MAX;
        for (uint32_t i = 0; i < local_node.pre_nbrs_count[depth]; i++) {
            uint32_t size = local_all_pred_neighbor_size[warp_id][i];
            if (size < neighbor_size) {
                neighbor_size = size;
                shortest_neighbor = i;
            }
        }
        local_shortest_neighbor[warp_id] = shortest_neighbor;
        if (local_node.nodeGreaterPosCount[depth] > 0) {
            local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeGreaterPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len] > local_max_min_target[warp_id][0]) {
                    local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len];
                }
            }
        }
        if (local_node.nodeLessPosCount[depth] > 0) {
            local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeLessPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len] < local_max_min_target[warp_id][1]) {
                    local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len];
                }
            }
        }
    }
    __syncwarp();
    uint32_t num_iters = (local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]] + WARP_SIZE - 1) / WARP_SIZE;
    for (uint32_t iter = 0; iter < num_iters; iter++) {
        uint32_t index = lane_id + iter * WARP_SIZE;
        if (index < local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]]) {
            uint32_t prob_vertex = local_all_pred_neighbor[warp_id][local_shortest_neighbor[warp_id]][index];
            bool found = gpuIntersection(
                            prob_vertex,
                            local_shortest_neighbor[warp_id],
                            local_node.pre_nbrs_count[depth],
                            local_all_pred_neighbor_size[warp_id],
                            local_all_pred_neighbor[warp_id]
            );
            if (found) {
                for (uint32_t i = 0; i < depth; i++) {
                    if (local_partial_embedding[warp_id][i + index_len] == prob_vertex) {
                        found = false;
                        break;
                    }
                }
                if (local_node.nodeGreaterPosCount[depth] > 0) {
                    if (prob_vertex < local_max_min_target[warp_id][0]) found = false;
                }
                if (local_node.nodeLessPosCount[depth] > 0) {
                    if (prob_vertex > local_max_min_target[warp_id][1]) found = false;
                }
            }
            if (found) {
                auto group = cooperative_groups::coalesced_threads();
                uint32_t rank = group.thread_rank();
                if (rank == 0) {
                    local_count[warp_id] += group.size();
                }
            }
        }
        __syncwarp();
    }

    // update the matching count
    if (lane_id == 0) {
        matching_count[global_warp_id] = local_count[warp_id];
    }
}

__global__ void writeKernel(uint32_t index_len, uint32_t depth, uint32_t* embedding, uint64_t embedding_size, uint64_t* exclusive_sum, uint32_t* new_embedding,
                            uint32_t next_level_max_embedding_count_, uint32_t* actual_start_index, NodeGPU* d_node) {
    __shared__ uint32_t local_partial_embedding[WARP_PER_BLOCK][MAX_PATTERN_SIZE + 2];
    __shared__ uint32_t* local_all_pred_neighbor[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_all_pred_neighbor_size[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint64_t local_exclusive_sum[WARP_PER_BLOCK][2];
    __shared__ uint32_t* local_write_position[WARP_PER_BLOCK];
    __shared__ uint32_t local_shortest_neighbor[WARP_PER_BLOCK];
    __shared__ uint32_t local_max_min_target[WARP_PER_BLOCK][2];
    __shared__ NodeGPU local_node;
    __shared__ uint64_t start_exclusive_sum;

    if (threadIdx.x == 0) {
        local_node = *d_node;
        start_exclusive_sum = exclusive_sum[0];
    }
    __syncthreads();

    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint64_t global_warp_id = blockIdx.x * WARP_PER_BLOCK + warp_id;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;

    if (global_warp_id >= embedding_size) {
        return;
    }
    if (lane_id < depth + index_len) {
        local_partial_embedding[warp_id][lane_id] = embedding[global_warp_id * (depth + index_len) + lane_id];
    }
    if (lane_id == 0) {
        local_exclusive_sum[warp_id][0] = exclusive_sum[global_warp_id];
        local_exclusive_sum[warp_id][1] = exclusive_sum[global_warp_id + 1];
        local_write_position[warp_id] = new_embedding + (local_exclusive_sum[warp_id][0] - start_exclusive_sum) * (depth + index_len + 1);
        local_max_min_target[warp_id][0] = 0;
        local_max_min_target[warp_id][1] = UINT32_MAX;
    }
    __syncwarp();
    if (local_exclusive_sum[warp_id][0] == local_exclusive_sum[warp_id][1] ||
        local_exclusive_sum[warp_id][0] > start_exclusive_sum + next_level_max_embedding_count_) {
        return;
    }
    if (lane_id == 0 &&
        local_exclusive_sum[warp_id][0] <= start_exclusive_sum + next_level_max_embedding_count_ &&
        local_exclusive_sum[warp_id][1] > start_exclusive_sum + next_level_max_embedding_count_) {
        *actual_start_index = global_warp_id;
    }

    if (lane_id < local_node.pre_nbrs_count[depth]) {
        uint32_t v = local_partial_embedding[warp_id][local_node.pre_nbrs_pos[depth][lane_id] + index_len];
        uint32_t type = local_node.pre_nbr_graph_type[depth][lane_id];
        local_all_pred_neighbor_size[warp_id][lane_id] = C_OFFSETS[type][v + 1] - C_OFFSETS[type][v];
        local_all_pred_neighbor[warp_id][lane_id] = C_NEIGHBORS[type] + C_OFFSETS[type][v];
    }
    __syncwarp();
    if (lane_id == 0) {
        uint32_t shortest_neighbor = UINT32_MAX;
        uint32_t neighbor_size = UINT32_MAX;
        for (uint32_t i = 0; i < local_node.pre_nbrs_count[depth]; i++) {
            uint32_t size = local_all_pred_neighbor_size[warp_id][i];
            if (size < neighbor_size) {
                neighbor_size = size;
                shortest_neighbor = i;
            }
        }
        local_shortest_neighbor[warp_id] = shortest_neighbor;
        if (local_node.nodeGreaterPosCount[depth] > 0) {
            local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeGreaterPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len] > local_max_min_target[warp_id][0]) {
                    local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len];
                }
            }
        }
        if (local_node.nodeLessPosCount[depth] > 0) {
            local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeLessPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len] < local_max_min_target[warp_id][1]) {
                    local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len];
                }
            }
        }
    }
    __syncwarp();

    uint32_t num_iters = (local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]] + WARP_SIZE - 1) / WARP_SIZE;
    for (uint32_t iter = 0; iter < num_iters; iter++) {
        uint32_t index = lane_id + iter * WARP_SIZE;
        if (index < local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]]) {
            uint32_t prob_vertex = local_all_pred_neighbor[warp_id][local_shortest_neighbor[warp_id]][index];
            bool found = gpuIntersection(
                            prob_vertex,
                            local_shortest_neighbor[warp_id],
                            local_node.pre_nbrs_count[depth],
                            local_all_pred_neighbor_size[warp_id],
                            local_all_pred_neighbor[warp_id]
            );
            if (found) {
                for (uint32_t i = 0; i < depth; i++) {
                    if (local_partial_embedding[warp_id][i + index_len] == prob_vertex) {
                        found = false;
                        break;
                    }
                }
                if (local_node.nodeGreaterPosCount[depth] > 0) {
                    if (prob_vertex < local_max_min_target[warp_id][0]) found = false;
                }
                if (local_node.nodeLessPosCount[depth] > 0) {
                    if (prob_vertex > local_max_min_target[warp_id][1]) found = false;
                }
            }

            if (found) {
                auto group = cooperative_groups::coalesced_threads();                
                uint32_t rank = group.thread_rank();
                // write the partial result
                for (uint32_t j = 0; j < depth + index_len; j++) {
                    local_write_position[warp_id][rank * (depth + 1 + index_len) + j] = local_partial_embedding[warp_id][j];
                }
                // write the new vertex
                local_write_position[warp_id][(rank + 1) * (depth + 1 + index_len) - 1] = prob_vertex;
                if (rank == 0) {
                    local_write_position[warp_id] += group.size() * (depth + 1 + index_len);
                }
            }
        }
        __syncwarp();
    }
}

__global__ void indexPrefixKernel(
        uint32_t index_len,
        uint32_t level,
        uint32_t* target,
        uint32_t next_level_count,
        uint32_t* nodesNeedsIndexing,
        uint32_t nodesCount
) {
    uint64_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_id >= next_level_count) {
        return;
    }
    for (uint32_t i = 0; i < nodesCount; i++) {
        target[global_thread_id * (index_len + 1 + level) + nodesNeedsIndexing[i]] = global_thread_id + 1;
    }
}

__global__ void indexPrefixWithinBatchKernel(
        uint32_t index_len,
        uint32_t level,
        uint32_t* target,
        uint32_t next_level_count,
        uint32_t* nodesNeedsIndexing,
        uint32_t nodesCount,
        uint32_t batchSize
) {
    uint64_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_id >= next_level_count) {
        return;
    }
    for (uint32_t i = 0; i < nodesCount; i++) {
        target[global_thread_id * (index_len + 1 + level) + nodesNeedsIndexing[i]] = global_thread_id % batchSize;
    }
}

__global__ void copyPrefixKernel(
    uint32_t* source,
    uint32_t source_count,
    uint32_t* target,
    uint32_t index_len,
    NodeGPU* d_node,
    uint32_t level
) {
    uint32_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid > source_count - 1) {
        return;
    }

    // copy the index
    for (int i = 0; i < index_len; i++) {
        target[tid * (index_len + d_node->prefixPosCount) + i] = source[tid * (index_len + level + 1) + i];
    }

    // copy the selected prefix
    for (int i = 0; i < d_node->prefixPosCount; i++) {
        target[tid * (index_len + d_node->prefixPosCount) + index_len + i] = source[tid * (index_len + level + 1) + index_len + d_node->prefixPos[i]];
    }

    
}

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
) {
    __shared__ uint32_t local_partial_embedding[WARP_PER_BLOCK][MAX_PATTERN_SIZE + 2];
    __shared__ uint32_t* local_all_pred_neighbor[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_all_pred_neighbor_size[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_max_min_target[WARP_PER_BLOCK][2];
    __shared__ uint32_t local_shortest_neighbor[WARP_PER_BLOCK];
    __shared__ NodeGPU local_node;
    __shared__ uint64_t* local_d_d_H[MAX_NUM_NODE];
    __shared__ void* local_d_d_partition_H[MAX_NUM_NODE];
    __shared__ uint64_t local_aggreRead[WARP_PER_BLOCK];

    if (threadIdx.x == 0) {
        local_node = *d_node;
    }
    if (threadIdx.x < numTreeNodes) {
        local_d_d_H[threadIdx.x] = d_d_H[threadIdx.x];
        local_d_d_partition_H[threadIdx.x] = d_d_partition_H[threadIdx.x];
    }
    __syncthreads();
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint64_t global_warp_id = blockIdx.x * WARP_PER_BLOCK + warp_id;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint64_t aggregateWrite = 0;

    if (global_warp_id > embedding_size - 1) return;
    if (global_warp_id > *failed_write) return;

    if (lane_id < depth + index_len) {
        local_partial_embedding[warp_id][lane_id] = embedding[global_warp_id * (depth + index_len) + lane_id];
    }
    __syncwarp();

    if (lane_id < local_node.pre_nbrs_count[depth]) {
        uint32_t v = local_partial_embedding[warp_id][local_node.pre_nbrs_pos[depth][lane_id] + index_len];
        uint32_t type = local_node.pre_nbr_graph_type[depth][lane_id];
        local_all_pred_neighbor_size[warp_id][lane_id] = C_OFFSETS[type][v + 1] - C_OFFSETS[type][v];
        local_all_pred_neighbor[warp_id][lane_id] = C_NEIGHBORS[type] + C_OFFSETS[type][v];
    }
    __syncwarp();
    if (lane_id == 0) {
        local_aggreRead[warp_id] = 1;
        local_max_min_target[warp_id][0] = 0;
        local_max_min_target[warp_id][1] = UINT32_MAX;
        uint32_t shortest_neighbor = UINT32_MAX;
        uint32_t neighbor_size = UINT32_MAX;
        for (uint32_t i = 0; i < local_node.pre_nbrs_count[depth]; i++) {
            uint32_t size = local_all_pred_neighbor_size[warp_id][i];
            if (size < neighbor_size) {
                neighbor_size = size;
                shortest_neighbor = i;
            }
        }
        local_shortest_neighbor[warp_id] = shortest_neighbor;
        if (local_node.nodeGreaterPosCount[depth] > 0) {
            local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeGreaterPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len] > local_max_min_target[warp_id][0]) {
                    local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len];
                }
            }
        }
        if (local_node.nodeLessPosCount[depth] > 0) {
            local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeLessPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len] < local_max_min_target[warp_id][1]) {
                    local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len];
                }
            }
        }
    }
    auto warp_group = cooperative_groups::tiled_partition<WARP_SIZE>(cooperative_groups::this_thread_block());
    auto ht_group = cooperative_groups::tiled_partition<1>(warp_group);
    if (lane_id == 0) {
        for (int j = 0; j < local_node.childCount; j++) {
            VertexID cID = local_node.children[j];
            VertexID key = local_partial_embedding[warp_id][local_node.childKeyPos[j][0] + index_len];
            if (local_node.childKeyPos[j][0] != depth) {
                local_aggreRead[warp_id] *= lookup(
                    local_d_d_H[cID],
                    local_node,
                    index_len,
                    local_d_d_partition_H[cID],
                    hashtable_size,
                    local_partial_embedding[warp_id],
                    d_node->indexPos[cID],
                    key,
                    ht_group,
                    prob_limit,
                    cID
                );
            }
        }
    }
    __syncwarp();
    if (local_aggreRead[warp_id] == 0) return;

    // start traversing all candidates
    uint32_t num_iters = (local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]] + WARP_SIZE - 1) / WARP_SIZE;
    for (uint32_t iter = 0; iter < num_iters; iter++) {
        uint32_t index = lane_id + iter * WARP_SIZE;
        uint32_t prob_vertex = 0;
        bool found = false;
        if (index < local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]]) {
            prob_vertex = local_all_pred_neighbor[warp_id][local_shortest_neighbor[warp_id]][index];
            found = gpuIntersection(
                            prob_vertex,
                            local_shortest_neighbor[warp_id],
                            local_node.pre_nbrs_count[depth],
                            local_all_pred_neighbor_size[warp_id],
                            local_all_pred_neighbor[warp_id]
            );
            if (found) {
                for (uint32_t i = 0; i < depth; i++) {
                    if (local_partial_embedding[warp_id][i + index_len] == prob_vertex) {
                        found = false;
                        break;
                    }
                }
                if (local_node.nodeGreaterPosCount[depth] > 0) {
                    if (prob_vertex < local_max_min_target[warp_id][0]) found = false;
                }
                if (local_node.nodeLessPosCount[depth] > 0) {
                    if (prob_vertex > local_max_min_target[warp_id][1]) found = false;
                }
            }
        }
        unsigned long long cnt = local_aggreRead[warp_id];
        for (int j = 0; j < local_node.childCount; j++) {
            VertexID cID = local_node.children[j];
            if (local_node.childKeyPos[j][0] == depth) {
                if (found) {
                    cnt *= lookup(
                        local_d_d_H[cID],
                        local_node,
                        index_len,
                        local_d_d_partition_H[cID],
                        hashtable_size,
                        local_partial_embedding[warp_id],
                        d_node->indexPos[cID],
                        prob_vertex,
                        ht_group,
                        prob_limit,
                        cID
                    );
                }
            }
        }
        unsigned long long* h = reinterpret_cast<unsigned long long*>(local_d_d_H[nID]);
        if (isRoot) {
            bool already_add_to_shared_mem = false;
            for (int j = 0; j < local_node.aggreVCount; j++) {
                if (local_node.aggrePos[j] == depth) {
                    if (found) atomicAdd(&h[prob_vertex], cnt * local_node.aggreWeight[j]);
                } else if (!already_add_to_shared_mem) {
                    already_add_to_shared_mem = true;
                    if (found) aggregateWrite += cnt;
                }
            }
        } else {
            if (local_node.aggrePos[0] == depth) {
                if (found) {
                    bool failed = insert(
                        h,
                        local_node,
                        index_len,
                        local_d_d_partition_H[nID],
                        hashtable_size,
                        local_partial_embedding[warp_id],
                        d_node->indexPos[nID],
                        prob_vertex,
                        ht_group,
                        cnt,
                        prob_limit,
                        nID
                    );
                    if (failed) {
                        atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_warp_id));
                    }
                }
            } else {
                if (found) aggregateWrite += cnt;
            }
        }
    }
    __syncwarp();
    if (global_warp_id >= *failed_write) return;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        aggregateWrite += __shfl_down_sync(0xFFFFFFFF, aggregateWrite, offset);
    }
    aggregateWrite = __shfl_sync(0xffffffff, aggregateWrite, 0);
    if (aggregateWrite == 0) return;
    unsigned long long* h = reinterpret_cast<unsigned long long*>(local_d_d_H[nID]);
    if (isRoot) {
        if (lane_id < local_node.aggreVCount) {
            if (local_node.aggrePos[lane_id] != depth) {
                VertexID key = local_partial_embedding[warp_id][index_len + local_node.aggrePos[lane_id]];
                atomicAdd(&h[key], aggregateWrite * local_node.aggreWeight[lane_id]);
            }
        }
    } else {
        if (lane_id == 0) {
            if (local_node.aggrePos[0] != depth) {
                VertexID key = local_partial_embedding[warp_id][index_len + local_node.aggrePos[0]];
                bool failed = insert(
                    h,
                    local_node,
                    index_len,
                    local_d_d_partition_H[nID],
                    hashtable_size,
                    local_partial_embedding[warp_id],
                    d_node->indexPos[nID],
                    key,
                    ht_group,
                    aggregateWrite,
                    prob_limit,
                    nID
                );
                if (failed) {
                    atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_warp_id));
                }
            }
        }
    }
}

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
) {
    __shared__ uint32_t local_partial_embedding[WARP_PER_BLOCK][MAX_PATTERN_SIZE + 2];
    __shared__ uint32_t* local_all_pred_neighbor[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_all_pred_neighbor_size[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_max_min_target[WARP_PER_BLOCK][2];
    __shared__ uint32_t local_shortest_neighbor[WARP_PER_BLOCK];
    __shared__ NodeGPU local_node;
    __shared__ uint64_t* local_d_d_H[MAX_NUM_NODE];
    __shared__ void* local_d_d_partition_H[MAX_NUM_NODE];
    __shared__ uint64_t local_aggreRead[WARP_PER_BLOCK];

    __shared__ uint32_t local_startOffset[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_cur_startOffset[WARP_PER_BLOCK];
    __shared__ uint32_t local_pos[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_childKey[WARP_PER_BLOCK][MAX_NUM_NODE];
    __shared__ uint32_t local_aggreKey[WARP_PER_BLOCK][MAX_PATTERN_SIZE];

    if (threadIdx.x == 0) {
        local_node = *d_node;
    }
    if (threadIdx.x < numTreeNodes) {
        local_d_d_H[threadIdx.x] = d_d_H[threadIdx.x];
        local_d_d_partition_H[threadIdx.x] = d_d_partition_H[threadIdx.x];
    }
    __syncthreads();
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint64_t global_warp_id = blockIdx.x * WARP_PER_BLOCK + warp_id;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;
    uint64_t aggregateWrite = 0;

    if (global_warp_id > embedding_size - 1) return;
    if (global_warp_id > *failed_write) return;
    
    if (lane_id < depth + index_len) {
        local_partial_embedding[warp_id][lane_id] = embedding[global_warp_id * (depth + index_len) + lane_id];
    }
    __syncwarp();

    if (lane_id < MAX_NUM_NODE) {
        local_childKey[warp_id][lane_id] = UINT32_MAX;
    }
    if (lane_id < MAX_PATTERN_SIZE) {
        local_aggreKey[warp_id][lane_id] = UINT32_MAX;
        local_startOffset[warp_id][lane_id] = UINT32_MAX;
        local_pos[warp_id][lane_id] = UINT32_MAX;
    }
    if (lane_id < local_node.pre_nbrs_count[depth]) {
        uint32_t v = local_partial_embedding[warp_id][local_node.pre_nbrs_pos[depth][lane_id] + index_len];
        uint32_t type = local_node.pre_nbr_graph_type[depth][lane_id];
        local_all_pred_neighbor_size[warp_id][lane_id] = C_OFFSETS[type][v + 1] - C_OFFSETS[type][v];
        local_all_pred_neighbor[warp_id][lane_id] = C_NEIGHBORS[type] + C_OFFSETS[type][v];
    }
    __syncwarp();
    if (lane_id == 0) {
        local_aggreRead[warp_id] = 1;
        local_max_min_target[warp_id][0] = 0;
        local_max_min_target[warp_id][1] = UINT32_MAX;
        uint32_t shortest_neighbor = UINT32_MAX;
        uint32_t neighbor_size = UINT32_MAX;
        for (uint32_t i = 0; i < local_node.pre_nbrs_count[depth]; i++) {
            uint32_t size = local_all_pred_neighbor_size[warp_id][i];
            if (size < neighbor_size) {
                neighbor_size = size;
                shortest_neighbor = i;
            }
        }
        local_cur_startOffset[warp_id] = C_OFFSETS[local_node.pre_nbr_graph_type[depth][shortest_neighbor]][local_partial_embedding[warp_id][local_node.pre_nbrs_pos[depth][shortest_neighbor] + index_len]];
        local_shortest_neighbor[warp_id] = shortest_neighbor;
        if (local_node.nodeGreaterPosCount[depth] > 0) {
            local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeGreaterPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len] > local_max_min_target[warp_id][0]) {
                    local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len];
                }
            }
        }
        if (local_node.nodeLessPosCount[depth] > 0) {
            local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeLessPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len] < local_max_min_target[warp_id][1]) {
                    local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len];
                }
            }
        }
    }
    // compute the start offset for all mapped vertices
    if (lane_id != 0 && lane_id < depth) {
        if (local_node.pre_nbrs_count[lane_id] == 1) {
            uint32_t v = local_partial_embedding[warp_id][index_len + lane_id];
            uint32_t nbr_v = local_partial_embedding[warp_id][local_node.pre_nbrs_pos[lane_id][0] + index_len];
            uint32_t type = local_node.pre_nbr_graph_type[lane_id][0];
            local_startOffset[warp_id][lane_id] = C_OFFSETS[type][nbr_v];
            local_pos[warp_id][lane_id] = GallopingBinarySearch(C_NEIGHBORS[type] + local_startOffset[warp_id][lane_id],
                                                        C_OFFSETS[type][nbr_v + 1] - local_startOffset[warp_id][lane_id], v);
        }
    }
    __syncwarp();

    // compute the aggregation key and children key
    for (int i = 0; i < depth; i++) {
        if (lane_id < local_node.posChildEdgeCount[i]) {
            uint32_t key = local_node.posChildEdge[i][lane_id];
            local_childKey[warp_id][key] = computeEdgeKey(local_startOffset[warp_id][i],
                                                    local_pos[warp_id][i],
                                                    local_node.childEdgeType[key],
                                                    local_partial_embedding[warp_id][index_len + local_node.childKeyPos[key][0]],
                                                    local_partial_embedding[warp_id][index_len + local_node.childKeyPos[key][1]]);
        }
        if (lane_id < local_node.posAggreEdgeCount[i]) {
            uint32_t key = local_node.posAggreEdge[i][lane_id];
            local_aggreKey[warp_id][key] = computeEdgeKey(local_startOffset[warp_id][i],
                                                    local_pos[warp_id][i],
                                                    local_node.aggreEdgeType[key],
                                                    local_partial_embedding[warp_id][index_len + local_node.aggrePos[2 * key]],
                                                    local_partial_embedding[warp_id][index_len + local_node.aggrePos[2 * key + 1]]);
        }
    }
    __syncwarp();
    auto warp_group = cooperative_groups::tiled_partition<WARP_SIZE>(cooperative_groups::this_thread_block());
    auto ht_group = cooperative_groups::tiled_partition<1>(warp_group);
    // pre-read all common child count
    if (lane_id == 0) {
        for (int j = 0; j < local_node.childCount; ++j) {
            VertexID cID = local_node.children[j];
            if (local_node.childKeyPosCount[j] == 1) {
                if (local_node.childKeyPos[j][0] != depth) {
                    VertexID key = local_partial_embedding[warp_id][index_len + local_node.childKeyPos[j][0]];
                    local_aggreRead[warp_id] *= lookup(
                        local_d_d_H[cID],
                        local_node,
                        index_len,
                        local_d_d_partition_H[cID],
                        hashtable_size,
                        local_partial_embedding[warp_id],
                        d_node->indexPos[cID],
                        key,
                        ht_group,
                        prob_limit,
                        cID
                    );
                }
            } else {
                if (local_childKey[warp_id][j] != UINT32_MAX) {
                    local_aggreRead[warp_id] *= lookup(
                        local_d_d_H[cID],
                        local_node,
                        index_len,
                        local_d_d_partition_H[cID],
                        hashtable_size,
                        local_partial_embedding[warp_id],
                        d_node->indexPos[cID],
                        local_childKey[warp_id][j],
                        ht_group,
                        prob_limit,
                        cID
                    );
                }
            }
        }
    }
    __syncwarp();
    if (local_aggreRead[warp_id] == 0) return;

    // start traversing all candidates
    const uint32_t num_iters = (local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]] + WARP_SIZE - 1) / WARP_SIZE;
    for (uint32_t iter = 0; iter < num_iters; iter++) {
        const uint32_t index = lane_id + iter * WARP_SIZE;
        uint32_t prob_vertex = UINT32_MAX;
        if (index < local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]]) {
            prob_vertex = local_all_pred_neighbor[warp_id][local_shortest_neighbor[warp_id]][index];
            bool found = gpuIntersection(
                            prob_vertex,
                            local_shortest_neighbor[warp_id],
                            local_node.pre_nbrs_count[depth],
                            local_all_pred_neighbor_size[warp_id],
                            local_all_pred_neighbor[warp_id]
            );
            if (found) {
                for (uint32_t i = 0; i < depth; i++) {
                    if (local_partial_embedding[warp_id][i + index_len] == prob_vertex) {
                        found = false;
                        break;
                    }
                }
                if (local_node.nodeGreaterPosCount[depth] > 0) {
                    if (prob_vertex < local_max_min_target[warp_id][0]) found = false;
                }
                if (local_node.nodeLessPosCount[depth] > 0) {
                    if (prob_vertex > local_max_min_target[warp_id][1]) found = false;
                }
            }
            if (!found) {
                prob_vertex = UINT32_MAX;
            }
        }
        unsigned long long cnt = local_aggreRead[warp_id];
        for (int j = 0; j < local_node.childCount; j++) {
            VertexID cID = local_node.children[j];
            if (local_node.childKeyPosCount[j] == 1 && local_node.childKeyPos[j][0] == depth) {
                if (prob_vertex != UINT32_MAX) {
                    cnt *= lookup(
                        local_d_d_H[cID],
                        local_node,
                        index_len,
                        local_d_d_partition_H[cID],
                        hashtable_size,
                        local_partial_embedding[warp_id],
                        d_node->indexPos[cID],
                        prob_vertex,
                        ht_group,
                        prob_limit,
                        cID
                    );
                }
            }
        }
        for (int j = 0; j < local_node.posChildEdgeCount[depth]; j++) {
            uint32_t key = local_node.posChildEdge[depth][j];
            uint32_t src = local_node.childKeyPos[key][0];
            uint32_t dst = local_node.childKeyPos[key][1];
            if (src == depth) {
                src = prob_vertex;
            } else {
                src = local_partial_embedding[warp_id][index_len + src];
            }
            if (dst == depth) {
                dst = prob_vertex;
            } else {
                dst = local_partial_embedding[warp_id][index_len + dst];
            }
            uint32_t childKey_cur = 0;
            if (prob_vertex != UINT32_MAX) {
                childKey_cur = computeEdgeKey(local_cur_startOffset[warp_id],
                                            index, local_node.childEdgeType[key],
                                            src, dst);
            }
            if (local_node.childKeyPosCount[key] != 1) {
                if (prob_vertex != UINT32_MAX) {
                    cnt *= lookup(
                        local_d_d_H[local_node.children[key]],
                        local_node,
                        index_len,
                        local_d_d_partition_H[local_node.children[key]],
                        hashtable_size,
                        local_partial_embedding[warp_id],
                        d_node->indexPos[local_node.children[key]],
                        childKey_cur,
                        ht_group,
                        prob_limit,
                        local_node.children[key]
                    );
                }
            }
        }
        unsigned long long* h = reinterpret_cast<unsigned long long*>(local_d_d_H[nID]);
        if (isRoot) {
            bool already_add_to_shared_mem = false;
            for (int j = 0; j < local_node.aggreVCount; j++) {
                if (local_node.aggrePos[j] == depth) {
                    if (prob_vertex != UINT32_MAX) atomicAdd(&h[prob_vertex], cnt * local_node.aggreWeight[j]);
                } else if (!already_add_to_shared_mem){
                    already_add_to_shared_mem = true;
                    if (prob_vertex != UINT32_MAX) aggregateWrite += cnt;
                }
            }
        } else {
            if (local_node.keySize < 2) {
                if (local_node.aggrePos[0] == depth) {
                    if (prob_vertex != UINT32_MAX) {
                        bool failed = insert(
                            h,
                            local_node,
                            index_len,
                            local_d_d_partition_H[nID],
                            hashtable_size,
                            local_partial_embedding[warp_id],
                            d_node->indexPos[nID],
                            prob_vertex,
                            ht_group,
                            cnt,
                            prob_limit,
                            nID
                        );
                        if (failed) {
                            atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_warp_id));
                        }
                    }
                } else {
                    if (prob_vertex != UINT32_MAX) aggregateWrite += cnt;
                }
            } else {
                if (local_aggreKey[warp_id][0] == UINT32_MAX) {
                    for (int j = 0; j < local_node.posAggreEdgeCount[depth]; j++) {
                        uint32_t key = local_node.posAggreEdge[depth][j];
                        uint32_t src = local_node.aggrePos[2 * key];
                        uint32_t dst = local_node.aggrePos[2 * key + 1];
                        if (src == depth) {
                            src = prob_vertex;
                        } else {
                            src = local_partial_embedding[warp_id][index_len + src];
                        }
                        if (dst == depth) {
                            dst = prob_vertex;
                        } else {
                            dst = local_partial_embedding[warp_id][index_len + dst];
                        }
                        uint32_t aggreKey_cur = 0;
                        if (prob_vertex != UINT32_MAX) {
                            aggreKey_cur = computeEdgeKey(local_cur_startOffset[warp_id],
                                                                index, local_node.aggreEdgeType[key],
                                                                src, dst);
                        }
                        if (prob_vertex != UINT32_MAX) {
                            bool failed = insert(
                                h,
                                local_node,
                                index_len,
                                local_d_d_partition_H[nID],
                                hashtable_size,
                                local_partial_embedding[warp_id],
                                d_node->indexPos[nID],
                                aggreKey_cur,
                                ht_group,
                                cnt * local_node.aggreWeight[key],
                                prob_limit,
                                nID
                            );
                            if (failed) {
                                atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_warp_id));
                            }
                        }
                    }
                } else {
                    if (prob_vertex != UINT32_MAX) aggregateWrite += cnt;
                }
            }
        }
    }
    __syncwarp();
    if (global_warp_id >= *failed_write) return;

    // accumulate the aggreWrite to the 0 position
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        aggregateWrite += __shfl_down_sync(0xFFFFFFFF, aggregateWrite, offset);
    }
    aggregateWrite = __shfl_sync(0xffffffff, aggregateWrite, 0);
    if (aggregateWrite == 0) return;
    unsigned long long* h = reinterpret_cast<unsigned long long*>(local_d_d_H[nID]);
    if (isRoot) {
        if (lane_id < local_node.aggreVCount) {
            if (local_node.aggrePos[lane_id] != depth) {
                VertexID key = local_partial_embedding[warp_id][index_len + local_node.aggrePos[lane_id]];
                atomicAdd(&h[key], aggregateWrite * local_node.aggreWeight[lane_id]);
            }
        }
    } else {
        if (lane_id == 0) {
            if (local_node.keySize < 2) {
                if (local_node.aggrePos[0] != depth) {
                    uint32_t key = local_partial_embedding[warp_id][index_len + local_node.aggrePos[0]];
                    bool failed = insert(
                        h,
                        local_node,
                        index_len,
                        local_d_d_partition_H[nID],
                        hashtable_size,
                        local_partial_embedding[warp_id],
                        d_node->indexPos[nID],
                        key,
                        ht_group,
                        aggregateWrite,
                        prob_limit,
                        nID
                    );
                    if (failed) {
                        atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_warp_id));
                    }
                }
            } else {
                if (local_aggreKey[warp_id][0] != UINT32_MAX) {
                    bool failed = insert(
                        h,
                        local_node,
                        index_len,
                        local_d_d_partition_H[nID],
                        hashtable_size,
                        local_partial_embedding[warp_id],
                        d_node->indexPos[nID],
                        local_aggreKey[warp_id][0],
                        ht_group,
                        aggregateWrite,
                        prob_limit,
                        nID
                    );
                    if (failed) {
                        atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_warp_id));
                    }
                }
            }
        }
    }
}

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
) {
    __shared__ uint32_t local_partial_embedding[WARP_PER_BLOCK][MAX_PATTERN_SIZE + 2];
    __shared__ uint32_t* local_all_pred_neighbor[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_all_pred_neighbor_size[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_max_min_target[WARP_PER_BLOCK][2];
    __shared__ uint32_t local_shortest_neighbor[WARP_PER_BLOCK];
    __shared__ NodeGPU local_node;
    __shared__ uint64_t* local_d_d_H[MAX_NUM_NODE];
    __shared__ void* local_d_d_partition_H[MAX_NUM_NODE];

    if (threadIdx.x == 0) {
        local_node = *d_node;
    }
    if (threadIdx.x < numTreeNodes) {
        local_d_d_H[threadIdx.x] = d_d_H[threadIdx.x];
        local_d_d_partition_H[threadIdx.x] = d_d_partition_H[threadIdx.x];
    }
    __syncthreads();
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint64_t global_warp_id = blockIdx.x * WARP_PER_BLOCK + warp_id;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;

    if (global_warp_id > embedding_size - 1) return;
    if (global_warp_id > *failed_write) return;

    if (lane_id < depth + index_len) {
        local_partial_embedding[warp_id][lane_id] = embedding[global_warp_id * (depth + index_len) + lane_id];
    }
    __syncwarp();

    if (lane_id < local_node.pre_nbrs_count[depth]) {
        uint32_t v = local_partial_embedding[warp_id][local_node.pre_nbrs_pos[depth][lane_id] + index_len];
        uint32_t type = local_node.pre_nbr_graph_type[depth][lane_id];
        local_all_pred_neighbor_size[warp_id][lane_id] = C_OFFSETS[type][v + 1] - C_OFFSETS[type][v];
        local_all_pred_neighbor[warp_id][lane_id] = C_NEIGHBORS[type] + C_OFFSETS[type][v];
    }
    __syncwarp();
    if (lane_id == 0) {
        local_max_min_target[warp_id][0] = 0;
        local_max_min_target[warp_id][1] = UINT32_MAX;
        uint32_t shortest_neighbor = UINT32_MAX;
        uint32_t neighbor_size = UINT32_MAX;
        for (uint32_t i = 0; i < local_node.pre_nbrs_count[depth]; i++) {
            uint32_t size = local_all_pred_neighbor_size[warp_id][i];
            if (size < neighbor_size) {
                neighbor_size = size;
                shortest_neighbor = i;
            }
        }
        local_shortest_neighbor[warp_id] = shortest_neighbor;
        if (local_node.nodeGreaterPosCount[depth] > 0) {
            local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeGreaterPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len] > local_max_min_target[warp_id][0]) {
                    local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len];
                }
            }
        }
        if (local_node.nodeLessPosCount[depth] > 0) {
            local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeLessPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len] < local_max_min_target[warp_id][1]) {
                    local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len];
                }
            }
        }
    }
    __syncwarp();
    auto warp_group = cooperative_groups::tiled_partition<WARP_SIZE>(cooperative_groups::this_thread_block());
    auto ht_group = cooperative_groups::tiled_partition<1>(warp_group);

    // start traversing all candidates
    uint32_t num_iters = (local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]] + WARP_SIZE - 1) / WARP_SIZE;
    for (uint32_t iter = 0; iter < num_iters; iter++) {
        uint32_t index = lane_id + iter * WARP_SIZE;
        uint32_t prob_vertex = 0;
        bool found = false;
        if (index < local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]]) {
            prob_vertex = local_all_pred_neighbor[warp_id][local_shortest_neighbor[warp_id]][index];
            found = gpuIntersection(
                            prob_vertex,
                            local_shortest_neighbor[warp_id],
                            local_node.pre_nbrs_count[depth],
                            local_all_pred_neighbor_size[warp_id],
                            local_all_pred_neighbor[warp_id]
            );
            if (found) {
                for (uint32_t i = 0; i < depth; i++) {
                    if (local_partial_embedding[warp_id][i + index_len] == prob_vertex) {
                        found = false;
                        break;
                    }
                }
                if (local_node.nodeGreaterPosCount[depth] > 0) {
                    if (prob_vertex < local_max_min_target[warp_id][0]) found = false;
                }
                if (local_node.nodeLessPosCount[depth] > 0) {
                    if (prob_vertex > local_max_min_target[warp_id][1]) found = false;
                }
            }
        }
        unsigned long long cnt = 1;
        for (int j = 0; j < local_node.childCount; j++) {
            VertexID cID = local_node.children[j];
            VertexID key = local_partial_embedding[warp_id][local_node.childKeyPos[j][0] + index_len];
            if (local_node.childKeyPos[j][0] == depth) {
                key = prob_vertex;
            }
            if (found) {
                cnt *= lookup(
                    local_d_d_H[cID],
                    local_node,
                    index_len,
                    local_d_d_partition_H[cID],
                    hashtable_size,
                    local_partial_embedding[warp_id],
                    d_node->indexPos[cID],
                    key,
                    ht_group,
                    prob_limit,
                    cID
                );
            }
        }
        unsigned long long* h = reinterpret_cast<unsigned long long*>(local_d_d_H[nID]);
        if (isRoot) {
            for (int j = 0; j < local_node.aggreVCount; j++) {
                uint32_t key = local_partial_embedding[warp_id][local_node.aggrePos[j] + index_len];
                if (local_node.aggrePos[j] == depth) {
                    key = prob_vertex;
                }
                if (found) atomicAdd(&h[key], cnt * local_node.aggreWeight[j]);
            }
        } else {
            uint32_t key = local_partial_embedding[warp_id][local_node.aggrePos[0] + index_len];
            if (local_node.aggrePos[0] == depth) {
                key = prob_vertex;
            }
            if (found) {
                bool failed = insert(
                    h,
                    local_node,
                    index_len,
                    local_d_d_partition_H[nID],
                    hashtable_size,
                    local_partial_embedding[warp_id],
                    d_node->indexPos[nID],
                    key,
                    ht_group,
                    cnt,
                    prob_limit,
                    nID
                );
                if (failed) {
                    atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_warp_id));
                }
            }
        }
    }
}

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
) {
    __shared__ uint32_t local_partial_embedding[WARP_PER_BLOCK][MAX_PATTERN_SIZE + 2];
    __shared__ uint32_t* local_all_pred_neighbor[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_all_pred_neighbor_size[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_max_min_target[WARP_PER_BLOCK][2];
    __shared__ uint32_t local_shortest_neighbor[WARP_PER_BLOCK];
    __shared__ NodeGPU local_node;
    __shared__ uint64_t* local_d_d_H[MAX_NUM_NODE];
    __shared__ void* local_d_d_partition_H[MAX_NUM_NODE];
    __shared__ uint32_t local_startOffset[WARP_PER_BLOCK][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_cur_startOffset[WARP_PER_BLOCK];
    __shared__ uint32_t local_pos[WARP_PER_BLOCK][MAX_PATTERN_SIZE];

    if (threadIdx.x == 0) {
        local_node = *d_node;
    }
    if (threadIdx.x < numTreeNodes) {
        local_d_d_H[threadIdx.x] = d_d_H[threadIdx.x];
        local_d_d_partition_H[threadIdx.x] = d_d_partition_H[threadIdx.x];
    }
    __syncthreads();
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint64_t global_warp_id = blockIdx.x * WARP_PER_BLOCK + warp_id;
    uint32_t lane_id = threadIdx.x % WARP_SIZE;

    if (global_warp_id > embedding_size - 1) return;
    if (global_warp_id > *failed_write) return;
    
    if (lane_id < depth + index_len) {
        local_partial_embedding[warp_id][lane_id] = embedding[global_warp_id * (depth + index_len) + lane_id];
    }
    __syncwarp();

    if (lane_id < MAX_PATTERN_SIZE) {
        local_startOffset[warp_id][lane_id] = UINT32_MAX;
        local_pos[warp_id][lane_id] = UINT32_MAX;
    }
    if (lane_id < local_node.pre_nbrs_count[depth]) {
        uint32_t v = local_partial_embedding[warp_id][local_node.pre_nbrs_pos[depth][lane_id] + index_len];
        uint32_t type = local_node.pre_nbr_graph_type[depth][lane_id];
        local_all_pred_neighbor_size[warp_id][lane_id] = C_OFFSETS[type][v + 1] - C_OFFSETS[type][v];
        local_all_pred_neighbor[warp_id][lane_id] = C_NEIGHBORS[type] + C_OFFSETS[type][v];
    }
    __syncwarp();
    if (lane_id == 0) {
        local_max_min_target[warp_id][0] = 0;
        local_max_min_target[warp_id][1] = UINT32_MAX;
        uint32_t shortest_neighbor = UINT32_MAX;
        uint32_t neighbor_size = UINT32_MAX;
        for (uint32_t i = 0; i < local_node.pre_nbrs_count[depth]; i++) {
            uint32_t size = local_all_pred_neighbor_size[warp_id][i];
            if (size < neighbor_size) {
                neighbor_size = size;
                shortest_neighbor = i;
            }
        }
        local_cur_startOffset[warp_id] = C_OFFSETS[local_node.pre_nbr_graph_type[depth][shortest_neighbor]][local_partial_embedding[warp_id][local_node.pre_nbrs_pos[depth][shortest_neighbor] + index_len]];
        local_shortest_neighbor[warp_id] = shortest_neighbor;
        if (local_node.nodeGreaterPosCount[depth] > 0) {
            local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeGreaterPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len] > local_max_min_target[warp_id][0]) {
                    local_max_min_target[warp_id][0] = local_partial_embedding[warp_id][local_node.nodeGreaterPos[depth][i] + index_len];
                }
            }
        }
        if (local_node.nodeLessPosCount[depth] > 0) {
            local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][0] + index_len];
            for (uint32_t i = 1; i < local_node.nodeLessPosCount[depth]; i++) {
                if (local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len] < local_max_min_target[warp_id][1]) {
                    local_max_min_target[warp_id][1] = local_partial_embedding[warp_id][local_node.nodeLessPos[depth][i] + index_len];
                }
            }
        }
    }
    // compute the start offset for all mapped vertices
    if (lane_id != 0 && lane_id < depth) {
        if (local_node.pre_nbrs_count[lane_id] == 1) {
            uint32_t v = local_partial_embedding[warp_id][index_len + lane_id];
            uint32_t nbr_v = local_partial_embedding[warp_id][local_node.pre_nbrs_pos[lane_id][0] + index_len];
            uint32_t type = local_node.pre_nbr_graph_type[lane_id][0];
            local_startOffset[warp_id][lane_id] = C_OFFSETS[type][nbr_v];
            local_pos[warp_id][lane_id] = GallopingBinarySearch(C_NEIGHBORS[type] + local_startOffset[warp_id][lane_id],
                                                        C_OFFSETS[type][nbr_v + 1] - local_startOffset[warp_id][lane_id], v);
        }
    }
    __syncwarp();

    auto warp_group = cooperative_groups::tiled_partition<WARP_SIZE>(cooperative_groups::this_thread_block());
    auto ht_group = cooperative_groups::tiled_partition<1>(warp_group);

    // start traversing all candidates
    const uint32_t num_iters = (local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]] + WARP_SIZE - 1) / WARP_SIZE;
    for (uint32_t iter = 0; iter < num_iters; iter++) {
        const uint32_t index = lane_id + iter * WARP_SIZE;
        uint32_t prob_vertex = UINT32_MAX;
        if (index < local_all_pred_neighbor_size[warp_id][local_shortest_neighbor[warp_id]]) {
            prob_vertex = local_all_pred_neighbor[warp_id][local_shortest_neighbor[warp_id]][index];
            bool found = gpuIntersection(
                            prob_vertex,
                            local_shortest_neighbor[warp_id],
                            local_node.pre_nbrs_count[depth],
                            local_all_pred_neighbor_size[warp_id],
                            local_all_pred_neighbor[warp_id]
            );
            if (found) {
                for (uint32_t i = 0; i < depth; i++) {
                    if (local_partial_embedding[warp_id][i + index_len] == prob_vertex) {
                        found = false;
                        break;
                    }
                }
                if (local_node.nodeGreaterPosCount[depth] > 0) {
                    if (prob_vertex < local_max_min_target[warp_id][0]) found = false;
                }
                if (local_node.nodeLessPosCount[depth] > 0) {
                    if (prob_vertex > local_max_min_target[warp_id][1]) found = false;
                }
            }
            if (!found) {
                prob_vertex = UINT32_MAX;
            }
        }
        unsigned long long cnt = 1;
        for (int j = 0; j < local_node.childCount; j++) {
            VertexID cID = local_node.children[j];
            if (local_node.childKeyPosCount[j] == 1) {
                uint32_t key = local_partial_embedding[warp_id][local_node.childKeyPos[j][0] + index_len];
                if (local_node.childKeyPos[j][0] == depth) {
                    key = prob_vertex;
                }
                if (prob_vertex != UINT32_MAX) {
                    cnt *= lookup(
                        local_d_d_H[cID],
                        local_node,
                        index_len,
                        local_d_d_partition_H[cID],
                        hashtable_size,
                        local_partial_embedding[warp_id],
                        d_node->indexPos[cID],
                        key,
                        ht_group,
                        prob_limit,
                        cID
                    );
                }
            }
        }
        for (int k = 0; k < depth + 1; k++) {
            for (int j = 0; j < local_node.posChildEdgeCount[k]; j++) {
                uint32_t key = local_node.posChildEdge[k][j];
                uint32_t src = local_node.childKeyPos[key][0];
                uint32_t dst = local_node.childKeyPos[key][1];
                if (src == depth) {
                    src = prob_vertex;
                } else {
                    src = local_partial_embedding[warp_id][index_len + src];
                }
                if (dst == depth) {
                    dst = prob_vertex;
                } else {
                    dst = local_partial_embedding[warp_id][index_len + dst];
                }
                uint32_t childKey_cur = 0;
                if (prob_vertex != UINT32_MAX) {
                    if (k == depth) {
                        childKey_cur = computeEdgeKey(local_cur_startOffset[warp_id],
                                                    index, local_node.childEdgeType[key],
                                                    src, dst);
                    } else {
                        childKey_cur = computeEdgeKey(local_startOffset[warp_id][k],
                                                    local_pos[warp_id][k], local_node.childEdgeType[key],
                                                    src, dst);
                    }
                }
                if (local_node.childKeyPosCount[key] != 1) {
                    if (prob_vertex != UINT32_MAX) {
                        cnt *= lookup(
                            local_d_d_H[local_node.children[key]],
                            local_node,
                            index_len,
                            local_d_d_partition_H[local_node.children[key]],
                            hashtable_size,
                            local_partial_embedding[warp_id],
                            d_node->indexPos[local_node.children[key]],
                            childKey_cur,
                            ht_group,
                            prob_limit,
                            local_node.children[key]
                        );
                    }
                }
            }
        }

        unsigned long long* h = reinterpret_cast<unsigned long long*>(local_d_d_H[nID]);
        if (isRoot) {
            for (int j = 0; j < local_node.aggreVCount; j++) {
                uint32_t key = local_partial_embedding[warp_id][local_node.aggrePos[j] + index_len];
                if (local_node.aggrePos[j] == depth) {
                    key = prob_vertex;
                }
                if (prob_vertex != UINT32_MAX) atomicAdd(&h[key], cnt * local_node.aggreWeight[j]);
            }
        } else {
            if (local_node.keySize < 2) {
                uint32_t key = local_partial_embedding[warp_id][local_node.aggrePos[0] + index_len];
                if (local_node.aggrePos[0] == depth) {
                    key = prob_vertex;
                }
                if (prob_vertex != UINT32_MAX) {
                    bool failed = insert(
                        h,
                        local_node,
                        index_len,
                        local_d_d_partition_H[nID],
                        hashtable_size,
                        local_partial_embedding[warp_id],
                        d_node->indexPos[nID],
                        key,
                        ht_group,
                        cnt,
                        prob_limit,
                        nID
                    );
                    if (failed) {
                        atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_warp_id));
                    }
                }
            } else {
                for (int k = 0; k < depth + 1; k++) {
                    for (int j = 0; j < local_node.posAggreEdgeCount[k]; j++) {
                        uint32_t key = local_node.posAggreEdge[k][j];
                        uint32_t src = local_node.aggrePos[2 * key];
                        uint32_t dst = local_node.aggrePos[2 * key + 1];
                        if (src == depth) {
                            src = prob_vertex;
                        } else {
                            src = local_partial_embedding[warp_id][index_len + src];
                        }
                        if (dst == depth) {
                            dst = prob_vertex;
                        } else {
                            dst = local_partial_embedding[warp_id][index_len + dst];
                        }
                        uint32_t aggreKey_cur = 0;
                        if (prob_vertex != UINT32_MAX) {
                            if (k == depth) {
                                aggreKey_cur = computeEdgeKey(local_cur_startOffset[warp_id],
                                                                index, local_node.aggreEdgeType[key],
                                                                src, dst);
                            } else {
                                aggreKey_cur = computeEdgeKey(local_startOffset[warp_id][k],
                                                                local_pos[warp_id][k], local_node.aggreEdgeType[key],
                                                                src, dst);
                            }
                        }
                        if (prob_vertex != UINT32_MAX) {
                            bool failed = insert(
                                h,
                                local_node,
                                index_len,
                                local_d_d_partition_H[nID],
                                hashtable_size,
                                local_partial_embedding[warp_id],
                                d_node->indexPos[nID],
                                aggreKey_cur,
                                ht_group,
                                cnt,
                                prob_limit,
                                nID
                            );
                            if (failed) {
                                atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_warp_id));
                            }
                        }
                    }
                }
            }
        }
    }
}

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
) {
    __shared__ NodeGPU local_node;
    __shared__ uint64_t* local_d_d_H[MAX_NUM_NODE];
    __shared__ void* local_d_d_partition_H[MAX_NUM_NODE];
    if (threadIdx.x == 0) {
        local_node = *d_node;
    }
    if (threadIdx.x < numTreeNodes) {
        local_d_d_H[threadIdx.x] = d_d_H[threadIdx.x];
        local_d_d_partition_H[threadIdx.x] = d_d_partition_H[threadIdx.x];
    }
    __syncthreads();
    uint64_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_thread_id > embedding_size - 1) return;
    if (global_thread_id > *failed_write) return;
    uint32_t* read_pos = embedding + global_thread_id * (depth + index_len);
    unsigned long long cnt = 1;
    auto warp_group = cooperative_groups::tiled_partition<WARP_SIZE>(cooperative_groups::this_thread_block());
    auto ht_group = cooperative_groups::tiled_partition<1>(warp_group);
    for (int j = 0; j < local_node.childCount; j++) {
        VertexID cID = local_node.children[j];
        VertexID key = read_pos[local_node.childKeyPos[j][0] + index_len];
        cnt *= lookup(
            local_d_d_H[cID],
            local_node,
            index_len,
            local_d_d_partition_H[cID],
            hashtable_size,
            read_pos,
            d_node->indexPos[cID],
            key,
            ht_group,
            prob_limit,
            cID
        );
    }
    unsigned long long* h = reinterpret_cast<unsigned long long*>(local_d_d_H[nID]);
    if (isRoot) {
        for (int j = 0; j < local_node.aggreVCount; j++) {
            uint32_t key = read_pos[local_node.aggrePos[j] + index_len];
            atomicAdd(&h[key], cnt * local_node.aggreWeight[j]);
        }
    } else {
        uint32_t key = read_pos[local_node.aggrePos[0] + index_len];
        bool failed = insert(
            h,
            local_node,
            index_len,
            local_d_d_partition_H[nID],
            hashtable_size,
            read_pos,
            d_node->indexPos[nID],
            key,
            ht_group,
            cnt,
            prob_limit,
            nID
        );
        if (failed) {
            atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_thread_id));
        }
    }
}

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
) {
    __shared__ NodeGPU local_node;
    __shared__ uint64_t* local_d_d_H[MAX_NUM_NODE];
    __shared__ void* local_d_d_partition_H[MAX_NUM_NODE];
    __shared__ uint32_t local_startOffset[BLOCK_DIM][MAX_PATTERN_SIZE];
    __shared__ uint32_t local_pos[BLOCK_DIM][MAX_PATTERN_SIZE];

    if (threadIdx.x == 0) {
        local_node = *d_node;
    }
    if (threadIdx.x < numTreeNodes) {
        local_d_d_H[threadIdx.x] = d_d_H[threadIdx.x];
        local_d_d_partition_H[threadIdx.x] = d_d_partition_H[threadIdx.x];
    }
    __syncthreads();
    uint64_t global_thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t thread_id = threadIdx.x;
    if (global_thread_id > embedding_size - 1) return;
    if (global_thread_id > *failed_write) return;
    uint32_t* read_pos = embedding + global_thread_id * (depth + index_len);

    // compute the start offset for all mapped vertices
    for (int i = 1; i < depth; i++) {
        if (local_node.pre_nbrs_count[i] == 1) {
            uint32_t v = read_pos[index_len + i];
            uint32_t nbr_v = read_pos[local_node.pre_nbrs_pos[i][0] + index_len];
            uint32_t type = local_node.pre_nbr_graph_type[i][0];
            local_startOffset[thread_id][i] = C_OFFSETS[type][nbr_v];
            local_pos[thread_id][i] = GallopingBinarySearch(C_NEIGHBORS[type] + local_startOffset[thread_id][i],
                                                        C_OFFSETS[type][nbr_v + 1] - local_startOffset[thread_id][i], v);
        }
    }

    auto warp_group = cooperative_groups::tiled_partition<WARP_SIZE>(cooperative_groups::this_thread_block());
    auto ht_group = cooperative_groups::tiled_partition<1>(warp_group);
    unsigned long long cnt = 1;
    for (int j = 0; j < local_node.childCount; j++) {
        VertexID cID = local_node.children[j];
        if (local_node.childKeyPosCount[j] == 1) {
            uint32_t key = read_pos[local_node.childKeyPos[j][0] + index_len];
            cnt *= lookup(
                local_d_d_H[cID],
                local_node,
                index_len,
                local_d_d_partition_H[cID],
                hashtable_size,
                read_pos,
                d_node->indexPos[cID],
                key,
                ht_group,
                prob_limit,
                cID
            );
        }
    }
    for (int k = 0; k < depth; k++) {
        for (int j = 0; j < local_node.posChildEdgeCount[k]; j++) {
            uint32_t key = local_node.posChildEdge[k][j];
            uint32_t src = local_node.childKeyPos[key][0];
            uint32_t dst = local_node.childKeyPos[key][1];
            src = read_pos[index_len + src];
            dst = read_pos[index_len + dst];
            uint32_t childKey_cur = 0;
            childKey_cur = computeEdgeKey(local_startOffset[thread_id][k],
                                        local_pos[thread_id][k], local_node.childEdgeType[key],
                                        src, dst);
            if (local_node.childKeyPosCount[key] != 1) {
                cnt *= lookup(
                    local_d_d_H[local_node.children[key]],
                    local_node,
                    index_len,
                    local_d_d_partition_H[local_node.children[key]],
                    hashtable_size,
                    read_pos,
                    d_node->indexPos[local_node.children[key]],
                    childKey_cur,
                    ht_group,
                    prob_limit,
                    local_node.children[key]
                );
            }
        }
    }

    unsigned long long* h = reinterpret_cast<unsigned long long*>(local_d_d_H[nID]);
    if (isRoot) {
        for (int j = 0; j < local_node.aggreVCount; j++) {
            uint32_t key = read_pos[local_node.aggrePos[j] + index_len];
            atomicAdd(&h[key], cnt * local_node.aggreWeight[j]);
        }
    } else {
        if (local_node.keySize < 2) {
            uint32_t key = read_pos[local_node.aggrePos[0] + index_len];
            bool failed = insert(
                h,
                local_node,
                index_len,
                local_d_d_partition_H[nID],
                hashtable_size,
                read_pos,
                d_node->indexPos[nID],
                key,
                ht_group,
                cnt,
                prob_limit,
                nID
            );
            if (failed) {
                atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_thread_id));
            }
        } else {
            for (int k = 0; k < depth; k++) {
                for (int j = 0; j < local_node.posAggreEdgeCount[k]; j++) {
                    uint32_t key = local_node.posAggreEdge[k][j];
                    uint32_t src = local_node.aggrePos[2 * key];
                    uint32_t dst = local_node.aggrePos[2 * key + 1];
                    src = read_pos[index_len + src];
                    dst = read_pos[index_len + dst];
                    uint32_t aggreKey_cur = 0;
                    aggreKey_cur = computeEdgeKey(local_startOffset[thread_id][k],
                                                    local_pos[thread_id][k], local_node.aggreEdgeType[key],
                                                    src, dst);
                    bool failed = insert(
                        h,
                        local_node,
                        index_len,
                        local_d_d_partition_H[nID],
                        hashtable_size,
                        read_pos,
                        d_node->indexPos[nID],
                        aggreKey_cur,
                        ht_group,
                        cnt,
                        prob_limit,
                        nID
                    );
                    if (failed) {
                        atomicMin(reinterpret_cast<unsigned long long*>(failed_write), static_cast<unsigned long long>(global_thread_id));
                    }
                }
            }
        }
    }
}