#include "lock_based_composite_hashtable.h"

// Boost-Style Hash Combine
__forceinline__ __device__ uint64_t hash_combine(uint64_t seed, uint64_t value) {
    // Use a 64-bit constant (the golden ratio for 64-bit numbers)
    return seed ^ (value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
}

__device__ uint64_t insertHash(uint32_t* partial_embeddings, NodeGPU& local_node, uint32_t key, uint32_t index_len, uint32_t insertNodeID) {
    // FNV‑1a 64‑bit offset basis and prime constants.
    uint64_t h = 14695981039346656037ULL;
    const uint64_t FNV_prime = 1099511628211ULL;
    
    // Process keys in reverse order so that latter (more variable) elements are prioritized.
    h ^= static_cast<uint64_t>(key);  // incorporate the key
    h *= FNV_prime;          // mix the bits with the FNV prime
    if (local_node.multiJoin == 0) {
        for (int i = 0; i < local_node.prefixLen[insertNodeID]; i++) {
            h ^= static_cast<uint64_t>(partial_embeddings[i + index_len]);  // incorporate the key
            h *= FNV_prime;          // mix the bits with the FNV prime
        }
    } else {
        h ^= static_cast<uint64_t>(partial_embeddings[local_node.indexPos[insertNodeID]]);  // incorporate the key
        h *= FNV_prime;          // mix the bits with the FNV prime
    }
    return h;
}

__device__ uint64_t lookUpHash(uint32_t* partial_embeddings, NodeGPU& local_node, uint32_t key, uint32_t index_len, uint32_t lookupNodeID) {
    // FNV‑1a 64‑bit offset basis and prime constants.
    uint64_t h = 14695981039346656037ULL;
    const uint64_t FNV_prime = 1099511628211ULL;
    
    // Process keys in reverse order so that latter (more variable) elements are prioritized.
    h ^= static_cast<uint64_t>(key);  // incorporate the key
    h *= FNV_prime;          // mix the bits with the FNV prime
    if (local_node.multiJoin == 0) {
        for (int i = 0; i < local_node.prefixLen[lookupNodeID]; i++) {
            h ^= static_cast<uint64_t>(partial_embeddings[i + index_len]);  // incorporate the key
            h *= FNV_prime;          // mix the bits with the FNV prime
        }
    } else {
        h ^= static_cast<uint64_t>(partial_embeddings[local_node.indexPos[lookupNodeID]]);  // incorporate the key
        h *= FNV_prime;          // mix the bits with the FNV prime
    }
    return h;
}

// hashtable_size: total number of buckets in the hashtable
// since bucket contains 8 bytes integers, hashtable must be 8 bytes aligned
__device__ bool insertLockHashTable(uint64_t* hashtable, uint64_t hashtable_size,
                                uint32_t* partial_embeddings, NodeGPU& local_node,
                                uint32_t key, uint32_t prob_limit, uint64_t value,
                                uint32_t index_len,  uint32_t insertNodeID) {
    if (value == 0) {
        return true;
    }
    uint64_t hash_value = insertHash(partial_embeddings, local_node, key, index_len, insertNodeID);
    uint64_t prob = 0;
    bool is_inserted = false;
    // 1 for state, 1 for key, 1 for value, rest for prefix keys
    uint64_t index = hash_value % hashtable_size;
    // break if the prob is larger than the limit
    // prob_limits < hashtable_size to avoid infinite loop
    while (prob < prob_limit) {
        // move to the next bucket
        uint64_t* bucket_pos = hashtable + ((index + prob) % hashtable_size) * 6;    // Calculate the starting byte offset of the bucket
        uint64_t prev_bucket_state = atomicCAS((unsigned long long int*)bucket_pos, 0, 1);                              // Try to lock the bucket
        if (prev_bucket_state == 0) {
            // the bucket is empty, insert the keys
            if (local_node.multiJoin == 0) {
                for (int i = 0; i < local_node.prefixLen[insertNodeID]; i++) {
                    *(bucket_pos + 1 + i) = partial_embeddings[index_len + i];
                }
                *(bucket_pos + 1 + local_node.prefixLen[insertNodeID]) = key;
                *(reinterpret_cast<uint64_t*>(bucket_pos + 1 + local_node.prefixLen[insertNodeID] + 1)) = value; // set the count to value
            } else {
                *(bucket_pos + 1) = partial_embeddings[local_node.indexPos[insertNodeID]];
                *(bucket_pos + 1 + 1) = key;
                *(reinterpret_cast<uint64_t*>(bucket_pos + 1 + 1 + 1)) = value; // set the count to value
            }
            is_inserted = true;
            __threadfence(); // make sure the bucket is written before setting the state
            *bucket_pos = 2; // set the bucket state to occupied
            break;
        } else if (prev_bucket_state == 1) {
            continue;
        } else {
            // the bucket is occupied, check if the keys are the same
            bool is_same = true;
            if (local_node.multiJoin == 0) {
                for (int i = 0; i < local_node.prefixLen[insertNodeID]; i++) {
                    if (*(bucket_pos + 1 + i) != partial_embeddings[index_len + i]) {
                        is_same = false;
                        break;
                    }
                }
                if (*(bucket_pos + 1 + local_node.prefixLen[insertNodeID]) != key) {
                    is_same = false;
                }
            } else {
                is_same = (*(bucket_pos + 1) == partial_embeddings[local_node.indexPos[insertNodeID]] && *(bucket_pos + 1 + 1) == key);
            }
            if (is_same) {
                if (local_node.multiJoin == 0) {
                    atomicAdd(reinterpret_cast<unsigned long long*>(bucket_pos + 1 + local_node.prefixLen[insertNodeID] + 1), value);
                } else {
                    atomicAdd(reinterpret_cast<unsigned long long*>(bucket_pos + 1 + 1 + 1), value);
                }
                is_inserted = true;
                break;
            }
        }
        prob++;
    }
    return is_inserted;
}

__device__ uint64_t lookupLockHashTable(uint64_t* hashtable, uint64_t hashtable_size,
                                uint32_t* partial_embeddings, NodeGPU& local_node,
                                uint32_t key, uint32_t prob_limit, uint32_t index_len, uint32_t lookupNodeID) {
    uint64_t count = 0;
    uint64_t hash_value = lookUpHash(partial_embeddings, local_node, key, index_len, lookupNodeID);
    uint64_t prob = 0;
    // 1 for state, 1 for key, 1 for value, rest for prefix keys
    uint64_t index = hash_value % hashtable_size;
    while (prob < prob_limit) {
        uint64_t* bucket_pos = hashtable + ((index + prob) % hashtable_size) * 6;    // Calculate the starting byte offset of the bucket
        uint32_t bucket_state = *bucket_pos;
        if (bucket_state == 0) {
            break;
        } else if (bucket_state == 1) {
            printf("error: the bucket state should not be 1\n");
            break;
        } else {
            bool is_same = true;
            if (local_node.multiJoin == 0) {
                for (int i = 0; i < local_node.prefixLen[lookupNodeID]; i++) {
                    if (*(bucket_pos + 1 + i) != partial_embeddings[index_len + i]) {
                        is_same = false;
                        break;
                    }
                }
                if (*(bucket_pos + 1 + local_node.prefixLen[lookupNodeID]) != key) {
                    is_same = false;
                }
            } else {
                is_same = (*(bucket_pos + 1) == partial_embeddings[local_node.indexPos[lookupNodeID]] && *(bucket_pos + 1 + 1) == key);
            }
            if (is_same) {
                if (local_node.multiJoin == 0) {
                    count = *reinterpret_cast<uint64_t*>(bucket_pos + 1 + local_node.prefixLen[lookupNodeID] + 1);
                } else {
                    count = *reinterpret_cast<uint64_t*>(bucket_pos + 1 + 1 + 1);
                }
                break;
            }
        }
        prob++;
    }
    return count;
}