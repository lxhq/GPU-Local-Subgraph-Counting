#include "lock_free_hashtable.h"

__forceinline__ __device__ uint64_t hash(uint32_t key_1, uint32_t key_2) {
    // FNV‑1a 64‑bit offset basis and prime constants.
    uint64_t h = 14695981039346656037ULL;
    const uint64_t FNV_prime = 1099511628211ULL;
    
    // Process keys in reverse order so that latter (more variable) elements are prioritized.
    h ^= static_cast<uint64_t>(key_2);  // incorporate the key
    h *= FNV_prime;          // mix the bits with the FNV prime
    h ^= static_cast<uint64_t>(key_1);  // incorporate the key
    h *= FNV_prime;          // mix the bits with the FNV prime
    return h;
}


__device__ bool insertHashTable(
    uint64_t* hashtable,
    uint64_t hashtable_size,
    uint32_t key_1,
    uint32_t key_2,
    uint32_t prob_limit,
    uint64_t value
) {
    if (value == 0) {
        return true;
    }
    uint32_t prob = 0;
    bool inserted = false;
    uint64_t index = hash(key_1, key_2);
    unsigned long long int combined = (static_cast<unsigned long long int>(key_1) << 32) | key_2;
    while (prob < prob_limit) {
        unsigned long long int* bucket_pos = reinterpret_cast<unsigned long long int*>(hashtable) + ((index + prob) % hashtable_size) * 2;
        unsigned long long  int old_value = bucket_pos[0];
        if (old_value == 0) {
            old_value = atomicCAS(bucket_pos, 0, combined);
            if (old_value == 0 || old_value == combined) {
                inserted = true;
                atomicAdd(bucket_pos + 1, value);
                break;
            }
        } else if (old_value == combined) {
            atomicAdd(bucket_pos + 1, value);
            inserted = true;
            break;
        }
        prob++;
    }
    return inserted;
}

__device__ uint64_t lookupHashTable(
    uint64_t* hashtable,
    uint64_t hashtable_size,
    uint32_t key_1,
    uint32_t key_2,
    uint32_t prob_limit
) {
    uint64_t count = 0;
    uint32_t prob = 0;
    uint64_t index = hash(key_1, key_2);
    uint64_t combined = (static_cast<uint64_t>(key_1) << 32) | key_2;
    while (prob < prob_limit) {
        uint64_t* bucket_pos = hashtable + ((index + prob) % hashtable_size) * 2;
        if (*bucket_pos == 0) {
            break;
        }
        if (*bucket_pos == combined) {
            count = *(bucket_pos + 1);
            break;
        }
        prob++;
    }
    return count;
}