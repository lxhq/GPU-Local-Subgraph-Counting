#pragma once

#include <cstdint>
#include <stdio.h>

__device__ bool insertHashTable(
    uint64_t* hashtable,
    uint64_t hashtable_size,
    uint32_t key_1,
    uint32_t key_2,
    uint32_t prob_limit,
    uint64_t value
);

__device__ uint64_t lookupHashTable(
    uint64_t* hashtable,
    uint64_t hashtable_size,
    uint32_t key_1,
    uint32_t key_2,
    uint32_t prob_limit
);