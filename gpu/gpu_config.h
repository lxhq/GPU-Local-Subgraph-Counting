#pragma once

/************************ size of GPU threads ********************************/
#define CG_SIZE 1u
#define GRID_DIM 1024u
#define BLOCK_DIM 256u
#define WARP_SIZE 32u
#define WARP_PER_BLOCK (BLOCK_DIM/WARP_SIZE)

#define IN_GRAPH 0u
#define OUT_GRAPH 1u
#define UN_GRAPH 2u

// hash table type:
// 0: warpcore
// 1: lock_free_hashtable
// 2: lock_based_hashtable
// 3: dense array

#ifndef HASH_TABLE_TYPE
  #define HASH_TABLE_TYPE 1u
#endif