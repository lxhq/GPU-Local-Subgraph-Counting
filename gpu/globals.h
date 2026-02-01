#pragma once

#include <cstdint>

extern __constant__ uint32_t* C_OFFSETS[3];
extern __constant__ uint32_t* C_NEIGHBORS[3];
extern __constant__ uint32_t* C_OUT_ID;
extern __constant__ uint32_t* C_UN_ID;
extern __constant__ uint32_t* C_REVERSE_ID;
extern __constant__ uint32_t C_EDGE_COUNT;
extern __constant__ uint32_t C_VERTEX_COUNT;
extern __constant__ uint32_t C_BASELINE_TABLE_SIZE;

// 'extern' tells other files this variable is defined in a .cu or .cpp file
extern double global_running_avg;
extern uint64_t global_total_count;

// Function to update the average from the CPU side
void update_global_average(double new_value);