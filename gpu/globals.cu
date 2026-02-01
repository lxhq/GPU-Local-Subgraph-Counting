#include "globals.h"

__constant__ uint32_t* C_OFFSETS[3];
__constant__ uint32_t* C_NEIGHBORS[3];
__constant__ uint32_t* C_OUT_ID;
__constant__ uint32_t* C_UN_ID;
__constant__ uint32_t* C_REVERSE_ID;
__constant__ uint32_t C_EDGE_COUNT;
__constant__ uint32_t C_VERTEX_COUNT;
__constant__ uint32_t C_BASELINE_TABLE_SIZE;

// Actual definitions
double global_running_avg = 0.0;
uint64_t global_total_count = 0;

void update_global_average(double new_value) {
    global_total_count++;
    // Running average formula: A = A + (new - A) / n
    global_running_avg += (new_value - global_running_avg) / global_total_count;
}