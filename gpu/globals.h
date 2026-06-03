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

extern uint64_t aggregation_hash_table_allocations;
extern uint64_t aggregation_hash_table_total_bytes;
extern uint64_t global_hash_table_allocations;
extern uint64_t global_hash_table_total_bytes;

void reset_aggregation_hash_table_stats();
void reset_global_aggregation_hash_table_stats();
void update_aggregation_hash_table_stats(uint64_t hash_table_size_bytes, uint32_t table_count);
double get_aggregation_average_hash_table_size_bytes();
double get_global_average_hash_table_size_bytes();

struct RestartProfileStats {
    uint64_t batchIterations = 0;
    uint64_t restartCount = 0;
    uint64_t attemptedPrefixes = 0;
    uint64_t completedPrefixes = 0;
    uint64_t truncatedPrefixes = 0;
    uint64_t safeguardCount = 0;
};

void reset_restart_profile_stats();
void reset_global_restart_profile_stats();
void record_restart_profile_batch(uint64_t attemptedPrefixes, uint64_t completedPrefixes, bool safeguardTriggered);
RestartProfileStats get_restart_profile_stats();
RestartProfileStats get_global_restart_profile_stats();
double get_restart_profile_rate(const RestartProfileStats &stats);
double get_restart_profile_truncated_fraction(const RestartProfileStats &stats);
