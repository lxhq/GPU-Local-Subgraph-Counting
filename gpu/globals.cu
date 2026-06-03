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
uint64_t aggregation_hash_table_allocations = 0;
uint64_t aggregation_hash_table_total_bytes = 0;
uint64_t global_hash_table_allocations = 0;
uint64_t global_hash_table_total_bytes = 0;
RestartProfileStats current_restart_profile_stats;
RestartProfileStats global_restart_profile_stats;
HashTableOccupancyStats current_hash_table_occupancy_stats;
HashTableOccupancyStats global_hash_table_occupancy_stats;

void update_global_average(double new_value) {
    global_total_count++;
    // Running average formula: A = A + (new - A) / n
    global_running_avg += (new_value - global_running_avg) / global_total_count;
}

void reset_aggregation_hash_table_stats() {
    aggregation_hash_table_allocations = 0;
    aggregation_hash_table_total_bytes = 0;
}

void reset_global_aggregation_hash_table_stats() {
    global_hash_table_allocations = 0;
    global_hash_table_total_bytes = 0;
}

void update_aggregation_hash_table_stats(uint64_t hash_table_size_bytes, uint32_t table_count) {
    if (table_count == 0) {
        return;
    }
    aggregation_hash_table_allocations += table_count;
    aggregation_hash_table_total_bytes += hash_table_size_bytes * table_count;
    global_hash_table_allocations += table_count;
    global_hash_table_total_bytes += hash_table_size_bytes * table_count;
}

double get_aggregation_average_hash_table_size_bytes() {
    if (aggregation_hash_table_allocations == 0) {
        return 0.0;
    }
    return static_cast<double>(aggregation_hash_table_total_bytes) /
           static_cast<double>(aggregation_hash_table_allocations);
}

double get_global_average_hash_table_size_bytes() {
    if (global_hash_table_allocations == 0) {
        return 0.0;
    }
    return static_cast<double>(global_hash_table_total_bytes) /
           static_cast<double>(global_hash_table_allocations);
}

void reset_restart_profile_stats() {
    current_restart_profile_stats = RestartProfileStats{};
}

void reset_global_restart_profile_stats() {
    global_restart_profile_stats = RestartProfileStats{};
}

void record_restart_profile_batch(uint64_t attemptedPrefixes, uint64_t completedPrefixes, bool safeguardTriggered) {
    RestartProfileStats delta;
    delta.batchIterations = 1;
    delta.restartCount = attemptedPrefixes != completedPrefixes ? 1 : 0;
    delta.attemptedPrefixes = attemptedPrefixes;
    delta.completedPrefixes = completedPrefixes;
    delta.truncatedPrefixes = attemptedPrefixes > completedPrefixes ? attemptedPrefixes - completedPrefixes : 0;
    delta.safeguardCount = safeguardTriggered ? 1 : 0;

    current_restart_profile_stats.batchIterations += delta.batchIterations;
    current_restart_profile_stats.restartCount += delta.restartCount;
    current_restart_profile_stats.attemptedPrefixes += delta.attemptedPrefixes;
    current_restart_profile_stats.completedPrefixes += delta.completedPrefixes;
    current_restart_profile_stats.truncatedPrefixes += delta.truncatedPrefixes;
    current_restart_profile_stats.safeguardCount += delta.safeguardCount;

    global_restart_profile_stats.batchIterations += delta.batchIterations;
    global_restart_profile_stats.restartCount += delta.restartCount;
    global_restart_profile_stats.attemptedPrefixes += delta.attemptedPrefixes;
    global_restart_profile_stats.completedPrefixes += delta.completedPrefixes;
    global_restart_profile_stats.truncatedPrefixes += delta.truncatedPrefixes;
    global_restart_profile_stats.safeguardCount += delta.safeguardCount;
}

RestartProfileStats get_restart_profile_stats() {
    return current_restart_profile_stats;
}

RestartProfileStats get_global_restart_profile_stats() {
    return global_restart_profile_stats;
}

double get_restart_profile_rate(const RestartProfileStats &stats) {
    if (stats.batchIterations == 0) {
        return 0.0;
    }
    return static_cast<double>(stats.restartCount) / static_cast<double>(stats.batchIterations);
}

double get_restart_profile_truncated_fraction(const RestartProfileStats &stats) {
    if (stats.attemptedPrefixes == 0) {
        return 0.0;
    }
    return static_cast<double>(stats.truncatedPrefixes) / static_cast<double>(stats.attemptedPrefixes);
}

void reset_hash_table_occupancy_stats() {
    current_hash_table_occupancy_stats = HashTableOccupancyStats{};
}

void reset_global_hash_table_occupancy_stats() {
    global_hash_table_occupancy_stats = HashTableOccupancyStats{};
}

void record_hash_table_occupancy(uint64_t occupiedBuckets, uint64_t totalBuckets) {
    if (totalBuckets == 0) {
        return;
    }
    double occupancy = static_cast<double>(occupiedBuckets) / static_cast<double>(totalBuckets);
    current_hash_table_occupancy_stats.samples += 1;
    current_hash_table_occupancy_stats.occupancySum += occupancy;

    global_hash_table_occupancy_stats.samples += 1;
    global_hash_table_occupancy_stats.occupancySum += occupancy;
}

HashTableOccupancyStats get_hash_table_occupancy_stats() {
    return current_hash_table_occupancy_stats;
}

HashTableOccupancyStats get_global_hash_table_occupancy_stats() {
    return global_hash_table_occupancy_stats;
}

double get_average_hash_table_occupancy(const HashTableOccupancyStats &stats) {
    if (stats.samples == 0) {
        return 0.0;
    }
    return stats.occupancySum / static_cast<double>(stats.samples);
}
