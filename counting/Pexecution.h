#pragma once

#include <tbb/tbb.h>
#include <tbb/task_group.h>
#include "graph.h"
#include "decompose.h"
#include "equation.h"
#include "compute_set_intersection.h"
#include "forest.h"
#include "triangle.h"

class ParallelProcessingMeta {
public:
    ui _num_threads;
    ui _node_partition_size;
    ui _prefix_partition_size;
    tbb::enumerable_thread_specific<int> _thread_id_ets;
    std::atomic<int> _next_thread_id;

    // allocate memory for each thread
    ui*** _dataV;
    ui*** _patternV;
    ui** _total_parallel_node_pos;
    HashTable** _total_hash_table;
    ui**** _total_candidates;
    ui*** _total_candidates_cnt;
    EdgeID*** _total_start_offset;
    ui*** _total_partition_candidates;
    ui** _total_partition_candidates_cnt;
    ui** _total_partition_candidates_pos;
    ui*** _total_key_pos;
    ui** _total_key_pos_size;
    ui** _total_size_bounds;
    ui*** _total_multi_join_pos;
    bool*** _total_visited_vertices;
    VertexID** _total_tmp;
    ParallelProcessingMeta(){}
    ParallelProcessingMeta(ui num_threads,
                    ui _node_partition_size,
                    ui _prefix_partition_size,
                    const DataGraph& din,
                    const DataGraph& dout, 
                    const DataGraph& dun);
    void setCandidates(const Tree& t, const DataGraph& dout);
    void clearCandidates(const Tree& t);
    void setPartitionCandidates(const Tree& t, const std::vector<VertexID>& partitionOrder, 
                                const std::vector<bool> &partitionCandPos, const DataGraph& dout, 
                                const std::vector<VertexID> &postOrder, int startPos, int endPos, ui n, ui m);
    void clearPartitionCandidates(const std::vector<VertexID>& partitionOrder, const std::vector<bool> &partitionCandPos, 
                                  const std::vector<VertexID> &postOrder, int startPos, int endPos);
    void setMultiJoinCandidates(const Tree&t, const DataGraph& dout);
    void clearMultiJoinCandidates(const Tree&t);
    ~ParallelProcessingMeta();
};