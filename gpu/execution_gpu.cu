#include "execution_gpu.h"

void copyMetaToGPU (
        const DataGraph &din,
        const DataGraph &dout,
        const DataGraph &dun,
        const EdgeID *outID,
        const EdgeID *unID,
        const EdgeID *reverseID,
        MemoryManager &memory_manager
) {
    void* tmp = nullptr;

    uint32_t data_graph_vertex_count = dun.getNumVertices();
    uint32_t data_graph_edge_count = dun.getNumEdges();

    // copy din, dout, dun to device memory
    uint32_t dun_edge_count = dun.getNumEdges();
    uint32_t dout_edge_count = dout.getNumEdges();
    uint32_t din_edge_count = din.getNumEdges();
    cudaErrorCheck(cudaMemcpyToSymbol(C_EDGE_COUNT, &data_graph_edge_count, sizeof(uint32_t)));
    cudaErrorCheck(cudaMemcpyToSymbol(C_VERTEX_COUNT, &data_graph_vertex_count, sizeof(uint32_t)));

    tmp = memory_manager.allocate((data_graph_vertex_count + 1) * sizeof(uint32_t), sizeof(uint32_t), "din offset");
    cudaErrorCheck(cudaMemcpy(tmp, din.getOffsets(), sizeof(uint32_t) * (data_graph_vertex_count + 1), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(C_OFFSETS, &tmp, sizeof(uint32_t*), sizeof(uint32_t*) * IN_GRAPH));

    tmp = memory_manager.allocate(din_edge_count * sizeof(uint32_t), sizeof(uint32_t), "din neighbors");
    cudaErrorCheck(cudaMemcpy(tmp, din.getNbors(), sizeof(uint32_t) * din_edge_count, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(C_NEIGHBORS, &tmp, sizeof(uint32_t*), sizeof(uint32_t*) * IN_GRAPH));

    tmp = memory_manager.allocate((data_graph_vertex_count + 1) * sizeof(uint32_t), sizeof(uint32_t), "dout offset");
    cudaErrorCheck(cudaMemcpy(tmp, dout.getOffsets(), sizeof(uint32_t) * (data_graph_vertex_count + 1), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(C_OFFSETS, &tmp, sizeof(uint32_t*), sizeof(uint32_t*) * OUT_GRAPH));

    tmp = memory_manager.allocate(dout_edge_count * sizeof(uint32_t), sizeof(uint32_t), "dout neighbors");
    cudaErrorCheck(cudaMemcpy(tmp, dout.getNbors(), sizeof(uint32_t) * dout_edge_count, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(C_NEIGHBORS, &tmp, sizeof(uint32_t*), sizeof(uint32_t*) * OUT_GRAPH));

    tmp = memory_manager.allocate((data_graph_vertex_count + 1) * sizeof(uint32_t), sizeof(uint32_t), "dun offset");
    cudaErrorCheck(cudaMemcpy(tmp, dun.getOffsets(), sizeof(uint32_t) * (data_graph_vertex_count + 1), cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(C_OFFSETS, &tmp, sizeof(uint32_t*), sizeof(uint32_t*) * UN_GRAPH));

    tmp = memory_manager.allocate(dun_edge_count * sizeof(uint32_t), sizeof(uint32_t), "dun neighbors");
    cudaErrorCheck(cudaMemcpy(tmp, dun.getNbors(), sizeof(uint32_t) * dun_edge_count, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(C_NEIGHBORS, &tmp, sizeof(uint32_t*), sizeof(uint32_t*) * UN_GRAPH));

    // copy outID, unID, reverseID to device memory
    tmp = memory_manager.allocate(din_edge_count * sizeof(uint32_t), sizeof(uint32_t), "outID");
    cudaErrorCheck(cudaMemcpy(tmp, outID, sizeof(uint32_t) * din_edge_count, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(C_OUT_ID, &tmp, sizeof(uint32_t*)));

    tmp = memory_manager.allocate(dun_edge_count * sizeof(uint32_t), sizeof(uint32_t), "unID");
    cudaErrorCheck(cudaMemcpy(tmp, unID, sizeof(uint32_t) * dun_edge_count, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(C_UN_ID, &tmp, sizeof(uint32_t*)));

    tmp = memory_manager.allocate(dun_edge_count * sizeof(uint32_t), sizeof(uint32_t), "reverseID");
    cudaErrorCheck(cudaMemcpy(tmp, reverseID, sizeof(uint32_t) * dun_edge_count, cudaMemcpyHostToDevice));
    cudaErrorCheck(cudaMemcpyToSymbol(C_REVERSE_ID, &tmp, sizeof(uint32_t*)));
}

void handleFinalLevel(
    uint32_t* source,
    uint32_t source_count,
    NodeGPU* d_node,
    NodeGPU &h_node,
    uint32_t level,
    const Tree &t,
    VertexID nID,
    AggregationTableWrapper &hashtables,
    uint32_t prob_limit,
    uint32_t index_len,
    uint64_t* hd_failed_write
) {
    if (t.getNode(nID).edgeKey) {
        finalLevelWithEdgeWithLocalCacheKernel<<<SDIV(source_count, WARP_PER_BLOCK), BLOCK_DIM>>>(
            index_len,
            level,
            source,
            source_count,
            d_node,
            hd_failed_write,
            hashtables.getHashTableDevicePointer(),
            (void**)hashtables.getPartitionHashTableDevicePointer(),
            hashtables.getPartitionHashTableBucketSize(),
            t.getRootID() == nID,
            t.getNumNodes(),
            nID,
            prob_limit
        );
    } else {
        finalLevelWithLocalCacheKernel<<<SDIV(source_count, WARP_PER_BLOCK), BLOCK_DIM>>>(
            index_len,
            level,
            source,
            source_count,
            d_node,
            hd_failed_write,
            hashtables.getHashTableDevicePointer(),
            (void**)hashtables.getPartitionHashTableDevicePointer(),
            hashtables.getPartitionHashTableBucketSize(),
            t.getRootID() == nID,
            t.getNumNodes(),
            nID,
            prob_limit
        );
    }
    cudaErrorCheck(cudaDeviceSynchronize());
    // if hd_failed_write is not UINT64_MAX, update it to the failed index
    if (*hd_failed_write != UINT64_MAX) {
        uint32_t new_end = 0;
        cudaErrorCheck(cudaMemcpy(&new_end, source + *hd_failed_write * (index_len + level) + h_node.indexPos[nID],
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));
        *hd_failed_write = new_end - 1;
    }
}

void handleFinalLevelFullEnumeration(
    uint32_t* source,
    uint32_t source_count,
    NodeGPU* d_node,
    NodeGPU &h_node,
    uint32_t level,
    const Tree &t,
    VertexID nID,
    AggregationTableWrapper &hashtables,
    uint32_t prob_limit,
    uint32_t index_len,
    uint64_t* hd_failed_write
) {
    if (t.getNode(nID).edgeKey) {
        writeToHashTableWithEdgeKeyKernel<<<SDIV(source_count, BLOCK_DIM), BLOCK_DIM>>>(
            index_len,
            level,
            source,
            source_count,
            d_node,
            hd_failed_write,
            hashtables.getHashTableDevicePointer(),
            (void**)hashtables.getPartitionHashTableDevicePointer(),
            hashtables.getPartitionHashTableBucketSize(),
            t.getRootID() == nID,
            t.getNumNodes(),
            nID,
            prob_limit
        );
    } else {
        writeToHashTableKernel<<<SDIV(source_count, BLOCK_DIM), BLOCK_DIM>>>(
            index_len,
            level,
            source,
            source_count,
            d_node,
            hd_failed_write,
            hashtables.getHashTableDevicePointer(),
            (void**)hashtables.getPartitionHashTableDevicePointer(),
            hashtables.getPartitionHashTableBucketSize(),
            t.getRootID() == nID,
            t.getNumNodes(),
            nID,
            prob_limit
        );
    }
    cudaErrorCheck(cudaDeviceSynchronize());
    // if hd_failed_write is not UINT64_MAX, update it to the failed index
    if (*hd_failed_write != UINT64_MAX) {
        uint32_t new_end = 0;
        cudaErrorCheck(cudaMemcpy(&new_end, source + *hd_failed_write * (index_len + level) + h_node.indexPos[nID],
                        sizeof(uint32_t), cudaMemcpyDeviceToHost));
        *hd_failed_write = new_end - 1;
    }
}

void trivalNodeHelper(
    uint32_t* source,
    uint32_t source_count,
    MemoryManager &memory_manager,
    const Graph &dun,
    NodeGPU* d_node,
    NodeGPU &h_node,
    uint32_t level,
    const Tree &t,
    VertexID nID,
    AggregationTableWrapper &hashtables,
    uint32_t prob_limit,
    uint64_t target_memory_size,
    uint32_t index_len,
    uint64_t* hd_failed_write
) {
    if (level == t.getNode(nID).numVertices - 1 && source_count != 0) {
        handleFinalLevel(source, source_count, d_node, h_node, level, t, nID, hashtables,
                        prob_limit, index_len, hd_failed_write);
        return;
    }
    SubgraphMatching iter(memory_manager, source, source_count, d_node,
                        target_memory_size, level, level == t.getNode(nID).numVertices - 2, index_len);
    while (iter.hashNext()) {
        uint32_t next_level_count = 0;
        uint32_t* target = iter.next(next_level_count);
        trivalNodeHelper(target, next_level_count, memory_manager, dun, d_node, h_node, level + 1, t, nID,
                        hashtables, prob_limit, target_memory_size, index_len, hd_failed_write);
        if (*hd_failed_write != UINT64_MAX) {
            break;
        }
    }
}

void simpleTreeHelper(
    NodeGPU* prefix_node,
    uint32_t prefix_size,
    uint32_t* source,
    uint32_t source_count,
    MemoryManager &memory_manager,
    const Graph &dun,
    std::vector<NodeGPU*> &d_nodes,
    std::vector<NodeGPU> &h_nodes,
    uint32_t level,
    const Tree &t,
    AggregationTableWrapper &hashtables,
    uint32_t prob_limit,
    uint64_t target_memory_size,
    uint32_t index_len,
    const std::vector<std::vector<uint32_t>> &nodesAtStep,
    uint64_t* hd_failed_write,
    uint32_t partitionRootNodeID,
    std::vector<uint32_t>& steps,
    uint32_t batchSize,
    std::vector<bool>& isSafeGuardMode
) {
    if (level == prefix_size) {
        return;
    }
    SubgraphMatching iter(memory_manager, source, source_count, prefix_node,
                        target_memory_size, level, level == prefix_size, index_len);
    uint32_t next_level_count = 0, start = 0, end = 0;
    uint32_t* target = nullptr;
    while (iter.hashNext() || end < next_level_count) {
        // decide the range to process
        if (end < next_level_count) {
            start = end;
        } else {
            target = iter.next(next_level_count);
            start = 0;
            if (nodesAtStep[level].size() != 0) {
                uint32_t* nodesNeedsIndexing = nullptr;
                cudaErrorCheck(cudaMallocManaged(&nodesNeedsIndexing, sizeof(uint32_t) * nodesAtStep[level].size()));
                uint32_t nodesCount = 0;
                for (uint32_t nID : nodesAtStep[level]) {
                    if (nID == partitionRootNodeID) {
                        continue;
                    }
                    nodesNeedsIndexing[nodesCount++] = h_nodes[nID].indexPos[nID];
                }
                if (nodesCount != 0) {
                    // index the embedding in the target
#if HASH_TABLE_TYPE == 0 || HASH_TABLE_TYPE == 1 || HASH_TABLE_TYPE == 2
                    indexPrefixKernel<<<SDIV(next_level_count, BLOCK_DIM), BLOCK_DIM>>>(
                        index_len,
                        level,
                        target,
                        next_level_count,
                        nodesNeedsIndexing,
                        nodesCount
                    );
#elif HASH_TABLE_TYPE == 3
                    indexPrefixWithinBatchKernel<<<SDIV(next_level_count, BLOCK_DIM), BLOCK_DIM>>>(
                        index_len,
                        level,
                        target,
                        next_level_count,
                        nodesNeedsIndexing,
                        nodesCount,
                        batchSize
                    );
#endif
                    cudaErrorCheck(cudaDeviceSynchronize());
                }
                cudaErrorCheck(cudaFree(nodesNeedsIndexing));
            }
        }
        if (nodesAtStep[level].size() != 0) {
            if (steps[level] == UINT32_MAX) {
                steps[level] = next_level_count;
            }
            end = std::min(start + steps[level], next_level_count);
#if HASH_TABLE_TYPE == 3
            end = std::min(start + batchSize, next_level_count);
#endif
        } else {
            end = next_level_count;
        }
        uint32_t original_end = end;
        for (uint32_t nID : nodesAtStep[level]) {
            // enumerate the node, pass target[start:end) with reference
            *hd_failed_write = UINT64_MAX;
            hashtables.clearPartitionHashTable(nID);
            trivalNodeHelper(target + start * (index_len + 1 + level), end - start, memory_manager, dun, d_nodes[nID], h_nodes[nID],
                level + 1, t, nID, hashtables, prob_limit, target_memory_size, index_len, hd_failed_write);
            
            // update end if there is any insert failed
            if (*hd_failed_write != UINT64_MAX) {
                end = *hd_failed_write;
            }
        }
        // pass the target[start:end) to the next level
        simpleTreeHelper(prefix_node, prefix_size, target, next_level_count, memory_manager, dun,
                d_nodes, h_nodes, level + 1, t, hashtables, prob_limit, target_memory_size, index_len,
                nodesAtStep, hd_failed_write, partitionRootNodeID, steps, batchSize, isSafeGuardMode);
        if (nodesAtStep[level].size() != 0) {
            if (original_end != end) {
                steps[level] = end - start;
            } else {
                steps[level] = std::floor(steps[level] * 1.5);
            }
            update_global_average(end - start);
            // ------------ safeguard ------------//
            if (isSafeGuardMode[level]) {
                isSafeGuardMode[level] = false;
                steps[level] = UINT32_MAX;
                // enable all children nodes' device partition, copy the content from hashtable host to device
                for (uint32_t nID : nodesAtStep[level]) {
                    cudaErrorCheck(
                        cudaMemcpy(hashtables.getPartitionHashTableDevicePointer() + nID,
                                    hashtables.getPartitionHashTableHostPointer() + nID,
                                    sizeof(uint64_t*), cudaMemcpyHostToDevice)
                    );
                }
            }
            if (steps[level] == 0) {
                std::cout << "safeguard is triggered, the performance may be degraded! Please consider increasing the hash table size" << std::endl;
                isSafeGuardMode[level] = true;
                steps[level] = 1;
                // disable all children nodes' device partition
                for (uint32_t nID : nodesAtStep[level]) {
                    cudaErrorCheck(
                        cudaMemset(hashtables.getPartitionHashTableDevicePointer() + nID,
                                    0, sizeof(uint64_t*))
                    );
                    // clear the array
                    if (nID != partitionRootNodeID) {
                        cudaErrorCheck(
                            cudaMemset(hashtables.getHashTableHostPointer()[nID], 0,
                                        (dun.getNumEdges() + 1) * sizeof(uint64_t))
                        );
                    }
                }
            }
            // ------------ safeguard ------------//
        }
    }
}

void executePartitionGPU(
        VertexID pID,
        const Tree &t,
        AggregationTableWrapper &hashtables,
        std::vector<NodeGPU*> &d_nodes,
        std::vector<NodeGPU> &h_nodes,
        const uint32_t prob_limit,
        float ratio,
        MemoryManager &memory_manager,
        const Graph &dun
) {
    const std::vector<std::vector<VertexID>> &globalOrder = t.getGlobalOrder();
    const std::vector<VertexID> &partitionOrder = globalOrder[pID];
    const std::vector<std::vector<std::vector<VertexID>>> &nodesAtStep = t.getNodesAtStep();
    const std::vector<VertexID> &postOrder = t.getPostOrder();
    const std::vector<int> &partitionPos = t.getPartitionPos();
    const std::vector<std::vector<int>> &partitionInPos = t.getPartitionInPos(pID);
    const std::vector<std::vector<int>> &partitionOutPos = t.getPartitionOutPos(pID);
    const std::vector<std::vector<int>> &partitionUnPos = t.getPartitionUnPos(pID);
    const std::vector<bool> &partitionInterPos = t.getPartitionInterPos(pID);
    const std::vector<std::vector<int>> &greaterPos = t.getPartitionGreaterPos(pID);
    const std::vector<std::vector<int>> &lessPos = t.getPartitionLessPos(pID);

    int startPos;
    if (pID == 0) startPos = 0;
    else startPos = partitionPos[pID - 1] + 1;
    // exclude the end position [startPos, endPos), the partition starts at postOrder[startPos], end at postOrder[endPos - 1]
    int endPos = partitionPos[pID] + 1;
    bool isRoot = endPos == postOrder.size();
    uint32_t partitionRootNodeID = postOrder[endPos - 1];
    uint32_t n = dun.getNumVertices(), m = dun.getNumEdges();

    /*** for node in this partition decide the indexPos position *****************/
    uint32_t index_len = endPos - startPos - 1;
    uint32_t indexPos[MAX_PATTERN_SIZE];
    uint32_t prefixLen[MAX_NUM_NODE];
    for (int i = startPos; i < endPos - 1; i++) {
        VertexID nID = postOrder[i];
        indexPos[nID] = i - startPos;
        prefixLen[nID] = 0;
        for (int j = 0; j < nodesAtStep[pID].size(); j++) {
            if (std::find(nodesAtStep[pID][j].begin(), nodesAtStep[pID][j].end(), nID) != nodesAtStep[pID][j].end()) {
                prefixLen[nID] = j + 1;
                break;
            }
        }
    }
    for (int i = startPos; i < endPos; i++) {
        VertexID nID = postOrder[i];
        for (int j = 0; j < MAX_PATTERN_SIZE; j++) {
            h_nodes[nID].indexPos[j] = indexPos[j];
        }
        for (int j = 0; j < MAX_NUM_NODE; j++) {
            h_nodes[nID].prefixLen[j] = prefixLen[j];
        }
        h_nodes[nID].multiJoin = 0;
        cudaErrorCheck(cudaMemcpy(d_nodes[nID], &h_nodes[nID], sizeof(NodeGPU), cudaMemcpyHostToDevice));
    }

    if (partitionOrder.empty()) {
        uint64_t* hd_failed_write;
        cudaErrorCheck(cudaMallocManaged(&hd_failed_write, sizeof(uint64_t)));
        *hd_failed_write = UINT64_MAX;
        for (int i = startPos; i < endPos; ++i) {
            VertexID nID = postOrder[i];
            isRoot = endPos == postOrder.size() && i == endPos - 1;
            uint64_t available_memory = memory_manager.getAvailableMemory();
            uint64_t target_memory_size = static_cast<uint64_t>(available_memory / (t.getNode(nID).numVertices - 1));
            trivalNodeHelper(nullptr, dun.getNumVertices(), memory_manager, dun, d_nodes[nID], h_nodes[nID], 0,
                            t, nID, hashtables, prob_limit, target_memory_size, index_len, hd_failed_write);
        }
        cudaErrorCheck(cudaFree(hd_failed_write));
        return;
    }
    uint32_t numHashTable = endPos - startPos - 1;
    uint64_t availableMemory = memory_manager.getAvailableMemory();
    uint64_t hashTableSizeMemoryLimit = 0, subgraphEnumerationMemoryLimit = 0, batchSize = 0;
    bool isEdgeTable = false;
    // hashTableSizeMemoryLimit * numHashTable + subgraphEnumerationMemoryLimit <= availableMemory
    // subgraphEnumerationMemoryLimit = hashTableSizeMemoryLimit * ratio
    // hashTableSizeMemoryLimit * numHashTable + hashTableSizeMemoryLimit * ratio <= availableMemory
    // hashTableSizeMemoryLimit = availableMemory / (numHashTable + ratio)
    hashTableSizeMemoryLimit = static_cast<uint64_t>(availableMemory / (numHashTable + ratio));
    subgraphEnumerationMemoryLimit = static_cast<uint64_t>(hashTableSizeMemoryLimit * ratio);
    for (int i = startPos; i < endPos; i++) {
        VertexID nID = postOrder[i];
        if (t.getNode(nID).edgeKey) {
            isEdgeTable = true;
        }
    }
    if (isEdgeTable) {
        batchSize = hashTableSizeMemoryLimit / sizeof(uint64_t) / m;
        cudaErrorCheck(cudaMemcpyToSymbol(C_BASELINE_TABLE_SIZE, &m, sizeof(uint32_t)));

    } else {
        batchSize = hashTableSizeMemoryLimit / sizeof(uint64_t) / n;
        cudaErrorCheck(cudaMemcpyToSymbol(C_BASELINE_TABLE_SIZE, &n, sizeof(uint32_t)));
    }
    
    if (hashTableSizeMemoryLimit < m * 2 * sizeof(uint64_t)) {
        std::cout <<" WARNING: The hash table size is very small. Please consider setting a smaller ratio value or allocate more memory for this program" << std::endl;
    }
    std::vector<uint32_t> nodesInPartition;
    for (int i = startPos; i < endPos; i++) {
        VertexID nID = postOrder[i];
        nodesInPartition.push_back(nID);
    }
    hashtables.allocatePartitionHashTable(t, memory_manager, hashTableSizeMemoryLimit, nodesInPartition, postOrder[endPos - 1]);
    NodeGPU h_prefix_node(partitionInterPos, partitionInPos, partitionOutPos, partitionUnPos, greaterPos, lessPos);
    NodeGPU* d_prefix_node = static_cast<NodeGPU*>(memory_manager.allocate(sizeof(NodeGPU), sizeof(uint32_t), "prefix node"));
    cudaErrorCheck(cudaMemcpy(d_prefix_node, &h_prefix_node, sizeof(NodeGPU), cudaMemcpyHostToDevice));

    // get the largest node
    uint32_t maxNodeSize = 0;
    for (int i = startPos; i < endPos; i++) {
        VertexID nID = postOrder[i];
        if (t.getNode(nID).numVertices > maxNodeSize) {
            maxNodeSize = t.getNode(nID).numVertices;
        }
    }
    uint64_t target_memory_size = static_cast<uint64_t>(subgraphEnumerationMemoryLimit / (maxNodeSize - 1)); // for simple case, no need to copy prefix
    uint64_t* hd_failed_write;
    cudaErrorCheck(cudaMallocManaged(&hd_failed_write, sizeof(uint64_t)));
    std::vector<uint32_t> steps(partitionOrder.size(), UINT32_MAX);
    std::vector<bool> isSafeGuardMode(partitionOrder.size(), false);
    simpleTreeHelper(
        d_prefix_node,
        partitionOrder.size(),
        nullptr,
        dun.getNumVertices(),
        memory_manager,
        dun,
        d_nodes,
        h_nodes,
        0,
        t,
        hashtables,
        prob_limit,
        target_memory_size,
        index_len,
        nodesAtStep[pID],
        hd_failed_write,
        partitionRootNodeID,
        steps,
        batchSize,
        isSafeGuardMode
    );
    cudaErrorCheck(cudaFree(hd_failed_write));
    memory_manager.release(d_prefix_node);
    hashtables.releasePartitionHashTable(t, memory_manager);
}

void  executeTreeGPU (
        const Tree &t,
        const DataGraph &dun,
        HashTable *H,
        MemoryManager &memory_manager,
        const uint32_t prob_limit,
        float ratio
) {

    /*************************** create the hashtable in the device memory **************************/
#if HASH_TABLE_TYPE == 0
    WarpcoreWrapper hashtables(t, memory_manager, dun);
#elif HASH_TABLE_TYPE == 1
    LockFreeHashTableWrapper hashtables(t, memory_manager, dun);
#elif HASH_TABLE_TYPE == 2
    LockBasedHashTableWrapper hashtables(t, memory_manager, dun);
#elif HASH_TABLE_TYPE == 3
    DenseArrayWrapper hashtables(t, memory_manager, dun);
#endif

    /*************************** create and copy all nodes to device memory **************************/
    std::vector<NodeGPU*> d_nodes;
    std::vector<NodeGPU> h_nodes;
    for (int nID = 0; nID < t.getNumNodes(); nID++) {
        NodeGPU* d_node = static_cast<NodeGPU*>(memory_manager.allocate(sizeof(NodeGPU), sizeof(uint32_t), "node " + std::to_string(nID)));
        NodeGPU h_node(t, nID);
        d_nodes.emplace_back(d_node);
        h_nodes.emplace_back(h_node);
    }

    /*************************** start compute each partitions **************************/
    for (VertexID pID = 0; pID < t.getPartitionPos().size(); ++pID) {
        executePartitionGPU(pID, t, hashtables, d_nodes, h_nodes, prob_limit, ratio, memory_manager, dun);
    }

    /********* copy the hash table to the host memory *********/
    cudaErrorCheck(cudaMemcpy(H[t.getRootID()], hashtables.getHashTableHostPointer()[t.getRootID()],
                    (dun.getNumEdges() + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    /*************************** release the memory **************************/
    for (int nID = t.getNumNodes() - 1; nID >= 0; nID--) {
        memory_manager.release(d_nodes[nID]);
    }
    hashtables.release(t, memory_manager);
}

void measureDepth(uint32_t nID, const Tree &t, std::vector<uint32_t> &nodeCallingSizes, uint32_t callingDepth) {
    Node tau = t.getNode(nID);
    const std::vector<std::vector<VertexID>> &nodesAtStep = t.getNodesAtStep()[nID];
    if (tau.prefixSize != 0) {
        for (int i = 0; i < nodesAtStep[0].size(); i++) {
            VertexID childID = nodesAtStep[0][i];
            measureDepth(childID, t, nodeCallingSizes, callingDepth + 1);
        }
    }
    uint32_t mappingSize = tau.prefixSize;
    while (mappingSize < tau.numVertices) {
        for (int i = 0; i < nodesAtStep[mappingSize].size(); i++) {
            VertexID childID = nodesAtStep[mappingSize][i];
            measureDepth(childID, t, nodeCallingSizes, callingDepth + 1 + (mappingSize - tau.prefixSize) + 1);
        }
        mappingSize++;
    }
    nodeCallingSizes[nID] = (mappingSize - tau.prefixSize - 1) + callingDepth;
}

void complexTreeHelper(
    uint32_t nID,
    uint32_t* source,
    uint32_t source_count,
    MemoryManager &memory_manager,
    const Graph &dun,
    std::vector<NodeGPU*> &d_nodes,
    std::vector<NodeGPU> &h_nodes,
    uint32_t level,
    const Tree &t,
    AggregationTableWrapper &hashtables,
    uint32_t prob_limit,
    uint64_t target_memory_size,
    uint32_t index_len,
    uint64_t* hd_failed_write,
    uint32_t partitionRootNodeID,
    std::vector<std::vector<uint32_t>> &steps,
    uint32_t batchSize,
    std::vector<std::vector<bool>> &isSafeGuardMode) {
    if (level == t.getNode(nID).numVertices && source_count != 0) {
        handleFinalLevelFullEnumeration(source, source_count, d_nodes[nID], h_nodes[nID], level, t, nID, hashtables,
            prob_limit, index_len, hd_failed_write);
        return;
    }
    bool isLastLevel = (t.getNode(nID).numVertices - 1 == level) && (t.getNodesAtStep()[nID][level].size() == 0);
    SubgraphMatching iter(memory_manager, source, source_count, d_nodes[nID],
                    target_memory_size, level, isLastLevel, index_len);
    uint32_t next_level_count = 0, start = 0, end = 0;
    uint32_t* target = nullptr;
    while (iter.hashNext() || end < next_level_count) {
        // decide the range to process
        if (end < next_level_count) {
            start = end;
        } else {
            target = iter.next(next_level_count);
            start = 0;
            if (t.getNodesAtStep()[nID][level].size() != 0) {
                uint32_t* nodesNeedsIndexing = nullptr;
                cudaErrorCheck(cudaMallocManaged(&nodesNeedsIndexing, sizeof(uint32_t) * t.getNodesAtStep()[nID][level].size()));
                uint32_t nodesCount = 0;
                for (uint32_t cID : t.getNodesAtStep()[nID][level]) {
                    if (cID == partitionRootNodeID) {
                        continue;
                    }
                    nodesNeedsIndexing[nodesCount++] = h_nodes[cID].indexPos[cID];
                }
                if (nodesCount != 0) {
                    // index the embedding in the target
#if HASH_TABLE_TYPE == 0 || HASH_TABLE_TYPE == 1 || HASH_TABLE_TYPE == 2
                    indexPrefixKernel<<<SDIV(next_level_count, BLOCK_DIM), BLOCK_DIM>>>(
                        index_len,
                        level,
                        target,
                        next_level_count,
                        nodesNeedsIndexing,
                        nodesCount
                    );
#elif HASH_TABLE_TYPE == 3
                    indexPrefixWithinBatchKernel<<<SDIV(next_level_count, BLOCK_DIM), BLOCK_DIM>>>(
                        index_len,
                        level,
                        target,
                        next_level_count,
                        nodesNeedsIndexing,
                        nodesCount,
                        batchSize
                    );
#endif
                    cudaErrorCheck(cudaDeviceSynchronize());
                }
                cudaErrorCheck(cudaFree(nodesNeedsIndexing));
            }
        }
        if (t.getNodesAtStep()[nID][level].size() != 0) {
            if (steps[nID][level] == UINT32_MAX) {
                steps[nID][level] = next_level_count;
            }
            end = std::min(start + steps[nID][level], next_level_count);
#if HASH_TABLE_TYPE == 3
            end = std::min(start + batchSize, next_level_count);
#endif
        } else {
            end = next_level_count;
        }
        uint32_t original_end = end;
        for (uint32_t cID : t.getNodesAtStep()[nID][level]) {
            // enumerate the node, pass target[start:end) with reference
            *hd_failed_write = UINT64_MAX;
            // copy prefix
            uint32_t* copied_target = static_cast<uint32_t*>(memory_manager.allocate(
                (end - start) * (index_len + h_nodes[cID].prefixPosCount) * sizeof(uint32_t), sizeof(uint32_t), "copied prefix"
            ));
            copyPrefixKernel<<<SDIV(end - start, BLOCK_DIM), BLOCK_DIM>>>(
                target + start * (index_len + level + 1),
                end - start,
                copied_target,
                index_len,
                d_nodes[cID],
                level);
            cudaErrorCheck(cudaDeviceSynchronize());
            hashtables.clearPartitionHashTable(cID);
            complexTreeHelper(cID, copied_target, end - start, memory_manager, dun, d_nodes, h_nodes,
                            h_nodes[cID].prefixPosCount, t, hashtables, prob_limit, target_memory_size,
                        index_len, hd_failed_write, partitionRootNodeID, steps, batchSize, isSafeGuardMode);
            // release copy prefix
            memory_manager.release(copied_target);

            // update end if there is any insert failed
            if (*hd_failed_write != UINT64_MAX) {
                end = *hd_failed_write;
                *hd_failed_write = UINT64_MAX;
            }
        }
        // pass the target[start:end) to the next level
        complexTreeHelper(nID, target + start * (index_len + level + 1), end - start, memory_manager, dun,
                        d_nodes, h_nodes, level + 1, t, hashtables, prob_limit, target_memory_size, 
                        index_len, hd_failed_write, partitionRootNodeID, steps, batchSize, isSafeGuardMode);
        if (t.getNodesAtStep()[nID][level].size() != 0) {
            if (original_end != end) {
                steps[nID][level] = end - start;
            } else {
                steps[nID][level] = std::floor(steps[nID][level] * 1.5);
            }
            update_global_average(end - start);
            // ------------ safeguard ------------//
            if (isSafeGuardMode[nID][level]) {
                isSafeGuardMode[nID][level] = false;
                steps[nID][level] = UINT32_MAX;
                // enable all children nodes' device partition, copy the content from hashtable host to device
                for (uint32_t cID : t.getNodesAtStep()[nID][level]) {
                    cudaErrorCheck(
                        cudaMemcpy(hashtables.getPartitionHashTableDevicePointer() + cID,
                                    hashtables.getPartitionHashTableHostPointer() + cID,
                                    sizeof(uint64_t*), cudaMemcpyHostToDevice)
                    );
                }
            }
            if (steps[nID][level] == 0) {
                std::cout << "safeguard is triggered, the performance may be degraded! Please consider increasing the hash table size" << std::endl;
                isSafeGuardMode[nID][level] = true;
                steps[nID][level] = 1;
                // disable all children nodes' device partition
                for (uint32_t cID : t.getNodesAtStep()[nID][level]) {
                    cudaErrorCheck(
                        cudaMemset(hashtables.getPartitionHashTableDevicePointer() + cID,
                                    0, sizeof(uint64_t*))
                    );
                    // clear the array
                    if (cID != partitionRootNodeID) {
                        cudaErrorCheck(
                            cudaMemset(hashtables.getHashTableHostPointer()[cID], 0,
                                        (dun.getNumEdges() + 1) * sizeof(uint64_t))
                        );
                    }
                }
            }
            // ------------ safeguard ------------//
        }
        if (*hd_failed_write != UINT64_MAX) {
            break;
        }
    }
}

void multiJoinTreeGPU(
        const Tree &t,
        const DataGraph &dun,
        HashTable *H,
        MemoryManager &memory_manager,
        uint32_t prob_limit,
        float ratio) {
#if HASH_TABLE_TYPE == 0
    WarpcoreWrapper hashtables(t, memory_manager, dun);
#elif HASH_TABLE_TYPE == 1
    LockFreeHashTableWrapper hashtables(t, memory_manager, dun);
#elif HASH_TABLE_TYPE == 2
    LockBasedHashTableWrapper hashtables(t, memory_manager, dun);
#elif HASH_TABLE_TYPE == 3
    DenseArrayWrapper hashtables(t, memory_manager, dun);
#endif
    
    /********** compute the size of the prefix and hashtable *********/
    std::vector<uint32_t> runningNodes;
    for (VertexID rootNodeID: t.getPostOrder()) {
        runningNodes.push_back(rootNodeID);
        if (t.getNode(rootNodeID).prefixSize == 0) {
            /********* get node ids that in this partition *********/
            std::vector<uint32_t> partition = runningNodes;
            runningNodes.clear();
            uint32_t index_len = partition.size() - 1;
            uint32_t indexPos[MAX_PATTERN_SIZE];
            for (int i = 0; i < partition.size() - 1; i++) {
                VertexID nID = partition[i];
                indexPos[nID] = i;
            }
            std::vector<NodeGPU*> d_nodes;
            std::vector<NodeGPU> h_nodes;
            for (int nID = 0; nID < t.getNumNodes(); nID++) {
                NodeGPU h_node(t, nID);
                // decide the index_len and the position of each node's index
                for (int i = 0; i < MAX_PATTERN_SIZE; i++) {
                    h_node.indexPos[i] = indexPos[i];
                }
                h_node.multiJoin = 1;
                NodeGPU* d_node = static_cast<NodeGPU*>(memory_manager.allocate(sizeof(NodeGPU), sizeof(uint32_t), "node " + std::to_string(nID)));
                cudaErrorCheck(cudaMemcpy(d_node, &h_node, sizeof(NodeGPU), cudaMemcpyHostToDevice));
                d_nodes.emplace_back(d_node);
                h_nodes.emplace_back(h_node);
            }

            /********* allocate memory for each node (subgraphenumeration, hashtable) *****************/
            uint32_t numHashTable = partition.size() - 1;
            uint64_t availableMemory = memory_manager.getAvailableMemory();
            uint64_t hashTableSizeMemoryLimit = 0, subgraphEnumerationMemoryLimit = 0, batchSize = 0;
            bool isEdgeTable = false;
            // hashTableSizeMemoryLimit * numHashTable + subgraphEnumerationMemoryLimit <= availableMemory
            // subgraphEnumerationMemoryLimit = hashTableSizeMemoryLimit * ratio
            // hashTableSizeMemoryLimit * numHashTable + hashTableSizeMemoryLimit * ratio <= availableMemory
            // hashTableSizeMemoryLimit = availableMemory / (numHashTable + ratio)
            hashTableSizeMemoryLimit = static_cast<uint64_t>(availableMemory / (numHashTable + ratio));
            subgraphEnumerationMemoryLimit = static_cast<uint64_t>(hashTableSizeMemoryLimit * ratio);
            for (uint32_t nID : partition) {
                if (t.getNode(nID).edgeKey) {
                    isEdgeTable = true;
                }
            }
            uint32_t m = dun.getNumEdges();
            uint32_t n = dun.getNumVertices();
            if (isEdgeTable) {
                batchSize = hashTableSizeMemoryLimit / sizeof(uint64_t) / m;
                cudaErrorCheck(cudaMemcpyToSymbol(C_BASELINE_TABLE_SIZE, &m, sizeof(uint32_t)));

            } else {
                batchSize = hashTableSizeMemoryLimit / sizeof(uint64_t) / n;
                cudaErrorCheck(cudaMemcpyToSymbol(C_BASELINE_TABLE_SIZE, &n, sizeof(uint32_t)));
            }
            if (hashTableSizeMemoryLimit < dun.getNumEdges() * 2 * sizeof(uint64_t)) {
                std::cout <<" WARNING: The hash table size is very small. Please consider setting a smaller ratio value or allocate more memory for this program" << std::endl;
            }
            hashtables.allocatePartitionHashTable(t, memory_manager, hashTableSizeMemoryLimit, partition, rootNodeID);

            // get the largest node
            std::vector<uint32_t> nodeCallingSizes(t.getNumNodes(), 0);
            measureDepth(rootNodeID, t, nodeCallingSizes, 0);
            uint32_t maxNodeSize = 0;
            for (int i = 0; i < t.getNumNodes(); i++) {
                maxNodeSize = std::max(maxNodeSize, nodeCallingSizes[i]);
            }
            uint64_t target_memory_size = static_cast<uint64_t>(subgraphEnumerationMemoryLimit / (maxNodeSize + 1));
            // call the complex tree helper
            uint64_t* hd_failed_write = nullptr;
            cudaErrorCheck(cudaMallocManaged(&hd_failed_write, sizeof(uint64_t)));
            *hd_failed_write = UINT64_MAX;
            std::vector<std::vector<uint32_t>> steps(t.getNumNodes(), std::vector<uint32_t>(MAX_PATTERN_SIZE, UINT32_MAX));
            std::vector<std::vector<bool>> isSafeGuardMode(t.getNumNodes(), std::vector<bool>(MAX_PATTERN_SIZE, false));
            complexTreeHelper(rootNodeID, nullptr, dun.getNumVertices(), memory_manager, dun, d_nodes, h_nodes,
                            0, t, hashtables, prob_limit, target_memory_size, index_len, hd_failed_write, rootNodeID,
                            steps, batchSize, isSafeGuardMode);
            
            /********* copy the hash table to the host memory *********/
            cudaErrorCheck(cudaMemcpy(H[rootNodeID], hashtables.getHashTableHostPointer()[rootNodeID],
                            (dun.getNumEdges() + 1) * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            
            /********* release memory *********/
            cudaErrorCheck(cudaFree(hd_failed_write));
            hashtables.releasePartitionHashTable(t, memory_manager);
            for (int nID = t.getNumNodes() - 1; nID >= 0; nID--) {
                memory_manager.release(d_nodes[nID]);
            }
        }
    }

    /********* release the memory ********************/
    hashtables.release(t, memory_manager);
}