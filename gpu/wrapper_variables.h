#pragma once

#include <cuda_runtime.h>
#include <warpcore/counting_hash_table.cuh>

#include "graph.h"
#include "tree.h"
#include "memory_manager.h"
#include "cuda_helpers.h"
#include "nodeGPU.h"
#include "gpu_config.h"

class AggregationTableWrapper {
protected:
    std::vector<uint64_t*> h_d_H;                   // host pointers to device hash tables for each aggregation vertex
    uint64_t** d_d_H = nullptr;                     // device pointer to device hash tables for each aggregation vertex
    uint64_t partitionHashtableBuckets = 0;         // number of buckets in each partition hashtable
    uint64_t partitionHashtableSize = 0;            // size in bytes of each partition hashtable
public:
    virtual void allocatePartitionHashTable(const Tree& t, MemoryManager &memory_manager, uint64_t hashtableSize,
                                    std::vector<uint32_t>& nodesInPartition, uint32_t parititionRootNode) = 0;
    virtual void clearPartitionHashTable(uint32_t nID) = 0;
    virtual void releasePartitionHashTable(const Tree &t, MemoryManager &memory_manager) = 0;
    virtual void release(const Tree &t, MemoryManager &memory_manager) = 0;
    virtual void** getPartitionHashTableDevicePointer() = 0;
    virtual void** getPartitionHashTableHostPointer() = 0;
    AggregationTableWrapper(const Tree &t, MemoryManager &memory_manager, const DataGraph &dun) {
        this->d_d_H = static_cast<uint64_t**>(
            memory_manager.allocate(t.getNumNodes() * sizeof(uint64_t*), sizeof(uint64_t*), "d_d_H")
        );
        for (uint32_t nID = 0; nID < t.getNumNodes(); nID++) {
            uint64_t* d_H = static_cast<uint64_t*>(
                memory_manager.allocate((dun.getNumEdges() + 1) * sizeof(uint64_t), sizeof(uint64_t), "d_H " + std::to_string(nID))
            );
            cudaErrorCheck(cudaMemset(d_H, 0, (dun.getNumEdges() + 1) * sizeof(uint64_t)));
            cudaErrorCheck(cudaMemcpy(d_d_H + nID, &d_H, sizeof(uint64_t*), cudaMemcpyHostToDevice));
            this->h_d_H.push_back(d_H);
        }
    }
    uint64_t** getHashTableHostPointer(){
        return h_d_H.data();
    }
    uint64_t** getHashTableDevicePointer(){
        return d_d_H;
    }
    uint64_t getPartitionHashTableBucketSize(){
        return partitionHashtableBuckets;
    }
};

class WarpcoreWrapper : public AggregationTableWrapper {
private:
    std::vector<warpcore::CountingHashTable<>*> h_d_partition_H;
    warpcore::CountingHashTable<>** d_d_partition_H = nullptr;
    std::vector<std::pair<void*, std::pair<void*, void*>>> partition_H_ptrs;
public:
    WarpcoreWrapper(const Tree &t, MemoryManager &memory_manager, const DataGraph &dun);
    void allocatePartitionHashTable(const Tree& t, MemoryManager &memory_manager, uint64_t hashtableSize,
                                    std::vector<uint32_t>& nodesInPartition, uint32_t parititionRootNode) override;
    void clearPartitionHashTable(uint32_t nID) override;
    void releasePartitionHashTable(const Tree &t, MemoryManager &memory_manager) override;
    void release(const Tree &t, MemoryManager &memory_manager) override;
    void** getPartitionHashTableDevicePointer() override;
    void** getPartitionHashTableHostPointer() override;
};

class LockFreeHashTableWrapper : public AggregationTableWrapper {
private:
    std::vector<uint64_t*> h_d_partition_H;
    uint64_t** d_d_partition_H = nullptr;
public:
    LockFreeHashTableWrapper(const Tree &t, MemoryManager &memory_manager, const DataGraph &dun);
    void allocatePartitionHashTable(const Tree& t, MemoryManager &memory_manager, uint64_t hashtableSize,
                                    std::vector<uint32_t>& nodesInPartition, uint32_t parititionRootNode) override;
    void clearPartitionHashTable(uint32_t nID) override;
    void releasePartitionHashTable(const Tree &t, MemoryManager &memory_manager) override;
    void release(const Tree &t, MemoryManager &memory_manager) override;
    void** getPartitionHashTableDevicePointer() override;
    void** getPartitionHashTableHostPointer() override;
};

class LockBasedHashTableWrapper : public AggregationTableWrapper {
private:
    std::vector<uint64_t*> h_d_partition_H;
    uint64_t** d_d_partition_H = nullptr;
public:
    LockBasedHashTableWrapper(const Tree &t, MemoryManager &memory_manager, const DataGraph &dun);
    void allocatePartitionHashTable(const Tree& t, MemoryManager &memory_manager, uint64_t hashtableSize,
                                    std::vector<uint32_t>& nodesInPartition, uint32_t parititionRootNode) override;
    void clearPartitionHashTable(uint32_t nID) override;
    void releasePartitionHashTable(const Tree &t, MemoryManager &memory_manager) override;
    void release(const Tree &t, MemoryManager &memory_manager) override;
    void** getPartitionHashTableDevicePointer() override;
    void** getPartitionHashTableHostPointer() override;
};

class DenseArrayWrapper : public AggregationTableWrapper {
private:
    std::vector<uint64_t*> h_d_partition_H;
    uint64_t** d_d_partition_H = nullptr;
public:
    DenseArrayWrapper(const Tree &t, MemoryManager &memory_manager, const DataGraph &dun);
    void allocatePartitionHashTable(const Tree& t, MemoryManager &memory_manager, uint64_t hashtableSize,
                                    std::vector<uint32_t>& nodesInPartition, uint32_t parititionRootNode) override;
    void clearPartitionHashTable(uint32_t nID) override;
    void releasePartitionHashTable(const Tree &t, MemoryManager &memory_manager) override;
    void release(const Tree &t, MemoryManager &memory_manager) override;
    void** getPartitionHashTableDevicePointer() override;
    void** getPartitionHashTableHostPointer() override;
};