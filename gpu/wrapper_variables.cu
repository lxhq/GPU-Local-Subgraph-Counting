#include "wrapper_variables.h"

namespace {
bool g_gpu_reset_profile_enabled = false;
GpuResetProfileStats g_gpu_reset_profile_stats;

bool isPrimeBucketCount(uint64_t value) {
    if (value < 2) {
        return false;
    }
    if (value == 2 || value == 3) {
        return true;
    }
    if (value % 2 == 0 || value % 3 == 0) {
        return false;
    }
    for (uint64_t divisor = 5; divisor <= value / divisor; divisor += 6) {
        if (value % divisor == 0 || value % (divisor + 2) == 0) {
            return false;
        }
    }
    return true;
}

uint64_t largestPrimeBucketCount(uint64_t value) {
    if (value <= 3) {
        return value;
    }
    if (value % 2 == 0) {
        --value;
    }
    while (value > 3 && !isPrimeBucketCount(value)) {
        value -= 2;
    }
    return value;
}
}

void setGpuResetProfileEnabled(bool enabled) {
    g_gpu_reset_profile_enabled = enabled;
}

bool getGpuResetProfileEnabled() {
    return g_gpu_reset_profile_enabled;
}

void resetGpuResetProfile() {
    g_gpu_reset_profile_stats = GpuResetProfileStats();
}

GpuResetProfileStats getGpuResetProfileStats() {
    return g_gpu_reset_profile_stats;
}

void gpuProfiledMemset(void *ptr, int value, size_t bytes, GpuResetKind kind) {
    if (!g_gpu_reset_profile_enabled) {
        cudaErrorCheck(cudaMemset(ptr, value, bytes));
        return;
    }

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaErrorCheck(cudaEventCreate(&start));
    cudaErrorCheck(cudaEventCreate(&stop));
    cudaErrorCheck(cudaEventRecord(start));
    cudaErrorCheck(cudaMemset(ptr, value, bytes));
    cudaErrorCheck(cudaEventRecord(stop));
    cudaErrorCheck(cudaEventSynchronize(stop));
    float elapsedMs = 0.0f;
    cudaErrorCheck(cudaEventElapsedTime(&elapsedMs, start, stop));
    cudaErrorCheck(cudaEventDestroy(start));
    cudaErrorCheck(cudaEventDestroy(stop));

    const double elapsedSeconds = elapsedMs / 1000.0;
    if (kind == GpuResetKind::InitialZero) {
        g_gpu_reset_profile_stats.initialZeroTime += elapsedSeconds;
        ++g_gpu_reset_profile_stats.initialZeroCalls;
        g_gpu_reset_profile_stats.initialZeroBytes += bytes;
    } else {
        g_gpu_reset_profile_stats.repeatedClearTime += elapsedSeconds;
        ++g_gpu_reset_profile_stats.repeatedClearCalls;
        g_gpu_reset_profile_stats.repeatedClearBytes += bytes;
    }
}

// --------------------- warpcore wrapper --------------------- //
WarpcoreWrapper::WarpcoreWrapper(const Tree &t, MemoryManager &memory_manager, const DataGraph &dun) : AggregationTableWrapper(t, memory_manager, dun) {
    this->d_d_partition_H = static_cast<warpcore::CountingHashTable<>**>(
        memory_manager.allocate(t.getNumNodes() * sizeof(warpcore::CountingHashTable<>*), sizeof(warpcore::CountingHashTable<>*), "d_d_partition_H")
    );
    cudaErrorCheck(cudaMemset(this->d_d_partition_H, 0, t.getNumNodes() * sizeof(warpcore::CountingHashTable<>*)));
    this->h_d_partition_H.resize(t.getNumNodes());
    this->partition_H_ptrs.resize(t.getNumNodes());
    std::fill(this->h_d_partition_H.begin(), this->h_d_partition_H.end(), nullptr);
}

void WarpcoreWrapper::allocatePartitionHashTable(const Tree& t, MemoryManager &memory_manager,
                                                    uint64_t hashtableSize, std::vector<uint32_t>& nodesInPartition,
                                                    uint32_t parititionRootNode) {
    this->partitionHashtableBuckets = hashtableSize / sizeof(warpcore::storage::key_value::detail::pair_t<uint64_t, uint64_t>);
    this->partitionHashtableBuckets = warpcore::detail::get_valid_max_capacity(this->partitionHashtableBuckets,warpcore::CountingHashTable<>::cg_size());
    this->partitionHashtableSize = this->partitionHashtableBuckets * sizeof(warpcore::storage::key_value::detail::pair_t<uint64_t, uint64_t>);
    for (int nID = 0; nID < t.getNumNodes(); nID++) {
        if (nID == parititionRootNode ||
            std::find(nodesInPartition.begin(), nodesInPartition.end(), nID) == nodesInPartition.end()) {
            continue;
        }
        void* table_ptr = memory_manager.allocate(hashtableSize, sizeof(uint64_t), "hash table");
        void* status_ptr = memory_manager.allocate(sizeof(warpcore::Status), sizeof(uint32_t), "hash table status");
        void* tmp_ptr = memory_manager.allocate(warpcore::defaults::temp_memory_bytes(), sizeof(uint64_t), "hash table tmp");
        warpcore::CountingHashTable<> hash_table(partitionHashtableBuckets, table_ptr, status_ptr, tmp_ptr);
        void* d_hash_table = memory_manager.allocate(sizeof(warpcore::CountingHashTable<>), sizeof(uint32_t), "d_hash_table " + std::to_string(nID));
        cudaErrorCheck(cudaMemcpy(d_hash_table, &hash_table, sizeof(warpcore::CountingHashTable<>), cudaMemcpyHostToDevice));
        cudaErrorCheck(cudaMemcpy(this->d_d_partition_H + nID, &d_hash_table, sizeof(void*), cudaMemcpyHostToDevice));
        this->h_d_partition_H[nID] = static_cast<warpcore::CountingHashTable<>*>(d_hash_table);
        this->partition_H_ptrs[nID] = std::make_pair(table_ptr, std::make_pair(status_ptr, tmp_ptr));
    }
}

void WarpcoreWrapper::clearPartitionHashTable(uint32_t nID) {
    if (h_d_partition_H[nID] != nullptr) {
        gpuProfiledMemset(partition_H_ptrs[nID].first, 0, this->partitionHashtableSize, GpuResetKind::RepeatedClear);
    }
}

void WarpcoreWrapper::releasePartitionHashTable(const Tree &t, MemoryManager &memory_manager) {
    for (int nID = t.getNumNodes() - 1; nID >= 0; nID--) {
        if (h_d_partition_H[nID] != nullptr) {
            memory_manager.release(this->h_d_partition_H[nID]);
            memory_manager.release(this->partition_H_ptrs[nID].second.second);
            memory_manager.release(this->partition_H_ptrs[nID].second.first);
            memory_manager.release(this->partition_H_ptrs[nID].first);

        }
        this->h_d_partition_H[nID] = nullptr;
    }
    cudaErrorCheck(cudaMemset(this->d_d_partition_H, 0, t.getNumNodes() * sizeof(warpcore::CountingHashTable<>*)));
}

void WarpcoreWrapper::release(const Tree &t, MemoryManager &memory_manager) {
    memory_manager.release(this->d_d_partition_H);
    for (int nID = t.getNumNodes() - 1; nID >= 0; nID--) {
        memory_manager.release(this->h_d_H[nID]);
    }
    memory_manager.release(this->d_d_H);
}

void** WarpcoreWrapper::getPartitionHashTableDevicePointer() {
    return reinterpret_cast<void**>(this->d_d_partition_H);
}
void** WarpcoreWrapper::getPartitionHashTableHostPointer() {
    return reinterpret_cast<void**>(this->h_d_partition_H.data());
}
// --------------------- lock free hash table wrapper --------------------- //
LockFreeHashTableWrapper::LockFreeHashTableWrapper(const Tree &t, MemoryManager &memory_manager, const DataGraph &dun) : AggregationTableWrapper(t, memory_manager, dun) {
    this->d_d_partition_H = static_cast<uint64_t**>(
        memory_manager.allocate(t.getNumNodes() * sizeof(uint64_t*), sizeof(uint64_t*), "d_d_partition_H")
    );
    cudaErrorCheck(cudaMemset(this->d_d_partition_H, 0, t.getNumNodes() * sizeof(uint64_t**)));
    this->h_d_partition_H.resize(t.getNumNodes());
    std::fill(this->h_d_partition_H.begin(), this->h_d_partition_H.end(), nullptr);
}

void LockFreeHashTableWrapper::allocatePartitionHashTable(const Tree& t, MemoryManager &memory_manager,
                                                    uint64_t hashtableSize, std::vector<uint32_t>& nodesInPartition,
                                                    uint32_t parititionRootNode) {
    this->partitionHashtableBuckets = hashtableSize / (2 * sizeof(uint64_t));
    this->partitionHashtableBuckets = largestPrimeBucketCount(this->partitionHashtableBuckets);
    this->partitionHashtableSize = this->partitionHashtableBuckets * (2 * sizeof(uint64_t));
    for (int nID = 0; nID < t.getNumNodes(); nID++) {
        if (nID == parititionRootNode ||
            std::find(nodesInPartition.begin(), nodesInPartition.end(), nID) == nodesInPartition.end()) {
            continue;
        }
        void* d_partition_H = memory_manager.allocate(this->partitionHashtableSize, sizeof(uint64_t), "d_partition_H " + std::to_string(nID));
        gpuProfiledMemset(d_partition_H, 0, this->partitionHashtableSize, GpuResetKind::InitialZero);
        cudaErrorCheck(cudaMemcpy(this->d_d_partition_H + nID, &d_partition_H, sizeof(void*), cudaMemcpyHostToDevice));
        this->h_d_partition_H[nID] = static_cast<uint64_t*>(d_partition_H);
    }
}

void LockFreeHashTableWrapper::clearPartitionHashTable(uint32_t nID) {
    if (h_d_partition_H[nID] != nullptr) {
        gpuProfiledMemset(h_d_partition_H[nID], 0, this->partitionHashtableSize, GpuResetKind::RepeatedClear);
    }
}

void LockFreeHashTableWrapper::releasePartitionHashTable(const Tree &t, MemoryManager &memory_manager) {
    for (int nID = t.getNumNodes() - 1; nID >= 0; nID--) {
        if (h_d_partition_H[nID] != nullptr) {
            memory_manager.release(this->h_d_partition_H[nID]);
        }
        this->h_d_partition_H[nID] = nullptr;
    }
    cudaErrorCheck(cudaMemset(this->d_d_partition_H, 0, t.getNumNodes() * sizeof(uint64_t*)));
}

void LockFreeHashTableWrapper::release(const Tree &t, MemoryManager &memory_manager) {
    memory_manager.release(this->d_d_partition_H);
    for (int nID = t.getNumNodes() - 1; nID >= 0; nID--) {
        memory_manager.release(this->h_d_H[nID]);
    }
    memory_manager.release(this->d_d_H);
}

void** LockFreeHashTableWrapper::getPartitionHashTableDevicePointer() {
    return reinterpret_cast<void**>(this->d_d_partition_H);
}

void** LockFreeHashTableWrapper::getPartitionHashTableHostPointer() {
    return reinterpret_cast<void**>(this->h_d_partition_H.data());
}

// --------------------- lock based hash table wrapper --------------------- //
LockBasedHashTableWrapper::LockBasedHashTableWrapper(const Tree &t, MemoryManager &memory_manager, const DataGraph &dun) : AggregationTableWrapper(t, memory_manager, dun) {
    this->d_d_partition_H = static_cast<uint64_t**>(
        memory_manager.allocate(t.getNumNodes() * sizeof(uint64_t*), sizeof(uint64_t*), "d_d_partition_H")
    );
    cudaErrorCheck(cudaMemset(this->d_d_partition_H, 0, t.getNumNodes() * sizeof(uint64_t**)));
    this->h_d_partition_H.resize(t.getNumNodes());
    std::fill(this->h_d_partition_H.begin(), this->h_d_partition_H.end(), nullptr);
}

void LockBasedHashTableWrapper::allocatePartitionHashTable(const Tree& t, MemoryManager &memory_manager,
                                                    uint64_t hashtableSize, std::vector<uint32_t>& nodesInPartition,
                                                    uint32_t parititionRootNode) {
    this->partitionHashtableBuckets = hashtableSize / (6 * sizeof(uint64_t));
    this->partitionHashtableBuckets = largestPrimeBucketCount(this->partitionHashtableBuckets);
    this->partitionHashtableSize = this->partitionHashtableBuckets * (6 * sizeof(uint64_t));
    for (int nID = 0; nID < t.getNumNodes(); nID++) {
        if (nID == parititionRootNode ||
            std::find(nodesInPartition.begin(), nodesInPartition.end(), nID) == nodesInPartition.end()) {
            continue;
        }
        void* d_partition_H = memory_manager.allocate(this->partitionHashtableSize, sizeof(uint64_t), "d_partition_H " + std::to_string(nID));
        gpuProfiledMemset(d_partition_H, 0, this->partitionHashtableSize, GpuResetKind::InitialZero);
        cudaErrorCheck(cudaMemcpy(this->d_d_partition_H + nID, &d_partition_H, sizeof(void*), cudaMemcpyHostToDevice));
        this->h_d_partition_H[nID] = static_cast<uint64_t*>(d_partition_H);
    }
}

void LockBasedHashTableWrapper::clearPartitionHashTable(uint32_t nID) {
    if (h_d_partition_H[nID] != nullptr) {
        gpuProfiledMemset(h_d_partition_H[nID], 0, this->partitionHashtableSize, GpuResetKind::RepeatedClear);
    }
}
    
void LockBasedHashTableWrapper::releasePartitionHashTable(const Tree &t, MemoryManager &memory_manager) {
    for (int nID = t.getNumNodes() - 1; nID >= 0; nID--) {
        if (h_d_partition_H[nID] != nullptr) {
            memory_manager.release(this->h_d_partition_H[nID]);
        }
        this->h_d_partition_H[nID] = nullptr;
    }
    cudaErrorCheck(cudaMemset(this->d_d_partition_H, 0, t.getNumNodes() * sizeof(uint64_t*)));
}

void LockBasedHashTableWrapper::release(const Tree &t, MemoryManager &memory_manager) {
    memory_manager.release(this->d_d_partition_H);
    for (int nID = t.getNumNodes() - 1; nID >= 0; nID--) {
        memory_manager.release(this->h_d_H[nID]);
    }
    memory_manager.release(this->d_d_H);
}

void** LockBasedHashTableWrapper::getPartitionHashTableDevicePointer() {
    return reinterpret_cast<void**>(this->d_d_partition_H);
}
void** LockBasedHashTableWrapper::getPartitionHashTableHostPointer() {
    return reinterpret_cast<void**>(this->h_d_partition_H.data());
}

// ------------------------- dense array wrapper ------------------------- //
DenseArrayWrapper::DenseArrayWrapper(const Tree &t, MemoryManager &memory_manager, const DataGraph &dun) : AggregationTableWrapper(t, memory_manager, dun) {
    this->d_d_partition_H = static_cast<uint64_t**>(
        memory_manager.allocate(t.getNumNodes() * sizeof(uint64_t*), sizeof(uint64_t*), "d_d_partition_H")
    );
    cudaErrorCheck(cudaMemset(this->d_d_partition_H, 0, t.getNumNodes() * sizeof(uint64_t**)));
    this->h_d_partition_H.resize(t.getNumNodes());
    std::fill(this->h_d_partition_H.begin(), this->h_d_partition_H.end(), nullptr);
}

void DenseArrayWrapper::allocatePartitionHashTable(const Tree& t, MemoryManager &memory_manager,
                                                    uint64_t hashtableSize, std::vector<uint32_t>& nodesInPartition,
                                                    uint32_t parititionRootNode) {
    this->partitionHashtableBuckets = hashtableSize / sizeof(uint64_t);                // this item is useless in this class
    this->partitionHashtableSize = this->partitionHashtableBuckets * sizeof(uint64_t);
    for (int nID = 0; nID < t.getNumNodes(); nID++) {
        if (nID == parititionRootNode ||
            std::find(nodesInPartition.begin(), nodesInPartition.end(), nID) == nodesInPartition.end()) {
            continue;
        }
        void* d_partition_H = memory_manager.allocate(hashtableSize, sizeof(uint64_t), "d_partition_H " + std::to_string(nID));
        gpuProfiledMemset(d_partition_H, 0, hashtableSize, GpuResetKind::InitialZero);
        cudaErrorCheck(cudaMemcpy(this->d_d_partition_H + nID, &d_partition_H, sizeof(void*), cudaMemcpyHostToDevice));
        this->h_d_partition_H[nID] = static_cast<uint64_t*>(d_partition_H);
    }
}

void DenseArrayWrapper::clearPartitionHashTable(uint32_t nID) {
    if (h_d_partition_H[nID] != nullptr) {
        gpuProfiledMemset(h_d_partition_H[nID], 0, this->partitionHashtableSize, GpuResetKind::RepeatedClear);
    }
}

void DenseArrayWrapper::releasePartitionHashTable(const Tree &t, MemoryManager &memory_manager) {
    for (int nID = t.getNumNodes() - 1; nID >= 0; nID--) {
        if (h_d_partition_H[nID] != nullptr) {
            memory_manager.release(this->h_d_partition_H[nID]);
        }
        this->h_d_partition_H[nID] = nullptr;
    }
    cudaErrorCheck(cudaMemset(this->d_d_partition_H, 0, t.getNumNodes() * sizeof(uint64_t*)));
}

void DenseArrayWrapper::release(const Tree &t, MemoryManager &memory_manager) {
    memory_manager.release(this->d_d_partition_H);
    for (int nID = t.getNumNodes() - 1; nID >= 0; nID--) {
        memory_manager.release(this->h_d_H[nID]);
    }
    memory_manager.release(this->d_d_H);
}

void** DenseArrayWrapper::getPartitionHashTableDevicePointer() {
    return reinterpret_cast<void**>(this->d_d_partition_H);
}

void** DenseArrayWrapper::getPartitionHashTableHostPointer() {
    return reinterpret_cast<void**>(this->h_d_partition_H.data());
}
