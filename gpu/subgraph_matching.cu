#include "subgraph_matching.h"


struct CastToUInt64 {
    __host__ __device__
    uint64_t operator()(const uint32_t &x) const {
        return static_cast<uint64_t>(x);
    }
};

SubgraphMatching::SubgraphMatching(MemoryManager& memory_manager, uint32_t *source, uint32_t source_embedding_count_,
                    NodeGPU* d_node, uint64_t target_memory_size, uint32_t level,
                    bool is_last_level, uint32_t index_len)
                    : memory_manager_(memory_manager),
                      source_(source),
                      source_embedding_count_(source_embedding_count_),
                      d_node_(d_node),
                      target_memory_size_(target_memory_size),
                      level_(level),
                      is_last_level_(is_last_level),
                      index_len_(index_len),
                      target_(nullptr),
                      exclusive_sum_(nullptr),
                      next_start_index_(0),
                      next_level_total_embedding_count_(0),
                      next_level_max_embedding_count_(0) {
    if (is_last_level_) {
        // next_level_max_embedding_count_ * (index_len_ + level + 1) * size_of(uint32_t) <= target_memory_size_
        // => next_level_max_embedding_count_ <= target_memory_size_ / (sizeof(uint32_t) * (index_len_ + level_ + 1))
        next_level_max_embedding_count_ = static_cast<uint32_t>(
            target_memory_size_ / (sizeof(uint32_t) * (index_len_ + level_ + 1))
        );
    } else {
        // next_level_max_embedding_count_ * (index_len_ + level + 1) * size_of(uint32_t) + (next_level_max_embedding_count_ + 1) * size_of(uint64_t) <= target_memory_size_
        // => next_level_max_embedding_count_ <= (target_memory_size_ - sizeof(uint64_t)) / (sizeof(uint32_t) * (index_len_ + level_ + 1) + sizeof(uint64_t))
        next_level_max_embedding_count_ = static_cast<uint32_t>(
            (target_memory_size_ - sizeof(uint64_t)) / (sizeof(uint32_t) * (index_len_ + level_ + 1) + sizeof(uint64_t))
        );
    }
    if (level != 0 && source_embedding_count_ > 0) {
        exclusive_sum_ = static_cast<uint64_t*>(memory_manager_.allocate(
            (source_embedding_count_ + 1) * sizeof(uint64_t), sizeof(uint64_t), "exclusive sum"));
        uint32_t* matching_counts = static_cast<uint32_t*>(memory_manager_.allocate(
            source_embedding_count_ * sizeof(uint32_t), sizeof(uint32_t), "matching counts"));
        countKernel<<<SDIV(source_embedding_count_, WARP_PER_BLOCK), BLOCK_DIM>>>(
            index_len_,
            level,
            source_,
            source_embedding_count_,
            matching_counts,
            d_node_);
        cudaErrorCheck(cudaDeviceSynchronize());
        cub::TransformInputIterator<uint64_t, CastToUInt64, uint32_t*> transform_iter(matching_counts, CastToUInt64());
        uint64_t temp_storage_size = 0;
        cudaErrorCheck(cub::DeviceScan::ExclusiveSum(
            nullptr, temp_storage_size,
            transform_iter,
            exclusive_sum_,
            source_embedding_count_ + 1));
        void* d_temp_storage = memory_manager_.allocate(
            temp_storage_size, sizeof(uint8_t), "temp storage for exclusive sum");
        cudaErrorCheck(cub::DeviceScan::ExclusiveSum(
            d_temp_storage, temp_storage_size,
            transform_iter,
            exclusive_sum_,
            source_embedding_count_ + 1));
        memory_manager_.release(d_temp_storage);
        memory_manager_.release(matching_counts);
        cudaErrorCheck(cudaMemcpy(&next_level_total_embedding_count_, &exclusive_sum_[source_embedding_count_], sizeof(uint64_t), cudaMemcpyDeviceToHost));
    } else {
        next_level_total_embedding_count_ = source_embedding_count_;
    }
    target_ = static_cast<uint32_t*>(memory_manager_.allocate(
            next_level_max_embedding_count_ * (index_len_ + level_ + 1) * sizeof(uint32_t), sizeof(uint32_t), "target embeddings"));
}

SubgraphMatching::~SubgraphMatching() {
    if (target_ != nullptr) {
        memory_manager_.release(target_);
    }
    if (exclusive_sum_ != nullptr) {
        memory_manager_.release(exclusive_sum_);
    }
}

uint32_t* SubgraphMatching::next(uint32_t& count) {
    if (level_ != 0) {
        uint32_t* actual_start_index;
        cudaErrorCheck(cudaMallocManaged(&actual_start_index, sizeof(uint32_t)));
        *actual_start_index = UINT32_MAX;
        writeKernel<<<SDIV(source_embedding_count_ - next_start_index_, WARP_PER_BLOCK), BLOCK_DIM>>>(
            index_len_,
            level_,
            source_ + next_start_index_ * (index_len_ + level_),
            source_embedding_count_ - next_start_index_,
            exclusive_sum_ + next_start_index_, 
            target_,
            next_level_max_embedding_count_,
            actual_start_index,
            d_node_);
        cudaErrorCheck(cudaDeviceSynchronize());
        uint64_t old_count, cur_count = 0;
        cudaErrorCheck(cudaMemcpy(&old_count, exclusive_sum_ + next_start_index_, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        if (*actual_start_index == UINT32_MAX) {
            // all embeddings are written
            cur_count = next_level_total_embedding_count_;
            next_start_index_ = source_embedding_count_;
        } else {
            next_start_index_ += *actual_start_index;
            cudaErrorCheck(cudaMemcpy(&cur_count, exclusive_sum_ + next_start_index_, sizeof(uint64_t), cudaMemcpyDeviceToHost));
        }
        cudaErrorCheck(cudaFree(actual_start_index));
        count = static_cast<uint32_t>(cur_count - old_count);
    } else {
        count = std::min(source_embedding_count_ - next_start_index_, next_level_max_embedding_count_);
        firstLevelKernel<<<SDIV(count, BLOCK_DIM), BLOCK_DIM>>>(
            index_len_,
            next_start_index_,
            count,
            target_);
        cudaErrorCheck(cudaDeviceSynchronize());
        next_start_index_ += count;
    }
    return target_;
}