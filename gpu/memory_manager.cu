# include "memory_manager.h"

MemoryManager::MemoryManager(uint64_t capacity):capacity_(capacity), pool_(nullptr) {
    cudaErrorCheck(cudaMalloc(&pool_, capacity_));
    available_memory_.pointer = pool_;
    available_memory_.start = 0;
    available_memory_.end = capacity_;
    available_memory_.name = "free memory";
    pre_allocated_ = false;
}

MemoryManager::MemoryManager(void* pool, uint64_t capacity):capacity_(capacity), pool_(pool) {
    available_memory_.pointer = pool_;
    available_memory_.start = 0;
    available_memory_.end = capacity_;
    available_memory_.name = "free memory";
    pre_allocated_ = true;
}

MemoryManager::~MemoryManager() {
    if (!pre_allocated_ && pool_ != nullptr) {
        cudaErrorCheck(cudaFree(pool_));
    }
}

uint64_t MemoryManager::getTotalMemory() {
    return capacity_;
}

uint64_t MemoryManager::getAvailableMemory() {
    return (available_memory_.end - available_memory_.start);
}

void MemoryManager::printMemoryUsage() {
    std::cout << "Allocated Memory is(are):" << std::endl;
    for (auto it = allocated_memory_.rbegin(); it != allocated_memory_.rend(); ++it) {
        std::cout << it->name << ": [" << it->start << ", " << it->end << ")" << std::endl;
    }
    std::cout << std::endl;
    std::cout << "Available Memory is(are): " << std::endl;
    std::cout << available_memory_.name << ": [" << available_memory_.start << ", " << available_memory_.end << ")" << std::endl;
    std::cout << std::endl;
}

void* MemoryManager::allocate(uint64_t size, uint32_t alignment, std::string name) {
    uint64_t start = available_memory_.start;
    while (reinterpret_cast<uintptr_t>(static_cast<char*>(pool_) + start) % alignment != 0) {
        start++;
    }
    if (start + size > available_memory_.end) {
        throw std::runtime_error("not enough memory to allocate " + name);
    }
    allocated_memory_.emplace_back(static_cast<char*>(pool_) + start, available_memory_.start, start + size, name);
    available_memory_.start = start + size;
    return static_cast<char*>(pool_) + start;
}

void MemoryManager::release(void* ptr) {
    // throw error if no allocated memory
    if (allocated_memory_.empty()) {
        throw std::runtime_error("no allocated memory to release");
    }

    // check if the ptr is from start allocations
    if (!allocated_memory_.empty() && allocated_memory_.back().pointer == ptr) {
        available_memory_.start = allocated_memory_.back().start;
        allocated_memory_.pop_back();
        return;
    }

    printMemoryUsage();
    // find the name of the memory block
    std::string name = "";
    for (const auto& block : allocated_memory_) {
        if (block.pointer == ptr) {
            name = block.name;
            throw std::runtime_error("memory block " + name + " must be released in reverse order of allocation (LIFO) from start");
        }
    }

    throw std::runtime_error("pointer not found in allocated memory blocks");
}