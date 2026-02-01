#pragma once

#include <cstdint>
#include <iostream>
#include <optional>
#include <vector>
#include <algorithm>
#include <string>

#include "cuda_helpers.h"

struct MemoryBlock {
    void* pointer;          // pointer to the allocated device memory
    uint64_t start;             // start of the memory range
    uint64_t end;               // end of the memory range
    std::string name;           // purpose of the memory block
    MemoryBlock(void* ptr, uint64_t s, uint64_t e, std::string n) : pointer(ptr), start(s), end(e), name(n) {}
    MemoryBlock() : pointer(nullptr), start(0), end(0), name("") {}
};

// A stack-like memory allocator for GPU device memory.
// Allocations behave like a stack (LIFO) that memory allocations must be released in reverse order of allocation,
class MemoryManager {
private:
    bool pre_allocated_;
    void* pool_;                                                        // memory pool
    uint64_t capacity_;                                                 // total bytes of the memory pool
    std::vector<MemoryBlock> allocated_memory_;                   // allocated memory blocks from the start
    MemoryBlock available_memory_;

public:
    MemoryManager(uint64_t capacity);
    MemoryManager(void* pool, uint64_t capacity);
    ~MemoryManager();
    void printMemoryUsage();                                            // print memory usage
    uint64_t getTotalMemory();                                          // get total memory capacity in bytes
    uint64_t getAvailableMemory();                                      // get available memory in bytes

    // allocate size bytes from the available, the address should be divisible with alignment, e.g.: 4 for uint32_t or 8 for uint64_t
    void* allocate(uint64_t size, uint32_t alignment, std::string name="");

    // the released memory must be adjacent to the available memory. either released.end = available.start or released.start = available.end
    void release(void* ptr);
};