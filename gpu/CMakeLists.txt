add_library(gpu-lsc
    globals.cu
    memory_manager.cu
    execution_gpu.cu
    counting_kernels.cu
    lock_based_composite_hashtable.cu
    lock_free_hashtable.cu
    wrapper_variables.cu
    subgraph_matching.cu
)

target_link_libraries(gpu-lsc PUBLIC graph plan counting helpers kiss_rng)
set_target_properties(gpu-lsc PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Add compile options only for CUDA files
target_compile_options(gpu-lsc PUBLIC
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:$<$<CONFIG:Debug>:-G>>
    $<$<COMPILE_LANGUAGE:CUDA>:-extended-lambda>  # Add this flag to support CUDA lambdas
)
