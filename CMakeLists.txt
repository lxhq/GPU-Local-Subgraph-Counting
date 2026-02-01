cmake_minimum_required(VERSION 3.19)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

project(scope LANGUAGES CUDA CXX)

include(cmake/CPM.cmake)

set(CMAKE_VERBOSE_MAKEFILE on)

set(CMAKE_CXX_STANDARD 17)
IF(NOT APPLE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
ENDIF()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
set(CMAKE_CUDA_STANDARD_REQUIRED True)


if(NOT DEFINED HASH_TABLE_TYPE)
    set(HASH_TABLE_TYPE 1)
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
    # set(CMAKE_BUILD_TYPE Debug)
endif()

CPMAddPackage(
  NAME helpers
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  URL https://gitlab.rlp.net/pararch/hpc_helpers/-/archive/restructure/hpc_helpers-restructure.zip
)

# lightweight GPU RNG
CPMAddPackage(
  NAME kiss_rng
  DOWNLOAD_EXTRACT_TIMESTAMP ON
  URL https://github.com/sleeepyjack/kiss_rng/archive/refs/heads/master.zip
)

# Pass to all subdirectories via global definition
add_compile_definitions(
  HASH_TABLE_TYPE=${HASH_TABLE_TYPE}
)

include_directories(
        ${PROJECT_SOURCE_DIR}/config
        ${PROJECT_SOURCE_DIR}/utility
        ${PROJECT_SOURCE_DIR}/utility/automorphism
        ${PROJECT_SOURCE_DIR}/graph
        ${PROJECT_SOURCE_DIR}/plan
        ${PROJECT_SOURCE_DIR}/counting
        ${PROJECT_SOURCE_DIR}/gpu
        ${PROJECT_SOURCE_DIR}/include
)

link_directories(
        ${PROJECT_SOURCE_DIR}/config
        ${PROJECT_SOURCE_DIR}/utility
        ${PROJECT_SOURCE_DIR}/utility/automorphism
        ${PROJECT_SOURCE_DIR}/graph
        ${PROJECT_SOURCE_DIR}/plan
        ${PROJECT_SOURCE_DIR}/counting
        ${PROJECT_SOURCE_DIR}/gpu
)

# link to GLPK. edit the paths based on your environment
IF(NOT APPLE)
    include_directories(/usr/local/include)
    link_directories(/usr/local/lib)
ELSE()
    include_directories(/opt/homebrew/Cellar/glpk/5.0/include)
    link_directories(/opt/homebrew/Cellar/glpk/5.0/lib)
ENDIF()

add_subdirectory(config)
add_subdirectory(graph)
add_subdirectory(utility)
add_subdirectory(executable)
add_subdirectory(plan)
add_subdirectory(counting)
add_subdirectory(gpu)