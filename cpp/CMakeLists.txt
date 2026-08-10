cmake_minimum_required(VERSION 3.16)

project(rgbd_mapping_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(
    semantic_voxel_map
    src/semantic_voxel_map.cpp
)

target_include_directories(
    semantic_voxel_map
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_executable(
    test_semantic_voxel_map
    tests/test_semantic_voxel_map.cpp
)

target_link_libraries(
    test_semantic_voxel_map
    PRIVATE
    semantic_voxel_map
)