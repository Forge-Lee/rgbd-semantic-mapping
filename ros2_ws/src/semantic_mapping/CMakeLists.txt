cmake_minimum_required(VERSION 3.8)
project(semantic_mapping)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

set(
  RGBD_CORE_DIR
  "${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp"
)

add_subdirectory(
  "${RGBD_CORE_DIR}"
  "${CMAKE_CURRENT_BINARY_DIR}/rgbd_core_build"
)

add_executable(
  semantic_mapping_node
  src/semantic_mapping_node.cpp
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()


ament_target_dependencies(
  semantic_mapping_node
  rclcpp
  sensor_msgs
  geometry_msgs
  message_filters
  cv_bridge
  tf2
  tf2_geometry_msgs
)

target_link_libraries(
  semantic_mapping_node
  semantic_voxel_map
)

install(
  TARGETS semantic_mapping_node
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
