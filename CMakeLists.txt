cmake_minimum_required(VERSION 3.8)
project(cibo)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED)

# Note: MediaPipe C++ API is not readily available in standard ROS2 repositories
# The C++ nodes are placeholders that maintain the ROS2 interface but without
# the MediaPipe landmark detection. To use MediaPipe, consider:
# 1. Using the Python nodes via rclpy
# 2. Integrating MediaPipe via external C++ bindings
# 3. Using alternative pose detection libraries

# Find MediaPipe (if available as CMake package)
# For now, we'll try to use pkg-config or system libraries
# MediaPipe is primarily a Python/Android library, so C++ usage may be limited
# For this conversion, we'll focus on the core ROS2 functionality without MediaPipe
# and will replicate the image processing using OpenCV instead

include_directories(
  ${OpenCV_INCLUDE_DIRS}
)

# Build executables
set(dependencies
  rclcpp
  sensor_msgs
  std_msgs
  geometry_msgs
  cv_bridge
  image_transport
  message_filters
  tf2_ros
)

# top_camera_node
add_executable(top_camera_node src/top_camera.cpp)
ament_target_dependencies(top_camera_node ${dependencies} OpenCV)
target_link_libraries(top_camera_node ${OpenCV_LIBRARIES})

# front_camera_node
add_executable(front_camera_node src/front_camera.cpp)
ament_target_dependencies(front_camera_node ${dependencies} OpenCV)
target_link_libraries(front_camera_node ${OpenCV_LIBRARIES})

# top_camera_depth_node
add_executable(top_camera_depth_node src/top_camera_depth.cpp)
ament_target_dependencies(top_camera_depth_node ${dependencies} OpenCV)
target_link_libraries(top_camera_depth_node ${OpenCV_LIBRARIES})

# front_camera_depth_node
add_executable(front_camera_depth_node src/front_camera_depth.cpp)
ament_target_dependencies(front_camera_depth_node ${dependencies} OpenCV)
target_link_libraries(front_camera_depth_node ${OpenCV_LIBRARIES})

# image_show_node
add_executable(image_show_node src/image_show.cpp)
ament_target_dependencies(image_show_node ${dependencies} OpenCV)
target_link_libraries(image_show_node ${OpenCV_LIBRARIES})

# check_image_compressed_node
add_executable(check_image_compressed_node src/check_image_compressed.cpp)
ament_target_dependencies(check_image_compressed_node ${dependencies} OpenCV)
target_link_libraries(check_image_compressed_node ${OpenCV_LIBRARIES})

# dual_camera_sync_node
add_executable(dual_camera_sync_node src/dual_camera_sync.cpp)
ament_target_dependencies(dual_camera_sync_node ${dependencies} OpenCV)
target_link_libraries(dual_camera_sync_node ${OpenCV_LIBRARIES})

# Install executables
install(TARGETS
  top_camera_node
  front_camera_node
  top_camera_depth_node
  front_camera_depth_node
  image_show_node
  check_image_compressed_node
  dual_camera_sync_node
  DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME})

# Install rviz configs
install(DIRECTORY rviz
  DESTINATION share/${PROJECT_NAME})

# Install urdf files
install(DIRECTORY urdf
  DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
