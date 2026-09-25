cmake_minimum_required(VERSION 3.8)
project(cibo_ros2)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME} PACKAGE_DIR src/${PROJECT_NAME})

# Install Python executables
install(
  PROGRAMS
    scripts/check_image_compressed.py
    scripts/dual_camera_sync.py
    scripts/front_camera_depth.py
    scripts/front_camera.py
    scripts/image_show.py
    scripts/top_camera_depth.py
    scripts/top_camera.py
    scripts/food_detection_and_access_tracking.py
    scripts/htm_chewing.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch rviz urdf models
  DESTINATION share/${PROJECT_NAME}
)

install(FILES
  pyproject.toml
  DESTINATION share/${PROJECT_NAME}
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

ament_package()