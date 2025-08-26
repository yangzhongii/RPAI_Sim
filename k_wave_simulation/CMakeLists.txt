cmake_minimum_required(VERSION 3.10)
project(kwave_simulation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  geometry_msgs
  cv_bridge
  std_msgs
)
find_package(OpenCV REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(OpenMP REQUIRED)

# Include directories
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${HDF5_INCLUDE_DIRS}
)

# Catkin package
catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs geometry_msgs cv_bridge std_msgs
)

# Add executable
add_executable(kwave_simulation src/kwave_ros_simulator_2d.cpp)

# Link libraries
target_link_libraries(kwave_simulation
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${HDF5_LIBRARIES}
  OpenMP::OpenMP_CXX
)

catkin_install_python(PROGRAMS
  scripts/kwave_ros_simulator_2d.py
  scripts/kwave_ros_simulator_3d.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
) 