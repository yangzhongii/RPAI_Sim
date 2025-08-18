cmake_minimum_required(VERSION 3.0.2)
project(pa_simulator)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  geometry_msgs
  sensor_msgs
  std_msgs
  visualization_msgs
  tf2_ros
  cv_bridge
)

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

find_package(OpenCV REQUIRED)

find_package(OpenMP)
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

catkin_package()

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${VTK_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(pa_simulator src/simulator.cpp)

target_link_libraries(pa_simulator
  ${catkin_LIBRARIES}
  ${VTK_LIBRARIES}
  ${OpenCV_LIBRARIES}
)