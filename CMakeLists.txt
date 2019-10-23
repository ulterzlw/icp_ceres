cmake_minimum_required(VERSION 2.8)
project(icp_ceres)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

add_executable(a main.cc main.cc)
target_link_libraries(a
  ${Sophus_LIBRARIES}
  ${CERES_LIBRARIES}
  )