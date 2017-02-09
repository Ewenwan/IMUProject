cmake_minimum_required(VERSION 3.0)

project(IMUOptimization)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package(OpenCV REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen REQUIRED)
find_package(Glog REQUIRED)
find_package(Gflags REQUIRED)

add_executable(IMUOptimization main.cpp imu_optimization.cpp imu_optimization.h)

target_link_libraries(IMUOptimization ${OpenCV_LIBS} ${Ceres_LIBRARIES}
        ${GLOG_LIBRARIES} ${GFLAGS_LIBRARIES} ${OPENMESH_LIBRARIES})

target_link_libraries(IMUOptimization IMUVisualization)