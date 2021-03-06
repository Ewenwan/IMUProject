cmake_minimum_required(VERSION 3.0)

list(APPEND CMAKE_PREFIX_PATH /opt/Qt/5.8/gcc_64 /usr/local/Cellar/qt5/5.9.3/)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

find_package(OpenCV REQUIRED)
find_package(Gflags REQUIRED)
find_package(Glog REQUIRED)
find_package(Eigen REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Core Widgets Gui)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
include_directories(${EIGEN_INCLUDE_DIRS})

add_executable(VisualizeTrajectory viewer_cli.cc
        main_widget.cpp main_widget.h
        renderable.h renderable.cpp
        navigation.h navigation.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/resource/resource.qrc)

target_link_libraries(VisualizeTrajectory IMUUtility ${OpenCV_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY}
        Qt5::Core Qt5::Widgets Qt5::Gui)
