cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(gprt-h5m CXX)

# Set module path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules)

# Allow user to specify <project>_ROOT variables
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.12)
  cmake_policy(SET CMP0074 NEW)
endif()

# Enable correct usage of CXX_EXTENSIONS
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.22)
  cmake_policy(SET CMP0128 NEW)
endif()


add_executable(gprt-h5m src/main.cpp)
