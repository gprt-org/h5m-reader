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

add_subdirectory(3rdParty/argparse)
add_subdirectory(3rdParty/gprt)

find_package(DAGMC REQUIRED PATH_SUFFIXES lib lib/cmake)

embed_devicecode(
  OUTPUT_TARGET
    dbl_deviceCode
  SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/deviceCode.hlsl
)

add_executable(gprt-h5m src/main.cpp src/MOABDirectAccess.cpp)
target_compile_features(gprt-h5m PUBLIC cxx_std_14)
target_include_directories(gprt-h5m PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gprt-h5m PUBLIC dbl_deviceCode argparse dagmc-shared gprt::gprt glfw ${OPENGL_gl_LIBRARY})
