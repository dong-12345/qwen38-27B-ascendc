cmake_minimum_required(VERSION 3.16)
project(md3_310b_ops)

set(RUN_MODE "npu" CACHE STRING "cpu/sim/npu")
set(SOC_VERSION "Ascend310B1" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest"
    CACHE STRING "ASCEND CANN package installation directory")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type Release/Debug (default Release)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp)
message(STATUS "kernel files: ${KERNEL_FILES}")

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message(FATAL_ERROR "invalid RUN_MODE: ${RUN_MODE}")
endif()

set(MD3_HOST_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/host/test_ops.cpp
)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/host/engine_md3.cpp)
    list(APPEND MD3_HOST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/host/engine_md3.cpp)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/host/main.cpp)
    list(APPEND MD3_HOST_SRC ${CMAKE_CURRENT_SOURCE_DIR}/host/main.cpp)
endif()

add_executable(md3_engine ${MD3_HOST_SRC})
target_include_directories(md3_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/host
    ${CMAKE_INSTALL_PREFIX}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_compile_options(md3_engine PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall)
target_compile_definitions(md3_engine PRIVATE SOC_VERSION="${SOC_VERSION}")
target_link_libraries(md3_engine PRIVATE
    host_intf_pub
    ascendc_kernels_${RUN_MODE}
    tiling_api register platform ascendalog dl
    opapi nnopbase ascendcl
)
install(TARGETS md3_engine RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
