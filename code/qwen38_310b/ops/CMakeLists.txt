cmake_minimum_required(VERSION 3.16)
project(qwen38_310b_ops)

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

# 普通 kernel 与 matmul kernel 分两个库（原因见 cmake/npu_lib.cmake）
file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp)
file(GLOB MM_KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/mm/*.cpp)
message(STATUS "kernel files: ${KERNEL_FILES}")
message(STATUS "mm kernel files: ${MM_KERNEL_FILES}")

if("${RUN_MODE}" STREQUAL "cpu")
    include(cmake/cpu_lib.cmake)
elseif("${RUN_MODE}" STREQUAL "sim" OR "${RUN_MODE}" STREQUAL "npu")
    include(cmake/npu_lib.cmake)
else()
    message(FATAL_ERROR "invalid RUN_MODE: ${RUN_MODE}")
endif()

add_executable(qwen38_ops_test
    ${CMAKE_CURRENT_SOURCE_DIR}/host/main.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host/matmul_tiling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host/engine_mini.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/host/engine_27b.cpp
)
target_include_directories(qwen38_ops_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/host
    ${CMAKE_INSTALL_PREFIX}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
)
target_compile_options(qwen38_ops_test PRIVATE
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:-g>>
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)
target_compile_definitions(qwen38_ops_test PRIVATE
    SOC_VERSION="${SOC_VERSION}"
)
target_link_libraries(qwen38_ops_test PRIVATE
    $<BUILD_INTERFACE:$<$<OR:$<STREQUAL:${RUN_MODE},npu>,$<STREQUAL:${RUN_MODE},sim>>:host_intf_pub>>
    $<BUILD_INTERFACE:$<$<STREQUAL:${RUN_MODE},cpu>:ascendcl>>
    ascendc_kernels_${RUN_MODE}
    ascendc_kernels_mm_${RUN_MODE}
    tiling_api
    opapi
    nnopbase
    ascendcl
    register
    platform
    ascendalog
    dl
)

install(TARGETS qwen38_ops_test
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
