if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist ,please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# 普通 kernel（无 workspace/tiling 约定）与 matmul kernel（HAVE_WORKSPACE/HAVE_TILING）
# 必须分成两个 library：HAVE_WORKSPACE 会让所有 kernel 的启动包装假定最后两个参数是
# workspace / tiling，而普通 kernel 的最后参数是标量。
if(NOT DEFINED MM_KERNEL_FILES)
    file(GLOB MM_KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/mm/*.cpp)
    file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/kernels/*.cpp)
endif()

ascendc_library(ascendc_kernels_${RUN_MODE} SHARED ${KERNEL_FILES})

if(MM_KERNEL_FILES)
    ascendc_library(ascendc_kernels_mm_${RUN_MODE} SHARED ${MM_KERNEL_FILES})
    ascendc_compile_definitions(ascendc_kernels_mm_${RUN_MODE} PRIVATE
        HAVE_WORKSPACE
        HAVE_TILING
    )
endif()
