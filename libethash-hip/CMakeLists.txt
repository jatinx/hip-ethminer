find_package(HIP PATHS /opt/rocm-5.1.3/hip/cmake/) 

set(HIPCC_FLAGS ${HIPCC_FLAGS};-use_fast_math)

if (NOT MSVC)
	list(APPEND HIPCC_FLAGS "--disable-warnings")
endif()

list(APPEND HIPCC_FLAGS_RELEASE -O3)
list(APPEND HIPCC_FLAGS_DEBUG -G)

# Only working for navi 21 atm
list(APPEND HIPCC_FLAGS "--offload-arch=gfx1030")

file(GLOB sources "*.cpp" "*.cpp")
file(GLOB headers "*.h" "*.cuh")

set(CMAKE_CXX_COMPILER hipcc)
set(CMAKE_C_COMPILER hipcc)

add_library(ethash-hip STATIC ethash_hip_miner_kernel.cpp HIPMiner.cpp)
target_compile_options(ethash-hip PUBLIC -fPIE)
target_link_libraries(ethash-hip ethcore ethash::ethash Boost::thread /opt/rocm-5.1.3/hip/lib/libamdhip64.so)
target_include_directories(ethash-hip PUBLIC /opt/rocm/include) # Make it dependent on some env Varaible
target_include_directories(ethash-hip PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
