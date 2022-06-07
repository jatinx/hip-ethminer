find_package(HIP REQUIRED)

set(HIPCC_FLAGS ${HIPCC_FLAGS};-use_fast_math)

if (NOT MSVC)
	list(APPEND HIPCC_FLAGS "--disable-warnings")
endif()

list(APPEND HIPCC_FLAGS_RELEASE -O3)
list(APPEND HIPCC_FLAGS_DEBUG -G)

# Only working for navi 21 atm
list(APPEND HIPCC_FLAGS "--offload-arch=gfx1030")

file(GLOB sources "*.cpp" "*.cu")
file(GLOB headers "*.h" "*.cuh")

hip_add_library(ethash-hip STATIC ${sources} ${headers})
target_link_libraries(ethash-hip ethcore ethash::ethash Boost::thread)
target_include_directories(ethash-hip PUBLIC /opt/rocm/include) # Make it dependent on some env Varaible
target_include_directories(ethash-hip PRIVATE .. ${CMAKE_CURRENT_BINARY_DIR})
