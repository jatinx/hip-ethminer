#pragma once

#include <stdint.h>
#include <sstream>
#include <stdexcept>
#include <string>

#include <hip/hip_runtime.h>

// It is virtually impossible to get more than
// one solution per stream hash calculation
// Leave room for up to 4 results. A power
// of 2 here will yield better CUDA optimization
// TODO verify this
#define MAX_SEARCH_RESULTS 8U

struct Search_Result
{
    // One word for gid and 8 for mix hash
    uint32_t gid;
    uint32_t mix[8];
    uint32_t pad[7];  // pad to size power of 2
};

struct Search_results
{
    Search_Result result[MAX_SEARCH_RESULTS];
    uint32_t count = 0;
};

#define ACCESSES 64
#define THREADS_PER_HASH (128 / 16)

typedef struct
{
    uint4 uint4s[32 / sizeof(uint4)];
} hash32_t;

typedef union
{
    uint32_t words[128 / sizeof(uint32_t)];
    uint2 uint2s[128 / sizeof(uint2)];
    uint4 uint4s[128 / sizeof(uint4)];
} hash128_t;

typedef union
{
    uint32_t words[64 / sizeof(uint32_t)];
    uint2 uint2s[64 / sizeof(uint2)];
    uint4 uint4s[64 / sizeof(uint4)];
} hash64_t;

void set_constants(hash128_t* _dag, uint32_t _dag_size, hash64_t* _light, uint32_t _light_size);
void get_constants(hash128_t** _dag, uint32_t* _dag_size, hash64_t** _light, uint32_t* _light_size);

void set_header(hash32_t _header);

void set_target(uint64_t _target);

void run_ethash_search(uint32_t gridSize, uint32_t blockSize, hipStream_t stream,
    volatile Search_results* g_output, uint64_t start_nonce);

void ethash_generate_dag(uint64_t dag_size, uint32_t blocks, uint32_t threads, hipStream_t stream);

struct hip_runtime_error : public virtual std::runtime_error
{
    hip_runtime_error(const std::string& msg) : std::runtime_error(msg) {}
};

#define HIP_SAFE_CALL(call)                                                              \
    do                                                                                   \
    {                                                                                    \
        hipError_t err = call;                                                           \
        if (hipSuccess != err)                                                           \
        {                                                                                \
            std::stringstream ss;                                                        \
            ss << "HIP error in func " << __FUNCTION__ << " at line " << __LINE__ << ' ' \
               << hipGetErrorString(err);                                                \
            throw hip_runtime_error(ss.str());                                           \
        }                                                                                \
    } while (0)
