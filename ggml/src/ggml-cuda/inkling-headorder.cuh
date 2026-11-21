#pragma once

struct inkling_vec_coords {
    int sequence, head;
    unsigned split;
};

// The original launch is (1, splits, 24*sequences), with y fastest after x.
// Transpose only the 3-by-splits rectangle belonging to one KV head.
#if defined(__CUDACC__) || defined(__HIP__)
__host__ __device__
#endif
static constexpr inkling_vec_coords inkling_headorder_coords(unsigned y, unsigned z, unsigned splits) {
    const unsigned linear = (z % 3)*splits + y;
    return {int(z / 24), int((z / 3 % 8)*3 + linear % 3), linear / 3};
}
