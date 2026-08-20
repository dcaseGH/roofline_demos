#pragma once

#include <vector>

#include "aligned_allocator.h"

void multiply_optimized(const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        int N, int tileSize = 64);

void multiply_avx2_aligned(const AlignedVectorFloat& A,
                           const AlignedVectorFloat& B,
                           AlignedVectorFloat& C,
                           int N, int tileSize = 64);
