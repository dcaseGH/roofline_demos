#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#include "matrix_multiply.h"

int main()
{
    constexpr int N = 8;
    constexpr int TILE_SIZE = 4;
    constexpr float expected_value = 2.0f * N;

    AlignedVectorFloat aligned_values(N * N, 1.0f);
    assert(reinterpret_cast<std::uintptr_t>(aligned_values.data()) % 32 == 0);

    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> optimized_result(N * N, 0.0f);
    AlignedVectorFloat avx2_A(N * N, 1.0f);
    AlignedVectorFloat avx2_B(N * N, 2.0f);
    AlignedVectorFloat avx2_result(N * N, 0.0f);

    multiply_optimized(A, B, optimized_result, N, TILE_SIZE);
    multiply_avx2_aligned(avx2_A, avx2_B, avx2_result, N, TILE_SIZE);

    for (std::size_t index = 0; index < optimized_result.size(); ++index) {
        assert(std::fabs(optimized_result[index] - expected_value) < 1e-5f);
        assert(std::fabs(avx2_result[index] - expected_value) < 1e-5f);
        assert(std::fabs(optimized_result[index] - avx2_result[index]) < 1e-5f);
    }

    std::cout << "All matrix multiplication tests passed.\n";
    return 0;
}
