#include <iostream>
#include <vector>
#include <chrono>
#include "matrix_multiply.h"


int main() {
    constexpr int N = 1024;        
    constexpr int TILE_SIZE = 64;  

    AlignedVectorFloat A(N * N, 1.0f);
    AlignedVectorFloat B(N * N, 2.0f);
    AlignedVectorFloat C_avx2(N * N, 0.0f);
    std::vector<float> A_optimized(A.begin(), A.end());
    std::vector<float> B_optimized(B.begin(), B.end());
    std::vector<float> C_optimized(N * N, 0.0f);

    auto optimized_start = std::chrono::high_resolution_clock::now();
    multiply_optimized(A_optimized, B_optimized, C_optimized, N, TILE_SIZE);
    auto optimized_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> optimized_elapsed = optimized_end - optimized_start;

    auto avx2_start = std::chrono::high_resolution_clock::now();
    multiply_avx2_aligned(A, B, C_avx2, N, TILE_SIZE);
    auto avx2_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> avx2_elapsed = avx2_end - avx2_start;

    std::cout << "Optimized execution time: " << optimized_elapsed.count() << " ms\n";
    std::cout << "Optimized verification (C[0]): " << C_optimized[0]
              << " (Expected: " << 2.0f * N << ")\n";
    std::cout << "Aligned AVX2 execution time: " << avx2_elapsed.count() << " ms\n";
    std::cout << "Aligned AVX2 verification (C[0]): " << C_avx2[0]
              << " (Expected: " << 2.0f * N << ")\n";

    return 0;
}