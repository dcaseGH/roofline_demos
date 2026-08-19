#include <iostream>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <immintrin.h>
#include <omp.h>

// Custom C++17 Allocator for 32-byte aligned vectors
template <typename T, std::size_t Alignment>
struct AlignedAllocator {
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {}

    T* allocate(std::size_t n) {
        std::size_t bytes = n * sizeof(T);
        // std::aligned_alloc requires the byte size to be a multiple of the alignment
        std::size_t remainder = bytes % Alignment;
        if (remainder != 0) {
            bytes += (Alignment - remainder);
        }
        void* ptr = std::aligned_alloc(Alignment, bytes);
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, std::size_t) noexcept {
        std::free(p);
    }
};

// Convenience type definition for 32-byte aligned float vectors
using AlignedVectorFloat = std::vector<float, AlignedAllocator<float, 32>>;

// Matrix Multiplication using 32-Byte Aligned AVX2 Loads & Stores
void multiply_avx2_aligned(const AlignedVectorFloat& A, 
                           const AlignedVectorFloat& B, 
                           AlignedVectorFloat& C, 
                           int N, int tileSize = 64) 
{
    #pragma omp parallel for collapse(2) schedule(static)
    for (int sj = 0; sj < N; sj += tileSize) {
        for (int si = 0; si < N; si += tileSize) {
            for (int sk = 0; sk < N; sk += tileSize) {

                int i_end = std::min(si + tileSize, N);
                int j_end = std::min(sj + tileSize, N);
                int k_end = std::min(sk + tileSize, N);

                for (int i = si; i < i_end; ++i) {
                    for (int k = sk; k < k_end; ++k) {
                        
                        __m256 vec_a = _mm256_set1_ps(A[i * N + k]);

                        int j = sj;
                        for (; j <= j_end - 8; j += 8) {
                            
                            // Fast Aligned Loads: memory address &matrix[row * N + j] is guaranteed 32-byte aligned
                            __m256 vec_c = _mm256_load_ps(&C[i * N + j]);
                            __m256 vec_b = _mm256_load_ps(&B[k * N + j]);

                            // Fused Multiply-Add
                            vec_c = _mm256_fmadd_ps(vec_a, vec_b, vec_c);

                            // Fast Aligned Store
                            _mm256_store_ps(&C[i * N + j], vec_c);
                        }

                        // Tail cleanup loop
                        for (; j < j_end; ++j) {
                            C[i * N + j] += A[i * N + k] * B[k * N + j];
                        }
                    }
                }

            }
        }
    }
}

int main() {
    // Note: N must be a multiple of 8 (or padded) so that inner row offsets remain 32-byte aligned
    constexpr int N = 1024;        
    constexpr int TILE_SIZE = 64;  

    AlignedVectorFloat A(N * N, 1.0f);
    AlignedVectorFloat B(N * N, 2.0f);
    AlignedVectorFloat C(N * N, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();

    multiply_avx2_aligned(A, B, C, N, TILE_SIZE);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Aligned AVX2 Execution time: " << elapsed.count() << " ms\n";
    std::cout << "Verification (C[0]): " << C[0] << " (Expected: " << 2.0f * N << ")\n";

    return 0;
}