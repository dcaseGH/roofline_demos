#include "matrix_multiply.h"

#include <algorithm>
#include <immintrin.h>

#include <omp.h>

void multiply_optimized(const std::vector<float>& A,
                        const std::vector<float>& B,
                        std::vector<float>& C,
                        int N, int tileSize)
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
                        float r = A[i * N + k];
                        for (int j = sj; j < j_end; ++j) {
                            C[i * N + j] += r * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

void multiply_avx2_aligned(const AlignedVectorFloat& A,
                           const AlignedVectorFloat& B,
                           AlignedVectorFloat& C,
                           int N, int tileSize)
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
                            __m256 vec_c = _mm256_load_ps(&C[i * N + j]);
                            __m256 vec_b = _mm256_load_ps(&B[k * N + j]);
                            vec_c = _mm256_fmadd_ps(vec_a, vec_b, vec_c);
                            _mm256_store_ps(&C[i * N + j], vec_c);
                        }

                        for (; j < j_end; ++j) {
                            C[i * N + j] += A[i * N + k] * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}
