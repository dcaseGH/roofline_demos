#include <iostream>
#include <vector>
#include <chrono>
#include <omp.h>

// Optimized Tiled & Multithreaded Matrix Multiplication
void multiply_optimized(const std::vector<float>& A, 
                        const std::vector<float>& B, 
                        std::vector<float>& C, 
                        int N, int tileSize = 64) 
{
    // OpenMP parallelizes the outer block iterations across available CPU cores
    #pragma omp parallel for collapse(2) schedule(static)
    for (int sj = 0; sj < N; sj += tileSize) {
        for (int si = 0; si < N; si += tileSize) {
            for (int sk = 0; sk < N; sk += tileSize) {

                // Tile bounds handling
                int i_end = std::min(si + tileSize, N);
                int j_end = std::min(sj + tileSize, N);
                int k_end = std::min(sk + tileSize, N);

                // Reordered loops (I -> K -> J) inside each tile for continuous memory access
                for (int i = si; i < i_end; ++i) {
                    for (int k = sk; k < k_end; ++k) {
                        float r = A[i * N + k]; // Cached scalar value
                        for (int j = sj; j < j_end; ++j) {
                            C[i * N + j] += r * B[k * N + j];
                        }
                    }
                }

            }
        }
    }
}

int main() {
    constexpr int N = 1024; // 1024x1024 float matrix
    constexpr int TILE_SIZE = 64; // Tuned for standard L1 cache sizes (~32KB-64KB)

    std::vector<float> A(N * N, 1.0f);
    std::vector<float> B(N * N, 2.0f);
    std::vector<float> C(N * N, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    
    multiply_optimized(A, B, C, N, TILE_SIZE);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed = end - start;

    std::cout << "Execution time: " << elapsed.count() << " ms\n";
    std::cout << "Verification (C[0]): " << C[0] << " (Expected: " << 2.0f * N << ")\n";

    return 0;
}