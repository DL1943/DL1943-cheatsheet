/* **************************************************************************
 * Copyright 2025. All Rights Reserved.
 *
 * Code for the book "Deep Learning: DL1943 Cheatsheet: DL/AI/ML Research, Engineering, Optimization & System Design"
 *
 * Licensed under the GNU General Public License, Version 3.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.gnu.org/licenses/gpl-3.0.html
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For details about the book, please visit: https://www.amazon.com/dp/B0DY7SXGVV
 * *************************************************************************/

#include <iostream>
#include <vector>
#include <chrono>
#include <immintrin.h> // For AVX-512 intrinsics

const int TILE_SIZE = 16; // Tune based on cache size

void matmul_pipelined(const std::vector<std::vector<float>> &A,
                      const std::vector<std::vector<float>> &B,
                      std::vector<std::vector<float>> &C, 
                      int M, int N, int K) {

    for (int i = 0; i < M; i += TILE_SIZE) {
        for (int j = 0; j < N; j += TILE_SIZE) {
            for (int k = 0; k < K; k += TILE_SIZE) {

                __m512 a_tile, b_tile, c_tile;
                
                for (int ii = i; ii < std::min(i + TILE_SIZE, M); ii++) {
                    for (int kk = k; kk < std::min(k + TILE_SIZE, K); kk++) {
                        
                        // Prefetch next A block before using it (if within bounds)
                        if (kk + TILE_SIZE < K) {
                            _mm_prefetch((char*)&A[ii][kk + TILE_SIZE], _MM_HINT_T0);
                        }

                        // Load current A block
                        a_tile = _mm512_set1_ps(A[ii][kk]);  

                        for (int jj = j; jj < std::min(j + TILE_SIZE, N); jj += 16) {
                            
                            // Load B into SIMD register
                            b_tile = _mm512_loadu_ps(&B[kk][jj]);

                            // Compute C = A * B (Fused Multiply-Add)
                            c_tile = _mm512_fmadd_ps(a_tile, b_tile, _mm512_loadu_ps(&C[ii][jj]));

                            // Store C back
                            _mm512_storeu_ps(&C[ii][jj], c_tile);
                        }
                    }
                }
            }
        }
    }
}

int main() {
    int M = 512, N = 512, K = 512; // Matrix dimensions

    std::vector<std::vector<float>> A(M, std::vector<float>(K, 1.0f));
    std::vector<std::vector<float>> B(K, std::vector<float>(N, 1.0f));
    std::vector<std::vector<float>> C(M, std::vector<float>(N, 0.0f));

    auto start = std::chrono::high_resolution_clock::now();
    matmul_pipelined(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds\n";

    return 0;
}
