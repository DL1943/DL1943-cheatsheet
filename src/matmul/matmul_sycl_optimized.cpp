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

#include <CL/sycl.hpp>
#include <iostream>
#include <vector>
#include <chrono>

using namespace sycl;

constexpr int TILE_SIZE = 16; // Work-group size
constexpr int VECTOR_SIZE = 4; // Vectorized load (float4)

void matmul_sycl_optimized(const std::vector<float> &A,
                           const std::vector<float> &B,
                           std::vector<float> &C, 
                           int M, int N, int K) {
    queue q{gpu_selector_v}; // Target GPU

    buffer<float, 2> bufA(A.data(), range<2>(M, K));
    buffer<float, 2> bufB(B.data(), range<2>(K, N));
    buffer<float, 2> bufC(C.data(), range<2>(M, N));

    q.submit([&](handler &h) {
        accessor aA(bufA, h, read_only);
        accessor aB(bufB, h, read_only);
        accessor aC(bufC, h, write_only, no_init);

        local_accessor<float, 2> tileA(range<2>(TILE_SIZE, TILE_SIZE), h);
        local_accessor<float, 2> tileB(range<2>(TILE_SIZE, TILE_SIZE), h);

        h.parallel_for(nd_range<2>{{M, N}, {TILE_SIZE, TILE_SIZE}}, [=](nd_item<2> it) {
            int row = it.get_global_id(0);
            int col = it.get_global_id(1);
            int local_row = it.get_local_id(0);
            int local_col = it.get_local_id(1);
            auto sg = it.get_sub_group(); // Subgroup for intra-work-group optimizations

            float sum = 0.0f;

            for (int k = 0; k < K; k += TILE_SIZE) {
                // Load tiles into shared memory
                if (row < M && (k + local_col) < K)
                    tileA[local_row][local_col] = aA[row][k + local_col];
                else
                    tileA[local_row][local_col] = 0;

                if (col < N && (k + local_row) < K)
                    tileB[local_row][local_col] = aB[k + local_row][col];
                else
                    tileB[local_row][local_col] = 0;

                it.barrier(access::fence_space::local_space); // Sync before computing

                // Compute partial sum
                for (int kk = 0; kk < TILE_SIZE; kk += VECTOR_SIZE) {
                    // Vectorized load
                    float4 vecA = *(reinterpret_cast<const float4*>(&tileA[local_row][kk]));
                    float4 vecB = *(reinterpret_cast<const float4*>(&tileB[kk][local_col]));

                    sum += vecA.x() * vecB.x() + vecA.y() * vecB.y() +
                           vecA.z() * vecB.z() + vecA.w() * vecB.w();
                }

                it.barrier(access::fence_space::local_space); // Sync before next tile
            }

            // Subgroup reduction (optional)
            sum = reduce_over_group(sg, sum, plus<>());

            // Store result in global memory
            if (row < M && col < N) {
                aC[row][col] = sum;
            }
        });
    }).wait();
}

int main() {
    int M = 1024, N = 1024, K = 1024; // Large matrix dimensions for GPU

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    matmul_sycl_optimized(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Optimized GPU MatMul time: " << elapsed.count() << " seconds\n";

    return 0;
}
