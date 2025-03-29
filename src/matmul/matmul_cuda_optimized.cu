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

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <chrono>

#define TILE_SIZE 16
#define VECTOR_SIZE 4  // Use float4 for vectorized loads

__global__ void matmul_cuda_optimized(const float *A, const float *B, float *C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int k = 0; k < K; k += TILE_SIZE) {
        // Vectorized loads using float4
        if (row < M && (k + threadIdx.x) < K)
            tileA[threadIdx.y][threadIdx.x] = A[row * K + k + threadIdx.x];
        else
            tileA[threadIdx.y][threadIdx.x] = 0;

        if (col < N && (k + threadIdx.y) < K)
            tileB[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + col];
        else
            tileB[threadIdx.y][threadIdx.x] = 0;

        __syncthreads(); // Sync before computing

        // Compute using tiling
        for (int kk = 0; kk < TILE_SIZE; kk += VECTOR_SIZE) {
            float4 vecA = *(float4*)&tileA[threadIdx.y][kk];
            float4 vecB = *(float4*)&tileB[kk][threadIdx.x];

            sum += vecA.x * vecB.x + vecA.y * vecB.y + vecA.z * vecB.z + vecA.w * vecB.w;
        }

        __syncthreads(); // Sync before next tile
    }

    // Store the result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void matmul_cuda(const std::vector<float> &A, const std::vector<float> &B, std::vector<float> &C, int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    cudaMalloc(&d_A, sizeA);
    cudaMalloc(&d_B, sizeB);
    cudaMalloc(&d_C, sizeC);

    cudaMemcpy(d_A, A.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.data(), sizeB, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C.data(), sizeC, cudaMemcpyHostToDevice);

    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (M + TILE_SIZE - 1) / TILE_SIZE);

    matmul_cuda_optimized<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaMemcpy(C.data(), d_C, sizeC, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    int M = 1024, N = 1024, K = 1024; // Large matrix sizes for GPU execution

    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 1.0f);
    std::vector<float> C(M * N, 0.0f);

    auto start = std::chrono::high_resolution_clock::now();
    matmul_cuda(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Optimized CUDA MatMul time: " << elapsed.count() << " seconds\n";

    return 0;
}
