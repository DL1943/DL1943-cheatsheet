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
#include <cuda_runtime.h>

using namespace std;

#define TILE_SIZE 16  // Block size for CUDA kernel

// CUDA Kernel for Fully Connected Layer
__global__ void fully_connected_cuda(const float *X, const float *W, const float *B, float *Y, 
                                     int batch_size, int input_size, int output_size) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < batch_size && col < output_size) {
        float sum = B[col];  // Start with bias

        // Compute dot product
        for (int k = 0; k < input_size; k++) {
            sum += X[row * input_size + k] * W[k * output_size + col];
        }

        Y[row * output_size + col] = sum;  // Store result
    }
}

// Helper function to run CUDA kernel
void run_fully_connected_cuda(const vector<vector<float>> &X, const vector<vector<float>> &W, 
                              const vector<float> &B, vector<vector<float>> &Y) {
    int batch_size = X.size();
    int input_size = X[0].size();
    int output_size = W[0].size();

    Y.assign(batch_size, vector<float>(output_size, 0.0f));

    // Flatten input and weights for CUDA
    vector<float> h_X(batch_size * input_size);
    vector<float> h_W(input_size * output_size);
    vector<float> h_B = B;
    vector<float> h_Y(batch_size * output_size, 0.0f);

    for (int i = 0; i < batch_size; i++)
        for (int j = 0; j < input_size; j++)
            h_X[i * input_size + j] = X[i][j];

    for (int i = 0; i < input_size; i++)
        for (int j = 0; j < output_size; j++)
            h_W[i * output_size + j] = W[i][j];

    // Allocate device memory
    float *d_X, *d_W, *d_B, *d_Y;
    cudaMalloc(&d_X, batch_size * input_size * sizeof(float));
    cudaMalloc(&d_W, input_size * output_size * sizeof(float));
    cudaMalloc(&d_B, output_size * sizeof(float));
    cudaMalloc(&d_Y, batch_size * output_size * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_X, h_X.data(), batch_size * input_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_W, h_W.data(), input_size * output_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), output_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Kernel
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((output_size + TILE_SIZE - 1) / TILE_SIZE, (batch_size + TILE_SIZE - 1) / TILE_SIZE);
    fully_connected_cuda<<<numBlocks, threadsPerBlock>>>(d_X, d_W, d_B, d_Y, batch_size, input_size, output_size);

    // Copy result back
    cudaMemcpy(h_Y.data(), d_Y, batch_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert back to 2D output
    for (int i = 0; i < batch_size; i++)
        for (int j = 0; j < output_size; j++)
            Y[i][j] = h_Y[i * output_size + j];

    // Free memory
    cudaFree(d_X);
    cudaFree(d_W);
    cudaFree(d_B);
    cudaFree(d_Y);
}

int main() {
    // Example input
    vector<vector<float>> X(16, vector<float>(16, 1.0f));  // Batch size = 16, Input size = 16
    vector<vector<float>> W(16, vector<float>(16, 0.1f));  // Input size = 16, Output size = 16
    vector<float> B(16, 0.1f);  // Bias (16)
    vector<vector<float>> Y;  // Output (16x16)

    run_fully_connected_cuda(X, W, B, Y);

    // Print output
    cout << "Fully Connected Output:\n";
    for (const auto &row : Y) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << "\n";
    }

    return 0;
}
