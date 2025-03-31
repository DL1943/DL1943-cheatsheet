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

// CUDA Kernel for AvgPooling
__global__ void avg_pooling_cuda(const float *input, float *output, int input_size, int output_size, int pool_size, int stride) {
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (out_x < output_size && out_y < output_size) {
        float sum = 0.0f;
        int in_x = out_x * stride;
        int in_y = out_y * stride;

        // Accumulate values over the pooling window
        for (int i = 0; i < pool_size; i++) {
            for (int j = 0; j < pool_size; j++) {
                sum += input[(in_y + i) * input_size + (in_x + j)];
            }
        }

        // Compute average
        output[out_y * output_size + out_x] = sum / (pool_size * pool_size);
    }
}

// Helper function to run CUDA kernel
void run_avg_pooling_cuda(const vector<vector<float>> &input, vector<vector<float>> &output, int pool_size, int stride) {
    int input_size = input.size();
    int output_size = (input_size - pool_size) / stride + 1;

    output.assign(output_size, vector<float>(output_size, 0.0f));

    // Flatten input for CUDA
    vector<float> h_input(input_size * input_size);
    vector<float> h_output(output_size * output_size, 0.0f);

    for (int i = 0; i < input_size; i++) {
        for (int j = 0; j < input_size; j++) {
            h_input[i * input_size + j] = input[i][j];
        }
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, input_size * input_size * sizeof(float));
    cudaMalloc(&d_output, output_size * output_size * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input.data(), input_size * input_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch Kernel
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((output_size + TILE_SIZE - 1) / TILE_SIZE, (output_size + TILE_SIZE - 1) / TILE_SIZE);
    avg_pooling_cuda<<<numBlocks, threadsPerBlock>>>(d_input, d_output, input_size, output_size, pool_size, stride);
    
    // Copy result back
    cudaMemcpy(h_output.data(), d_output, output_size * output_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Convert back to 2D output
    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            output[i][j] = h_output[i * output_size + j];
        }
    }

    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Example 16x16 input
    vector<vector<float>> input(16, vector<float>(16, 1.0f)); // All ones for easy validation

    int pool_size = 2, stride = 2;
    vector<vector<float>> output;

    run_avg_pooling_cuda(input, output, pool_size, stride);

    // Print output
    cout << "AvgPooling Output:\n";
    for (const auto &row : output) {
        for (float val : row) {
            cout << val << " ";
        }
        cout << "\n";
    }

    return 0;
}
