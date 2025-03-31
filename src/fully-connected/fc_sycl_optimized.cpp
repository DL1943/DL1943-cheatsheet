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
#include <CL/sycl.hpp>

using namespace sycl;
using namespace std;

#define TILE_SIZE 16  // Work-group size for SYCL kernel

// Fully Connected Layer using SYCL
void fully_connected_sycl(const vector<vector<float>> &X, const vector<vector<float>> &W,
                          const vector<float> &B, vector<vector<float>> &Y) {
    int batch_size = X.size();
    int input_size = X[0].size();
    int output_size = W[0].size();

    Y.assign(batch_size, vector<float>(output_size, 0.0f));

    // Flatten input and weights for SYCL
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

    // Create SYCL queue for device execution
    queue q(default_selector_v);

    // Allocate device memory
    buffer<float, 1> d_X(h_X.data(), range<1>(batch_size * input_size));
    buffer<float, 1> d_W(h_W.data(), range<1>(input_size * output_size));
    buffer<float, 1> d_B(h_B.data(), range<1>(output_size));
    buffer<float, 1> d_Y(h_Y.data(), range<1>(batch_size * output_size));

    // Launch SYCL kernel
    q.submit([&](handler &h) {
        accessor X(d_X, h, read_only);
        accessor W(d_W, h, read_only);
        accessor B(d_B, h, read_only);
        accessor Y(d_Y, h, write_only, no_init);

        h.parallel_for(nd_range<2>({batch_size, output_size}, {TILE_SIZE, TILE_SIZE}),
                       [=](nd_item<2> item) {
                           int row = item.get_global_id(0);
                           int col = item.get_global_id(1);

                           if (row < batch_size && col < output_size) {
                               float sum = B[col];  // Start with bias

                               // Compute dot product
                               for (int k = 0; k < input_size; k++) {
                                   sum += X[row * input_size + k] * W[k * output_size + col];
                               }

                               Y[row * output_size + col] = sum;  // Store result
                           }
                       });
    }).wait();

    // Copy result back to host
    host_accessor h_Y_acc(d_Y, read_only);
    for (int i = 0; i < batch_size; i++)
        for (int j = 0; j < output_size; j++)
            Y[i][j] = h_Y_acc[i * output_size + j];
}

int main() {
    // Example input
    vector<vector<float>> X(16, vector<float>(16, 1.0f));  // Batch size = 16, Input size = 16
    vector<vector<float>> W(16, vector<float>(16, 0.1f));  // Input size = 16, Output size = 16
    vector<float> B(16, 0.1f);  // Bias (16)
    vector<vector<float>> Y;  // Output (16x16)

    fully_connected_sycl(X, W, B, Y);

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
