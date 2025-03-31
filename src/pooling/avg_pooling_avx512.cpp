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
#include <immintrin.h>  // AVX-512 header

using namespace std;

// AVX-512 optimized AvgPooling
void avg_pooling_avx512(const vector<vector<float>> &input, vector<vector<float>> &output, 
                        int pool_size, int stride) {
    int input_size = input.size();
    int output_size = (input_size - pool_size) / stride + 1;

    output.assign(output_size, vector<float>(output_size, 0.0f));

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            __m512 sum_vec = _mm512_setzero_ps(); // Initialize sum vector

            // Process 16 elements at a time using AVX-512
            for (int m = 0; m < pool_size; m++) {
                for (int n = 0; n < pool_size; n += 16) {
                    __m512 val = _mm512_loadu_ps(&input[i * stride + m][j * stride + n]);
                    sum_vec = _mm512_add_ps(sum_vec, val);
                }
            }

            // Sum all 16 values in sum_vec
            float sum[16];
            _mm512_storeu_ps(sum, sum_vec);
            float total_sum = 0.0f;
            for (int k = 0; k < 16; k++) {
                total_sum += sum[k];
            }

            // Compute average
            output[i][j] = total_sum / (pool_size * pool_size);
        }
    }
}

int main() {
    // Example 16x16 input (AVX-512 requires at least 16 elements per row)
    vector<vector<float>> input(16, vector<float>(16, 1.0f)); // All ones for easy validation

    int pool_size = 2, stride = 2;
    vector<vector<float>> output;

    avg_pooling_avx512(input, output, pool_size, stride);

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
