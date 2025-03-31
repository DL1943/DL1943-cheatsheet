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
#include <immintrin.h>  // AVX-512
#include <omp.h>        // OpenMP

using namespace std;

// Fully Connected Layer - Optimized with OpenMP & AVX-512
void fully_connected_omp_avx512(const vector<vector<float>> &X, const vector<vector<float>> &W, 
                                const vector<float> &B, vector<vector<float>> &Y) {
    int batch_size = X.size();
    int input_size = X[0].size();
    int output_size = W[0].size();

    // Initialize output matrix
    Y.assign(batch_size, vector<float>(output_size, 0.0f));

    #pragma omp parallel for collapse(2) num_threads(8)
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < output_size; j++) {
            __m512 sum_vec = _mm512_setzero_ps();  // Initialize SIMD sum

            // Vectorized computation: Process 16 elements at a time
            for (int k = 0; k < input_size; k += 16) {
                __m512 x_vec = _mm512_loadu_ps(&X[i][k]);  // Load 16 elements from input
                __m512 w_vec = _mm512_loadu_ps(&W[k][j]);  // Load 16 elements from weights
                sum_vec = _mm512_fmadd_ps(x_vec, w_vec, sum_vec); // Fused multiply-add
            }

            // Sum all 16 elements in the SIMD register
            float sum[16];
            _mm512_storeu_ps(sum, sum_vec);
            float total_sum = B[j];  // Start with bias
            for (int t = 0; t < 16; t++) {
                total_sum += sum[t];
            }

            Y[i][j] = total_sum;
        }
    }
}

int main() {
    // Example input
    vector<vector<float>> X(16, vector<float>(16, 1.0f));  // Batch size = 16, Input size = 16
    vector<vector<float>> W(16, vector<float>(16, 0.1f));  // Input size = 16, Output size = 16
    vector<float> B(16, 0.1f);  // Bias (16)
    vector<vector<float>> Y;  // Output (16x16)

    fully_connected_omp_avx512(X, W, B, Y);

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
