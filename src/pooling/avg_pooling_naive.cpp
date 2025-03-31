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

using namespace std;

// Performs AvgPooling on a single-channel 2D input
void avg_pooling(const vector<vector<float>> &input, vector<vector<float>> &output, 
                 int pool_size, int stride) {
    int input_size = input.size();
    int output_size = (input_size - pool_size) / stride + 1;

    output.assign(output_size, vector<float>(output_size, 0.0f));

    for (int i = 0; i < output_size; i++) {
        for (int j = 0; j < output_size; j++) {
            float sum = 0.0f;

            // Compute sum over the pooling window
            for (int m = 0; m < pool_size; m++) {
                for (int n = 0; n < pool_size; n++) {
                    sum += input[i * stride + m][j * stride + n];
                }
            }

            // Compute the average
            output[i][j] = sum / (pool_size * pool_size);
        }
    }
}

int main() {
    // Example 4x4 input
    vector<vector<float>> input = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12},
        {13, 14, 15, 16}
    };

    int pool_size = 2, stride = 2;
    vector<vector<float>> output;

    avg_pooling(input, output, pool_size, stride);

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
