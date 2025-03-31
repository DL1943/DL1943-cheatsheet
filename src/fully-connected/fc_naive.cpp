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

// Fully Connected (Dense) Layer - Naive Implementation
void fully_connected_naive(const vector<vector<float>> &X, const vector<vector<float>> &W, 
                           const vector<float> &B, vector<vector<float>> &Y) {
    int batch_size = X.size();
    int input_size = X[0].size();
    int output_size = W[0].size();

    // Initialize output with bias
    Y.assign(batch_size, vector<float>(output_size, 0.0f));

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < output_size; j++) {
            Y[i][j] = B[j]; // Add bias
            for (int k = 0; k < input_size; k++) {
                Y[i][j] += X[i][k] * W[k][j]; // Matrix multiplication
            }
        }
    }
}

int main() {
    // Example input
    vector<vector<float>> X = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}}; // (2x3) batch_size=2, input_size=3
    vector<vector<float>> W = {{0.1, 0.2}, {0.3, 0.4}, {0.5, 0.6}}; // (3x2) input_size=3, output_size=2
    vector<float> B = {0.1, 0.2}; // Bias (2)
    vector<vector<float>> Y; // Output (2x2)

    fully_connected_naive(X, W, B, Y);

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
