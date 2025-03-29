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

void matmul(const std::vector<std::vector<float>> &A,
            const std::vector<std::vector<float>> &B,
            std::vector<std::vector<float>> &C, 
            int M, int N, int K) {
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++) {
                C[i][j] += A[i][k] * B[k][j]; // Poor cache reuse
            }
        }
    }
}

int main() {
    int M = 500, N = 500, K = 500; // Matrix dimensions

    std::vector<std::vector<float>> A(M, std::vector<float>(K, 1.0f));
    std::vector<std::vector<float>> B(K, std::vector<float>(N, 1.0f));
    std::vector<std::vector<float>> C(M, std::vector<float>(N, 0.0f));

    auto start = std::chrono::high_resolution_clock::now();
    matmul(A, B, C, M, N, K);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Time taken: " << elapsed.count() << " seconds\n";

    return 0;
}
