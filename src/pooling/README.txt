## AvgPool

Read the explanation quickly in "[Deep Learning: DL1943 Cheatsheet: DL/AI/ML Research, Engineering, Optimization & System Design](https://www.amazon.com/dp/B0DY7SXGVV)" (v2.0 edition)

Run the different variants:

```cpp
g++ -O3 -std=c++17 avg_pooling_naive.cpp -o avg_pooling_naive
./avg_pooling_naive
```

```cpp
g++ -O3 -std=c++17 -march=native -mavx512f avg_pooling_avx512.cpp -o avg_pooling_avx512
./avg_pooling_avx512
```

```cpp
g++ -O2 -std=c++17 -march=native -mavx512f -fopenmp avg_pooling_omp_avx512.cpp -o avg_pooling_omp_avx512
export OMP_NUM_THREADS=8  # Adjust based on CPU cores
./avg_pooling_omp_avx512
```

```cpp
nvcc -O2 -std=c++17 avg_pooling_cuda.cu -o avg_pooling_cuda
./avg_pooling_cuda
```
