Read the explanation quickly in "[Deep Learning: DL1943 Cheatsheet: DL/AI/ML Research, Engineering, Optimization & System Design](https://www.amazon.com/dp/B0DY7SXGVV)" (v2.0 edition)

Run all Fully Connected variants as follows:

```cpp
g++ -O2 -std=c++17 fc_naive.cpp -o fc_naive
./fc_naive
```

```cpp
g++ -O2 -std=c++17 -march=native -mavx512f -fopenmp fc_omp_avx512.cpp -o fc_omp_avx512
export OMP_NUM_THREADS=8  # Adjust based on CPU cores
./fc_omp_avx512
```

```cpp
nvcc -O2 -std=c++17 fc_cuda_optimized.cu -o fc_cuda_optimized
./fc_cuda_optimized
```

```cpp
dpcpp -O2 -std=c++17 fc_sycl_optimized.cpp -o fc_sycl_optimized
./fc_sycl_optimized
```
