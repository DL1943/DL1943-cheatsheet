Read the explanation quickly in Chapter 1 of "[Deep Learning: DL1943 Cheatsheet: DL/AI/ML Research, Engineering, Optimization & System Design](https://www.amazon.com/dp/B0DY7SXGVV)" (v2.0 edition)

Run all MatMul variants as follows:

| **Variant**            | **Description**                                       |
|------------------------|------------------------------------------------------|
| **Reference (Brute Force)** | Naive **O(M×N×K)** triple loop implementation (poor cache reuse) |
| **Tiled Implementation** | Uses **blocking (tiling) technique** for better cache reuse |
| **Prefetch Optimization** | Uses **_mm_prefetch()** to fetch next block of data into cache |
| **Pipelined Execution** | Uses **AVX-512 instructions** (`_mm512_fmadd_ps`) and software pipelining |
| **OpenMP + AVX-512** | Uses **multi-threading (OpenMP) + AVX-512 vectorization** |
| **SYCL Optimized (GPU)** | Uses **Tiling, Vectorized Loads, Subgroup Reductions** |
| **CUDA Optimized (GPU)** | Uses **Shared Memory, Warp-Level Ops, Float4 Vectorization** |

```
g++ -O3 -std=c++17 matmul_ref.cpp -o matmul
./matmul
```

```
g++ -O3 -std=c++17 matmul_tiled.cpp -o matmul_tiled
./matmul_tiled
```

```
g++ -O3 -std=c++17 -march=native matmul_prefetch.cpp -o matmul_prefetch
./matmul_prefetch
```

```
g++ -O2 -std=c++17 -march=native -mavx512f matmul_pipelined.cpp -o matmul_pipelined
./matmul_pipelined
```

```
g++ -O2 -std=c++17 -march=native -mavx512f -fopenmp matmul_omp_avx512.cpp -o matmul_omp_avx512
export OMP_NUM_THREADS=8  # Adjust based on CPU cores
./matmul_omp_avx512
```

```
dpcpp -O2 -std=c++17 matmul_sycl_optimized.cpp -o matmul_sycl_optimized
./matmul_sycl_optimized
```

```
nvcc -O2 -std=c++17 matmul_cuda_optimized.cu -o matmul_cuda_optimized
./matmul_cuda_optimized
```
