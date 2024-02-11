#include <algorithm>
#include <cstdio>
#include <feval_gpu_xx.h>

// __global__ means this is called from the CPU, and runs on the GPU
__global__ void matrixMul(const double *a, double *b, uint64_t n_col, uint64_t len) {
    // Compute each thread's global row and column index
    uint64_t row = blockIdx.y * blockDim.y + threadIdx.y;
    uint64_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Iterate over row, and down column
    if (row < n_col && col < n_col) {
        double val = 0;
        for (uint64_t k = 0; k < len; k++) {
            val += a[row * len + k] * a[col * len + k];
        }
        b[row * n_col + col] = val;
    }
}

FevalGpuXX::~FevalGpuXX() {
    release();
}

void FevalGpuXX::release() {
    if (d_datum) {
        cudaError_t et = cudaFree(d_datum);
        if (et == cudaSuccess) {
            d_datum = nullptr;
        } else {
            printf("FevalGpuXX::release cudaFree d_datum failed %d\n", et);
        }
    }
    if (d_XTX) {
       cudaError_t et = cudaFree(d_XTX);
        if (et == cudaSuccess) {
            d_XTX = nullptr;
        } else {
            printf("FevalGpuXX::release cudaFree d_XTX failed %d\n", et);
        } 
    }
}

bool FevalGpuXX::init(uint64_t _n_col, uint64_t _len) {
    n_col = _n_col;
    uint32_t THREADS = 32;
    BLOCKS = (n_col + THREADS - 1) / THREADS;
    cudaMalloc(&d_XTX, n_col * n_col * sizeof(double));
    if (d_XTX == nullptr) return false;
    cudaError_t et = cudaMalloc(&d_datum, _len * sizeof(double));
    if (et == cudaSuccess) return true;
    else {
        printf("FevalGpuXX::init cudaMalloc failed ec=%d, %p,%zu,%zu\n", et, d_datum, n_col, _len);
        return false;
    }
}

void FevalGpuXX::calc(const std::vector<std::vector<double>*>& pXs, double * h_XTX, uint64_t offset, uint64_t _len) {
    uint32_t THREADS = 32;
    dim3 threads(THREADS, THREADS);
    dim3 blocks(BLOCKS, BLOCKS);
    for (size_t i = 0; i < n_col; i++) {
        cudaMemcpy(d_datum + i * _len, pXs[i]->data() + offset, _len * sizeof(double), cudaMemcpyHostToDevice);
    }
    matrixMul<<<blocks, threads>>>(d_datum, d_XTX, n_col, _len);
    cudaMemcpy(h_XTX, d_XTX, n_col * n_col * sizeof(double), cudaMemcpyDeviceToHost);
    // printf("FevalGpuXX::calc cudaMemcpyDeviceToHost %f %zu,%zu\n", h_XTX[0], n_col, new_len);
}