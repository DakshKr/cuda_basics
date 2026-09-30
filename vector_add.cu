#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vector_add(const int *a, const int *b, int *c, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index == 0 || index == 23479) {
        printf("[GPU Kernel] Executing on Block %d, Thread %d\n", blockIdx.x, threadIdx.x);
    }
    if (index < size) {
        c[index] = a[index] + b[index];
    }
}

int main() {
    const int size = 1000000;
    const int bytes = size * sizeof(int);

    int *h_a = (int*)malloc(bytes);
    int *h_b = (int*)malloc(bytes);
    int *h_c = (int*)malloc(bytes);

    for (int i = 0; i < size; i++) {
        h_a[i] = rand() % 100;
        h_b[i] = rand() % 100;
    }

    int *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, bytes);
    cudaMalloc(&dev_b, bytes);
    cudaMalloc(&dev_c, bytes);

    cudaMemcpy(dev_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, h_b, bytes, cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block; 

    // Correct function name and triple angle brackets <<< ... >>>
    vector_add<<<blocks_per_grid, threads_per_block>>>(dev_a, dev_b, dev_c, size);

    cudaMemcpy(h_c, dev_c, bytes, cudaMemcpyDeviceToHost);

    printf("Addition successful for %d elements\n", size);
    printf("Verification: c[0]=%d, expected=%d\n", h_c[0], h_a[0] + h_b[0]);
    printf("Verification: c[99999]=%d, expected=%d\n", h_c[99999], h_a[99999] + h_b[99999]);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}