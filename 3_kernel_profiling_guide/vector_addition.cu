#include <cstdio>
#include <cuda.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vector_add(float *a, float *b, float *c)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    c[idx] = a[idx] + b[idx];
}

bool check(float *gpu_result, float *cpu_result, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (abs(gpu_result[i] - cpu_result[i]) > 0.005)
            return false;
    }
    return true;
}

int main()
{
    const int N = 1024 * 1024 * 32;

    float *a = (float *)malloc(N * sizeof(float));
    float *b = (float *)malloc(N * sizeof(float));
    float *gpu_result = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++)
    {
        a[i] = 2.0 * (float)drand48() - 1.0;
        b[i] = 2.0 * (float)drand48() - 1.0;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, N * sizeof(float));
    cudaMalloc((void **)&d_b, N * sizeof(float));
    cudaMalloc((void **)&d_c, N * sizeof(float));

    cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice);

    const int grid_size = N / 256;
    dim3 Grid(grid_size);
    // only compute first 1/4 elements in vector a and b 
    dim3 Block(64);

    for (int i = 0; i < 2; i++)
    {
        vector_add<<<Grid, Block>>>(d_a, d_b, d_c, N);
        cudaDeviceSynchronize();
    }

    cudaMemcpy(gpu_result, d_c, N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("The answer has been computed.\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(a);
    free(b);
    free(gpu_result);

    return 0;
}