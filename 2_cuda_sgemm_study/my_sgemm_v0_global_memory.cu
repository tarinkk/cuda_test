#include <cstdio>             // printf
#include <cuda.h>              
#include <stdlib.h>           // malloc, free, drand48, abs
#include <cuda_runtime.h>     // cudaMalloc, cudaMemcpy

// Utility: Fill a host matrix with random numbers in the range [-1, 1]
void random_matrix(int M, int N, float *A_ptr)
{
    for (int m = 0; m < M; m++)
        for(int n = 0; n < N; n++)
        {
            A_ptr[m * N + n] = 2.0 * (float)drand48() - 1.0;
        } 
}

// Reference SGEMM on CPU: triple-nested loop (O(M.N.K))
void cpu_sgemm_v0(float *A_ptr, float *B_ptr, float *C_ptr, int M, int N, int K)
{
    for (int m = 0; m < M; m++)
        for(int n = 0; n < N; n++)
        {
            for(int k = 0; k < K; k++)
            {
                C_ptr[m * N + n] += A_ptr[m * K + k] * B_ptr[k * N + n];
            }
        }
}

// Compare two matrices element-wise; return max difference
float compare_matrices(int M, int N, float *C_gpu, float *C_cpu)
{
    float diff = 0.0f;
    float max_diff = 0.0f;
    for (int m = 0; m < M; m++)
        for(int n = 0; n < N; n++)
        {
            diff = abs(C_gpu[m * N + n] - C_cpu[m * N + n]);
            if (diff > max_diff)
            {
                max_diff = diff;
            }
        }
    return max_diff;
}

// Naïve CUDA SGEMM kernel (no tiling / shared memory, purely register compute).
// Each thread computes one element of the output matrix C.
//   Grid : ceil(N / BLOCK) × ceil(M / BLOCK)
//   Block: BLOCK × BLOCK threads (2D grid)
__global__ void cuda_sgemm_v0(float *A_ptr, float *B_ptr, float *C_ptr, const int M, const int N, const int K)
{
    // Global coordinates of the thread in output matrix C.
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // column n
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // row    m

    // Bounds check: skip out-of-range threads (when N or M not divisible by BLOCK).
    if (x >= N || y >= M) return;

    // Pointers to the start of the A and B matrices for this block.
    float *A_ptr_start = A_ptr + blockDim.y * blockIdx.y * K;
    float *B_ptr_start = B_ptr + blockDim.x * blockIdx.x;

    float temp = 0.f; // accumulator in register

    // Compute the dot product for the (x,y) element of C.
    // The code is structured to emphasize per-block execution.
    for (int k = 0; k < K; k++)
    {
        temp += A_ptr_start[threadIdx.y * K + k] * B_ptr_start[k * N + threadIdx.x];
    }

    // Store result.
    C_ptr[y * N + x] = temp;
}

int main()
{
    // Matrix sizes
    const int M = 512;
    const int N = 512;
    const int K = 512;
    printf("Matrix sizes: M = %d, N = %d, K = %d\n", M, N, K);

    // Host memory allocation
    const size_t mem_size_A = M * K * sizeof(float);
    const size_t mem_size_B = K * N * sizeof(float);
    const size_t mem_size_C = M * N * sizeof(float);
    
    float *matrix_A_host = (float *)malloc(mem_size_A);
    float *matrix_B_host = (float *)malloc(mem_size_B);
    float *matrix_C_host_gpu_calc = (float *)malloc(mem_size_C);
    float *matrix_C_host_cpu_calc = (float *)malloc(mem_size_C);

    // Generate random inputs
    random_matrix(M, K, matrix_A_host);
    random_matrix(K, N, matrix_B_host);
    printf("Random matrices generated.\n");

    // Initialize output matrices to zero
    memset(matrix_C_host_gpu_calc, 0, mem_size_C);
    memset(matrix_C_host_cpu_calc, 0, mem_size_C);
    printf("Memory allocated and initialized.\n");

    // Device memory allocation
    float *matrix_A_device, *matrix_B_device, *matrix_C_device;
    cudaMalloc((void **)&matrix_A_device, mem_size_A);
    cudaMalloc((void **)&matrix_B_device, mem_size_B);
    cudaMalloc((void **)&matrix_C_device, mem_size_C);

    // Copy inputs to GPU
    cudaMemcpy(matrix_A_device, matrix_A_host, mem_size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(matrix_B_device, matrix_B_host, mem_size_B, cudaMemcpyHostToDevice);
    printf("Data copied to device.\n");

    // CPU (reference) computation
    cpu_sgemm_v0(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, M, N, K);
    printf("CPU SGEMM completed.\n");
    
    // Launch GPU SGEMM kernel
    constexpr int BLOCK = 16; // 16 * 16 = 256 threads per block
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);
    cuda_sgemm_v0<<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, M, N, K);
    printf("GPU SGEMM kernel launched.\n");

    // Retrieve result from GPU
    cudaMemcpy(matrix_C_host_gpu_calc, matrix_C_device, mem_size_C, cudaMemcpyDeviceToHost);
    printf("Data copied from device to host.\n");

    // Verify results
    float diff = compare_matrices(M, N, matrix_C_host_gpu_calc, matrix_C_host_cpu_calc);
    printf("Comparison of GPU and CPU results completed.\n");

    if (diff > 0.5f)
    {
        printf("Error: GPU and CPU results do not match! Difference: %f\n", diff);

    }
    else
    {
        printf("Success: GPU and CPU results match! Difference: %f\n", diff);
    }

    // Cleanup
    free(matrix_A_host);
    free(matrix_B_host);
    free(matrix_C_host_cpu_calc);
    free(matrix_C_host_gpu_calc);

    cudaFree(matrix_A_device);
    cudaFree(matrix_B_device);
    cudaFree(matrix_C_device);
    printf("Memory freed.\n");
    return 0;
}