#include <cstdio>               // printf
#include <cuda.h>
#include <stdlib.h>             // malloc, free, drand48, abs
#include <cuda_runtime.h>       // cudaMalloc, cudaMemcpy

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
void cpu_sgemm(float *A_ptr, float *B_ptr, float *C_ptr, int M, int N, int K)
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

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])

template <unsigned int N_TILE_SIZE,
          unsigned int M_TILE_SIZE,
          unsigned int K_TILE_SIZE, 
          unsigned int N_PER_THREAD,
          unsigned int M_PER_THREAD>
__global__ void cuda_sgemm_v5(float *A_ptr, float *B_ptr, float *C_ptr, int M, int N, int K)
{
    // Thread & block coordinates
    const int tx = threadIdx.x;                 // local column id inside the block 
    const int ty = threadIdx.y;                 // local row id  inside the block
    const int xBase = blockIdx.x * N_TILE_SIZE; // global col of block’s left edge
    const int yBase = blockIdx.y * M_TILE_SIZE; // global row of block’s top edge

    // shared memory staging buffers
    __shared__ float a_shared[K_TILE_SIZE][M_TILE_SIZE];
    __shared__ float b_shared[K_TILE_SIZE][N_TILE_SIZE];

    // per-thread accumulator
    float temp[M_PER_THREAD][N_PER_THREAD] = {0.0};
    float r_load_a[4];
    float r_load_b[4];
    float r_comp_a[M_PER_THREAD]; 
    float r_comp_b[N_PER_THREAD];

    // Total number of K tiles (ceil division)
    const int NUM_TILES = (K + K_TILE_SIZE - 1) / K_TILE_SIZE;

    // 1D threadID
    const int tid = ty * blockDim.x + tx; 

    // Loading address for A and B (initially)
    int load_a_smem_m = tid / (K_TILE_SIZE / 4);
    int load_a_smem_k = (tid % (K_TILE_SIZE / 4)) * 4;
    int load_b_smem_k = tid / (N_TILE_SIZE / 4);
    int load_b_smem_n = (tid % (N_TILE_SIZE / 4)) * 4;

    // step size for computing A and B
    int comp_a_step_m = 4 * blockDim.x; 
    int comp_b_step_n = 4 * blockDim.y;
    int comp_a_NUM_m = M_TILE_SIZE / comp_a_step_m;
    int comp_b_NUM_n = N_TILE_SIZE / comp_b_step_n;


    // Load A and B tiles into shared memory
    for (int tile = 0; tile < NUM_TILES; ++tile)
    {  
        int kBase = tile * K_TILE_SIZE; // starting K index of this tile
        
        // Load A and B tile into register
        int y = yBase + load_a_smem_m; 
        int k_a = kBase + load_a_smem_k;
        int load_a_gmem_addr = OFFSET(y, k_a, K);
        FETCH_FLOAT4(r_load_a[0]) = FETCH_FLOAT4(A_ptr[load_a_gmem_addr]);
        int k_b = kBase + load_b_smem_k;
        int x = xBase + load_b_smem_n;
        int load_b_gmem_addr = OFFSET(k_b, x, N);
        FETCH_FLOAT4(r_load_b[0]) = FETCH_FLOAT4(B_ptr[load_b_gmem_addr]);

        // From register to shared memory
        a_shared[load_a_smem_k][load_a_smem_m] = r_load_a[0];
        a_shared[load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        a_shared[load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        a_shared[load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FETCH_FLOAT4(b_shared[load_b_smem_k][load_b_smem_n]) = FETCH_FLOAT4(r_load_b[0]);
        
        __syncthreads();   // ensures tiles fully populated

        // Compute the outer product for this tile
        for (int kk = 0; kk < K_TILE_SIZE; ++kk)
        {
            for (int i = 0; i < comp_a_NUM_m; ++i)
            {
                int yy = 4 * ty + i * comp_a_step_m; 
                FETCH_FLOAT4(r_comp_a[4 * i]) = FETCH_FLOAT4(a_shared[kk][yy]);
            }
            for(int j = 0; j < comp_b_NUM_n; ++j)
            {
                int xx = 4 * tx + j * comp_b_step_n;
                FETCH_FLOAT4(r_comp_b[4 * j]) = FETCH_FLOAT4(b_shared[kk][xx]);
            }   
            for (int tm = 0; tm < M_PER_THREAD; ++tm)
                for (int tn = 0; tn < N_PER_THREAD; ++tn)
                {
                    temp[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                }

        }
        __syncthreads(); // avoid data hazard before next load
    }

    // Write back the result in C using Float4 for vectorized stores

    for (int i = 0; i < comp_a_NUM_m; ++i)
        for(int j = 0; j < comp_b_NUM_n; ++j)
        {
            int y = yBase + i * comp_a_step_m + 4 * ty;
            int x = xBase + j * comp_b_step_n + 4 * tx; 
            FETCH_FLOAT4(C_ptr[y * N + x]) = FETCH_FLOAT4(temp[4*i][4*j]);
            FETCH_FLOAT4(C_ptr[(y + 1) * N + x]) = FETCH_FLOAT4(temp[4*i + 1][4*j]);
            FETCH_FLOAT4(C_ptr[(y + 2) * N + x]) = FETCH_FLOAT4(temp[4*i + 2][4*j]);
            FETCH_FLOAT4(C_ptr[(y + 3) * N + x]) = FETCH_FLOAT4(temp[4*i + 3][4*j]);
        }    

}

int main()
{
    // Matrix sizes
    int M = 1024;
    int N = 1024;
    int K = 1024;

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
    cpu_sgemm(matrix_A_host, matrix_B_host, matrix_C_host_cpu_calc, M, N, K);
    printf("CPU SGEMM completed.\n");
    
    // Launch GPU SGEMM kernel
    constexpr int N_BLOCK_SIZE = 16;
    constexpr int M_BLOCK_SIZE = 16;
    constexpr int K_TILE_SIZE = 8;
    constexpr int N_TILE_SIZE = 128;
    constexpr int M_TILE_SIZE = 128;
    constexpr int N_PER_THREAD = N_TILE_SIZE / N_BLOCK_SIZE;
    constexpr int M_PER_THREAD = M_TILE_SIZE / M_BLOCK_SIZE;
    dim3 block(N_BLOCK_SIZE, M_BLOCK_SIZE);
    dim3 grid((N + N_TILE_SIZE - 1) / N_TILE_SIZE, (M + M_TILE_SIZE - 1) / M_TILE_SIZE);
    cuda_sgemm_v5<N_TILE_SIZE, M_TILE_SIZE, K_TILE_SIZE, N_PER_THREAD, M_PER_THREAD><<<grid, block>>>(matrix_A_device, matrix_B_device, matrix_C_device, M, N, K);
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