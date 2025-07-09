# Note on CUDA programming

## Table of Contents
1. [CUDA Reduction Kernel](#cuda-reduction-kernel)
    1. [Baseline](#baseline)
    2. [No Divergence Branch](#no-divergence-branch)

## CUDA Reduction Kernel:

The CUDA kernel performs a parallel reduction operation to compute the sum of elements in an input array. Each block processes a section of the input array, computes the sum of its elements, and writes the result to an output array.

1. **Data Loading**:

    ```cuda
    // Find the start of this block's data in the global array
    float *input_begin = d_input + blockDim.x * blockIdx.x;
    // Initialize shared memory for this block
    __shared__ float input_shared[THREAD_PER_BLOCK];
    input_shared[threadIdx.x] = input_begin[threadIdx.x];
    __syncthreads(); // Ensure all threads have written their data to shared memory
    ```

2. **Parallel Reduction:**

    we will compare several reduction algorithms below.

    - Baseline (Binary-tree type)
    - No divergence branch

3. **Output**:

    ```cuda
    // Only the first thread in each block writes the final sum to the output array
    if (threadIdx.x == 0)
        d_output[blockIdx.x] = input_shared[0];
    ```

### Baseline

1. **For a block with 8 threads (Visualization)**:

    - **Initial State**: Each thread loads one element into shared memory.
        ```
        input_shared: [a0, a1, a2, a3, a4, a5, a6, a7]
        ```

    - **Step size 1**: Threads 0, 2, 4, 6 add elements at offset 1 (e.g., thread 0 adds a1 to a0).
        ```
        input_shared: [a0+a1, a1, a2+a3, a3, a4+a5, a5, a6+a7, a7]
        ```

    - **Step size 2**: Threads 0, 4 add elements at offset 2 (e.g., thread 0 adds a2+a3 to a0+a1).
        ```
        input_shared: [a0+a1+a2+a3, a1, a2+a3, a3, a4+a5+a6+a7, a5, a6+a7, a7]
        ```

    - **Step size 4**: Thread 0 adds the element at offset 4 (a4+a5+a6+a7).
        ```
        input_shared: [a0+a1+a2+a3+a4+a5+a6+a7, a1, a2+a3, a3, a4+a5+a6+a7, a5, a6+a7, a7]
        ```

2. **For a block with 2^N threads**:

    ```cuda
    // double the step size i for each iteration
    for (int i = 1; i < blockDim.x; i *= 2)
    {   
        //threads with indices that are multiples of 2*i (where 
        //i is the current step size) add the value from the 
        //shared memory location at offset i to their own.
        if (threadIdx.x % (i * 2) == 0)
        {
            input_shared[threadIdx.x] += input_shared[threadIdx.x + i];
        }
        __syncthreads();
    }
    ```


### No Divergence Branch

1. **A block with 8 threads (Visualization)**:

    - **Initial state**: Each thread loads one element into shared memory.
        ```
        input_shared: [a0, a1, a2, a3, a4, a5, a6, a7]
        ```

    - **Step size 1**: Threads 0, 1, 2, 3 add elements at indices `threadIdx.x * 2 and threadIdx.x * 2 + 1` (e.g., thread 0 adds a1 to a0, thread 1 adds a3 to a2).
        ```
        input_shared: [a0+a1, a1, a2+a3, a3, a4+a5, a5, a6+a7, a7]
        ```

    - **Step size 2**: Threads 0, 1 add elements at indices `threadIdx.x * 4` and `threadIdx.x * 4 + 2` (e.g., thread 0 adds a2+a3 to a0+a1).
        ```
        input_shared: [a0+a1+a2+a3, a1, a2+a3, a3, a4+a5+a6+a7, a5, a6+a7, a7]
        ```

    - **Step size 4**: Thread 0 adds elements at indices `threadIdx.x * 8` and `threadIdx.x * 8 + 4` (i.e., a4+a5+a6+a7 to a0+a1+a2+a3).
        ```
        input_shared: [a0+a1+a2+a3+a4+a5+a6+a7, a1, a2+a3, a3, a4+a5+a6+a7, a5, a6+a7, a7]
        ```

2. **A block with 2^N threads**:

    ```cuda
    // double the step size i for each iteration
    // On each step, threads with indices less than blockDim.x/(2*i) add the value 
    // at index threadIdx.x*2*i to the value at threadIdx.x*2*i + i
    for (int i = 1; i < blockDim.x; i *= 2)
    {
        if (threadIdx.x <  blockDim.x / (i * 2))
        {   
            int index = threadIdx.x * 2 * i;
            input_shared[index] += input_shared[index + i];
        }
        __syncthreads(); // Synchronize to make sure all threads are done before the next step
    }
    ```
3. **Discussions**:
    - **Thread Utilization**:
    Comparing to the baseline, for each iteration, only the first `blockDim.x/(2*i)` threads are active, utilizing a contiguous block of threads. This reduces thread divergence within warps, as active threads are grouped together, improving warp execution efficiency.
    - **Bank Conflicts**:
    Consider `THREAD_PER_BLOCK = 64`, `i = 16`:
        - **Active Threads**: `threadIdx.x < 64 / (2 * 16) = 64 / 32 = 2` (threads 0 to 1).
        -  `index = threadIdx.x * 2 * 16 = 32 * threadIdx.x`
        - Thread 0: Accesses `input_shared[0]` (bank 0) and `input_shared[16]` (bank 16).
        - Thread 1: Accesses `input_shared[32]` (bank 0) and `input_shared[48]` (bank 16).
    
    Both Thread 0 and 1 access bank 0, 16 (different addresses) where bank conflicts arise.
    In general, when `THREAD_PER_BLOCK = 2^N` with `N>5`, step size i can reach `16`, where bank conflicts occur.  


## Reference:
1. [[CUDA]Reduce规约求和（已完结~）](https://www.bilibili.com/video/BV1HvBSY2EJW?spm_id_from=333.788.videopod.episodes&vd_source=aa41d00aebd84e6f99f529df7f83258a)
2. [深入浅出GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026)