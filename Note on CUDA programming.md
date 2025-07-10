# Note on CUDA programming

## Table of Contents
1. [CUDA Reduction Kernel](#cuda-reduction-kernel)
    1. [Baseline](#baseline)
    2. [No Divergence Branch](#no-divergence-branch)
    3. [No Bank Conflict](#no-bank-conflict)
    4. [Add During Load](#add-during-load)

## CUDA Reduction Kernel:

The CUDA kernel performs a parallel reduction operation to compute the sum of elements in an input array. Each block processes a section of the input array, computes the sum of its elements, and writes the result to an output array.

**Input**: an array of length N.

**Config**: M: split the array into M portions, or the number of blocks. (N is divisible by M.)


**Output**: an array of length M.

1. **Data Loading**:

    ```cuda
    // Find the start of this block's data in the global array
    float *input_begin = d_input + blockDim.x * blockIdx.x;
    // Initialize shared memory for this block
    __shared__ float input_shared[THREAD_PER_BLOCK];
    // Threadwisely load data from global memory to shared memory
    input_shared[threadIdx.x] = input_begin[threadIdx.x];
    __syncthreads(); // Ensure all threads have written their data to shared memory
    ```

2. **Parallel Reduction:**

    we will compare several optimization strategies below.

    - Baseline (Binary-tree type)
    - No divergence branch
    - No bank conflict
    - Add during Load

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

2. **Key Code**:

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
Assign work to threads based on a stride that aligns with warp boundaries, ensuring all threads in a warp perform similar operations.
1. **A block with 8 threads (Visualization)**:
    
    warp size is 2:

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

2. **Key Code**:

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
    Let us assume `THREAD_PER_BLOCK` is large:
        1. `i = 1`:
            - **Active Threads**: `threadIdx.x < 64 / (2 * 1) = 32` (threads 0 to 31). 
            -  `index = threadIdx.x * 2 * 1 = 2 * threadIdx.x`
            - Thread 0: Accesses `input_shared[0]` (bank 0) and `input_shared[1]` (bank 1).
            - Thread 16: Accesses `input_shared[32]` (bank 0) and `input_shared[33]` (bank 1).
            - Thread N and Thread N+16 have conflict.
        2. `i = 2`:
            - **Active Threads**: `threadIdx.x < 64 / (2 * 2) = 16` (threads 0 to 31). 
            -  `index = threadIdx.x * 2 * 2 = 4 * threadIdx.x`
            - Threads N, N+8, N+16, N+32 have conflicts.
        3. `i = 2^k`, `i <= 16`: 
            - Threads `{N+ n (16/i) | n = 0,1, ...(2i-1)}`  have conflicts.
        4.  `i>=16`:
            - All threads have bank conflicts between each other. 

### No Bank Conflict
To avoid bank conflict, it's better to ensure threads in a warp access consecutive addresses that map to different banks.

1. **A block with 8 threads (Visualization)**:
    
    With warp size 2, addresses 0, 2, 4, 6 are in bank 0, addresses 1,3,5,7 are in bank 1.

    - **Initial state**: Each thread loads one element into shared memory.
        ```
        input_shared: [a0, a1, a2, a3, a4, a5, a6, a7]
        ```

    - **Step size 4**: Threads 0, 1, 2, 3 add elements at indices `threadIdx.x and threadIdx.x + 4` (e.g., thread 0 adds a4 to a0, thread 1 adds a5 to a1).
        ```
        input_shared: [a0+a4, a1+a5, a2+a6, a3+a7, a4, a5, a6, a7]
        ```

    - **Step size 2**: Threads 0, 1 add elements at indices `threadIdx.x` and `threadIdx.x + 2` (e.g., thread 0 adds a0+a4 to a2+a6).
        ```
        input_shared: [a0+a4+a2+a6, a1+a5+a3+a7, a2+a6, a3+a7, a4, a5, a6, a7]
        ```

    - **Step size 1**: Thread 0 adds elements at indices `threadIdx.x ` and `threadIdx.x + 1` (i.e., a0+a4+a2+a6 to a1+a5+a3+a7).
        ```
        input_shared: [a0+a4+a2+a6+a1+a5+a3+a7, a1+a5+a3+a7, a2+a6, a3+a7, a4, a5, a6, a7]
        ```
2. **Key Code**:
    ```cuda
    // half the step size for each iteration
    // for each step size i, threads with indices less than i add the value 
    // at index threadIdx.x to the value at threadIdx.x + i
    for (int i = blockDim.x / 2; i > 0; i /= 2)
    {
        if (threadIdx.x < i)
        {   
            input_shared[threadIdx.x] += input_shared[threadIdx.x + i];
        }
        __syncthreads(); // Synchronize to make sure all threads are done before the next step
    }
    ```
3. **Discussion**:
    - When step size i >= 32, thread `threadIdx.x` access bank `threadIdx.x % 32`, ensuring threads within the same warp access distinct banks.
    - When step size i < 32, data is stored in addresses mapped to different banks, preventing bank conflicts.

### Add During Load
Each thread sums two elements from global memory before storing the result in shared memory

1. **A block with 8 threads (Visualization)**:

    `THREAD_PER_BlOCK` is halved. 

    - **Initial state**: Starting from an array in global memory:
        ```
        input_begin: [a0, a1, a2, a3, a4, a5, a6, a7]
        ```
    
    - **Add during load**: Threads 0, 1, 2, 3 sum elements at indices `threadIdx.x and threadIdx.x + 4` in global memory and then store the result at index `threadIdx.x in the shared memory.
        ```
        input_shared: [a0+a4, a1+a5, a2+a6, a3+a7]
        ```

    - **Step size 2**: Threads 0, 1 add elements at indices `threadIdx.x` and `threadIdx.x + 2` (e.g., thread 0 adds a0+a4 to a2+a6).
        ```
        input_shared: [a0+a4+a2+a6, a1+a5+a3+a7, a2+a6, a3+a7]
        ```

    - **Step size 1**: Thread 0 adds elements at indices `threadIdx.x ` and `threadIdx.x + 1` (i.e., a0+a4+a2+a6 to a1+a5+a3+a7).
        ```
        input_shared: [a0+a4+a2+a6+a1+a5+a3+a7, a1+a5+a3+a7, a2+a6, a3+a7]
        ```

2. **Key code**
    ```cuda
    // Find the start of this block's data in the global array
    // subarray_size = 2 * blockDim.x
    float *input_begin = d_input + blockDim.x * blockIdx.x * 2;
    // Initialize shared memory for this block
    __shared__ float input_shared[THREAD_PER_BLOCK];
    // Each thread sum two elements from global memory into shared memory
    input_shared[threadIdx.x] = input_begin[threadIdx.x] + input_begin[threadIdx.x + blockDim.x];
    __syncthreads(); // Ensure all threads have written their data to shared memory
    ```

3. **Discussion**
    - This strategy reduces shared memory accesses and synchronization, while leaves global memory access unchanged, decreasing the total computation time.
    - It reduce shared memory usage as well as threads per blocks, potentially allowing more blocks to run concurrently on a streaming multiprocessor.

## Reference:
1. [[CUDA]Reduce规约求和（已完结~）](https://www.bilibili.com/video/BV1HvBSY2EJW?spm_id_from=333.788.videopod.episodes&vd_source=aa41d00aebd84e6f99f529df7f83258a)
2. [深入浅出GPU优化系列：reduce优化](https://zhuanlan.zhihu.com/p/426978026)