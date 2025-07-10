add_executable(my_reduce_v1_shared_memory my_reduce_v1_shared_memory.cu)

target_link_libraries(my_reduce_v1_shared_memory PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v1_shared_memory PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_reduce_v2_no_divergence_branch my_reduce_v2_no_divergence_branch.cu)

target_link_libraries(my_reduce_v2_no_divergence_branch PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v2_no_divergence_branch PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_reduce_v3_no_bank_conflict my_reduce_v3_no_bank_conflict.cu)

target_link_libraries(my_reduce_v3_no_bank_conflict PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v3_no_bank_conflict PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

add_executable(my_reduce_v4_add_during_load my_reduce_v4_add_during_load.cu)

target_link_libraries(my_reduce_v4_add_during_load PRIVATE CUDA::cudart ${CUDA_cublas_LIBRARY})
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(my_reduce_v4_add_during_load PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
endif()

