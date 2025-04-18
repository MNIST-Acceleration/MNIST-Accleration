#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>;

#define CHECK_CUDA(call)                                                    
{                                                                       
    cudaError_t err = call;                                             
    if (err != cudaSuccess) {                                           
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << "\n";
        exit(EXIT_FAILURE);                                             
    }                                                                   
}

#define CHECK_CUBLAS(call)                                                  
{                                                                       
    cublasStatus_t stat = call;                                         
    if (stat != CUBLAS_STATUS_SUCCESS) {                                
        std::cerr << "cuBLAS error\n";                                  
        exit(EXIT_FAILURE);                                             
    }                                                                   
}

int main() {
    const int N = 128;  // Hidden layer size
    const int input_size = 784;  // Input size (28x28 flattened)

    // Host matrices in half precision (FP16)
    half input[input_size];      // 1 x 784
    half weights[input_size * N];  // 784 x 128
    half bias[N];                // 1 x 128
    half output[N];              // 1 x 128

    // Initialize matrices with ones
    for (int i = 0; i < input_size; ++i) {
        input[i] = __float2half(1.0f);  // All ones
    }

    for (int i = 0; i < input_size * N; ++i) {
        weights[i] = __float2half(1.0f);  // All ones
    }

    for (int i = 0; i < N; ++i) {
        bias[i] = __float2half(1.0f);  // All ones
    }

    // Device pointers
    half *d_input, *d_weights, *d_bias, *d_output;

    CHECK_CUDA(cudaMalloc(&d_input, sizeof(half) * input_size));
    CHECK_CUDA(cudaMalloc(&d_weights, sizeof(half) * input_size * N));
    CHECK_CUDA(cudaMalloc(&d_bias, sizeof(half) * N));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(half) * N));

    CHECK_CUDA(cudaMemcpy(d_input, input, sizeof(half) * input_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, weights, sizeof(half) * input_size * N, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_bias, bias, sizeof(half) * N, cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    // Enable Tensor Core operations
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    float alpha = 1.0f;
    float beta = 0.0f;

    // Matrix multiplication: output = input * weights + bias
    // Using cublasGemmEx: C = alpha * A * B + beta * C
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,  // No transpose on A and B
        N, 1, input_size,          // M = N, N = 1 (batch size), K = input_size (784)
        &alpha,
        d_weights, CUDA_R_16F, input_size,  // Weights (784 x 128), leading dimension = input_size
        d_input, CUDA_R_16F, input_size,   // Input (1 x 784), leading dimension = input_size
        &beta,
        d_output, CUDA_R_16F, N,  // Output (1 x 128), leading dimension = N (hidden size)
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // Add the bias to the output (output += bias)
    // Bias is a vector and needs to be added to each element in the output vector
    float alpha_bias = 1.0f;
    CHECK_CUBLAS(cublasSaxpy(handle, N, &alpha_bias, d_bias, 1, d_output, 1));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(output, d_output, sizeof(half) * N, cudaMemcpyDeviceToHost));

    // Print result
    std::cout << "Output of the first hidden layer (using Tensor Cores):\n";
    for (int i = 0; i < N; ++i) {
        std::cout << __half2float(output[i]) << " ";
        if ((i + 1) % 16 == 0) std::cout << std::endl;
    }

    // Cleanup
    cudaFree(d_input); 
    cudaFree(d_weights); 
    cudaFree(d_bias); 
    cudaFree(d_output); 
    cublasDestroy(handle);

    return 0;
}
