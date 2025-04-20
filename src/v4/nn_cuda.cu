#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 64
#define NUM_CLASSES 10

#define CHECK_CUDA(call)                                                         \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    }

#define CHECK_CUBLAS(call)                                             \
    {                                                                  \
        cublasStatus_t status = call;                                  \
        if (status != CUBLAS_STATUS_SUCCESS) {                         \
            std::cerr << "cuBLAS error code: " << status << std::endl; \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    }

// Timer function
float get_time(clock_t start) {
    return (float)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
float **allocateMatrix(int rows, int cols) {
    float **mat = (float **)malloc(rows * sizeof(float *));
    for (int i = 0; i < rows; i++) {
        mat[i] = (float *)malloc(cols * sizeof(float));
    }
    return mat;
}
void allocateMatrixDevice(float **mat, int rows, int cols) {
    if (cudaMalloc((void **)mat, rows * cols * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for matrix\n");
        exit(EXIT_FAILURE);
    }
}

// Free allocated matrix memory
void freeMatrix(float **mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Activation functions
void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(float *x, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        x[i] = exp(x[i]);
        sum += x[i];
    }
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

// Neural network structure
typedef struct {
    float **W1;
    float **W2;
    float *b1;
    float *b2;
} NeuralNetwork;

typedef struct {
    float *W1;
    float *W2;
    float *b1;
    float *b2;
} NeuralNetworkDevice;

void copyMatrixtoDevice(float **mat, float *mat_dev, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        if (cudaMemcpy(mat_dev + (i * cols), mat[i], sizeof(float) * cols, cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "Error copying matrix to device\n");
            exit(EXIT_FAILURE);
        }
    }
}

void copyMatrixtoHost(float **mat, float *mat_dev, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        if (cudaMemcpy(mat[i], mat_dev + (i * cols), sizeof(float) * cols, cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("%d Error in copying copyMatrixtoHost\n", i);
        }
    }
}

// Initialize neural network
NeuralNetwork *createNetwork() {
    NeuralNetwork *net = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (float *)calloc(HIDDEN_SIZE, sizeof(float));
    net->b2 = (float *)calloc(OUTPUT_SIZE, sizeof(float));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((float)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((float)rand() / RAND_MAX) * 0.01;

    return net;
}

// Initialize neural network
NeuralNetworkDevice *createNetworkDevice(NeuralNetwork *networkHost) {
    NeuralNetworkDevice *net;
    NeuralNetworkDevice netHost;

    float *w1;
    float *w2;
    float *b1;
    float *b2;

    if (cudaMalloc((void **)&net, sizeof(NeuralNetworkDevice)) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for network structure\n");
        exit(EXIT_FAILURE);
    }

    allocateMatrixDevice(&(w1), HIDDEN_SIZE, INPUT_SIZE);
    allocateMatrixDevice(&(w2), OUTPUT_SIZE, HIDDEN_SIZE);

    copyMatrixtoDevice(networkHost->W1, w1, HIDDEN_SIZE, INPUT_SIZE);
    copyMatrixtoDevice(networkHost->W2, w2, OUTPUT_SIZE, HIDDEN_SIZE);

    if (cudaMalloc((void **)&b1, HIDDEN_SIZE * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for b1\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&b2, OUTPUT_SIZE * sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for b2\n");
        exit(EXIT_FAILURE);
    }

    cudaMemset(b1, 0, HIDDEN_SIZE * sizeof(float));
    cudaMemset(b2, 0, OUTPUT_SIZE * sizeof(float));

    netHost.W1 = w1;
    netHost.W2 = w2;
    netHost.b1 = b1;
    netHost.b2 = b2;

    if (cudaMemcpy(net, &netHost, sizeof(NeuralNetworkDevice), cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Error copying network structure to device\n");
        exit(EXIT_FAILURE);
    }
    printf("network created\n");
    return net;
}

void forward(NeuralNetwork *net, float *input, float *hidden, float *output) {
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        hidden[i] = net->b1[i];
        for (int j = 0; j < INPUT_SIZE; j++)
            hidden[i] += net->W1[i][j] * input[j];
    }
    relu(hidden, HIDDEN_SIZE);

    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output[i] = net->b2[i];
        for (int j = 0; j < HIDDEN_SIZE; j++)
            output[i] += net->W2[i][j] * hidden[j];
    }
    softmax(output, OUTPUT_SIZE);
}

__global__ void computerOutputGradient(float *output, float *target, float *d_output_device) {
    d_output_device[threadIdx.x] = output[threadIdx.x] - target[threadIdx.x];
}


__global__ void reluBackward(
    const float *hidden,
    float *d_hidden_device) {

    int i = threadIdx.x;
    d_hidden_device[i] = (hidden[i] > 0.0f) ? d_hidden_device[i] : 0.0f;
}

__global__ void updateOutputLayer(NeuralNetworkDevice *net, float *d_output_device, float *hidden) {

    int i = blockIdx.x;
    int j = threadIdx.x;
    net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output_device[i] * hidden[j];

    net->b2[i] -= (threadIdx.x == 0) * LEARNING_RATE * d_output_device[i];
}
__global__ void updateHiddenLayer(NeuralNetworkDevice *net, float *d_hidden_device, float *input) {

    int i = blockIdx.x;
    int j = threadIdx.x;
    net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden_device[i] * input[j];

    net->b1[i] -= (threadIdx.x == 0) * LEARNING_RATE * d_hidden_device[i];
}

__global__ void updateBias(float *b2, float *d_output) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    b2[i] -= LEARNING_RATE * d_output[i];
}

void backwardDevice(cublasHandle_t handle, NeuralNetworkDevice net_obj, NeuralNetworkDevice *net, float *input, float *hidden, float *output, float *target, float *d_hidden_device, float *d_output_device) {

    float alpha = 1;
    float beta = 0;

    computerOutputGradient<<<1, OUTPUT_SIZE>>>(output, target, d_output_device);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

    // calculating the gradient of hidden layer
    CHECK_CUBLAS(cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        1, HIDDEN_SIZE, OUTPUT_SIZE,
        &alpha,
        d_output_device, CUDA_R_32F, 1,
        net_obj.W2, CUDA_R_32F, HIDDEN_SIZE,
        &beta,
        d_hidden_device, CUDA_R_32F, 1,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    reluBackward<<<1, HIDDEN_SIZE>>>(hidden, d_hidden_device);

    updateOutputLayer<<<OUTPUT_SIZE, HIDDEN_SIZE>>>(net, d_output_device, hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // updating the output layer weights

    updateHiddenLayer<<<HIDDEN_SIZE, INPUT_SIZE>>>(net, d_hidden_device, input);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    alpha = -LEARNING_RATE;
    beta = 1.0f;
    // CHECK_CUBLAS(cublasGemmEx(
    //     handle,
    //     CUBLAS_OP_T, CUBLAS_OP_N,
    //     HIDDEN_SIZE, OUTPUT_SIZE, 1,
    //     &alpha,
    //     hidden, CUDA_R_32F, HIDDEN_SIZE,
    //     d_output_device, CUDA_R_32F, 1,
    //     &beta,
    //     net_obj.W2, CUDA_R_32F, OUTPUT_SIZE,
    //     CUDA_R_32F,
    //     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    
    // // // Update output layer bias
    // CHECK_CUBLAS(cublasSaxpy(
    //     handle,
    //     OUTPUT_SIZE,
    //     &alpha,
    //     d_output_device, 1,
    //     net_obj.b2, 1));
    
    // Replace updateHiddenLayer kernel with tensor core optimized version
    // Original: updateHiddenLayer<<<HIDDEN_SIZE, INPUT_SIZE>>>(net, d_hidden_device, input);
    
    // Update hidden layer weights using tensor cores
    // CHECK_CUBLAS(cublasGemmEx(
    //     handle,
    //     CUBLAS_OP_N, CUBLAS_OP_T,
    //     HIDDEN_SIZE, INPUT_SIZE, 1,
    //     &alpha,
    //     d_hidden_device, CUDA_R_32F, HIDDEN_SIZE,
    //     input, CUDA_R_32F, INPUT_SIZE,
    //     &beta,
    //     net_obj.W1, CUDA_R_32F, HIDDEN_SIZE,
    //     CUDA_R_32F,
    //     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    
    // // Update hidden layer bias
    // CHECK_CUBLAS(cublasSaxpy(
    //     handle,
    //     HIDDEN_SIZE,
    //     &alpha,
    //     d_hidden_device, 1,
    //     net_obj.b1, 1));

  
}

__global__ void findCorrect(float *loss_device, int *correct_device, float *output, float *labels_dev) {

    float loss_loc = *loss_device;
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        loss_loc -= labels_dev[k] * log(output[k]);
    }
    *loss_device = loss_loc;
    int pred = 0, actual = 0;

    for (int j = 0; j < OUTPUT_SIZE; j++) {
        if (output[j] > output[pred])
            pred = j;
        if (labels_dev[j] > labels_dev[actual])
            actual = j;
    }

    (*correct_device) += (pred == actual);
}

__global__ void addBias(float *hidden, const float *bias) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    hidden[idx] += bias[idx];
    hidden[idx] = hidden[idx] > 0 ? hidden[idx] : 0.0;
}

__global__ void addBiasForHidden(float *output, const float *bias) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float ans = output[idx];
    ans += bias[idx];

    ans = (ans > 0) * ans;

    __shared__ float sum;

    int expv = exp(ans);
    if (idx == 0)
        sum = 0.0f;
    __syncthreads();
    atomicAdd(&sum, expv);
    __syncthreads();
    output[idx] = expv / sum;
}

void train(NeuralNetwork *net, NeuralNetworkDevice *net_device, float **images, float **labels, int numImages) {
    float *images_dev;

    float *hidden_device;
    float *output_device;
    float *labels_dev;

    float *d_hidden_device;
    float *d_output_device;

    NeuralNetworkDevice net_device_obs; // for assessing the device network on host side;
    cudaMemcpy(&net_device_obs, net_device, sizeof(NeuralNetworkDevice), cudaMemcpyDeviceToHost);

    allocateMatrixDevice(&images_dev, numImages, INPUT_SIZE);
    allocateMatrixDevice(&labels_dev, numImages, OUTPUT_SIZE);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH))

    if (cudaMalloc((void **)&hidden_device, sizeof(float) * HIDDEN_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for hidden layer\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMalloc((void **)&output_device, sizeof(float) * OUTPUT_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for output layer\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMalloc((void **)&d_hidden_device, sizeof(float) * HIDDEN_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for hidden layer gradient\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&d_output_device, sizeof(float) * OUTPUT_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for output layer gradient\n");
        exit(EXIT_FAILURE);
    }

    float *hidden1 = (float *)malloc(sizeof(float) * HIDDEN_SIZE);
    float *hidden2 = (float *)malloc(sizeof(float) * HIDDEN_SIZE);

    copyMatrixtoDevice(images, images_dev, numImages, INPUT_SIZE);
    copyMatrixtoDevice(labels, labels_dev, numImages, OUTPUT_SIZE);

    float *loss_device;
    int *correct_device;

    if (cudaMalloc((void **)&loss_device, sizeof(float)) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for loss\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&correct_device, sizeof(int)) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for correct\n");
        exit(EXIT_FAILURE);
    }
    clock_t total_start = clock();
    int epoch = 0;
    for (; epoch < EPOCHS; epoch++) {
        clock_t epoch_start = clock();
        float loss = 0.0f;
        int correct = 0;

        float alpha = 1.0f;
        float beta = 0.0f;

        if (cudaMemset(loss_device, 0, sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "Error initializing loss_device\n");
            exit(EXIT_FAILURE);
        }
        if (cudaMemset(correct_device, 0, sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "Error initializing correct_device\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < numImages; i++) {

            cudaError_t err = cudaGetLastError();
            cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                1, HIDDEN_SIZE, INPUT_SIZE,
                &alpha,
                images_dev + (i * INPUT_SIZE), CUDA_R_32F, 1,
                net_device_obs.W1, CUDA_R_32F, INPUT_SIZE,
                &beta,
                hidden_device, CUDA_R_32F, 1,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            addBias<<<1, HIDDEN_SIZE>>>(hidden_device, net_device_obs.b1);
            cudaDeviceSynchronize();

            cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                1, OUTPUT_SIZE, HIDDEN_SIZE,
                &alpha,
                hidden_device, CUDA_R_32F, 1,
                net_device_obs.W2, CUDA_R_32F, HIDDEN_SIZE,
                &beta,
                output_device, CUDA_R_32F, 1,
                CUDA_R_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP);

            addBiasForHidden<<<1, OUTPUT_SIZE>>>(output_device, net_device_obs.b2);
            cudaDeviceSynchronize();

            backwardDevice(handle, net_device_obs, net_device, images_dev + (i * INPUT_SIZE), hidden_device, output_device, labels_dev + (i * OUTPUT_SIZE), d_hidden_device, d_output_device);
            findCorrect<<<1, 1>>>(loss_device, correct_device, output_device, labels_dev + (i * OUTPUT_SIZE));
            cudaDeviceSynchronize();
        }
        if (cudaMemcpy(&loss, loss_device, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error in copying loss from device to host");
        }

        if (cudaMemcpy(&correct, correct_device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error in copying correct from device to host");
        }
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }

    printf("Total training time: %.3fs\n", get_time(total_start));
    NeuralNetworkDevice net_dev;
    cudaMemcpy(&net_dev, net_device, sizeof(NeuralNetworkDevice), cudaMemcpyDeviceToHost);

    copyMatrixtoHost(net->W1, net_dev.W1, HIDDEN_SIZE, INPUT_SIZE);
    copyMatrixtoHost(net->W2, net_dev.W2, OUTPUT_SIZE, HIDDEN_SIZE);

    cudaMemcpy(net->b1, net_dev.b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->b2, net_dev.b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork *net, float **images, float **labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        float hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
        forward(net, images[i], hidden, output);
        int pred = 0, actual = 0;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            if (output[j] > output[pred])
                pred = j;
            if (labels[i][j] > labels[i][actual])
                actual = j;
        }
        if (pred == actual)
            correct++;
    }
    printf("Test Accuracy: %.2f%%\n", (correct / (float)numImages) * 100);
}

// Read MNIST dataset
float **loadMNISTImages(const char *filename, int numImages) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    float **images = allocateMatrix(numImages, INPUT_SIZE);
    for (int i = 0; i < numImages; i++) {
        for (int j = 0; j < INPUT_SIZE; j++) {
            unsigned char pixel;

            // fread(&pixel, sizeof(unsigned char), 1, file);
            if (fread(&pixel, sizeof(unsigned char), 1, file) != 1) {
                fprintf(stderr, "Error: Failed to read pixel\n");
                fclose(file);
                exit(EXIT_FAILURE);
            }

            images[i][j] = pixel / 255.0;
        }
    }
    fclose(file);
    return images;
}

float **loadMNISTLabels(const char *filename, int numLabels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    float **labels = allocateMatrix(numLabels, OUTPUT_SIZE);
    for (int i = 0; i < numLabels; i++) {
        unsigned char label;
        // fread(&label, sizeof(unsigned char), 1, file);
        if (fread(&label, sizeof(unsigned char), 1, file) != 1) {
            fprintf(stderr, "Error: Failed to read label\n");
            fclose(file);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < OUTPUT_SIZE; j++) {
            labels[i][j] = (j == label) ? 1.0 : 0.0;
        }
    }
    fclose(file);
    return labels;
}

// Free network memory
void freeNetwork(NeuralNetwork *net) {
    freeMatrix(net->W1, HIDDEN_SIZE);
    freeMatrix(net->W2, OUTPUT_SIZE);
    free(net->b1);
    free(net->b2);
    free(net);
}

// Main function
int main() {
    printf("MNIST Neural Network\n\n");

    float **train_images = loadMNISTImages("./../../data/train-images.idx3-ubyte", 60000);
    float **train_labels = loadMNISTLabels("./../../data/train-labels.idx1-ubyte", 60000);
    float **test_images = loadMNISTImages("./../../data/t10k-images.idx3-ubyte", 10000);
    float **test_labels = loadMNISTLabels("./../../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork *net = createNetwork();
    NeuralNetworkDevice *net_device = createNetworkDevice(net);
    train(net, net_device, train_images, train_labels, 60000);

    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    return 0;
}