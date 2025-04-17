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

__global__ void forwardKernalHidden(NeuralNetworkDevice *net_dev, float *image, float *hidden) {
    
    int neuron_no = threadIdx.x;
    int g_index = INPUT_SIZE * neuron_no; // suspicious line

    __shared__ float shared_image[INPUT_SIZE];
    int one_part = INPUT_SIZE / HIDDEN_SIZE;

    for (int i = one_part * neuron_no; i < (one_part * (neuron_no + 1)); i++)
        shared_image[i] = image[i];

    if (neuron_no == 127) {
        for (int i = 768; i < 784; i++)
            shared_image[i] = image[i];
    }
    __syncthreads();

    float ans = net_dev->b1[neuron_no];
    for (int j = 0; j < INPUT_SIZE; j++)
        ans += net_dev->W1[g_index + j] * shared_image[j];

    hidden[neuron_no] = (ans > 0) ? ans : 0;
    // hidden[neuron_no] = fmax(ans, 0.0);

}
__global__ void forwardKernalOutput(NeuralNetworkDevice *net_dev, float *hidden, float *output_device) {
   
    int neuron_no = threadIdx.x;
    __shared__ float shared_hidden[OUTPUT_SIZE];

    int one_part = HIDDEN_SIZE / OUTPUT_SIZE;
    
    for (int i = one_part * neuron_no; i < (one_part * (neuron_no + 1)); i++)
        shared_hidden[i] = hidden[i];
    if (neuron_no == 9) {
        for (int i = 120; i < 128; i++)
            shared_hidden[i] = hidden[i];
    }
    __syncthreads();

    int g_index = HIDDEN_SIZE * neuron_no; // suspicious line

    float ans = net_dev->b2[neuron_no];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        ans += net_dev->W2[g_index + j] * shared_hidden[j];
    }

    output_device[neuron_no] = (ans > 0) ? ans : 0;
}

__global__ void applySoftMaxDevice(float *output_device) {

    float sum = 0;
    float exps[10];
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        exps[i] =  output_device[i];
        exps[i] = exp(exps[i]);
        sum += exps[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_device[i] = exps[i] /  sum;
    }
}

__global__ void computerOutputGradient(float *output, float *target, float *d_output_device) {
    d_output_device[threadIdx.x] = output[threadIdx.x] - target[threadIdx.x];
}

__global__ void computerHiddenGradient(NeuralNetworkDevice *net, float *d_output_device, float *d_hidden_device, float *hidden) {

    int i = threadIdx.x;

    float ans = 0;
    for (int j = 0; j < OUTPUT_SIZE; j++)
        ans += net->W2[j * HIDDEN_SIZE + i] * d_output_device[j];
    d_hidden_device[i] = ans * (hidden[i] > 0);
}

__global__ void updateOutputLayer(NeuralNetworkDevice *net, float *d_output_device, float *hidden) {

    int i = blockIdx.x;
    int j = threadIdx.x;
    net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output_device[i] * hidden[j];

    if (threadIdx.x == 0)
        net->b2[i] -= LEARNING_RATE * d_output_device[i];
}
__global__ void updateHiddenLayer(NeuralNetworkDevice *net, float *d_hidden_device, float *input) {

    int i = blockIdx.x;
    int j = threadIdx.x;
    net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden_device[i] * input[j];

    if (threadIdx.x == 0)
        net->b1[i] -= LEARNING_RATE * d_hidden_device[i];
}

void backwardDevice(NeuralNetworkDevice *net, float *input, float *hidden, float *output, float *target, float *d_hidden_device, float *d_output_device) {

    computerOutputGradient<<<1, OUTPUT_SIZE>>>(output, target, d_output_device);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

    computerHiddenGradient<<<1, HIDDEN_SIZE>>>(net, d_output_device, d_hidden_device, hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();

    updateOutputLayer<<<OUTPUT_SIZE, HIDDEN_SIZE>>>(net, d_output_device, hidden);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    updateHiddenLayer<<<HIDDEN_SIZE, INPUT_SIZE>>>(net, d_hidden_device, input);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaDeviceSynchronize();
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
    if (pred == actual)
        (*correct_device)++;
}
// Train network
void train(NeuralNetwork *net, NeuralNetworkDevice *net_device, float **images, float **labels, int numImages) {
    float *images_dev;

    float *hidden_device;
    float *output_device;
    float *labels_dev;

    float *d_hidden_device;
    float *d_output_device;

    allocateMatrixDevice(&images_dev, numImages, INPUT_SIZE);
    allocateMatrixDevice(&labels_dev, numImages, OUTPUT_SIZE);

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
        float loss = 0.0;
        int correct = 0;

        if (cudaMemset(loss_device, 0, sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "Error initializing loss_device\n");
            exit(EXIT_FAILURE);
        }
        if (cudaMemset(correct_device, 0, sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "Error initializing correct_device\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < numImages; i++) {

            cudaEvent_t start, stop;
            float elapsed;

            // cudaEventCreate(&start);
            // cudaEventCreate(&stop);

            // cudaEventRecord(start);
            forwardKernalHidden<<<1, HIDDEN_SIZE>>>(net_device, images_dev + (i * INPUT_SIZE), hidden_device);
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            if (cudaDeviceSynchronize() != cudaSuccess) {
                fprintf(stderr, "Error in cudaDeviceSynchronize\n");
                exit(EXIT_FAILURE);
            }

            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&elapsed, start, stop);
            // printf("forwardKernalHidden: %.3f ms\n", elapsed);

            // cudaEventRecord(start);
            forwardKernalOutput<<<1, OUTPUT_SIZE>>>(net_device, hidden_device, output_device);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            // cudaDeviceSynchronize();
            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&elapsed, start, stop);
            // printf("forwardKernaOutput: %.3f ms\n", elapsed);

            // cudaEventRecord(start);
            applySoftMaxDevice<<<1, 1>>>(output_device);
            
            cudaDeviceSynchronize();
            
            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&elapsed, start, stop);
            // printf("applySoftMaxDevice: %.3f ms\n", elapsed);

            // cudaEventRecord(start);
            backwardDevice(net_device, images_dev + (i * INPUT_SIZE), hidden_device, output_device, labels_dev + (i * OUTPUT_SIZE), d_hidden_device, d_output_device);
            
            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&elapsed, start, stop);
            // printf("backwardDevice: %.3f ms\n", elapsed);
            
            
            // cudaEventRecord(start);
            findCorrect<<<1, 1>>>(loss_device, correct_device, output_device, labels_dev + (i * OUTPUT_SIZE));
            cudaDeviceSynchronize();
           
            // cudaEventRecord(stop);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&elapsed, start, stop);
            // printf("findCorrect: %.3f ms\n", elapsed);
            // brak;
        }

        if (cudaMemcpy(&loss, loss_device, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error in copying loss from device to host");
        }

        if (cudaMemcpy(&correct, correct_device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error in copying correct from device to host");
        }
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (float)numImages) * 100, get_time(epoch_start));
    }

    NeuralNetworkDevice net_dev;
    cudaMemcpy(&net_dev, net_device, sizeof(NeuralNetworkDevice), cudaMemcpyDeviceToHost);

    copyMatrixtoHost(net->W1, net_dev.W1, HIDDEN_SIZE, INPUT_SIZE);
    copyMatrixtoHost(net->W2, net_dev.W2, OUTPUT_SIZE, HIDDEN_SIZE);

    cudaMemcpy(net->b1, net_dev.b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->b2, net_dev.b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Total training time: %.3fs\n", get_time(total_start));
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