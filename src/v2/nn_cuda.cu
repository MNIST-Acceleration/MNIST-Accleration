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
double get_time(clock_t start) {
    return (double)(clock() - start) / CLOCKS_PER_SEC;
}

// Allocate memory for a matrix
double **allocateMatrix(int rows, int cols) {
    double **mat = (double **)malloc(rows * sizeof(double *));
    for (int i = 0; i < rows; i++) {
        mat[i] = (double *)malloc(cols * sizeof(double));
    }
    return mat;
}
void allocateMatrixDevice(double **mat, int rows, int cols) {
    if (cudaMalloc((void **)mat, rows * cols * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for matrix\n");
        exit(EXIT_FAILURE);
    }
}

// Free allocated matrix memory
void freeMatrix(double **mat, int rows) {
    for (int i = 0; i < rows; i++) {
        free(mat[i]);
    }
    free(mat);
}

// Activation functions
void relu(double *x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = (x[i] > 0) ? x[i] : 0;
    }
}

void softmax(double *x, int size) {
    double sum = 0;
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
    double **W1;
    double **W2;
    double *b1;
    double *b2;
} NeuralNetwork;

typedef struct {
    double *W1;
    double *W2;
    double *b1;
    double *b2;
} NeuralNetworkDevice;

void copyMatrixtoDevice(double **mat, double *mat_dev, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        if (cudaMemcpy(mat_dev + (i * cols), mat[i], sizeof(double) * cols, cudaMemcpyHostToDevice) != cudaSuccess) {
            fprintf(stderr, "Error copying matrix to device\n");
            exit(EXIT_FAILURE);
        }
    }
}

void copyMatrixtoHost(double **mat, double *mat_dev, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        if (cudaMemcpy(mat[i], mat_dev + (i * cols), sizeof(double) * cols, cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("%d Error in copying copyMatrixtoHost\n", i);
        }
    }
}

// Initialize neural network
NeuralNetwork *createNetwork() {
    NeuralNetwork *net = (NeuralNetwork *)malloc(sizeof(NeuralNetwork));
    net->W1 = allocateMatrix(HIDDEN_SIZE, INPUT_SIZE);
    net->W2 = allocateMatrix(OUTPUT_SIZE, HIDDEN_SIZE);
    net->b1 = (double *)calloc(HIDDEN_SIZE, sizeof(double));
    net->b2 = (double *)calloc(OUTPUT_SIZE, sizeof(double));

    srand(time(NULL));
    for (int i = 0; i < HIDDEN_SIZE; i++)
        for (int j = 0; j < INPUT_SIZE; j++)
            net->W1[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    for (int i = 0; i < OUTPUT_SIZE; i++)
        for (int j = 0; j < HIDDEN_SIZE; j++)
            net->W2[i][j] = ((double)rand() / RAND_MAX) * 0.01;

    return net;
}

// Initialize neural network
NeuralNetworkDevice *createNetworkDevice(NeuralNetwork *networkHost) {
    NeuralNetworkDevice *net;
    NeuralNetworkDevice netHost;

    double *w1;
    double *w2;
    double *b1;
    double *b2;

    if (cudaMalloc((void **)&net, sizeof(NeuralNetworkDevice)) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for network structure\n");
        exit(EXIT_FAILURE);
    }

    allocateMatrixDevice(&(w1), HIDDEN_SIZE, INPUT_SIZE);
    allocateMatrixDevice(&(w2), OUTPUT_SIZE, HIDDEN_SIZE);

    copyMatrixtoDevice(networkHost->W1, w1, HIDDEN_SIZE, INPUT_SIZE);
    copyMatrixtoDevice(networkHost->W2, w2, OUTPUT_SIZE, HIDDEN_SIZE);

    if (cudaMalloc((void **)&b1, HIDDEN_SIZE * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for b1\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&b2, OUTPUT_SIZE * sizeof(double)) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for b2\n");
        exit(EXIT_FAILURE);
    }

    cudaMemset(b1, 0, HIDDEN_SIZE * sizeof(double));
    cudaMemset(b2, 0, OUTPUT_SIZE * sizeof(double));

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

void forward(NeuralNetwork *net, double *input, double *hidden, double *output) {
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

__global__ void forwardKernalHidden(NeuralNetworkDevice *net_dev, double *image, double *hidden) {
    int neuron_no = threadIdx.x;
    int g_index = INPUT_SIZE * neuron_no; // suspicious line

    hidden[neuron_no] = net_dev->b1[neuron_no];
    for (int j = 0; j < INPUT_SIZE; j++)
        hidden[neuron_no] += net_dev->W1[g_index + j] * image[j];

    hidden[neuron_no] = (hidden[neuron_no] > 0) ? hidden[neuron_no] : 0;
}
__global__ void forwardKernalOutput(NeuralNetworkDevice *net_dev, double *hidden, double *output_device) {
    int neuron_no = threadIdx.x;
    int g_index = HIDDEN_SIZE * neuron_no; // suspicious line

    double ans = net_dev->b2[neuron_no];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
        ans += net_dev->W2[g_index + j] * hidden[j];
    }
    // printf("ans = %lf \n", ans);
    // softmax(output, OUTPUT_SIZE);

    output_device[neuron_no] = (ans > 0) ? ans : 0;
}

__global__ void applySoftMaxDevice(double *output_device) {

    double sum = 0;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_device[i] = exp(output_device[i]);
        sum += output_device[i];
    }
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        output_device[i] /= sum;
    }
}

__global__ void computerOutputGradient(double *output, double *target, double *d_output_device) {
    d_output_device[threadIdx.x] = output[threadIdx.x] - target[threadIdx.x];
}

__global__ void computerHiddenGradient(NeuralNetworkDevice *net, double *d_output_device, double *d_hidden_device, double *hidden) {

    int i = threadIdx.x;
    d_hidden_device[i] = 0;
    for (int j = 0; j < OUTPUT_SIZE; j++)
        d_hidden_device[i] += net->W2[j * HIDDEN_SIZE + i] * d_output_device[j];
    d_hidden_device[i] *= (hidden[i] > 0);
}

__global__ void updateOutputLayer(NeuralNetworkDevice *net, double *d_output_device, double *hidden) {

    int i = blockIdx.x;
    int j = threadIdx.x;
    net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * d_output_device[i] * hidden[j];

    if (threadIdx.x == 0)
        net->b2[i] -= LEARNING_RATE * d_output_device[i];
}
__global__ void updateHiddenLayer(NeuralNetworkDevice *net, double *d_hidden_device, double *input) {

    int i = blockIdx.x;
    int j = threadIdx.x;
    net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * d_hidden_device[i] * input[j];

    if (threadIdx.x == 0)
        net->b1[i] -= LEARNING_RATE * d_hidden_device[i];
}

void backwardDevice(NeuralNetworkDevice *net, double *input, double *hidden, double *output, double *target, double *d_hidden_device, double *d_output_device) {

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

__global__ void findCorrect(double *loss_device, int *correct_device, double *output, double *labels_dev) {
    for (int k = 0; k < OUTPUT_SIZE; k++) {
        *loss_device -= labels_dev[k] * log(output[k]);
    }
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
void train(NeuralNetwork *net, NeuralNetworkDevice *net_device, double **images, double **labels, int numImages) {
    double *images_dev;

    double *hidden_device;
    double *output_device;
    double *labels_dev;

    double *d_hidden_device;
    double *d_output_device;

    allocateMatrixDevice(&images_dev, numImages, INPUT_SIZE);
    allocateMatrixDevice(&labels_dev, numImages, OUTPUT_SIZE);

    if (cudaMalloc((void **)&hidden_device, sizeof(double) * HIDDEN_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for hidden layer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&output_device, sizeof(double) * OUTPUT_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for output layer\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMalloc((void **)&d_hidden_device, sizeof(double) * HIDDEN_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for hidden layer gradient\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&d_output_device, sizeof(double) * OUTPUT_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for output layer gradient\n");
        exit(EXIT_FAILURE);
    }

    copyMatrixtoDevice(images, images_dev, numImages, INPUT_SIZE);
    copyMatrixtoDevice(labels, labels_dev, numImages, OUTPUT_SIZE);

    double *loss_device;
    int *correct_device;

    if (cudaMalloc((void **)&loss_device, sizeof(double)) != cudaSuccess) {
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
        double loss = 0.0;
        int correct = 0;

        if (cudaMemset(loss_device, 0, sizeof(double)) != cudaSuccess) {
            fprintf(stderr, "Error initializing loss_device\n");
            exit(EXIT_FAILURE);
        }
        if (cudaMemset(correct_device, 0, sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "Error initializing correct_device\n");
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < numImages; i++) {

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

            forwardKernalOutput<<<1, OUTPUT_SIZE>>>(net_device, hidden_device, output_device);
            err = cudaGetLastError();
            if (err != cudaSuccess) {
                fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            cudaDeviceSynchronize();
            applySoftMaxDevice<<<1, 1>>>(output_device);
            cudaDeviceSynchronize();

            backwardDevice(net_device, images_dev + (i * INPUT_SIZE), hidden_device, output_device, labels_dev + (i * OUTPUT_SIZE), d_hidden_device, d_output_device);
            cudaDeviceSynchronize();
            findCorrect<<<1, 1>>>(loss_device, correct_device, output_device, labels_dev + (i * OUTPUT_SIZE));
            cudaDeviceSynchronize();
            // break;
        }

        if (cudaMemcpy(&loss, loss_device, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error in copying loss from device to host");
        }

        if (cudaMemcpy(&correct, correct_device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error in copying correct from device to host");
        }
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (double)numImages) * 100, get_time(epoch_start));
    }

    NeuralNetworkDevice net_dev;
    cudaMemcpy(&net_dev, net_device, sizeof(NeuralNetworkDevice), cudaMemcpyDeviceToHost);

    copyMatrixtoHost(net->W1, net_dev.W1, HIDDEN_SIZE, INPUT_SIZE);
    copyMatrixtoHost(net->W2, net_dev.W2, OUTPUT_SIZE, HIDDEN_SIZE);

    cudaMemcpy(net->b1, net_dev.b1, HIDDEN_SIZE * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->b2, net_dev.b2, OUTPUT_SIZE * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy on test data
void evaluate(NeuralNetwork *net, double **images, double **labels, int numImages) {
    int correct = 0;
    for (int i = 0; i < numImages; i++) {
        double hidden[HIDDEN_SIZE], output[OUTPUT_SIZE];
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
    printf("Test Accuracy: %.2f%%\n", (correct / (double)numImages) * 100);
}

// Read MNIST dataset
double **loadMNISTImages(const char *filename, int numImages) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 16, SEEK_SET);
    double **images = allocateMatrix(numImages, INPUT_SIZE);
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

double **loadMNISTLabels(const char *filename, int numLabels) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        printf("Error opening %s\n", filename);
        exit(1);
    }
    fseek(file, 8, SEEK_SET);
    double **labels = allocateMatrix(numLabels, OUTPUT_SIZE);
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

    double **train_images = loadMNISTImages("./../../data/train-images.idx3-ubyte", 60000);
    double **train_labels = loadMNISTLabels("./../../data/train-labels.idx1-ubyte", 60000);
    double **test_images = loadMNISTImages("./../../data/t10k-images.idx3-ubyte", 10000);
    double **test_labels = loadMNISTLabels("./../../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork *net = createNetwork();
    NeuralNetworkDevice *net_device = createNetworkDevice(net);
    train(net, net_device, train_images, train_labels, 60000);

    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    return 0;
}
