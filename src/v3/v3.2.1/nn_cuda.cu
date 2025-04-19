#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 2
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

    hidden[neuron_no] = (ans > 0) * ans;
}
__global__ void forwardKernalOutput_Softmax_Gradient_Accum(NeuralNetworkDevice *net_dev, float *hidden, float *output_device, float *labels_dev, float *d_output_device, float *accum_d_output_device) {

    int neuron_no = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ float shared_hidden[HIDDEN_SIZE];

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

    // ans = (ans > 0) * ans;
    // softmax
    __shared__ float sum;

    int tid = threadIdx.x;
    float expv = exp(ans);
    if (tid == 0)
        sum = 0.0f;
    __syncthreads();
    atomicAdd(&sum, expv);
    __syncthreads();

    float cal = expv / sum;
    output_device[neuron_no] = cal;
    __syncthreads();
    // gradient
    float gradient = cal - labels_dev[neuron_no];
    d_output_device[neuron_no] = gradient;

    // accumulate gradient
    accum_d_output_device[neuron_no] += gradient;
}

__global__ void computerHiddenGradient_Accum(NeuralNetworkDevice *net, float *d_output_device, float *d_hidden_device, float *hidden, float *accum_d_hidden_device) {

    int i = threadIdx.x;
    int is_nonzero = (hidden[i] > 0);
    float ans = 0;
    // accumulate gradient
    for (int j = 0; j < OUTPUT_SIZE; j++)
        ans += net->W2[j * HIDDEN_SIZE + i] * d_output_device[j];
    d_hidden_device[i] = ans * is_nonzero;

    // accumulate gradient
    accum_d_hidden_device[i] += ans * is_nonzero;
}

__global__ void updateOutputLayer(NeuralNetworkDevice *net, float *accum_d_output_device, float *hidden) {

    int i = blockIdx.x;
    int j = threadIdx.x;
    net->W2[i * HIDDEN_SIZE + j] -= LEARNING_RATE * accum_d_output_device[i] / (float)BATCH_SIZE * hidden[j];

    net->b2[i] -= (threadIdx.x == 0) * LEARNING_RATE * accum_d_output_device[i] / (float)BATCH_SIZE;
}
__global__ void updateHiddenLayer(NeuralNetworkDevice *net, float *accum_d_hidden_device, float *input) {

    int i = blockIdx.x;
    int j = threadIdx.x;
    net->W1[i * INPUT_SIZE + j] -= LEARNING_RATE * accum_d_hidden_device[i] / (float)BATCH_SIZE * input[j];

    net->b1[i] -= (threadIdx.x == 0) * LEARNING_RATE * accum_d_hidden_device[i] / (float)BATCH_SIZE;
}

__global__ void updateWeightsUnified(
    NeuralNetworkDevice *net,
    const float *accum_d_output_device,
    const float *hidden,
    const float *accum_d_hidden_device,
    const float *input) {
    int layer = blockIdx.x;
    int tid = threadIdx.x;

    if (layer < OUTPUT_SIZE) {
        // Output layer update for neuron i = layer
        if (tid < HIDDEN_SIZE) {
            net->W2[layer * HIDDEN_SIZE + tid] -=
                LEARNING_RATE * accum_d_output_device[layer] / (float)BATCH_SIZE * hidden[tid];
        }
        if (tid == 0) {
            net->b2[layer] -=
                LEARNING_RATE * accum_d_output_device[layer] / (float)BATCH_SIZE;
        }
    } else {
        // Hidden layer update for neuron i = layer - OUTPUT_SIZE
        int i = layer - OUTPUT_SIZE;
        if (tid < INPUT_SIZE) {
            net->W1[i * INPUT_SIZE + tid] -=
                LEARNING_RATE * accum_d_hidden_device[i] / (float)BATCH_SIZE * input[tid];
        }
        if (tid == 0) {
            net->b1[i] -=
                LEARNING_RATE * accum_d_hidden_device[i] / (float)BATCH_SIZE;
        }
    }
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

__global__ void forwardBackwardFused(
    NeuralNetworkDevice *net,
    float *input,
    float *label,
    float *d_hidden_out,
    float *hidden_out) {
    __shared__ float hidden[HIDDEN_SIZE];
    __shared__ float output[OUTPUT_SIZE];
    __shared__ float d_output[OUTPUT_SIZE];

    int tid = threadIdx.x;

    // 1. Forward hidden
    if (tid < HIDDEN_SIZE) {
        float sum = net->b1[tid];
        for (int i = 0; i < INPUT_SIZE; ++i)
            sum += net->W1[tid * INPUT_SIZE + i] * input[i];
        hidden[tid] = tanhf(sum);
    }
    __syncthreads();

    // 2. Forward output
    if (tid < OUTPUT_SIZE) {
        float sum = net->b2[tid];
        for (int i = 0; i < HIDDEN_SIZE; ++i)
            sum += net->W2[tid * HIDDEN_SIZE + i] * hidden[i];
        output[tid] = expf(sum); // softmax numerator
    }
    __syncthreads();

    // Normalize softmax
    if (tid == 0) {
        float sum = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; ++i)
            sum += output[i];
        for (int i = 0; i < OUTPUT_SIZE; ++i)
            output[i] /= sum;
    }
    __syncthreads();

    // 3. Compute d_output = output - label
    if (tid < OUTPUT_SIZE)
        d_output[tid] = output[tid] - label[tid];
    __syncthreads();

    // 4. Backprop to hidden
    if (tid < HIDDEN_SIZE) {
        float sum = 0.0f;
        for (int i = 0; i < OUTPUT_SIZE; ++i)
            sum += d_output[i] * net->W1[i * HIDDEN_SIZE + tid];
        d_hidden_out[tid] = sum * (1.0f - hidden[tid] * hidden[tid]);
        if (hidden_out != nullptr)
            hidden_out[tid] = hidden[tid]; // optional
    }
}

__global__ void forwardBackwardUnified(
    NeuralNetworkDevice *net_dev,
    const float *image,
    float *hidden_device,
    float *output_device,
    const float *labels_dev,
    float *d_output_device,
    float *accum_d_output_device,
    float *d_hidden_device,
    float *accum_d_hidden_device) {
    int tid = threadIdx.x;

    __shared__ float shared_image[INPUT_SIZE];
    __shared__ float shared_hidden[HIDDEN_SIZE];
    __shared__ float sum;

    if (tid == 0)
        sum = 0.0f;

    if (tid < HIDDEN_SIZE) {
        int one_part = INPUT_SIZE / HIDDEN_SIZE;
        int start = one_part * tid;
        int end = start + one_part;
        for (int i = start; i < end; ++i)
            shared_image[i] = image[i];
        if (tid == HIDDEN_SIZE - 1) {
            for (int i = one_part * HIDDEN_SIZE; i < INPUT_SIZE; ++i)
                shared_image[i] = image[i];
        }
    }
    __syncthreads();

    if (tid < HIDDEN_SIZE) {
        int gidx = INPUT_SIZE * tid;
        float acc = net_dev->b1[tid];
        for (int j = 0; j < INPUT_SIZE; ++j)
            acc += net_dev->W1[gidx + j] * shared_image[j];
        float h = (acc > 0 ? acc : 0);
        shared_hidden[tid] = h;
        hidden_device[tid] = h;
    }
    __syncthreads();
    float e;
    if (tid < OUTPUT_SIZE) {
        int gidx = HIDDEN_SIZE * tid;
        float acc = net_dev->b2[tid];
        for (int j = 0; j < HIDDEN_SIZE; ++j)
            acc += net_dev->W2[gidx + j] * shared_hidden[j];
        e = exp(acc);
        atomicAdd(&sum, e);
    }
    __syncthreads();

    if (tid < OUTPUT_SIZE) {
        float p = e / sum;
        output_device[tid] = p;
        float grad = p - labels_dev[tid];
        d_output_device[tid] = grad;
        accum_d_output_device[tid] += grad;
    }
    __syncthreads();

    if (tid < HIDDEN_SIZE) {
        float sumg = 0;
        for (int j = 0; j < OUTPUT_SIZE; ++j)
            sumg += net_dev->W2[j * HIDDEN_SIZE + tid] * d_output_device[j];
        float dh = sumg * (shared_hidden[tid] > 0);
        d_hidden_device[tid] = dh;
        accum_d_hidden_device[tid] += dh;
    }
}

void train(NeuralNetwork *net, NeuralNetworkDevice *net_device, float **images, float **labels, int numImages) {
    float *images_dev;

    float *hidden_device;
    float *output_device;
    float *labels_dev;

    float *d_hidden_device;
    float *d_output_device;

    float *accum_d_output_device;
    float *accum_d_hidden_device;

    float *d_input_avg;
    float *d_hidden_avg;

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
    if (cudaMalloc((void **)&accum_d_output_device, sizeof(float) * OUTPUT_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for output layer gradient\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&accum_d_hidden_device, sizeof(float) * HIDDEN_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for hidden layer gradient\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&d_input_avg, sizeof(float) * INPUT_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for input average\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&d_hidden_avg, sizeof(float) * HIDDEN_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for hidden average\n");
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
    int epoch = 0;
    float total_time = 0.0f;
    for (; epoch < EPOCHS; epoch++) {
        float epoch_time = 0.0f;
        float loss = 0.0f;
        int correct = 0;

        if (cudaMemset(loss_device, 0, sizeof(float)) != cudaSuccess) {
            fprintf(stderr, "Error initializing loss_device\n");
            exit(EXIT_FAILURE);
        }
        if (cudaMemset(correct_device, 0, sizeof(int)) != cudaSuccess) {
            fprintf(stderr, "Error initializing correct_device\n");
            exit(EXIT_FAILURE);
        }

        for (int start = 0; start < numImages; start += BATCH_SIZE) {

            int batch_size = (start + BATCH_SIZE <= numImages) ? BATCH_SIZE : (numImages - start);

            cudaEvent_t e_start, e_stop;
            float elapsed;

            cudaEventCreate(&e_start);
            cudaEventCreate(&e_stop);
            cudaEventRecord(e_start);

            cudaMemset(accum_d_output_device, 0, sizeof(float) * OUTPUT_SIZE);
            cudaMemset(accum_d_hidden_device, 0, sizeof(float) * HIDDEN_SIZE);
            for (int b = 0; b < batch_size; b++) {
                int index = start + b;
                int threads = max(HIDDEN_SIZE, OUTPUT_SIZE);  // why this?
                forwardBackwardUnified<<<1, threads>>>(
                    net_device,
                    images_dev + index * INPUT_SIZE,
                    hidden_device,
                    output_device,
                    labels_dev + index * OUTPUT_SIZE,
                    d_output_device,
                    accum_d_output_device,
                    d_hidden_device,
                    accum_d_hidden_device);

                // forwardKernalHidden<<<1, HIDDEN_SIZE>>>(net_device, images_dev + index * INPUT_SIZE, hidden_device);
                // cudaError_t err = cudaGetLastError();

                // cudaDeviceSynchronize();
                // forwardKernalOutput_Softmax_Gradient_Accum<<<1, OUTPUT_SIZE>>>(net_device, hidden_device, output_device, labels_dev + index * OUTPUT_SIZE, d_output_device, accum_d_output_device);
                // cudaDeviceSynchronize();
                // computerHiddenGradient_Accum<<<1, HIDDEN_SIZE>>>(net_device, d_output_device, d_hidden_device, hidden_device, accum_d_hidden_device);
                // cudaDeviceSynchronize();
            }
            cudaDeviceSynchronize();
            
            int threads = max(HIDDEN_SIZE, INPUT_SIZE);
            int blocks = OUTPUT_SIZE + HIDDEN_SIZE;
            updateWeightsUnified<<<blocks, threads>>>(
                net_device,
                accum_d_output_device,
                hidden_device,
                accum_d_hidden_device,
                images_dev + start * INPUT_SIZE);
            // updateOutputLayer<<<OUTPUT_SIZE, HIDDEN_SIZE>>>(net_device, accum_d_output_device, hidden_device);
            // cudaDeviceSynchronize();
            // updateHiddenLayer<<<HIDDEN_SIZE, INPUT_SIZE>>>(net_device, accum_d_hidden_device, images_dev + start * INPUT_SIZE);
            cudaDeviceSynchronize();
            cudaEventRecord(e_stop);
            cudaEventSynchronize(e_stop);
            cudaEventElapsedTime(&elapsed, e_start, e_stop);
            float time = elapsed / 1000.0f;
            epoch_time += time;
            total_time += time;
            findCorrect<<<1, 1>>>(loss_device, correct_device, output_device, labels_dev + start * OUTPUT_SIZE);
        }
        if (cudaMemcpy(&loss, loss_device, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error in copying loss from device to host");
        }

        if (cudaMemcpy(&correct, correct_device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error in copying correct from device to host");
        }
        // printf("Epoch %d: Time: %.3fs\n", epoch + 1, epoch_time);
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / numImages, (correct / (float)numImages) * 100, epoch_time);
    }
    printf("Total training time: %.3fs\n", total_time);

    NeuralNetworkDevice net_dev;
    cudaMemcpy(&net_dev, net_device, sizeof(NeuralNetworkDevice), cudaMemcpyDeviceToHost);

    copyMatrixtoHost(net->W1, net_dev.W1, HIDDEN_SIZE, INPUT_SIZE);
    copyMatrixtoHost(net->W2, net_dev.W2, OUTPUT_SIZE, HIDDEN_SIZE);

    cudaMemcpy(net->b1, net_dev.b1, HIDDEN_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(net->b2, net_dev.b2, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // printf("Total training time: %.3fs\n", get_time(total_start));
}

// Evaluate accuracy on test data;
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

    float **train_images = loadMNISTImages("./../../../data/train-images.idx3-ubyte", 60000);
    float **train_labels = loadMNISTLabels("./../../../data/train-labels.idx1-ubyte", 60000);
    float **test_images = loadMNISTImages("./../../../data/t10k-images.idx3-ubyte", 10000);
    float **test_labels = loadMNISTLabels("./../../../data/t10k-labels.idx1-ubyte", 10000);

    NeuralNetwork *net = createNetwork();
    NeuralNetworkDevice *net_device = createNetworkDevice(net);
    train(net, net_device, train_images, train_labels, 60000);

    evaluate(net, test_images, test_labels, 10000);

    freeNetwork(net);
    return 0;
}