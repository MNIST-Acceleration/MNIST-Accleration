#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10
#define LEARNING_RATE 0.01
#define EPOCHS 3
#define BATCH_SIZE 32
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


//forward pass kernel, each block will process a batch of samples and each block will have a thread for each layer
__global__ void forwardAndBackwardsKernel(NeuralNetworkDevice *net, float *input, float *hidden, float *output, int batch_size, float* labels_dev, float *d_hidden, float *d_output) {
    int batch_index = blockIdx.x;
    int thread_index = threadIdx.x;
    int no_threads   = blockDim.x;
    __shared__ float sum;
    __shared__ float shared_hidden[OUTPUT_SIZE];
    __shared__ float shared_image[INPUT_SIZE];

    const float* img = input + batch_index * INPUT_SIZE;

    for (int i = thread_index; i < INPUT_SIZE; i += no_threads) {
        shared_image[i] = img[i];
    }
    __syncthreads();


    if (thread_index < HIDDEN_SIZE) {
        float hidden_bias = net->b1[thread_index];
        float val = 0.0f;
        for (int j = 0; j < INPUT_SIZE; j++) {
            val += net->W1[thread_index * INPUT_SIZE + j] * shared_image[j] + (j == 0) * hidden_bias;
        }
        val = (val > 0) ? val : 0;
        hidden[batch_index * HIDDEN_SIZE + thread_index] = val;
        shared_hidden[thread_index] = val;
    }

    sum = 0.0f;
    __syncthreads();
    float expv = 0.0f;
    if (thread_index < OUTPUT_SIZE) {
        float val = 0.0f;
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            val += net->W2[thread_index * HIDDEN_SIZE + j] * shared_hidden[j] + (j == 0) * net->b2[thread_index];
        }
        expv = exp(val);
        atomicAdd(&sum, expv);
    }
    __syncthreads();

    if (thread_index < OUTPUT_SIZE) {
        float val = expv / sum;
        output[batch_index * OUTPUT_SIZE + thread_index] = val;
        d_output[batch_index * OUTPUT_SIZE + thread_index] = val - labels_dev[batch_index * OUTPUT_SIZE + thread_index];
    }
    __syncthreads();

    if (thread_index < HIDDEN_SIZE) {
        float val = 0.0f;
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            val += net->W2[j * HIDDEN_SIZE + thread_index] * d_output[batch_index * OUTPUT_SIZE + j];
        }
        d_hidden[batch_index * HIDDEN_SIZE + thread_index] = hidden[batch_index * HIDDEN_SIZE + thread_index] > 0 ? val : 0;
    }

}

__global__ void update_grads(float* d_accum_W1, float* d_accum_W2, float* d_accum_b1, float* d_accum_b2, 
                            float* d_hidden_device, float* d_output_device, float* images_dev, float* hidden_device) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < HIDDEN_SIZE * OUTPUT_SIZE) {
        int i = idx / HIDDEN_SIZE;
        int j = idx % HIDDEN_SIZE;
        float sum = 0.0f;
        
        for (int b = 0; b < BATCH_SIZE; b++) {
            sum += d_output_device[b * OUTPUT_SIZE + i] * hidden_device[b * HIDDEN_SIZE + j];
        }
        
        d_accum_W2[i * HIDDEN_SIZE + j] = sum;
    }

    if (idx < OUTPUT_SIZE) {
        float sum = 0.0f;
        for (int b = 0; b < BATCH_SIZE; b++) {
            sum += d_output_device[b * OUTPUT_SIZE + idx];
        }
        d_accum_b2[idx] = sum;
    }

    if (idx < INPUT_SIZE * HIDDEN_SIZE) {
        int i = idx / INPUT_SIZE;
        int j = idx % INPUT_SIZE;
        float sum = 0.0f;
        
        for (int b = 0; b < BATCH_SIZE; b++) {
            sum += d_hidden_device[b * HIDDEN_SIZE + i] * images_dev[b * INPUT_SIZE + j];
        }
        
        d_accum_W1[i * INPUT_SIZE + j] = sum;
    }

    if (idx < HIDDEN_SIZE) {
        float sum = 0.0f;
        for (int b = 0; b < BATCH_SIZE; b++) {
            sum += d_hidden_device[b * HIDDEN_SIZE + idx];
        }
        d_accum_b1[idx] = sum;
    }
}

__global__ void update_weights(NeuralNetworkDevice* net_dev, float *d_accum_W1, float *d_accum_W2, float *d_accum_b1, float *d_accum_b2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < HIDDEN_SIZE * INPUT_SIZE) {
     
        net_dev->W1[idx] -= LEARNING_RATE * d_accum_W1[idx]/BATCH_SIZE;
    }

    if (idx < OUTPUT_SIZE * HIDDEN_SIZE) {

        net_dev->W2[idx] -= LEARNING_RATE * d_accum_W2[idx]/BATCH_SIZE;
    }

    if (idx < HIDDEN_SIZE) {
        net_dev->b1[idx] -= LEARNING_RATE * d_accum_b1[idx]/BATCH_SIZE;
    }

    if (idx < OUTPUT_SIZE) {
        net_dev->b2[idx] -= LEARNING_RATE * d_accum_b2[idx]/BATCH_SIZE;
    }
}


__global__
void find_correct(
    const float *outputs,    
    const float *labels,     
    int           *correct,  
    int            numSamples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numSamples) return;

    const float* out = outputs + idx * OUTPUT_SIZE;
    const float* lbl = labels  + idx * OUTPUT_SIZE;

    int   pred = 0;
    float maxv = out[0];
    for (int j = 1; j < OUTPUT_SIZE; ++j) {
        float v = out[j];
        if (v > maxv) {
            maxv = v;
            pred = j;
        }
    }

    int actual = 0;
    for (int j = 1; j < OUTPUT_SIZE; ++j) {
        if (lbl[j] > lbl[actual]) {
            actual = j;
        }
    }

    if (pred == actual) {
        atomicAdd(correct, 1);
    }
}

__global__ void statsKernel(const float* __restrict__ output,const float* __restrict__ labels,int batch_size,int output_size,float* d_loss,int*   d_correct) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    float sample_loss = 0.0f;
    const float* out_row = output + tid * output_size;
    const float* lb_row  = labels + tid * output_size;
    for (int k = 0; k < output_size; ++k) {
        sample_loss -= lb_row[k] * logf(out_row[k]);
    }

    int pred = 0, actual = 0;
    for (int k = 1; k < output_size; ++k) {
        if (out_row[k] > out_row[pred]) pred = k;
        if (lb_row[k] > lb_row[actual]) actual = k;
    }
    int hit = (pred == actual) ? 1 : 0;

    atomicAdd(d_loss,    sample_loss);
    atomicAdd(d_correct,  hit);
}

void train(NeuralNetwork *net, NeuralNetworkDevice *net_device, float **images, float **labels, int numImages) {
    float *images_dev;

    float *hidden_device;
    float *output_device;
    float *labels_dev;


    //ok so the problem inthe previous commit was that each batch sample was being processed sequentially, we need to parallelize it
    //we can use a kernal with block size of batch size and thread size of layer size
    //we need gradient buffers of size BATCH_SIZE*HIDDEN_SIZE and BATCH_SIZE*OUTPUT_SIZE
    float *d_hidden_device;
    float *d_output_device;

    float* d_accum_W1;
    float* d_accum_W2;
    float* d_accum_b1;
    float* d_accum_b2;  

    allocateMatrixDevice(&images_dev, numImages, INPUT_SIZE);
    allocateMatrixDevice(&labels_dev, numImages, OUTPUT_SIZE);

    if (cudaMalloc((void **)&hidden_device, sizeof(float) * BATCH_SIZE * HIDDEN_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for hidden layer\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&output_device, sizeof(float) * BATCH_SIZE * OUTPUT_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for output layer\n");
        exit(EXIT_FAILURE);
    }

    if (cudaMalloc((void **)&d_hidden_device, sizeof(float) * BATCH_SIZE * HIDDEN_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for hidden layer gradient\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&d_output_device, sizeof(float) * BATCH_SIZE * OUTPUT_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for output layer gradient\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&d_accum_W1, sizeof(float) * HIDDEN_SIZE * INPUT_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for W1 gradient\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&d_accum_W2, sizeof(float) * OUTPUT_SIZE * HIDDEN_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for W2 gradient\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&d_accum_b1, sizeof(float) * HIDDEN_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for b1 gradient\n");
        exit(EXIT_FAILURE);
    }
    if (cudaMalloc((void **)&d_accum_b2, sizeof(float) * OUTPUT_SIZE) != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory for b2 gradient\n");
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
        cudaMemset(loss_device,    0, sizeof(float));
        cudaMemset(correct_device, 0, sizeof(int));
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

            int batch_size = (start + BATCH_SIZE < numImages) ? BATCH_SIZE : (numImages - start);

            cudaEvent_t e_start, e_stop;
            float elapsed;

            cudaEventCreate(&e_start);
            cudaEventCreate(&e_stop);
            cudaEventRecord(e_start);

            //batch processing in a unified kernel
            forwardAndBackwardsKernel<<<batch_size, HIDDEN_SIZE>>>(net_device, images_dev + start * INPUT_SIZE, hidden_device, output_device, batch_size, labels_dev + start * OUTPUT_SIZE, d_hidden_device, d_output_device);
            //setting all accumulators to 0
            cudaMemset(d_accum_W1, 0, sizeof(float) * HIDDEN_SIZE * INPUT_SIZE);
            cudaMemset(d_accum_W2, 0, sizeof(float) * OUTPUT_SIZE * HIDDEN_SIZE);
            cudaMemset(d_accum_b1, 0, sizeof(float) * HIDDEN_SIZE);
            cudaMemset(d_accum_b2, 0, sizeof(float) * OUTPUT_SIZE);
            cudaDeviceSynchronize();

            //using a multiple of 32 so that warps are fully utilized
            int threads_per_block = 32;
            int blocks = (HIDDEN_SIZE * INPUT_SIZE + threads_per_block - 1) / threads_per_block;

            update_grads<<<blocks, threads_per_block>>>(d_accum_W1, d_accum_W2, d_accum_b1, d_accum_b2, d_hidden_device, d_output_device, images_dev + start * INPUT_SIZE, hidden_device);
            update_weights<<<blocks, threads_per_block>>>(net_device, d_accum_W1, d_accum_W2, d_accum_b1, d_accum_b2);

            threads_per_block = 128;                            
            blocks  = (batch_size + threads_per_block - 1) / threads_per_block;

            statsKernel<<<blocks, threads_per_block>>>(output_device,labels_dev + start * OUTPUT_SIZE,batch_size,OUTPUT_SIZE,loss_device,correct_device);
            cudaEventRecord(e_stop);
            cudaEventSynchronize(e_stop);
            cudaEventElapsedTime(&elapsed, e_start, e_stop);
          

            float time = elapsed / 1000.0f;
            epoch_time += time;
            total_time += time;
            
            
        }
        if (cudaMemcpy(&loss, loss_device, sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error in copying loss from device to host");
        }

        if (cudaMemcpy(&correct, correct_device, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) {
            printf("Error in copying correct from device to host");
        }
     
        // printf("Epoch %d: Time: %.3fs\n", epoch + 1, epoch_time);
        printf("Epoch %d - Loss: %.4f - Train Accuracy: %.2f%% - Time: %.3fs\n",
               epoch + 1, loss / float(numImages), (correct / (float)numImages) * 100.0f, epoch_time);
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