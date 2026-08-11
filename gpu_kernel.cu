/*
 * gpu_kernel.cu
 * ------------------------------------------------------------
 * Kernel CUDA para normalizacion Min-Max de un arreglo numerico.
 * Este archivo es el codigo GPU "real" que se entrega como parte
 * del pipeline de preprocesamiento (paso 1 de la actividad).
 *
 * NOTA IMPORTANTE:
 * Este entorno de desarrollo (VS Code en Windows 11, sin GPU NVIDIA
 * dedicada) no permite compilar/ejecutar CUDA directamente. Por eso
 * gpu_module.py incluye un "modo simulado" que reproduce el mismo
 * algoritmo en CPU (con multiprocessing, emulando OpenMP) para que
 * todo el flujo se pueda demostrar y medir sin hardware CUDA.
 *
 * Si el usuario cuenta con una GPU NVIDIA + CUDA Toolkit instalado:
 *   nvcc gpu_kernel.cu -o gpu_kernel.exe
 *   gpu_kernel.exe
 *
 * Algoritmo: normalizacion Min-Max -> x' = (x - min) / (max - min)
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>

#define N 1000000
#define THREADS_PER_BLOCK 256

// Kernel 1: reduccion para encontrar min y max del arreglo
__global__ void reduceMinMax(const float *input, float *blockMin, float *blockMax, int n) {
    __shared__ float sMin[THREADS_PER_BLOCK];
    __shared__ float sMax[THREADS_PER_BLOCK];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sMin[tid] = (idx < n) ? input[idx] : FLT_MAX;
    sMax[tid] = (idx < n) ? input[idx] : -FLT_MAX;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sMin[tid] = fminf(sMin[tid], sMin[tid + stride]);
            sMax[tid] = fmaxf(sMax[tid], sMax[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        blockMin[blockIdx.x] = sMin[0];
        blockMax[blockIdx.x] = sMax[0];
    }
}

// Kernel 2: aplica la normalizacion Min-Max usando min/max globales
__global__ void normalizeMinMax(float *data, int n, float minVal, float maxVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float range = (maxVal - minVal);
        data[idx] = (range > 1e-8f) ? (data[idx] - minVal) / range : 0.0f;
    }
}

int main() {
    size_t bytes = N * sizeof(float);
    float *h_data = (float *)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)(rand() % 100000) / 100.0f;
    }

    float *d_data, *d_blockMin, *d_blockMax;
    int numBlocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_blockMin, numBlocks * sizeof(float));
    cudaMalloc(&d_blockMax, numBlocks * sizeof(float));
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    reduceMinMax<<<numBlocks, THREADS_PER_BLOCK>>>(d_data, d_blockMin, d_blockMax, N);

    float *h_blockMin = (float *)malloc(numBlocks * sizeof(float));
    float *h_blockMax = (float *)malloc(numBlocks * sizeof(float));
    cudaMemcpy(h_blockMin, d_blockMin, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_blockMax, d_blockMax, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    float globalMin = FLT_MAX, globalMax = -FLT_MAX;
    for (int i = 0; i < numBlocks; i++) {
        if (h_blockMin[i] < globalMin) globalMin = h_blockMin[i];
        if (h_blockMax[i] > globalMax) globalMax = h_blockMax[i];
    }

    normalizeMinMax<<<numBlocks, THREADS_PER_BLOCK>>>(d_data, N, globalMin, globalMax);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    printf("Min=%.2f Max=%.2f\n", globalMin, globalMax);
    printf("Tiempo GPU: %.4f ms\n", ms);
    printf("Primeros 5 valores normalizados: %.4f %.4f %.4f %.4f %.4f\n",
           h_data[0], h_data[1], h_data[2], h_data[3], h_data[4]);

    cudaFree(d_data); cudaFree(d_blockMin); cudaFree(d_blockMax);
    free(h_data); free(h_blockMin); free(h_blockMax);
    return 0;
}
