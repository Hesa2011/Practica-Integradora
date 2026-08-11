/*
 * openmp_normalize.c
 * ------------------------------------------------------------
 * Version CPU paralela (OpenMP) del mismo algoritmo de normalizacion
 * Min-Max. Sirve como comparativo GPU vs CPU paralelo (paso 1) y
 * como respaldo cuando no hay GPU disponible.
 *
 * Compilar (Windows con MinGW o WSL/Linux con gcc):
 *   gcc -fopenmp openmp_normalize.c -o openmp_normalize -O2
 *   ./openmp_normalize
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <float.h>
#include <time.h>

#define N 1000000

int main() {
    float *data = (float *)malloc(N * sizeof(float));
    srand(42);
    for (int i = 0; i < N; i++) {
        data[i] = (float)(rand() % 100000) / 100.0f;
    }

    double t0 = omp_get_wtime();

    float globalMin = FLT_MAX, globalMax = -FLT_MAX;

    #pragma omp parallel for reduction(min:globalMin) reduction(max:globalMax)
    for (int i = 0; i < N; i++) {
        if (data[i] < globalMin) globalMin = data[i];
        if (data[i] > globalMax) globalMax = data[i];
    }

    float range = globalMax - globalMin;

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        data[i] = (range > 1e-8f) ? (data[i] - globalMin) / range : 0.0f;
    }

    double t1 = omp_get_wtime();

    printf("Min=%.2f Max=%.2f\n", globalMin, globalMax);
    printf("Tiempo OpenMP (%d hilos): %.4f ms\n", omp_get_max_threads(), (t1 - t0) * 1000.0);
    printf("Primeros 5 valores normalizados: %.4f %.4f %.4f %.4f %.4f\n",
           data[0], data[1], data[2], data[3], data[4]);

    free(data);
    return 0;
}
