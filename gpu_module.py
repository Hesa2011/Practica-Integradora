"""
gpu_module.py
------------------------------------------------------------
Paso 1: Preprocesamiento en GPU (con respaldo CPU/OpenMP).

Este modulo es el que realmente se ejecuta en Windows 11 / VS Code.
Intenta usar CuPy (equivalente Python de CUDA) si hay GPU NVIDIA
disponible; si no, cae automaticamente a un modo "CPU paralelo"
(multiprocessing) que emula el comportamiento de openmp_normalize.c,
y deja documentado en gpu_kernel.cu / openmp_normalize.c el codigo
CUDA/OpenMP nativo equivalente.

Expone la funcion normalize_array(data) que usa la funcion serverless
(main.py) en el endpoint /preprocess.
"""

import time
import numpy as np

try:
    import cupy as cp  # type: ignore
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False


def _normalize_cpu_naive(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Version secuencial (1 hilo) - referencia de linea base."""
    t0 = time.perf_counter()
    mn, mx = arr.min(), arr.max()
    rng = (mx - mn) if (mx - mn) > 1e-8 else 1.0
    out = np.empty_like(arr)
    for i in range(arr.shape[0]):
        out[i] = (arr[i] - mn) / rng
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def _normalize_cpu_vectorized(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Version vectorizada con NumPy. NumPy delega en BLAS/SIMD y usa
    varios nucleos internamente, por lo que sirve como equivalente
    practico al pragma omp parallel for de openmp_normalize.c.
    """
    t0 = time.perf_counter()
    mn, mx = arr.min(), arr.max()
    rng = (mx - mn) if (mx - mn) > 1e-8 else 1.0
    out = (arr - mn) / rng
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def _normalize_gpu(arr: np.ndarray) -> tuple[np.ndarray, float]:
    """Version GPU real usando CuPy (equivale al kernel de gpu_kernel.cu)."""
    t0 = time.perf_counter()
    g = cp.asarray(arr)
    mn, mx = g.min(), g.max()
    rng = (mx - mn) if (mx - mn) > 1e-8 else 1.0
    g = (g - mn) / rng
    out = cp.asnumpy(g)
    cp.cuda.Stream.null.synchronize()
    t1 = time.perf_counter()
    return out, (t1 - t0) * 1000.0


def normalize_array(data: list[float]) -> dict:
    """
    Punto de entrada usado por la funcion serverless /preprocess.
    Ejecuta las 3 variantes (naive, vectorizada/OpenMP-equivalente,
    GPU real o simulada) y regresa el resultado + metricas comparativas.
    """
    arr = np.asarray(data, dtype=np.float64)

    naive_out, t_naive = _normalize_cpu_naive(arr)
    vec_out, t_vec = _normalize_cpu_vectorized(arr)

    if GPU_AVAILABLE:
        gpu_out, t_gpu = _normalize_gpu(arr)
        gpu_mode = "GPU real (CuPy/CUDA)"
    else:
        # Sin GPU disponible en este entorno: se reporta el tiempo
        # vectorizado como aproximacion documentada (ver informe,
        # seccion "Limitaciones de hardware").
        gpu_out, t_gpu = vec_out, t_vec
        gpu_mode = "Simulado en CPU (sin GPU NVIDIA/CUDA disponible)"

    speedup_gpu_vs_naive = t_naive / t_gpu if t_gpu > 0 else float("inf")

    return {
        "normalized": gpu_out.tolist(),
        "n": int(arr.shape[0]),
        "gpu_mode": gpu_mode,
        "timings_ms": {
            "cpu_naive_1_hilo": round(t_naive, 4),
            "cpu_vectorizado_openmp_equiv": round(t_vec, 4),
            "gpu": round(t_gpu, 4),
        },
        "speedup_gpu_vs_cpu_naive": round(speedup_gpu_vs_naive, 2),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    sample = rng.uniform(0, 1000, size=200_000).tolist()
    result = normalize_array(sample)
    print("Modo GPU:", result["gpu_mode"])
    print("Tiempos (ms):", result["timings_ms"])
    print("Speedup GPU/simulado vs CPU naive:", result["speedup_gpu_vs_cpu_naive"])
