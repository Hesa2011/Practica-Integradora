"""
actors.py
------------------------------------------------------------
Paso 3: Modelo de actores para orquestacion (equivalente a Akka).

Se usa Pykka (biblioteca de actores para Python, con la misma
filosofia que Akka: cada actor tiene su propio hilo/buzon de
mensajes y no comparte estado mutable con otros actores). Si el
curso requiere especificamente Scala/Akka, la logica de cada actor
de aqui es 1:1 trasladable a clases `Actor` de Akka (se documenta
la equivalencia en el informe).

Cuatro actores, uno por etapa (segun el enunciado):
  1. ValidationActor   -> valida el input recibido por HTTP.
  2. GpuJobActor        -> ejecuta gpu_module.normalize_array (paso 1).
  3. SparkJobActor       -> ejecuta spark_module.run_comparison (paso 2).
  4. ResponseActor      -> arma la respuesta final / analiza resultados.

OrchestratorActor coordina el envio de mensajes entre ellos y aplica
reintentos (retries) ante fallos transitorios, aprovechando que cada
actor procesa su buzon de forma aislada (tolerancia a fallos al
estilo "let it crash" de Akka).
"""

import time
import pykka

import gpu_module
import spark_module


MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 0.5


class ValidationActor(pykka.ThreadingActor):
    def validate(self, data):
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("El input debe ser una lista numerica no vacia.")
        if not all(isinstance(x, (int, float)) for x in data):
            raise ValueError("Todos los elementos deben ser numericos.")
        return {"status": "valid", "n": len(data)}


class GpuJobActor(pykka.ThreadingActor):
    def run_gpu_job(self, data):
        return gpu_module.normalize_array(data)


class SparkJobActor(pykka.ThreadingActor):
    def run_spark_job(self, data):
        return spark_module.run_comparison(data)


class ResponseActor(pykka.ThreadingActor):
    def build_response(self, validation, gpu_result, spark_result, total_time_ms):
        return {
            "validacion": validation,
            "preprocesamiento_gpu": {
                "modo": gpu_result["gpu_mode"],
                "tiempos_ms": gpu_result["timings_ms"],
                "speedup_gpu_vs_cpu_naive": gpu_result["speedup_gpu_vs_cpu_naive"],
                "n_normalizados": gpu_result["n"],
            },
            "procesamiento_spark": {
                "rdd": spark_result["rdd"],
                "dataframe": spark_result["dataframe"],
                "speedup_dataframe_vs_rdd": spark_result["speedup_dataframe_vs_rdd"],
            },
            "tiempo_total_pipeline_ms": round(total_time_ms, 2),
        }


def _with_retries(fn, *args, stage_name="etapa"):
    """Reintenta una llamada a un actor ante fallos transitorios."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args).get()  # .get() espera la respuesta del actor
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[retry] {stage_name} fallo (intento {attempt}/{MAX_RETRIES}): {exc}")
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"La etapa '{stage_name}' fallo tras {MAX_RETRIES} intentos: {last_error}")


def run_pipeline(data: list[float]) -> dict:
    """
    Orquesta el flujo completo: validacion -> GPU -> Spark -> respuesta.
    Cada etapa corre en su propio actor (hilo aislado) con reintentos.
    Este es el punto de entrada usado por el endpoint HTTP /process.
    """
    t0 = time.perf_counter()

    validation_ref = ValidationActor.start()
    gpu_ref = GpuJobActor.start()
    spark_ref = SparkJobActor.start()
    response_ref = ResponseActor.start()

    validation_actor = validation_ref.proxy()
    gpu_actor = gpu_ref.proxy()
    spark_actor = spark_ref.proxy()
    response_actor = response_ref.proxy()

    try:
        validation = _with_retries(
            lambda d: validation_actor.validate(d), data, stage_name="validacion"
        )

        gpu_result = _with_retries(
            lambda d: gpu_actor.run_gpu_job(d), data, stage_name="gpu_job"
        )

        spark_result = _with_retries(
            lambda d: spark_actor.run_spark_job(d), data, stage_name="spark_job"
        )

        t1 = time.perf_counter()

        final_response = _with_retries(
            lambda v, g, s, t: response_actor.build_response(v, g, s, t),
            validation, gpu_result, spark_result, (t1 - t0) * 1000.0,
            stage_name="respuesta",
        )
        return final_response
    finally:
        validation_ref.stop()
        gpu_ref.stop()
        spark_ref.stop()
        response_ref.stop()


if __name__ == "__main__":
    import numpy as np
    sample = np.random.default_rng(1).uniform(0, 500, size=50_000).tolist()
    result = run_pipeline(sample)
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))
