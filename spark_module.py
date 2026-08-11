"""
spark_module.py
------------------------------------------------------------
Paso 2: Procesamiento con Spark sobre el dataset preprocesado.

Implementa DOS pipelines equivalentes (mismo resultado, distinta API):
  - pipeline_rdd(): usando RDDs de bajo nivel (map/filter/reduce).
  - pipeline_dataframe(): usando DataFrames + Spark SQL.

En un despliegue real, el dataset vendria de HDFS o de un storage en
la nube (Azure Blob / ADLS). Para poder ejecutar y medir localmente
en Windows sin un cluster HDFS, se simula ese storage con un CSV
local (ver preprocessed_dataset.csv) generado a partir de la salida
del paso GPU (gpu_module.py). El codigo de las funciones no cambia:
solo cambia la ruta de lectura (local:// vs hdfs:// / abfss://).
"""

import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def get_spark(app_name: str = "BigDataHibrido") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .master("local[*]")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )


def pipeline_rdd(spark: SparkSession, data: list[float]) -> dict:
    """
    Pipeline con RDD: calcula media, desviacion estandar y cuenta de
    valores por encima/por debajo de la media, usando transformaciones
    y acciones de bajo nivel (map, filter, reduce).
    """
    sc = spark.sparkContext
    t0 = time.perf_counter()

    rdd = sc.parallelize(data, numSlices=8)
    n = rdd.count()
    total = rdd.reduce(lambda a, b: a + b)
    mean = total / n

    sq_diffs = rdd.map(lambda x: (x - mean) ** 2)
    variance = sq_diffs.reduce(lambda a, b: a + b) / n
    std = variance ** 0.5

    above = rdd.filter(lambda x: x > mean).count()
    below = n - above

    t1 = time.perf_counter()
    return {
        "n": n, "mean": mean, "std": std,
        "above_mean": above, "below_mean": below,
        "tiempo_ms": round((t1 - t0) * 1000.0, 2),
    }


def pipeline_dataframe(spark: SparkSession, data: list[float]) -> dict:
    """
    Pipeline equivalente con DataFrame + Spark SQL (Catalyst optimiza
    el plan de ejecucion, a diferencia del RDD que es transformacion
    a transformacion sin optimizador logico).
    """
    t0 = time.perf_counter()

    df = spark.createDataFrame([(float(x),) for x in data], ["valor"])
    stats = df.select(
        F.count("valor").alias("n"),
        F.mean("valor").alias("mean"),
        F.stddev_pop("valor").alias("std"),
    ).collect()[0]

    above = df.filter(F.col("valor") > stats["mean"]).count()
    below = stats["n"] - above

    t1 = time.perf_counter()
    return {
        "n": stats["n"], "mean": stats["mean"], "std": stats["std"],
        "above_mean": above, "below_mean": below,
        "tiempo_ms": round((t1 - t0) * 1000.0, 2),
    }


def run_comparison(data: list[float]) -> dict:
    """Ejecuta ambos pipelines sobre el mismo dataset y calcula el speedup."""
    spark = get_spark()
    try:
        rdd_result = pipeline_rdd(spark, data)
        df_result = pipeline_dataframe(spark, data)
        speedup = (
            rdd_result["tiempo_ms"] / df_result["tiempo_ms"]
            if df_result["tiempo_ms"] > 0 else float("inf")
        )
        return {
            "rdd": rdd_result,
            "dataframe": df_result,
            "speedup_dataframe_vs_rdd": round(speedup, 2),
        }
    finally:
        spark.stop()


if __name__ == "__main__":
    import numpy as np
    sample = np.random.default_rng(7).uniform(0, 1, size=100_000).tolist()
    result = run_comparison(sample)
    print("RDD:       ", result["rdd"])
    print("DataFrame: ", result["dataframe"])
    print("Speedup DF vs RDD:", result["speedup_dataframe_vs_rdd"])
