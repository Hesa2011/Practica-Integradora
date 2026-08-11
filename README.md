# Proyecto: Aplicación Híbrida de Procesamiento Big-Data en Entorno Serverless

Combina GPU (CUDA/OpenMP), Spark (RDD/DataFrame) y modelo de actores (Pykka,
equivalente Python a Akka), expuesto mediante endpoints HTTP que emulan
funciones serverless (Azure Functions).

## Archivos

| Archivo | Contenido |
|---|---|
| `gpu_kernel.cu` | Kernel CUDA de referencia (normalización Min-Max en GPU). |
| `openmp_normalize.c` | Versión CPU paralela con OpenMP del mismo algoritmo. |
| `gpu_module.py` | **Ejecutable.** Corre y mide naive / vectorizado (equiv. OpenMP) / GPU (CuPy si hay, si no simula). |
| `spark_module.py` | **Ejecutable.** Pipelines RDD y DataFrame + cálculo de speedup. |
| `actors.py` | **Ejecutable.** Sistema de actores (Pykka) que orquesta validación → GPU → Spark → respuesta, con reintentos. |
| `main.py` | **Ejecutable.** API Flask con los endpoints `/preprocess` y `/process` (simulan las Azure Functions). |
| `requirements.txt` | Dependencias. |
| `Informe_Proyecto_BigData.docx` | Informe con arquitectura, código explicado y análisis de rendimiento. |

## Requisitos (Windows 11)

- Python 3.10+ instalado y en el PATH.
- Java 11+ (JDK) instalado — requerido por PySpark. Verifica con `java -version`.
  Si no lo tienes: instala Temurin JDK 17 y define `JAVA_HOME`.
- (Opcional) GPU NVIDIA + CUDA Toolkit + `cupy-cuda12x` si quieres usar GPU real
  en `gpu_module.py`. Sin esto, el proyecto igual corre completo en modo simulado.

## Instalación

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución

### 1. Probar cada módulo por separado (útil para depurar en VS Code)

```powershell
python gpu_module.py
python spark_module.py
python actors.py
```

### 2. Levantar la API (endpoints "serverless")

```powershell
python main.py
```

Luego, en otra terminal (o con Thunder Client / Postman dentro de VS Code):

```powershell
curl -X POST http://127.0.0.1:5000/preprocess -H "Content-Type: application/json" -d "{\"data\": [1,2,3,4,5]}"
curl -X POST http://127.0.0.1:5000/process     -H "Content-Type: application/json" -d "{\"data\": [1,2,3,4,5]}"
```

- `/preprocess` → ejecuta solo el paso GPU (Función serverless 1).
- `/process` → ejecuta el flujo completo orquestado por actores (Función serverless 2).

## De local a serverless real (Azure Functions)

Cada endpoint de `main.py` se traslada a una Azure Function Python v2 así:

```python
import azure.functions as func
app = func.FunctionApp()

@app.route(route="preprocess", methods=["POST"])
def preprocess(req: func.HttpRequest) -> func.HttpResponse:
    data = req.get_json()["data"]
    result = gpu_module.normalize_array(data)
    return func.HttpResponse(json.dumps(result), mimetype="application/json")
```

El job de Spark (`spark_module.py`) en producción se apuntaría a un dataset en
Azure Data Lake Storage (`abfss://...`) o HDFS en lugar de una lista en memoria,
y correría sobre un clúster (Azure Databricks / HDInsight) en vez de `local[*]`.

## Notas sobre limitaciones de este entorno

- No se cuenta con GPU NVIDIA física para compilar/ejecutar `gpu_kernel.cu`
  directamente aquí; por eso `gpu_module.py` detecta automáticamente si hay
  CuPy/CUDA disponible y, si no, usa una ruta CPU vectorizada como
  aproximación medible y documentada (ver informe).
- Spark corre en `local[*]` (un solo nodo) en vez de un clúster distribuido,
  lo que afecta las magnitudes absolutas de tiempo pero no la comparación
  metodológica RDD vs DataFrame (ver informe, sección de análisis).
