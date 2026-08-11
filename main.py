"""
main.py
------------------------------------------------------------
Paso 4: Endpoints HTTP + naturaleza serverless.

Expone dos endpoints, equivalentes a dos Azure Functions con HTTP
Trigger:

  POST /preprocess  -> ejecuta SOLO el paso GPU (gpu_module).
                        Equivale a la "Funcion 1" del enunciado.
  POST /process      -> ejecuta el flujo COMPLETO (validacion ->
                        GPU -> Spark -> respuesta) a traves del
                        sistema de actores (actors.py).
                        Equivale a la "Funcion 2" del enunciado.

Se usa Flask en lugar del emulador de Azure Functions Core Tools
para que el proyecto se pueda ejecutar con un solo comando en
VS Code / Windows 11, sin instalar herramientas adicionales. La
logica de negocio (gpu_module, spark_module, actors) es identica
a la que se usaria dentro de una Azure Function real; solo cambia
la capa de entrada HTTP. En el informe se documenta como migrar
cada endpoint a `azure.functions` (decorador @app.route de las
Python v2 Azure Functions) para el despliegue en la nube.

Ejecutar:
    python main.py
Luego probar con curl / Postman / Thunder Client (VS Code):
    curl -X POST http://127.0.0.1:5000/preprocess -H "Content-Type: application/json" -d "{\"data\": [1,2,3,4,5]}"
    curl -X POST http://127.0.0.1:5000/process     -H "Content-Type: application/json" -d "{\"data\": [1,2,3,4,5]}"
"""

from flask import Flask, request, jsonify

import gpu_module
import actors

app = Flask(__name__)


@app.errorhandler(Exception)
def handle_error(exc):
    return jsonify({"error": str(exc)}), 400


@app.route("/preprocess", methods=["POST"])
def preprocess():
    """Funcion serverless 1: preprocesamiento GPU (normalizacion)."""
    body = request.get_json(force=True)
    data = body.get("data")
    if not data:
        return jsonify({"error": "Falta el campo 'data' (lista numerica)."}), 400

    result = gpu_module.normalize_array(data)
    return jsonify(result), 200


@app.route("/process", methods=["POST"])
def process():
    """Funcion serverless 2: flujo completo orquestado por actores."""
    body = request.get_json(force=True)
    data = body.get("data")
    if not data:
        return jsonify({"error": "Falta el campo 'data' (lista numerica)."}), 400

    result = actors.run_pipeline(data)
    return jsonify(result), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
