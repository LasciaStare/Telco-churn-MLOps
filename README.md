# Predicción de Churn en Telecomunicaciones

## Descripción

Este proyecto implementa un sistema completo de Machine Learning para predecir la probabilidad de churn (abandono) de clientes en una empresa de telecomunicaciones. El sistema incluye análisis exploratorio de datos, entrenamiento de modelos con búsqueda de hiperparámetros, interpretabilidad de predicciones y una API REST para servir el modelo en producción.

El proyecto utiliza técnicas avanzadas de MLOps, incluyendo pipelines de preprocesamiento, validación automática, contenedorización con Docker y CI/CD con GitHub Actions.

## Componentes del Proyecto

### `notebooks/`
Contiene los notebooks de Jupyter con el flujo completo de desarrollo:

- **`1_eda_preprocessing.ipynb`**: Análisis exploratorio de datos (EDA) y generación de pipelines de preprocesamiento. Incluye visualizaciones de distribuciones, correlaciones y creación de transformadores para variables numéricas y categóricas.

- **`2_model_training.ipynb`**: Entrenamiento de modelos con búsqueda de hiperparámetros usando GridSearchCV. Evalúa cuatro algoritmos (Random Forest, XGBoost, CatBoost, LightGBM) y guarda automáticamente el mejor modelo en `app/model.joblib`.

- **`3_interpretability.ipynb`**: Análisis de interpretabilidad del modelo usando LIME (Local Interpretable Model-agnostic Explanations). Explica predicciones individuales y visualiza la importancia de features.

### `app/`
Aplicación FastAPI para servir el modelo en producción:

- **`api.py`**: API REST con endpoints para predicciones individuales y por lote. Incluye validación de entrada con Pydantic y health checks.

- **`schemas.py`**: Esquemas de validación Pydantic que definen la estructura de datos de entrada y salida de la API.

### `tests/`
Pruebas unitarias con pytest:

- **`test_api.py`**: Tests para todos los endpoints de la API, validación de datos y casos extremos.

- **`test_model.py`**: Tests para la carga del modelo, consistencia de predicciones y performance.

### `Dockerfile`
Archivo de configuración para construir la imagen Docker del servicio. Utiliza Python 3.10-slim como base e incluye health checks automáticos.

### `.github/workflows/ci.yml`
Pipeline de CI/CD que ejecuta linting, pruebas unitarias y construcción de imagen Docker en cada push o pull request.

## Instalación

### Requisitos Previos
- Python 3.9 o superior
- pip

### Instalación Local

1. Clonar el repositorio:
```bash
git clone https://github.com/LasciaStare/Telco-churn-MLOps.git
cd Telco-churn-MLOps
```

2. Instalar las dependencias:
```bash
pip install -r requirements.txt
```

3. Entrenar el modelo (ejecutar el notebook 2):
```bash
jupyter notebook notebooks/2_model_training.ipynb
```
Esto generará el archivo `app/model.joblib` necesario para la API.

4. Ejecutar la API localmente:
```bash
cd app
uvicorn api:app --host 0.0.0.0 --port 8000
```

La API estará disponible en `http://localhost:8000`. La documentación interactiva se encuentra en `http://localhost:8000/docs`.

## Uso con Docker

### Construcción de la Imagen

```bash
docker build -t telco-churn-api .
```

### Ejecución del Contenedor

```bash
docker run -d -p 8000:8000 --name telco-churn telco-churn-api
```

### Verificar el Estado del Contenedor

```bash
docker ps
docker logs telco-churn
```

### Detener y Eliminar el Contenedor

```bash
docker stop telco-churn
docker rm telco-churn
```

## Endpoints de la API

### `GET /`
Mensaje de bienvenida de la API.

**Respuesta:**
```json
{
  "message": "Telco Churn Prediction API"
}
```

### `GET /health`
Endpoint de health check para verificar el estado del servicio.

**Respuesta:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### `POST /predict`
Realiza una predicción de churn para un cliente individual.

**Payload de ejemplo:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

**Respuesta:**
```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.7234
}
```

### `POST /predict-batch`
Realiza predicciones de churn para múltiples clientes.

**Payload de ejemplo:**
```json
{
  "customers": [
    {
      "gender": "Female",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 1,
      "PhoneService": "No",
      "MultipleLines": "No phone service",
      "InternetService": "DSL",
      "OnlineSecurity": "No",
      "OnlineBackup": "Yes",
      "DeviceProtection": "No",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 29.85,
      "TotalCharges": 29.85
    },
    {
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "No",
      "Dependents": "No",
      "tenure": 34,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "DSL",
      "OnlineSecurity": "Yes",
      "OnlineBackup": "No",
      "DeviceProtection": "Yes",
      "TechSupport": "No",
      "StreamingTV": "No",
      "StreamingMovies": "No",
      "Contract": "One year",
      "PaperlessBilling": "No",
      "PaymentMethod": "Mailed check",
      "MonthlyCharges": 56.95,
      "TotalCharges": 1889.5
    }
  ]
}
```

**Respuesta:**
```json
{
  "predictions": [
    {
      "churn_prediction": "Yes",
      "churn_probability": 0.7234
    },
    {
      "churn_prediction": "No",
      "churn_probability": 0.1456
    }
  ]
}
```

## Ejecución de Pruebas

Ejecutar todas las pruebas con cobertura:

```bash
pytest tests/ -v --cov=app --cov-report=html
```

Ejecutar pruebas específicas:

```bash
# Solo pruebas de API
pytest tests/test_api.py -v

# Solo pruebas del modelo
pytest tests/test_model.py -v
```

## Estructura del Dataset

El dataset `data/telco_churn.csv` contiene información de clientes con las siguientes características:

- **Demográficas**: gender, SeniorCitizen, Partner, Dependents
- **Servicios**: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
- **Contrato**: tenure, Contract, PaperlessBilling, PaymentMethod
- **Facturación**: MonthlyCharges, TotalCharges
- **Target**: Churn (Yes/No)

## Tecnologías Utilizadas

- **Machine Learning**: scikit-learn, XGBoost, CatBoost, LightGBM
- **API**: FastAPI, Pydantic, Uvicorn
- **Análisis de Datos**: pandas, numpy, matplotlib, seaborn
- **Interpretabilidad**: LIME
- **Testing**: pytest, pytest-cov
- **Containerización**: Docker
- **CI/CD**: GitHub Actions
- **Linting**: flake8

