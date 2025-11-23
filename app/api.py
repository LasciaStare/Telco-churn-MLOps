from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import logging
try:
    # Allow running both as package (uvicorn app.api:app) and as script (python app/api.py)
    from app.schemas import CustomerData, PredictionResponse
except ImportError:  # pragma: no cover - fallback for local script execution
    from schemas import CustomerData, PredictionResponse

# Configuración de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicializar FastAPI
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="API para predecir la probabilidad de churn de clientes de telecomunicaciones",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ruta del modelo
MODEL_PATH = Path(__file__).parent / "model.joblib"

# Variable global para el modelo
model = None


def load_model():
    """Carga el modelo desde el archivo"""
    global model
    try:
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            logger.info("Modelo cargado exitosamente desde model.joblib")
        else:
            logger.warning(f"Archivo de modelo no encontrado en {MODEL_PATH}")
            logger.info("Por favor, ejecuta el notebook 2_model_training.ipynb completamente para generar el modelo")
    except Exception as e:
        logger.error(f"Error al cargar el modelo: {str(e)}")
        raise


@app.on_event("startup")
async def startup_event():
    """Evento que se ejecuta al iniciar la aplicación"""
    logger.info("Iniciando API de predicción de churn...")
    load_model()
    
    if model is None:
        logger.warning("API iniciada sin modelo. Ejecuta el notebook 2_model_training.ipynb para generar model.joblib")
    else:
        logger.info("API iniciada correctamente con el modelo cargado")


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "API de Predicción de Churn de Clientes Telco",
        "version": "1.0.0",
        "status": "active" if model is not None else "waiting for model",
        "endpoints": {
            "predict": "/predict (POST)",
            "predict_batch": "/predict-batch (POST)",
            "health": "/health (GET)"
        }
    }


@app.get("/health")
async def health_check():
    """Endpoint para verificar el estado de la API"""
    model_loaded = model is not None
    return {
        "status": "healthy" if model_loaded else "model not loaded",
        "model_loaded": model_loaded,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists()
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_churn(customer: CustomerData):
    """
    Endpoint para predecir la probabilidad de churn de un cliente.
    
    Args:
        customer: Datos del cliente en formato JSON
    
    Returns:
        PredictionResponse: Probabilidad de churn y predicción final
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Por favor, ejecuta el notebook 2_model_training.ipynb para generar el modelo."
        )
    
    try:
        # Convertir los datos de entrada a DataFrame
        input_data = pd.DataFrame([customer.model_dump()])
        
        # Realizar la predicción
        prediction_proba = model.predict_proba(input_data)[0]
        churn_probability = float(prediction_proba[1])
        
        # Determinar la predicción final
        churn_prediction = "Yes" if churn_probability >= 0.5 else "No"
        
        # Calcular la confianza (distancia desde 0.5)
        confidence = float(max(prediction_proba))
        
        logger.info(f"Predicción realizada: Probabilidad={churn_probability:.4f}, Predicción={churn_prediction}")
        
        return PredictionResponse(
            churn_probability=churn_probability,
            churn_prediction=churn_prediction,
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"Error durante la predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(e)}"
        )


@app.post("/predict-batch")
async def predict_batch(customers: list[CustomerData]):
    """
    Endpoint para predecir la probabilidad de churn para múltiples clientes.
    
    Args:
        customers: Lista de datos de clientes en formato JSON
    
    Returns:
        Lista de predicciones
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Por favor, ejecuta el notebook 2_model_training.ipynb para generar el modelo."
        )
    
    try:
        # Convertir los datos de entrada a DataFrame
        input_data = pd.DataFrame([customer.model_dump() for customer in customers])
        
        # Realizar las predicciones
        predictions_proba = model.predict_proba(input_data)
        
        results = []
        for i, proba in enumerate(predictions_proba):
            churn_probability = float(proba[1])
            churn_prediction = "Yes" if churn_probability >= 0.5 else "No"
            confidence = float(max(proba))
            
            results.append({
                "index": i,
                "churn_probability": churn_probability,
                "churn_prediction": churn_prediction,
                "confidence": confidence
            })
        
        logger.info(f"Predicciones batch realizadas para {len(customers)} clientes")
        
        return {"predictions": results, "total": len(results)}
    
    except Exception as e:
        logger.error(f"Error durante la predicción batch: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar las predicciones: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
