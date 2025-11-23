from pydantic import BaseModel, Field
from typing import Literal


class CustomerData(BaseModel):
    """Modelo Pydantic para validar los datos de entrada del cliente"""
    
    gender: Literal['Male', 'Female'] = Field(..., description="Género del cliente")
    SeniorCitizen: Literal[0, 1] = Field(..., description="Si es ciudadano mayor (1) o no (0)")
    Partner: Literal['Yes', 'No'] = Field(..., description="Si tiene pareja")
    Dependents: Literal['Yes', 'No'] = Field(..., description="Si tiene dependientes")
    tenure: int = Field(..., ge=0, description="Número de meses con la compañía")
    PhoneService: Literal['Yes', 'No'] = Field(..., description="Si tiene servicio telefónico")
    MultipleLines: Literal['Yes', 'No', 'No phone service'] = Field(..., description="Si tiene múltiples líneas")
    InternetService: Literal['DSL', 'Fiber optic', 'No'] = Field(..., description="Tipo de servicio de internet")
    OnlineSecurity: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Si tiene seguridad online")
    OnlineBackup: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Si tiene backup online")
    DeviceProtection: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Si tiene protección de dispositivos")
    TechSupport: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Si tiene soporte técnico")
    StreamingTV: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Si tiene streaming de TV")
    StreamingMovies: Literal['Yes', 'No', 'No internet service'] = Field(..., description="Si tiene streaming de películas")
    Contract: Literal['Month-to-month', 'One year', 'Two year'] = Field(..., description="Tipo de contrato")
    PaperlessBilling: Literal['Yes', 'No'] = Field(..., description="Si tiene facturación sin papel")
    PaymentMethod: Literal['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'] = Field(..., description="Método de pago")
    MonthlyCharges: float = Field(..., gt=0, description="Cargo mensual")
    TotalCharges: float = Field(..., ge=0, description="Cargo total acumulado")

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Modelo Pydantic para la respuesta de predicción"""
    
    churn_probability: float = Field(..., description="Probabilidad de churn (0-1)")
    churn_prediction: Literal['Yes', 'No'] = Field(..., description="Predicción de churn")
    confidence: float = Field(..., description="Confianza de la predicción")

    class Config:
        json_schema_extra = {
            "example": {
                "churn_probability": 0.7234,
                "churn_prediction": "Yes",
                "confidence": 0.7234
            }
        }
