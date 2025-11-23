# Predicción de Churn en Telecomunicaciones

## Introducción al Proyecto

El abandono de clientes (churn) representa uno de los desafíos más críticos para las empresas de telecomunicaciones, impactando directamente la rentabilidad y el crecimiento sostenible del negocio. Este proyecto implementa un sistema completo de Machine Learning Operations (MLOps) para predecir el comportamiento de abandono de clientes y desarrollar programas de retención enfocados basados en datos.

## Contexto del Problema

En el sector de telecomunicaciones, adquirir un nuevo cliente es significativamente más costoso que retener uno existente. El churn no solo implica la pérdida de ingresos recurrentes, sino también la inversión realizada en marketing, ventas y onboarding. La capacidad de identificar proactivamente clientes en riesgo permite a las empresas implementar intervenciones personalizadas antes de que ocurra el abandono.

## Objetivo del Proyecto

Desarrollar un pipeline end-to-end de MLOps que incluya:

- Análisis exploratorio exhaustivo del comportamiento de clientes y sus patrones de abandono
- Entrenamiento y optimización de múltiples algoritmos de machine learning con búsqueda sistemática de hiperparámetros
- Evaluación comparativa de modelos basada en métricas de clasificación robustas
- Análisis de interpretabilidad para entender los factores que impulsan las predicciones
- API REST de producción para servir predicciones en tiempo real
- Infraestructura containerizada con Docker y pipeline de CI/CD automatizado

## Dataset

El proyecto utiliza el dataset Telco Customer Churn, que contiene información de 7,043 clientes con las siguientes dimensiones:

- **Información demográfica**: género, edad (senior citizen), estado civil, dependientes
- **Servicios contratados**: telefonía, líneas múltiples, internet (DSL/Fibra óptica/No), servicios adicionales (seguridad online, respaldo, protección de dispositivos, soporte técnico, streaming)
- **Datos de cuenta**: antigüedad (tenure), tipo de contrato (mes a mes, anual, bianual), método de pago, facturación electrónica
- **Información financiera**: cargos mensuales, cargos totales acumulados
- **Variable objetivo**: Churn (Yes/No) - indica si el cliente abandonó el servicio en el último mes

El desbalance de clases (aproximadamente 73.5% no churn vs 26.5% churn) refleja la distribución real del problema y requiere estrategias específicas de evaluación y validación.


## Estructura del Proyecto

El repositorio está organizado para facilitar la reproducibilidad y el despliegue:

```
Telco-churn-MLOps/
├── notebooks/          # Jupyter notebooks con análisis y entrenamiento
├── app/               # Aplicación FastAPI de producción
├── tests/             # Pruebas unitarias con pytest
├── data/              # Dataset original
├── data_processed/    # Datos preprocesados para modelado
├── models/            # Modelos entrenados serializados
├── .github/workflows/ # Pipelines de CI/CD
└── requirements.txt   # Dependencias del proyecto
```

## Resultados Principales

El análisis revela factores críticos para la retención de clientes:

- La antigüedad del cliente (tenure) emerge como el predictor más robusto: los primeros meses son la ventana crítica de retención
- Los contratos de largo plazo (anuales y bianuales) reducen dramáticamente el riesgo de churn comparado con contratos mes a mes
- El servicio de internet por fibra óptica muestra una asociación contraintuitiva con churn elevado, sugiriendo una disonancia entre expectativas y valor percibido
- Los servicios adicionales (seguridad online, soporte técnico) actúan como "anclas de valor" que aumentan el costo de cambiar de proveedor
- El método de pago con cheque electrónico está asociado con mayores tasas de abandono comparado con métodos automáticos

El modelo XGBoost optimizado alcanza un ROC-AUC de 0.8476 en el conjunto de prueba, demostrando capacidad predictiva sólida para identificar clientes en riesgo y priorizar intervenciones de retención.
w