# Clasificador de Números Manuscritos con FastAPI + TensorFlow

## Descripción
Aplicación web que permite reconocer dígitos escritos a mano y está desplegada en Google Cloud Run.

## Arquitectura
- FastAPI para la API REST
- TensorFlow para el modelo CNN
- HTML/CSS/JS para el frontend infantil
- Docker para containerización
- Google Cloud Run para el despliegue

## Cómo ejecutar localmente
git clone https://github.com/usuario/mnist-app
cd mnist-app
pip install -r requirements.txt
uvicorn app.main:app --reload

## Endpoints
GET /
POST /predict
GET /health
GET /docs

## URL del despliegue
https://mi-servicio-cloudrun.run.app
