# Agridrone API

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://www.python.org/) [![FastAPI](https://img.shields.io/badge/FastAPI-API-green?logo=fastapi)](https://fastapi.tiangolo.com/) [![YOLOv8](https://img.shields.io/badge/YOLOv8-vision-orange?logo=github)](https://github.com/ultralytics/ultralytics) [![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://www.docker.com/)

## Overview

**Agridrone API** is an advanced agricultural image analysis system that combines NDVI (Normalized Difference Vegetation Index) prediction and object detection using deep learning. It enables automated crop health monitoring, field analysis, and feature detection in aerial or drone imagery. The API supports a full pipeline from RGB images to NDVI computation and YOLO-based detection, making it ideal for precision agriculture and research.

---

## Key Features
- **NDVI prediction** from RGB images for crop health assessment
- **YOLOv8 object detection** on 4-channel TIFFs for field feature analysis
- **Full pipeline**: RGB → NDVI → 4-channel TIFF → YOLO detection
- **FastAPI** backend with async endpoints and auto-generated docs
- **Dockerized** for easy deployment
- **Nginx reverse proxy** for production
- **Modular**: Easily extend to new crops or detection tasks

---

## Architecture
- **Models**: NDVI predictor (Keras/TensorFlow), YOLOv8 (PyTorch)
- **Backend**: FastAPI (Python 3.9+)
- **Deployment**: Docker Compose, Nginx (optional)
- **Workflow**:
  1. Upload RGB image
  2. NDVI prediction (vegetation health)
  3. Convert to 4-channel TIFF
  4. YOLOv8 detection (objects, crops, anomalies)
  5. Return results (JSON, annotated images, NDVI bands)

---

## API Endpoints
- `GET /` — API info
- `POST /predict_ndvi/` — Predict NDVI from RGB image (returns NDVI visualization and band)
- `POST /predict_yolo/` — YOLO detection on 4-channel TIFF (returns JSON)
- `POST /predict_pipeline/` — Full RGB→NDVI→YOLO pipeline (returns JSON)
- `POST /predict_yolo_image/` — YOLO detection with annotated image (PNG)
- `POST /predict_pipeline_image/` — Full pipeline with annotated image (PNG)

---

## Example Use Cases
- **Precision agriculture**: Automated crop health monitoring
- **Field scouting**: Detecting weeds, disease, or anomalies
- **Research**: Large-scale field trials, remote sensing
- **Agri-tech products**: Integrate with farm management platforms

---

## Quickstart

### Prerequisites
- Docker & Docker Compose
- Model files in the `Agridrone-API/` directory

### 1. Build & Run (API only)
```bash
docker-compose up --build agridrone
```

### 2. Build & Run (API + nginx proxy)
```bash
docker-compose up --build
```
- API: http://localhost:8001/
- Via nginx: http://localhost/agridrone/

---

## File Structure
```
agridrone_api/
├── Agridrone-API/
│   ├── app.py
│   ├── best_yolo_model.pt
│   ├── ndvi_best_model/
│   └── ...
├── Dockerfile.agridrone
├── requirements.agridrone.txt
├── docker-compose.yml
├── nginx.conf
└── README.md
```

---

## Example Request
```bash
curl -X POST "http://localhost:8001/predict_ndvi/" -F "file=@test.jpg"
```

---
