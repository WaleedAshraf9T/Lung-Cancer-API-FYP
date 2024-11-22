# Lung Cancer Detection API

A FastAPI-based REST API for detecting lung cancer from CT scan images using a hybrid deep learning approach.

## Features

- Hybrid CNN framework using AlexNet and VGG16
- Feature extraction and concatenation
- Lasso regularization
- KNN classification
- REST API endpoints
- API key authentication
- Comprehensive logging
- Docker support

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lung-cancer-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained models in the `models/` directory:
- alexnet_features.pth
- vgg16_weights.weights.h5
- knn_model.pkl
- lasso_selector.pkl
- preprocessing_params.joblib

## Running the API

### Using Python

pFYZrubm54gvXxNjZoC634yYQvMtoTJe
```bash
export API_KEY=your-secret-key
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Using Docker

```bash
docker build -t lung-cancer-api .
docker run -p 8000:8000 -e API_KEY=your-secret-key lung-cancer-api
```

## API Documentation

Access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### POST /api/v1/predict

Make a prediction on a CT scan image.

**Headers:**
- X-API-Key: your-secret-key

**Body:**
- file: Image file (multipart/form-data)

**Response:**
```json
{
    "prediction": "string",
    "confidence": float,
    "class_probabilities": {
        "Normal": float,
        "Benign": float,
        "Malignant": float
    }
}
```

## Testing

Run tests using pytest:
```bash
pytest tests/
```