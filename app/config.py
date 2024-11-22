# app/config.py
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Lung Cancer Detection API"
    
    # Security
    API_KEY: str = os.getenv("API_KEY", "your-secret-api-key")
    
    # Model paths
    ALEXNET_PATH: str = "models/alexnet_features.pth"
    VGG16_PATH: str = "models/vgg16_weights.weights.h5"
    KNN_PATH: str = "models/knn_model.pkl"
    LASSO_PATH: str = "models/lasso_selector.pkl"
    PREPROCESS_PARAMS_PATH: str = "models/preprocessing_params.joblib"
    
    # Logging
    LOG_FILE: str = "logs/api.log"
    
    class Config:
        case_sensitive = True

settings = Settings()