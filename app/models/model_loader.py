# app/models/model_loader.py
import torch
import joblib
from tensorflow.keras.applications.vgg16 import VGG16
import torch.nn as nn
from app.config import settings
import logging

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.alexnet = None
        self.vgg16 = None
        self.knn = None
        self.lasso = None
        self.load_models()
    
    def load_models(self):
        """Load all required models"""
        try:
            # Load AlexNet
            self.alexnet = torch.load(settings.ALEXNET_PATH)
            
            # Load VGG16
            self.vgg16 = VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
            self.vgg16.load_weights(settings.VGG16_PATH)  # Load saved VGG16 weights
            
            # Load KNN and Lasso
            self.knn = joblib.load(settings.KNN_PATH)
            self.lasso = joblib.load(settings.LASSO_PATH)
            
            logger.info("All models loaded successfully")
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            raise RuntimeError("Failed to load models")