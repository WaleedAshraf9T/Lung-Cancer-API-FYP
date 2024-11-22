# app/utils/image_processing.py
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from app.config import settings
import joblib
from io import BytesIO

class ImageProcessor:
    def __init__(self):
        self.preprocess_params = joblib.load(settings.PREPROCESS_PARAMS_PATH)
        self.transform = transforms.Compose([
            transforms.Resize(self.preprocess_params['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.preprocess_params['means'],
                std=self.preprocess_params['stds']
            )
        ])
    
    def process_image(self, image_bytes):
        """Process uploaded image bytes into tensor"""
        try:
            # Convert bytes to BytesIO object
            image_buffer = BytesIO(image_bytes)
            # Open image from BytesIO
            image = Image.open(image_buffer).convert('RGB')
            # Apply transformations
            tensor = self.transform(image)
            # Add batch dimension
            tensor = tensor.unsqueeze(0)
            return tensor
        except Exception as e:
            raise ValueError(f"Error processing image: {str(e)}")