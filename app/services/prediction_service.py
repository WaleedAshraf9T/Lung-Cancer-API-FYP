# app/services/prediction_service.py
import numpy as np
import torch
from app.models.model_loader import ModelLoader
from app.utils.image_processing import ImageProcessor
from tensorflow.keras.applications.vgg16 import preprocess_input
import logging
import traceback  # Add this import

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self):
        self.model_loader = ModelLoader()
        self.image_processor = ImageProcessor()
        self.class_mapping = {0: 'Normal', 1: 'Benign', 2: 'Malignant'}
    
    def predict(self, image_bytes):
        """Make prediction on processed image"""
        try:
            # Process image
            tensor = self.image_processor.process_image(image_bytes)
            
            # Extract features
            alexnet_features = self._extract_alexnet_features(tensor)
            vgg_features = self._extract_vgg_features(tensor)
            
            # Concatenate features
            combined_features = np.hstack((vgg_features, alexnet_features))
            
            # Add debug logging
            #logger.info(f"Combined features shape: {combined_features.shape}")
            
            # Apply Lasso feature selection
            selected_features = self.model_loader.lasso.transform(combined_features)
            
            # Get predictions
            predictions = self.model_loader.knn.predict_proba(selected_features)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # Prepare response
            class_probabilities = {
                self.class_mapping[i]: float(prob) 
                for i, prob in enumerate(predictions[0])
            }
            
            return {
                'prediction': self.class_mapping[predicted_class],
                'confidence': float(confidence),
                'class_probabilities': class_probabilities
            }
            
        except Exception as e:
            # Detailed error logging
            logger.error(f"Prediction error: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to make prediction: {str(e)}")
    
    def _extract_alexnet_features(self, tensor):
        try:
            with torch.no_grad():
                features = self.model_loader.alexnet(tensor)
                return features.view(features.size(0), -1).detach().numpy()
        except Exception as e:
            logger.error(f"AlexNet feature extraction error: {str(e)}")
            logger.error(f"Tensor shape: {tensor.shape}")
            raise
    
    def _extract_vgg_features(self, tensor):
        try:
            # Convert tensor to numpy for VGG16
            np_image = tensor.numpy().transpose(0, 2, 3, 1)
            #logger.info(f"VGG16 input shape: {np_image.shape}")
            features = self.model_loader.vgg16.predict(preprocess_input(np_image))
            return features.reshape(features.shape[0], -1)
        except Exception as e:
            logger.error(f"VGG16 feature extraction error: {str(e)}")
            raise