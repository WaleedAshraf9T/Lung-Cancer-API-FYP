# app/schemas/request_response.py
from pydantic import BaseModel
from typing import List
import enum

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    class_probabilities: dict

class ErrorResponse(BaseModel):
    error: str
    details: str = None

class SuccessResponse(BaseModel):
    message: str