# app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.services.prediction_service import PredictionService
from app.schemas.request_response import PredictionResponse, ErrorResponse, SuccessResponse
import logging
import uvicorn
import traceback

# Setup logging
logging.basicConfig(
    filename=settings.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for lung cancer detection from CT scan images",
    version="1.0.0"
)

# Security
api_key_header = APIKeyHeader(name="X-API-Key")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction service
prediction_service = PredictionService()

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

@app.get("/", response_model=SuccessResponse)
async def root():
    """
    Healthcheck endpoint
    """
    return {"message": "API is launched and working fine"}
    
@app.post(
    f"{settings.API_V1_STR}/predict",
    response_model=PredictionResponse,
    responses={
        403: {"model": ErrorResponse},
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse}
    }
)
async def predict(
    file: UploadFile = File(...),
    api_key: str = Security(verify_api_key)
):
    """
    Predict lung cancer from CT scan image
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an image"
            )
        
        # Read image as bytes
        contents = await file.read()
        
        # Make prediction
        try:
            result = prediction_service.predict(contents)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        
        logger.info(f"Successful prediction for file: {file.filename}")
        return result
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)