from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import pickle
import numpy as np
import os
from typing import List
import traceback

# Initialize FastAPI app
app = FastAPI(
    title="House Price Prediction API",
    description="ML model for predicting house prices",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Debug: Check files and environment
print("🚀 Starting House Price Prediction API")
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

# Load model
model = None
model_loaded = False

try:
    # Try different possible file names
    possible_model_files = [
        'housing_price_model.pkl',
        'housing_price_model (2).pkl',
        'model.pkl'
    ]
    
    model_file = None
    for file in possible_model_files:
        if os.path.exists(file):
            model_file = file
            break
    
    if model_file:
        print(f"📁 Found model file: {model_file}")
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        model_loaded = True
        print("✅ Model loaded successfully!")
        print(f"📊 Model type: {type(model)}")
        
        # Try to get feature information
        if hasattr(model, 'n_features_in_'):
            print(f"🎯 Model expects {model.n_features_in_} features")
        if hasattr(model, 'feature_names_in_'):
            print(f"📝 Feature names: {list(model.feature_names_in_)}")
            
    else:
        print("❌ No model file found. Available files:")
        for f in os.listdir('.'):
            print(f"   - {f}")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print(f"🔍 Full error: {traceback.format_exc()}")
    model = None
    model_loaded = False

# Request model with validation
class PredictionRequest(BaseModel):
    features: List[float]
    
    @validator('features', each_item=True)
    def convert_to_float(cls, v):
        """Ensure all features are converted to float"""
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0  # Default value if conversion fails

# Response model
class PredictionResponse(BaseModel):
    prediction: float
    status: str
    features_received: int

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "House Price Prediction API is running!",
        "model_loaded": model_loaded,
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict"
        }
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model_loaded or model is None:
        raise HTTPException(
            status_code=500, 
            detail="Model not loaded. Please check server logs."
        )
    
    try:
        features = request.features
        print(f"📨 Received prediction request with {len(features)} features")
        print(f"🔢 Raw features: {features}")
        print(f"📊 Feature types: {[type(f) for f in features]}")
        
        # Validate feature count
        expected_features = 20  # From your model
        if len(features) != expected_features:
            raise HTTPException(
                status_code=400, 
                detail=f"Expected {expected_features} features, but got {len(features)}"
            )
        
        # Convert each feature to float with explicit handling
        features_float = []
        for i, feature in enumerate(features):
            try:
                # Handle None, string, or other types
                if feature is None:
                    features_float.append(0.0)
                    print(f"⚠️ Feature {i} was None, using 0.0")
                else:
                    # Convert to float explicitly
                    float_val = float(feature)
                    features_float.append(float_val)
            except (ValueError, TypeError) as e:
                print(f"❌ Error converting feature {i} ({feature}) to float: {e}")
                features_float.append(0.0)  # Default value
        
        print(f"✅ Converted features: {features_float}")
        
        # Convert to numpy array with explicit dtype and error handling
        try:
            features_array = np.array(features_float, dtype=np.float64).reshape(1, -1)
            print(f"✅ Array created - Shape: {features_array.shape}, Dtype: {features_array.dtype}")
            
            # Check for NaN or Inf values
            if np.any(np.isnan(features_array)):
                print("❌ NaN values detected in array")
                raise ValueError("Array contains NaN values")
            if np.any(np.isinf(features_array)):
                print("❌ Infinite values detected in array")
                raise ValueError("Array contains infinite values")
                
        except Exception as array_error:
            print(f"❌ Array creation error: {array_error}")
            # Fallback: try with different dtype
            features_array = np.array(features_float, dtype=float).reshape(1, -1)
            print(f"🔧 Fallback array - Shape: {features_array.shape}, Dtype: {features_array.dtype}")
        
        # Debug: Check the actual array contents
        print("🔍 Final array inspection:")
        print(f"Array: {features_array}")
        print(f"Shape: {features_array.shape}")
        print(f"Dtype: {features_array.dtype}")
        print(f"Min: {np.min(features_array)}, Max: {np.max(features_array)}")

        # Check each element individually
        for i in range(features_array.shape[1]):
            val = features_array[0, i]
            print(f"Feature {i}: {val} (type: {type(val)})")
        
        # Make prediction
        print("🎯 Making prediction...")
        prediction_result = model.predict(features_array)
        predicted_price = float(prediction_result[0])
        
        print(f"💰 Prediction result: ${predicted_price:,.2f}")
        
        return PredictionResponse(
            prediction=predicted_price,
            status="success",
            features_received=len(features)
        )
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"❌ Prediction error: {str(e)}")
        print(f"🔍 Full traceback: {error_trace}")
        
        raise HTTPException(
            status_code=400, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    health_status = {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_type": str(type(model)) if model else "None",
        "expected_features": getattr(model, 'n_features_in_', "Unknown")
    }
    return health_status

@app.get("/info")
async def model_info():
    """Get model information"""
    if not model_loaded:
        return {"error": "Model not loaded"}
    
    info = {
        "model_type": str(type(model)),
        "expected_features": getattr(model, 'n_features_in_', "Unknown"),
        "has_feature_names": hasattr(model, 'feature_names_in_')
    }
    
    if hasattr(model, 'feature_names_in_'):
        info["feature_names"] = list(model.feature_names_in_)
    
    return info

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)