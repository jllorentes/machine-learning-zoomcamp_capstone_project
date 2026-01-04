import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io

app = FastAPI(
    title="SmartClaim API",
    description="API for detecting damage to vehicles using MobileNetV3 + ONNX",
    version="1.0.0"
)

# --- CONFIGURACIÓN ---
MODEL_PATH = "car_damage.onnx"
CLASS_NAMES = ["Damaged", "Whole"] 

print("Loading ONNX model...")
try:
    session = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print("✅ Model successfully loaded.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

def preprocess_image(image_bytes):
    """
    Prepare the image so that it is identical to what the model expects:
    1. Resize to 224x224
    2. Convert to Float32 Array
    3. Normalise (ImageNet Mean/Std)
    4. Transpose (HWC -> CHW)
    5. Batch Dimension
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        img_data = np.array(image).astype('float32') / 255.0
        
        # ImageNet standard normalisation
        # mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
        mean = np.array([0.485, 0.456, 0.406], dtype='float32')
        std = np.array([0.229, 0.224, 0.225], dtype='float32')
        
        img_data = (img_data - mean) / std
        
        # Transpose from (Height, Width, Channel) to (Channel, Height, Width)
        # PyTorch/ONNX expect channels first
        img_data = img_data.transpose(2, 0, 1)
        
        # Add batch dimension: (1, 3, 224, 224)
        img_data = np.expand_dims(img_data, axis=0)
        
        return img_data
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
def home():
    return {"message": "SmartClaim API is running. Go to /docs for testing."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo debe ser una imagen.")
    
    content = await file.read()
    
    input_tensor = preprocess_image(content)
    
    # Inference
    outputs = session.run(None, {input_name: input_tensor})
    
    logits = outputs[0][0]
    
    # Apply Softmax to obtain probabilities
    probs = np.exp(logits) / np.sum(np.exp(logits))
    
    predicted_idx = np.argmax(probs)
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    return {
        "filename": file.filename,
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": {
            "damaged": float(probs[0]),
            "whole": float(probs[1])
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9696)