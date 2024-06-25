import sys
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import base64
import tempfile
import json
import logging
import os
import warnings
from utils.image_processing import UndertonePredictor


# f the warnings (sorry :( )
warnings.filterwarnings("ignore", category=DeprecationWarning, module='google._upb._message')
warnings.filterwarnings("ignore", category=DeprecationWarning, module='fdlite.iris_landmark')
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.base')

app = FastAPI()

# Initialize the model with the path to the pretrained model
model = UndertonePredictor('models/skin_tone_classifier.pkl')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        print("Starting prediction:")
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            img.save(tmp.name)
            img_path = tmp.name

        try:
            # Perform predictions using the saved image path
            undertone = model.predict_undertone(img_path)
            tone_palette = ['#533023', '#6C4131', '#A36F48', '#BF8861', '#ECD0BA', '#F8E5D6']
            tone_labels = ['deep', 'medium-deep', 'medium', 'light-medium', 'light', 'fair']
            tone = model.predict_tone(img_path, tone_palette, tone_labels)
        finally:
            # Ensure the temporary file is removed
            os.remove(img_path)
            img.close()
        
        response = {
            'undertone': undertone,
            'tone': tone
        }
        print("Returning response:\n", response)
        return JSONResponse(content=response)
    
    except Exception as e:
        print("error:", e)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)