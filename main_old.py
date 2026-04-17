from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import ml_utils
from typing import Optional
import json

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for the current dataframe
# Note: In a production app, you'd use a more robust way to handle state
current_data = {"df": None}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        df = ml_utils.parse_file(content, file.filename)
        current_data["df"] = df
        
        stats = ml_utils.get_basic_stats(df)
        preview = df.head(5).to_dict(orient='records')
        
        return {
            "filename": file.filename,
            "preview": preview,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/visualize")
async def visualize():
    if current_data["df"] is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    try:
        visuals = ml_utils.get_visualizations(current_data["df"])
        return visuals
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
async def predict(target_col: str = Form(...)):
    if current_data["df"] is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    
    df = current_data["df"]
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column {target_col} not found")

    try:
        results = ml_utils.perform_prediction(df, target_col)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
