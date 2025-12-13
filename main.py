print("LOADING MAIN.PY REVISION 5")
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from predictor import CPVPredictor
import os

app = FastAPI(title="CPV Decoder API", description="API for predicting CPV codes from text.")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

# Global predictor instance
predictor = None

class PredictionRequest(BaseModel):
    text: str

class MatchResult(BaseModel):
    cpv_code: str
    name: str

class ScoredResult(BaseModel):
    cpv_code: str
    name: str
    score: Optional[float] = None
    probability: Optional[float] = None

class FinalDecision(BaseModel):
    cpv_code: str
    name: str
    confidence: float
    sources: List[str]

class PredictionResponse(BaseModel):
    keyword_matches: List[MatchResult]
    rag_matches: List[ScoredResult]
    model_predictions: List[ScoredResult]
    final_decision: List[FinalDecision]

@app.on_event("startup")
async def startup_event():
    global predictor
    try:
        cwd = os.getcwd()
        print(f"DEBUG: CWD is {cwd}")
        print(f"DEBUG: Directory contents: {os.listdir(cwd)}")
        
        # Check enviroment variables or default paths
        # Force absolute paths
        model_dir = os.path.join(cwd, "models", "cpv-decoder")
        cpv_codes = os.path.join(cwd, "cpv_codes.csv")
        embeddings_file = os.path.join(cwd, "cpv_embeddings.pt")
        
        print(f"DEBUG: Checking model_dir: {model_dir}")
        
        # Fallback to test model if main doesn't exist (for development)
        if not os.path.exists(model_dir):
            test_dir = os.path.join(cwd, "models", "cpv-decoder-test")
            if os.path.exists(test_dir):
                print("Warning: Main model not found, falling back to test model.")
                model_dir = test_dir
            else:
                print(f"ERROR: Model dir not found at {model_dir} or {test_dir}")

        print(f"Loading resources from: {model_dir}")
        predictor = CPVPredictor(
            model_dir=model_dir, 
            cpv_codes_path=cpv_codes, 
            embeddings_path=embeddings_file
        )
        print("Predictor initialized successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR in startup: {e}")
        import traceback
        traceback.print_exc()

@app.get("/health")
async def health_check():
    if predictor and predictor.model:
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.get("/debug_load")
async def debug_load():
    debug_info = {}
    try:
        import sys
        cwd = os.getcwd()
        debug_info["cwd"] = cwd
        debug_info["dir_contents"] = os.listdir(cwd)
        debug_info["model_dir_env"] = os.getenv("MODEL_DIR")
        
        # Check specific paths
        model_path = os.path.join(cwd, "models", "cpv-decoder")
        debug_info["model_path_absolute"] = model_path
        debug_info["model_path_exists"] = os.path.exists(model_path)
        
        if os.path.exists(model_path):
             debug_info["model_dir_contents"] = os.listdir(model_path)
        
        # Manually trigger load
        if predictor:
            debug_info["predictor_id"] = id(predictor)
            debug_info["predictor_model_dir"] = predictor.model_dir
            debug_info["predictor_model_dir_exists"] = os.path.exists(predictor.model_dir)
            
            predictor._load_resources()
            debug_info["load_status"] = "Resources reloaded"
            debug_info["model_loaded"] = predictor.model is not None
            debug_info["model_type"] = str(type(predictor.model))
            debug_info["cpv_codes_len"] = len(predictor.cpv_descriptions)
        else:
            debug_info["error"] = "Predictor instance is None"
            
    except Exception as e:
        debug_info["exception"] = str(e)
        import traceback
        debug_info["traceback"] = traceback.format_exc()
        
    return debug_info

@app.post("/predict", response_model=PredictionResponse)
async def predict_cpv(request: PredictionRequest):
    if not predictor:
        raise HTTPException(status_code=503, detail="Predictor not initialized")
    
    try:
        raw_results = predictor.predict(request.text)
        
        # Transform to Pydantic models
        response = PredictionResponse(
            keyword_matches=[
                MatchResult(cpv_code=r['cpv_code'], name=r['name'])
                for r in raw_results['keyword_matches']
            ],
            rag_matches=[
                ScoredResult(cpv_code=r['cpv_code'], name=r['name'], score=r['score']) 
                for r in raw_results['rag_matches']
            ],
            model_predictions=[
                ScoredResult(cpv_code=r['cpv_code'], name=r['name'], probability=r['probability']) 
                for r in raw_results['model_predictions']
            ],
            final_decision=[
                FinalDecision(cpv_code=r['cpv_code'], name=r['name'], confidence=r['confidence'], sources=r['sources'])
                for r in raw_results['final_decision']
            ]
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
