import os
from predictor import CPVPredictor

def test_startup():
    print(f"CWD: {os.getcwd()}")
    
    model_dir = os.getenv("MODEL_DIR", "models/cpv-decoder")
    cpv_codes = os.getenv("CPV_CODES", "cpv_codes.csv")
    embeddings_file = "cpv_embeddings.pt"
    
    print(f"Checking model_dir: '{model_dir}' -> Exists: {os.path.exists(model_dir)}")
    print(f"Checking cpv_codes: '{cpv_codes}' -> Exists: {os.path.exists(cpv_codes)}")
    
    if not os.path.exists(model_dir) and os.path.exists("models/cpv-decoder-test"):
        print("Fallback condition met.")
        model_dir = "models/cpv-decoder-test"

    print(f"Final model_dir: {model_dir}")
    
    try:
        predictor = CPVPredictor(
            model_dir=model_dir, 
            cpv_codes_path=cpv_codes, 
            embeddings_path=embeddings_file
        )
        print(f"Predictor initialized.")
        print(f"Model loaded: {predictor.model is not None}")
        if predictor.model is None:
            print("WARNING: predictor.model is None!")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_startup()
