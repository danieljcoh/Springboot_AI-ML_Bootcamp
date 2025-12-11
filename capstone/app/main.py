# This will use FastAPI to create a web server.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import model_utils
import logging

# 1. Setup Logging (Rubric Requirement)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BattleBrainAPI")

# 2. Initialize App
app = FastAPI(
    title="Pokémon BattleBrain API",
    description="Predicts the winner of a Pokemon battle using a Random Forest model.",
    version="1.0"
)

# 3. Load Model on Startup
@app.on_event("startup")
def startup_event():
    logger.info("Loading model artifacts...")
    model_utils.load_artifacts()

# 4. Define Input Schema
class BattleRequest(BaseModel):
    pokemon_1_id: int
    pokemon_2_id: int

# 5. Define API Endpoints
@app.get("/")
def home():
    return {"message": "Welcome to BattleBrain! Go to /docs for the UI."}

@app.post("/predict")
def predict_battle(request: BattleRequest):
    """
    Accepts two Pokemon IDs and returns the predicted winner.
    """
    logger.info(f"Received battle request: {request.pokemon_1_id} vs {request.pokemon_2_id}")
    
    try:
        result = model_utils.predict_battle(request.pokemon_1_id, request.pokemon_2_id)
        return result
    except KeyError:
        logger.error("Pokemon ID not found.")
        raise HTTPException(status_code=404, detail="Pokemon ID not found in Pokedex.")
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))