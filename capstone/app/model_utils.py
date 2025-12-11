# This is the Data Pipeline. Handling loading, feature engineering, and prediction.
import pandas as pd
import joblib
import numpy as np

# Load assets (Global variables for caching)
MODEL = None
POKEDEX = None

# Simplified Type Chart (You can expand this with the full chart if you have it)
TYPE_CHART = {
    'Fire': {'Grass': 2.0, 'Water': 0.5, 'Fire': 0.5},
    'Water': {'Fire': 2.0, 'Grass': 0.5, 'Water': 0.5},
    'Grass': {'Water': 2.0, 'Fire': 0.5, 'Grass': 0.5},
    'Electric': {'Water': 2.0, 'Ground': 0.0},
    'Normal': {'Ghost': 0.0},
    # ... (Add key types as needed for the demo)
}

def load_artifacts():
    """Loads the ML model and Pokemon Data into memory."""
    global MODEL, POKEDEX
    # Load the Random Forest Model
    MODEL = joblib.load('pokemon_battle_model.pkl')
    
    # Load the Pokedex (Database of stats)
    # Ensure pokemon.csv is in the same directory!
    POKEDEX = pd.read_csv('pokemon.csv').set_index('#')
    print("✅ Model and Pokedex Loaded successfully.")

def get_type_advantage(attacker_type, defender_type):
    """Calculates type multiplier."""
    if not attacker_type or not defender_type:
        return 1.0
    
    # Default to 1.0 if type interaction not in our simple map
    return TYPE_CHART.get(attacker_type, {}).get(defender_type, 1.0)

def prepare_battle_data(p1_id, p2_id):
    """
    Data Pipeline: 
    1. Fetches stats for both IDs.
    2. Calculates Differentials (Speed_Diff, etc.).
    3. Calculates Type Advantages.
    4. Returns a single-row DataFrame ready for the model.
    """
    p1 = POKEDEX.loc[p1_id]
    p2 = POKEDEX.loc[p2_id]
    
    # 1. Feature Engineering (Differentials)
    features = {
        'Speed_Diff': p1['Speed'] - p2['Speed'],
        'Attack_Diff': p1['Attack'] - p2['Attack'],
        'Defense_Diff': p1['Defense'] - p2['Defense'],
        'Sp. Atk_Diff': p1['Sp. Atk'] - p2['Sp. Atk'],
        'Sp. Def_Diff': p1['Sp. Def'] - p2['Sp. Def'],
        'HP_Diff': p1['HP'] - p2['HP'],
    }
    
    # 2. Type Advantage Logic
    p1_adv = get_type_advantage(p1['Type 1'], p2['Type 1'])
    p2_adv = get_type_advantage(p2['Type 1'], p1['Type 1'])
    features['Type_Win_Score'] = p1_adv - p2_adv
    
    return pd.DataFrame([features])

def predict_battle(p1_id, p2_id):
    """Runs the model and returns prediction + probability."""
    if MODEL is None:
        load_artifacts()
        
    input_data = prepare_battle_data(p1_id, p2_id)
    
    # Get Probability (Confidence)
    probs = MODEL.predict_proba(input_data)[0] # Returns [Prob_Loss, Prob_Win]
    win_probability = probs[1]
    
    winner = "Player 1" if win_probability > 0.5 else "Player 2"
    
    return {
        "winner": winner,
        "win_probability": round(win_probability, 4),
        "p1_name": POKEDEX.loc[p1_id]['Name'],
        "p2_name": POKEDEX.loc[p2_id]['Name']
    }