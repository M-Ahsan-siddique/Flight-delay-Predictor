import os
import sys
import pickle
import pandas as pd
import numpy as np

# Ensure parent directory is in sys.path so pickled objects can be resolved
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

def load_prediction_artifacts(models_dir, return_threshold=False):
    """
    Loads saved model, preprocessor, and optional optimal threshold artifacts.
    """
    model_path = os.path.join(models_dir, 'final_model.pkl')
    prep_path = os.path.join(models_dir, 'preprocessor.pkl')
    thresh_path = os.path.join(models_dir, 'optimal_threshold.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(prep_path):
        raise FileNotFoundError("Saved model or preprocessor not found. Run src/retrain_improved.py first.")
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(prep_path, 'rb') as f:
        preprocessor = pickle.load(f)
    
    # Load optimal threshold (fall back to 0.5 if not found)
    threshold = 0.5
    if os.path.exists(thresh_path):
        with open(thresh_path, 'rb') as f:
            threshold = pickle.load(f)
        
    if return_threshold:
        return model, preprocessor, threshold
    return model, preprocessor

def predict_flight_delay(flight_data, models_dir='models'):
    """
    Predicts flight delay probability and status for input data using CatBoost.
    """
    # Convert dict to DataFrame if needed
    if isinstance(flight_data, dict):
        df_input = pd.DataFrame([flight_data])
    else:
        df_input = flight_data.copy()
        
    # If standard keys are not present, add defaults
    if 'CANCELLED' not in df_input.columns: df_input['CANCELLED'] = 0
    if 'DIVERTED' not in df_input.columns: df_input['DIVERTED'] = 0
        
    # Load model and preprocessor
    model, preprocessor, threshold = load_prediction_artifacts(models_dir, return_threshold=True)
    
    # Preprocess inputs
    df_clean = preprocessor.clean_data(df_input)
    
    # Check if cleaning emptied the dataframe
    if df_clean.empty:
        df_clean = df_input.copy()
        
    df_feats = preprocessor.engineer_features(df_clean)
    X_input, _ = preprocessor.transform(df_feats, is_train=False)
    
    # Run prediction with optimal threshold
    probabilities = model.predict_proba(X_input)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    results = []
    for idx, (prob, pred) in enumerate(zip(probabilities, predictions)):
        results.append({
            'Delay Probability': float(prob),
            'Is Delayed (>=15 min)': int(pred),
            'Status': 'Delayed' if pred == 1 else 'On-Time'
        })
        
        # Add back calculated details
        if 'DISTANCE' in df_feats.columns:
            results[-1]['DISTANCE'] = float(df_feats.iloc[idx]['DISTANCE'])
        if 'SCHEDULED_TIME' in df_feats.columns:
            results[-1]['SCHEDULED_TIME'] = float(df_feats.iloc[idx]['SCHEDULED_TIME'])
            
    return pd.DataFrame(results)

if __name__ == '__main__':
    # Test sample flight prediction
    sample_flight = {
        'YEAR': 2015,
        'MONTH': 11,
        'DAY': 20,
        'DAY_OF_WEEK': 5, # Friday
        'AIRLINE': 'AA',
        'ORIGIN_AIRPORT': 'ORD',
        'DESTINATION_AIRPORT': 'LAX',
        'SCHEDULED_DEPARTURE': 1830, # 6:30 PM
        'SCHEDULED_ARRIVAL': 2115, # 9:15 PM
        'SCHEDULED_TIME': 285.0, # 4h 45m
        'DISTANCE': 1744.0,
        'CANCELLED': 0,
        'DIVERTED': 0,
        'WEATHER_CONDITION': 'Stormy',
        'INBOUND_DELAY': 15.0,
        'TURNAROUND_BUFFER': 45.0
    }
    
    print("Testing inference with sample flight:")
    print(sample_flight)
    try:
        res = predict_flight_delay(sample_flight)
        print("\nPrediction Results:")
        print(res.to_string())
    except FileNotFoundError as e:
        print("\nError:", e)
        print("Please train the model first by running train_catboost.py.")
