import os
import sys
import pickle
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime

# Add the project directory to python path
sys.path.append(os.path.abspath('.'))
from src.predict import load_prediction_artifacts

def geocode_airport(name):
    """
    Queries Nominatim API to get lat/lon coordinates for a city or airport name.
    """
    url = f"https://nominatim.openstreetmap.org/search?q={requests.utils.quote(name)}&format=json&limit=1"
    headers = {"User-Agent": "flight-delay-predictor-agent"}
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon']), data[0]['display_name']
    except Exception as e:
        print(f"Geocoding error for {name}: {e}")
    return None, None, None

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculates Haversine distance in miles.
    """
    d_lat = np.radians(lat2 - lat1)
    d_lon = np.radians(lon2 - lon1)
    a = np.sin(d_lat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return 3956 * c

def get_forecast_weather(lat, lon, scheduled_dt):
    """
    Fetches forecasted weather for a specific latitude, longitude, and date-time.
    """
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=temperature_2m,weathercode,windspeed_10m&timezone=auto"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            hourly = data.get('hourly', {})
            times = hourly.get('time', [])
            
            # Find the index of the closest forecasted hour
            target_str = scheduled_dt.strftime("%Y-%m-%dT%H:00")
            closest_idx = 0
            if target_str in times:
                closest_idx = times.index(target_str)
            else:
                # Find closest by parsing
                min_diff = float('inf')
                for idx, t_str in enumerate(times):
                    t_val = datetime.strptime(t_str, "%Y-%m-%dT%H:%M")
                    diff = abs((t_val - scheduled_dt).total_seconds())
                    if diff < min_diff:
                        min_diff = diff
                        closest_idx = idx
            
            temp_c = hourly.get('temperature_2m', [20.0])[closest_idx]
            wind_k = hourly.get('windspeed_10m', [10.0])[closest_idx]
            w_code = hourly.get('weathercode', [0])[closest_idx]
            
            # Map WMO weather code to model categories
            if wind_k > 25.0:
                weather_cond = "Windy"
                weather_desc = "Windy"
            elif w_code in [95, 96, 99]:
                weather_cond = "Stormy"
                weather_desc = "Stormy / Thunderstorm"
            elif w_code in [71, 73, 75, 77, 81, 82, 85, 86]:
                weather_cond = "Snowy"
                weather_desc = "Snowy"
            elif w_code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80]:
                weather_cond = "Rainy"
                weather_desc = "Rainy / Showers"
            else:
                weather_cond = "Sunny"
                weather_desc = "Clear / Sunny"
                
            return {
                'condition': weather_cond,
                'description': weather_desc,
                'temp_c': temp_c,
                'wind_k': wind_k
            }
    except Exception as e:
        print(f"Weather API error: {e}")
        
    return {
        'condition': 'Sunny',
        'description': 'Clear / Sunny (Default Fallback)',
        'temp_c': 20.0,
        'wind_k': 10.0
    }

def predict_custom_flight(carrier, origin_name, dest_name, departure_str):
    """
    Orchestrates coordinates fetching, weather API call, features scaling and model prediction.
    """
    # 1. Parse Date/Time
    try:
        dt = datetime.strptime(departure_str, "%Y-%m-%d %H:%M")
    except ValueError:
        print("Invalid date format. Use YYYY-MM-DD HH:MM")
        return
        
    print(f"\nAnalyzing Custom Flight: {carrier} from '{origin_name}' to '{dest_name}'")
    print(f"Departure Date/Time: {dt.strftime('%A, %b %d, %Y at %I:%M %p')}")
    
    # 2. Geocode airports
    print("Geocoding airports...")
    o_lat, o_lon, o_display = geocode_airport(origin_name)
    d_lat, d_lon, d_display = geocode_airport(dest_name)
    
    if not o_lat or not d_lat:
        print("Failed to geocode origin or destination airport coordinates.")
        return
        
    o_display_ascii = o_display.encode('ascii', errors='ignore').decode('ascii')
    d_display_ascii = d_display.encode('ascii', errors='ignore').decode('ascii')
    print(f"Origin Coordinates: {o_display_ascii[:60]}... ({o_lat:.4f}, {o_lon:.4f})")
    print(f"Dest Coordinates: {d_display_ascii[:60]}... ({d_lat:.4f}, {d_lon:.4f})")
    
    # 3. Calculate distance & scheduled duration
    distance = haversine_distance(o_lat, o_lon, d_lat, d_lon)
    # Estimate time: 450 mph average jet speed + 40 mins ground taxiing
    scheduled_time = round(distance / 7.5 + 40, 0)
    print(f"Calculated Flight Distance: {distance:.1f} miles")
    print(f"Estimated Flight Duration: {scheduled_time:.0f} minutes")
    
    # 4. Fetch Weather Forecast
    print("Fetching weather forecast from Open-Meteo...")
    weather = get_forecast_weather(o_lat, o_lon, dt)
    print(f"Forecasted weather: {weather['description']} ({weather['temp_c']}°C, Wind: {weather['wind_k']} km/h)")
    
    # 5. Build Model Input
    # Extract temporal properties
    month = dt.month
    day = dt.day
    day_of_week = dt.isoweekday() # 1=Mon, 7=Sun
    scheduled_departure = dt.hour * 100 + dt.minute
    # Approximate arrival time
    arr_dt = dt + pd.Timedelta(minutes=scheduled_time)
    scheduled_arrival = arr_dt.hour * 100 + arr_dt.minute
    
    query = {
        'YEAR': dt.year,
        'MONTH': month,
        'DAY': day,
        'DAY_OF_WEEK': day_of_week,
        'AIRLINE': carrier,
        'ORIGIN_AIRPORT': 'ZZZ', # Placeholder custom
        'DESTINATION_AIRPORT': 'ZZZ',
        'SCHEDULED_DEPARTURE': scheduled_departure,
        'SCHEDULED_ARRIVAL': scheduled_arrival,
        'SCHEDULED_TIME': scheduled_time,
        'DISTANCE': distance,
        'WEATHER_CONDITION': weather['condition'],
        'CANCELLED': 0,
        'DIVERTED': 0
    }
    
    # Load model & preprocessor
    print("Loading model artifacts...")
    model, preprocessor, threshold = load_prediction_artifacts('models', return_threshold=True)
    
    # Preprocess
    df_input = pd.DataFrame([query])
    df_feats = preprocessor.engineer_features(df_input)
    X_input, _ = preprocessor.transform(df_feats, is_train=False)
    
    # Predict
    prob = model.predict_proba(X_input)[0, 1]
    is_delayed = int(prob >= threshold)
    
    print("\n" + "="*50)
    print("DELAY PREDICTION RESULTS:")
    print("="*50)
    print(f"Flight Code:          {carrier} (Unseen/International carrier)")
    print(f"Estimated Distance:   {distance:.1f} miles")
    print(f"Estimated Duration:   {scheduled_time:.0f} minutes")
    print(f"Scheduled Weather:    {weather['description']} ({weather['temp_c']} C, Wind: {weather['wind_k']} km/h)")
    print(f"Optimal Threshold:    {threshold:.2f}")
    print(f"Calculated Delay Risk: {prob * 100:.2f}%")
    print("="*50)
    
    if prob < 0.25:
        print("[LOW RISK] STATUS: Flight is highly likely to depart on-time.")
    elif prob < 0.45:
        print("[MODERATE RISK] STATUS: Potential minor air traffic or weather delays.")
    else:
        print("[HIGH RISK] STATUS: Weather or schedule congestion introduces extreme delay risks.")
    print("="*50)

if __name__ == '__main__':
    # Test for Etihad EY-289 Abu Dhabi to Dublin today 19:45
    predict_custom_flight("EY", "Abu Dhabi Airport", "Dublin Airport", "2026-08-24 19:45")
