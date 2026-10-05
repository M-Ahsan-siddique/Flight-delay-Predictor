import os
import json
import re
from openai import OpenAI

# Check Streamlit secrets, environment variables, or fallback
_FALLBACK_KEY = "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"
DEFAULT_NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY", _FALLBACK_KEY)
try:
    import streamlit as st
    if hasattr(st, "secrets") and "NVIDIA_API_KEY" in st.secrets:
        DEFAULT_NVIDIA_API_KEY = st.secrets["NVIDIA_API_KEY"]
except Exception:
    pass

DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_MODEL = "meta/llama-3.2-11b-vision-instruct"

# Robust list of verified operational models on NVIDIA NIM
VERIFIED_NVIDIA_MODELS = [
    "meta/llama-3.2-11b-vision-instruct",
    "nvidia/nemotron-3-super-120b-a12b",
    "meta/muse-glimmer-30b",
    "nvidia/nemotron-3-ultra-550b-a55b"
]

def query_nvidia_flight_analysis(
    flight_dict,
    weather_info,
    ml_delay_prob,
    api_key=DEFAULT_NVIDIA_API_KEY,
    base_url=DEFAULT_NVIDIA_BASE_URL,
    model=DEFAULT_MODEL,
    timeout=15.0
):
    """
    Queries NVIDIA hosted LLM (e.g. meta/llama-3.2-11b-vision-instruct) to analyze flight risk factors
    and generate an operational assessment, risk percentage, and passenger guidance.
    Includes automated fallback to verified alternative models if the primary model times out.
    """
    if not api_key:
        return {
            "success": False,
            "error": "NVIDIA API Key not provided.",
            "llm_prob": ml_delay_prob,
            "raw_text": "API key missing.",
            "model_used": model
        }

    client = OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=timeout
    )

    system_prompt = (
        "You are an expert Chief Airline Flight Operations Officer and Meteorologist. "
        "Your task is to analyze flight parameters, weather forecast, route characteristics, "
        "and turnaround delays to evaluate the probability of flight delay (>=15 min delay). "
        "You must provide an operational assessment, an estimated delay risk percentage (0-100), "
        "and clear, actionable traveler recommendations."
    )

    user_prompt = f"""
Analyze the flight delay risk for the following scheduled flight:
- Carrier / Airline: {flight_dict.get('AIRLINE', 'Unknown')}
- Route: {flight_dict.get('ORIGIN', 'Origin')} -> {flight_dict.get('DEST', 'Destination')}
- Scheduled Departure: {flight_dict.get('DEP_TIME', 'N/A')} ({flight_dict.get('DATE', 'Today')})
- Estimated Distance: {flight_dict.get('DISTANCE', 0):.1f} miles
- Flight Duration: {flight_dict.get('DURATION', 0):.0f} minutes
- Forecast Weather: {weather_info.get('desc', 'Clear')} (Temp: {weather_info.get('temp', 20)}°C, Wind: {weather_info.get('wind', 10)} km/h)
- Inbound Aircraft Delay: {flight_dict.get('INBOUND_DELAY', 0):.0f} mins
- Ground Turnaround Buffer: {flight_dict.get('TURNAROUND_BUFFER', 45):.0f} mins
- Local ML Baseline Delay Probability: {ml_delay_prob * 100:.1f}%

Format your response in structured Markdown with these exact sections:
### 🌌 Operational Risk Assessment
(2-3 sentences analyzing air traffic, weather impact, and inbound delay cascading)

### 📊 LLM Estimated Delay Probability
Estimated Risk: [Specify XX%]
Risk Level: [LOW / MODERATE / HIGH / SEVERE]

### 🛰️ Critical Risk Factors
- Factor 1: ...
- Factor 2: ...
- Factor 3: ...

### 🧭 Actionable Passenger Recommendations
- Recommendation 1: ...
- Recommendation 2: ...
"""

    # Build sequence of models to try (primary chosen model first, then verified fallbacks)
    models_to_try = [model]
    for fb in VERIFIED_NVIDIA_MODELS:
        if fb not in models_to_try:
            models_to_try.append(fb)

    last_error = ""
    for current_model in models_to_try:
        try:
            response = client.chat.completions.create(
                model=current_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=650
            )

            content = response.choices[0].message.content or ""
            if not content:
                continue
            
            # Parse out estimated risk percentage from the LLM text if available
            llm_prob = None
            match = re.search(r'Estimated Risk:\s*(\d+(?:\.\d+)?)\s*%', content, re.IGNORECASE)
            if match:
                llm_prob = float(match.group(1)) / 100.0
            else:
                pct_matches = re.findall(r'(\d+(?:\.\d+)?)\s*%', content)
                if pct_matches:
                    llm_prob = float(pct_matches[0]) / 100.0

            return {
                "success": True,
                "raw_text": content,
                "llm_prob": llm_prob if llm_prob is not None else ml_delay_prob,
                "model_used": current_model
            }

        except Exception as e:
            last_error = str(e)
            continue

    # Fallback response if all API attempts exhausted
    return {
        "success": False,
        "error": last_error,
        "llm_prob": ml_delay_prob,
        "raw_text": f"⚠️ **Model Inference Notice**: Live model inference timed out ({last_error}). Using trained baseline.",
        "model_used": model
    }
