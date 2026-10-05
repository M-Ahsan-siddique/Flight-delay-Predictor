# 🛡️ DelayGuard — Flight Delay Prediction & Risk Intelligence Dashboard

An end-to-end Machine Learning and AI-powered Flight Delay and Cancellation Risk Intelligence platform built with **Streamlit**, **CatBoost**, and **NVIDIA NIM LLMs**.

---

## ✈️ Overview

DelayGuard analyzes scheduled flight parameters, route characteristics, turnaround buffers, and weather conditions to deliver actionable operational risk assessments:
- **CatBoost ML Engine**: Predicts flight delay probabilities (>=15 min delay) trained on millions of historical flights.
- **AI Operations Briefing**: LLM-synthesized operational risk analysis powered by NVIDIA NIM.
- **Airport & Airline Intelligence**: Global airport database and carrier delay tendency metrics.
- **Interactive Visualizations**: Interactive route congestion, turnaround buffer analysis, and key risk driver breakdowns.

---

## 🚀 Getting Started Locally

### 1. Clone the repository
```bash
git clone https://github.com/M-Ahsan-siddique/Flight-delay-Predictor.git
cd Flight-delay-Predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## ☁️ Deployment

This application is ready to deploy on **Streamlit Community Cloud**:
1. Connect this GitHub repository on [share.streamlit.io](https://share.streamlit.io).
2. Set the main file path to `app.py`.
3. In **Advanced Settings > Secrets**, optionally add your NVIDIA API key:
   ```toml
   NVIDIA_API_KEY = "your-nvidia-api-key"
   ```
4. Click **Deploy**!
