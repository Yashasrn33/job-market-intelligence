"""Forecasts page -- placeholder for ML-driven skill forecasting."""
import streamlit as st

st.set_page_config(page_title="Forecasts", layout="wide")

st.title("🔮 Skill Demand Forecasts")
st.info(
    "This page will display time-series forecasts once the ML training pipeline "
    "(`ml/training/skill_forecast.py`) has been executed and results are stored in S3."
)

st.markdown("""
### Planned features
- 30/60/90-day demand projections per skill
- Emerging skill anomaly detection
- Confidence intervals via Prophet / ARIMA
""")
