from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).with_name("model.joblib")


@st.cache_resource(show_spinner=False)
def load_bundle() -> tuple[Any, list[str], list[str]]:
    """Load the persisted model bundle and return model, features, and categorical columns."""
    bundle = joblib.load(MODEL_PATH)

    if isinstance(bundle, dict):
        model = (
            bundle.get("model")
            or bundle.get("lgb_model")
            or bundle.get("cb_model")
            or next(
                (
                    value
                    for value in bundle.values()
                    if hasattr(value, "predict")
                ),
                None,
            )
        )
        features = list(bundle.get("features", []))
        cat_cols = list(bundle.get("cat_cols", []))
    else:
        model = bundle
        features = list(getattr(bundle, "feature_name_", []) or getattr(bundle, "feature_names_in_", []))
        cat_cols = []

    if model is None:
        raise ValueError("No predictive model was found in model.joblib.")
    if not features:
        raise ValueError("No training feature list was found in model.joblib.")

    return model, features, cat_cols


def infer_probability(prediction: float) -> float:
    """Normalize model output to a 0-1 accident risk score."""
    if 0.0 <= prediction <= 1.0:
        return float(prediction)

    # Fallback for regression / log-odds style outputs.
    return 1.0 / (1.0 + math.exp(-prediction))


def risk_label(score: float) -> str:
    if score < 0.3:
        return "Low"
    if score <= 0.6:
        return "Medium"
    return "High"


def build_feature_row(inputs: dict[str, Any], features: list[str], cat_cols: list[str]) -> pd.DataFrame:
    """Recreate feature engineering and align a single row to the training schema."""
    row = dict(inputs)

    # Requested feature engineering.
    row["traffic_complexity"] = row["num_lanes"] * row["curvature"]
    row["speed_risk"] = row["speed_limit"] * row["curvature"]
    row["curvature_sq"] = row["curvature"] ** 2
    row["road_weather"] = f"{row['road_type']}_{row['weather']}"
    row["time_weather"] = f"{row['time_of_day']}_{row['weather']}"

    # Sensible fallbacks for common engineered features if the training bundle expects them.
    row.setdefault("accident_per_lane", row["num_reported_accidents"] / max(row["num_lanes"], 1))
    row.setdefault("risk_interaction", row["speed_limit"] * row["num_reported_accidents"] * row["curvature"])
    row.setdefault("log_speed", math.log1p(row["speed_limit"]))
    row.setdefault("public_road", 1 if row["road_type"] in {"urban", "rural"} else 0)
    row.setdefault("road_signs_present", 1 if row["road_type"] != "highway" else 0)
    row.setdefault("holiday", 0)
    row.setdefault("school_season", 1)

    aligned = {}
    for feature in features:
        if feature in row:
            aligned[feature] = row[feature]
        elif feature.endswith("_te"):
            aligned[feature] = 0.0
        elif feature in cat_cols:
            aligned[feature] = "unknown"
        else:
            aligned[feature] = 0.0

    frame = pd.DataFrame([aligned], columns=features)
    for col in cat_cols:
        if col in frame.columns:
            frame[col] = frame[col].astype("category")

    return frame


st.set_page_config(page_title="Accident Risk Dashboard", page_icon="🚧", layout="wide")
st.title("🚧 Accident Risk Prediction Dashboard")
st.write(
    "Use the sidebar to describe current road conditions and estimate the predicted accident risk."
)

try:
    model, features, cat_cols = load_bundle()
except Exception as exc:  # pragma: no cover - Streamlit runtime feedback
    st.error(f"Unable to load model.joblib: {exc}")
    st.stop()

with st.sidebar:
    st.header("Input Features")
    num_lanes = st.slider("num_lanes", min_value=1, max_value=4, value=2, step=1)
    curvature = st.slider("curvature", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    speed_limit = st.slider("speed_limit", min_value=20, max_value=80, value=45, step=5)
    num_reported_accidents = st.slider(
        "num_reported_accidents", min_value=0, max_value=7, value=1, step=1
    )
    road_type = st.selectbox("road_type", ["urban", "rural", "highway"])
    weather = st.selectbox("weather", ["clear", "rainy", "foggy"])
    lighting = st.selectbox("lighting", ["daylight", "dim"])
    time_of_day = st.selectbox(
        "time_of_day", ["morning", "afternoon", "evening", "night"]
    )

input_payload = {
    "num_lanes": num_lanes,
    "curvature": curvature,
    "speed_limit": speed_limit,
    "num_reported_accidents": num_reported_accidents,
    "road_type": road_type,
    "weather": weather,
    "lighting": lighting,
    "time_of_day": time_of_day,
}

feature_frame = build_feature_row(input_payload, features, cat_cols)
raw_prediction = float(model.predict(feature_frame)[0])
score = infer_probability(raw_prediction)
label = risk_label(score)

left, right = st.columns([1.3, 1])
with left:
    st.subheader("Prediction Result")
    st.metric("Accident risk score", f"{score:.2%}")
    label_colors = {
        "Low": "#16a34a",
        "Medium": "#d97706",
        "High": "#dc2626",
    }
    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:0.75rem;background:{label_colors[label]}15;border:1px solid {label_colors[label]};">
            <div style="font-size:0.95rem;color:#475569;">Risk label</div>
            <div style="font-size:2rem;font-weight:700;color:{label_colors[label]};">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "Labels: Low (<0.30), Medium (0.30–0.60), High (>0.60)."
    )

with right:
    st.subheader("Current Scenario")
    st.dataframe(pd.DataFrame([input_payload]), hide_index=True, use_container_width=True)

with st.expander("Aligned model features"):
    st.dataframe(feature_frame, hide_index=True, use_container_width=True)
    st.caption(
        "Categorical columns are cast to pandas category dtype and columns are ordered to match the training feature schema."
    )
