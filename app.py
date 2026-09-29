import streamlit as st
import pandas as pd
import joblib
import numpy as np


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="F1 Podium Predictor",
    page_icon="🏎️",
    layout="wide"
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏎️ F1 Predictor")

    st.markdown("---")

    st.markdown("### 🤖 Model")
    st.write("Random Forest V2")

    st.markdown("### 📊 Features")
    st.write("16 engineered features")

    st.markdown("### 🌳 Trees")
    st.write("500 decision trees")

    st.markdown("---")

    st.markdown(
        """
        **About**

        This application uses machine learning
        and historical Formula 1 data to estimate
        podium probabilities for each driver.
        """
    )

    st.markdown("---")

    st.caption("F1 Race Predictor • ML Project")
# ============================================================
# FILE PATHS
# ============================================================

MODEL_FILE = "notebooks/f1_podium_predictor_v2.pkl"
RACE_DATA_FILE = "notebooks/future_race_features.csv"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    model_package = joblib.load(MODEL_FILE)
    return model_package


model_package = load_model()

model = model_package["model"]
features = model_package["features"]


# ============================================================
# TITLE
# ============================================================

st.title("🏎️ F1 Podium Predictor")

st.markdown(
    """
    ### Machine Learning Based Formula 1 Podium Prediction

    This application uses a **Random Forest machine learning model**
    trained on historical Formula 1 race data to estimate each
    driver's probability of finishing on the podium.
    """
)

st.divider()



# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader("🤖 Model Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Random Forest"
    )

with col2:
    st.metric(
        "Features",
        len(features)
    )

with col3:
    st.metric(
        "Trees",
        model.n_estimators
    )


st.divider()


# ============================================================
# LOAD RACE DATA
# ============================================================

try:

    race_data = pd.read_csv(RACE_DATA_FILE)

except FileNotFoundError:

    st.error(
        "Race feature file was not found:\n\n"
        f"{RACE_DATA_FILE}"
    )

    st.stop()

    # ============================================================
# RACE INFORMATION
# ============================================================

st.subheader("🏁 Race Information")

race_col1, race_col2, race_col3 = st.columns(3)

with race_col1:
    st.metric(
        "Race",
        "Dutch Grand Prix"
    )

with race_col2:
    st.metric(
        "Season",
        "2026"
    )

with race_col3:
    st.metric(
        "Drivers",
        len(race_data)
    )

st.divider()


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in race_data.columns
]

if missing_features:

    st.error("Required model features are missing:")

    for feature in missing_features:
        st.write(f"- {feature}")

    st.stop()


# ============================================================
# MAKE PREDICTIONS
# ============================================================

X = race_data[features]

race_data["podium_probability"] = (
    model.predict_proba(X)[:, 1] * 100
)


# ============================================================
# SORT PREDICTIONS
# ============================================================

predictions = race_data[
    ["driver_name", "podium_probability"]
].copy()

predictions = predictions.sort_values(
    "podium_probability",
    ascending=False
).reset_index(drop=True)

predictions["predicted_podium_position"] = 0

predictions.loc[:2, "predicted_podium_position"] = [
    1,
    2,
    3
]

predictions["podium_probability"] = (
    predictions["podium_probability"].round(2)
)


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.subheader("🎯 Model Performance")

perf_col1, perf_col2, perf_col3 = st.columns(3)

with perf_col1:
    st.metric(
        "Average Correct Podium Drivers",
        "1.88 / 3"
    )

with perf_col2:
    st.metric(
        "3/3 Podium Driver Accuracy",
        "20.78%"
    )

with perf_col3:
    st.metric(
        "Exact P1-P2-P3 Accuracy",
        "7.50%"
    )

st.caption(
    "Historical retrospective evaluation of the V2 model. "
    "These metrics are in-sample because the model was trained "
    "on the full historical dataset before evaluation."
)

st.divider()


# ============================================================
# PODIUM PREDICTION
# ============================================================

st.subheader("🏆 Predicted Podium")

top3 = predictions.head(3)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "🥇 P1",
        top3.iloc[0]["driver_name"],
        f"{top3.iloc[0]['podium_probability']:.2f}%"
    )

with col2:

    st.metric(
        "🥈 P2",
        top3.iloc[1]["driver_name"],
        f"{top3.iloc[1]['podium_probability']:.2f}%"
    )

with col3:

    st.metric(
        "🥉 P3",
        top3.iloc[2]["driver_name"],
        f"{top3.iloc[2]['podium_probability']:.2f}%"
    )


st.divider()


# ============================================================
# PREDICTION TABLE
# ============================================================

st.subheader("📊 Driver Predictions")

display_df = predictions.copy()

display_df["Rank"] = range(
    1,
    len(display_df) + 1
)

display_df = display_df[
    [
        "Rank",
        "driver_name",
        "podium_probability",
        "predicted_podium_position"
    ]
]

display_df.columns = [
    "Rank",
    "Driver",
    "Podium Probability (%)",
    "Predicted Podium Position"
]

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


st.divider()

# ============================================================
# DOWNLOAD PREDICTIONS
# ============================================================

csv_data = display_df.to_csv(index=False)

st.download_button(
    label="📥 Download Predictions CSV",
    data=csv_data,
    file_name="f1_podium_predictions.csv",
    mime="text/csv"
)
# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.subheader("🧠 Feature Importance")

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
).head(10)

feature_importance = feature_importance.set_index("Feature")

st.bar_chart(feature_importance)

# ============================================================
# PROBABILITY CHART
# ============================================================

st.subheader("📈 Podium Probability")

chart_data = predictions[
    ["driver_name", "podium_probability"]
].copy()

chart_data = chart_data.set_index(
    "driver_name"
)

st.bar_chart(
    chart_data
)


# ============================================================
# ACTUAL VS PREDICTED PODIUM
# ============================================================

st.subheader("🏁 Actual vs Predicted Podium")

actual_col, predicted_col = st.columns(2)

with actual_col:
    st.markdown("### 🏆 Actual Podium")

    st.write("🥇 Lando Norris")
    st.write("🥈 Andrea Kimi Antonelli")
    st.write("🥉 George Russell")

with predicted_col:
    st.markdown("### 🤖 Predicted Podium")

    st.write("🥇 George Russell")
    st.write("🥈 Lando Norris")
    st.write("🥉 Andrea Kimi Antonelli")

st.divider()



# ============================================================
# PREDICTION SUMMARY
# ============================================================

st.subheader("📌 Prediction Summary")

st.success(
    "The model correctly identified all 3 drivers "
    "who finished on the actual podium."
)

st.info(
    "The predicted podium order differed from the actual "
    "finishing order."
)

st.divider()

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.subheader("📌 About This Project")

st.write(
    """
    The F1 Podium Predictor uses historical Formula 1 data and
    machine learning to estimate podium probabilities.

    The V2 model uses 16 engineered features including:

    • Grid position
    • Qualifying position
    • Previous race performance
    • Recent driver form
    • Recent constructor form
    • Qualifying/grid gap
    • Combined form scores
    • Qualifying strength
    """
)

st.caption(
    "F1 Podium Predictor — Random Forest V2"
)