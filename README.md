# 🏎️ F1 Podium Predictor

A Machine Learning project that predicts the probability of Formula 1 drivers finishing on the podium using historical F1 race data.

## 📌 Project Overview

The F1 Podium Predictor uses a Random Forest classification model to estimate the podium probability of each driver.

The project uses historical race performance, qualifying performance, driver form, and constructor form to generate predictions.

## 🤖 Machine Learning Model

- Model: Random Forest Classifier
- Version: V2
- Number of trees: 500
- Features: 16 engineered features
- Target: Podium finish

## 📊 Features Used

The model uses the following features:

- Grid position
- Qualifying position
- Qualifying missing indicator
- Previous finish position
- Previous podium
- Previous points
- Recent 5 podiums
- Recent 5 points
- Constructor recent 5 podiums
- Constructor recent 5 points
- Qualifying/grid gap
- Recent form score
- Constructor form score
- Combined form score
- Previous result score
- Qualifying strength

## 🎯 Model Performance

Historical retrospective evaluation of the V2 model:

| Metric | Result |
|---|---:|
| Average correct podium drivers | 1.88 / 3 |
| 3/3 podium-driver accuracy | 20.78% |
| Exact P1-P2-P3 accuracy | 7.50% |

> Note: These metrics are retrospective/in-sample evaluation results because the V2 model was trained on the full historical dataset before evaluation. They should not be interpreted as unbiased out-of-sample performance.

## 🏁 Application Features

The Streamlit application provides:

- 🏆 Predicted podium
- 📊 Driver podium probabilities
- 🧠 Feature importance
- 🎯 Model performance metrics
- 🏁 Actual vs predicted podium comparison
- 📥 Download predictions as CSV
- 🤖 Model information
- 🏎️ Race information

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook
- Git & GitHub

## 📁 Project Structure

```text
F1-Race-Predictor/
│
├── app/
├── data/
│   └── raw/
│
├── models/
│
├── notebooks/
│   ├── f1_podium_predictor_v2.pkl
│   ├── future_race_features.csv
│   └── f1_podium_predictor_config.json
│
├── src/
│
├── app.py
├── requirements.txt
├── README.md
└── test_setup.py

▶️ How to Run
1. Clone the repository
git clone <your-github-repository-url>
2. Open the project
cd F1-Race-Predictor
3. Install dependencies
pip install -r requirements.txt
4. Run the Streamlit application
streamlit run app.py

The application will open in your browser.

🔮 Future Improvements

Possible future improvements include:

Real-time F1 race data integration
Automatic qualifying data updates
Driver and constructor dashboards
More advanced ML models
Model comparison
Race-by-race prediction history
Automatic prediction updates
Deployment to a cloud platform
👨‍💻 Project

F1 Race Predictor

Machine Learning project for Formula 1 podium prediction.


Then press **Ctrl + S**.

**Don't push to GitHub yet.** Tomorrow we'll do the GitHub cleanup and push together.

Tell me **“saved”** when the README is saved.