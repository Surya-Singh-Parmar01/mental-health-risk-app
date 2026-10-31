# Explainable Mental Health Risk Explorer

Pipeline: survey answers -> Random Forest (Low / Medium / High) -> SHAP explanation -> personalised suggestions -> what-if explorer.

## Run on localhost
1. Keep all files in ONE folder: app.py, risk_model_final.pkl, label_encoder_final.pkl, presets.json, requirements.txt
2. Open a terminal in that folder:
   pip install -r requirements.txt
   streamlit run app.py
3. Open http://localhost:8501 (it usually opens automatically).

scikit-learn is pinned to 1.6.1 because the model was trained with that version.

## Using it
- Left sidebar: answer the questions (results update instantly, no submit button).
- Try the Low / Medium / High example buttons, or Reset.
- Tabs: Result, Why this result (hover the bars), Suggestions, What-if explorer.

Research prototype only: not a diagnostic or crisis tool.
