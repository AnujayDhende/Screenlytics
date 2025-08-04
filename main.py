from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib
import os
from collections import defaultdict
import re

# Initialize FastAPI and templates
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load pre-trained model pipeline
model = joblib.load("logistic_model_pipeline.pkl")
preproc = model.named_steps['preproc']
clf = model.named_steps['clf']

# Feature definitions
numeric_features = ['Age', 'Avg_Daily_Screen_Time_hr', 'Educational_to_Recreational_Ratio']
categorical_features = ['Gender', 'Primary_Device', 'Health_Impacts', 'Urban_or_Rural']
feature_aliases = {
    "Age": "Age",
    "Avg_Daily_Screen_Time_hr": "Average Daily Screen Time (hours)",
    "Educational_to_Recreational_Ratio": "Educational vs. Recreational Usage Ratio",
    "Gender": "Gender",
    "Primary_Device": "Main Device Used",
    "Health_Impacts": "Reported Health Impacts",
    "Urban_or_Rural": "Living Environment"
}

# Utility functions
def consolidate_features(features):
    merged = defaultdict(float)
    for name, val in features:
        merged[name] += val
    return list(merged.items())

def clean_feedback(text):
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    seen, result = set(), []
    for s in parts:
        norm = s.lower().strip()
        if norm not in seen:
            seen.add(norm)
            result.append(s)
    return " ".join(result)

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("form.html", {"request": request, "result": None, "explanation": None, "feedback": None})

@app.post("/", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: float = Form(...),
    screen_time: float = Form(...),
    ratio: float = Form(...),
    gender: str = Form(...),
    device: str = Form(...),
    health: str = Form(...),
    location: str = Form(...)
):
    # 1. Prepare input
    feature_values = {
        "Age": age,
        "Avg_Daily_Screen_Time_hr": screen_time,
        "Educational_to_Recreational_Ratio": ratio,
        "Gender": gender,
        "Primary_Device": device,
        "Health_Impacts": health,
        "Urban_or_Rural": location
    }
    input_df = pd.DataFrame([feature_values])

    # 2. Predict
    pred = model.predict(input_df)[0]
    result = "Exceeded Limit" if pred else "Within Limit"

    # 3. Compute SHAP lazily
    import shap
    df_bg = pd.read_csv("Indian_Kids_Screen_Time.csv")[numeric_features + categorical_features]
    df_bg_sample = df_bg.sample(n=50, random_state=0)
    X_bg = preproc.transform(df_bg_sample)
    explainer = shap.LinearExplainer(clf, X_bg)
    X_proc = preproc.transform(input_df)
    shap_vals = explainer.shap_values(X_proc)
    # Ensure single-sample array
    if isinstance(shap_vals, list):
        arr = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
    else:
        arr = shap_vals
    # arr shape: (n_samples, n_features) or (n_features,)
    if arr.ndim == 2:
        sample_contribs = arr[0]
    else:
        sample_contribs = arr

    # Map contributions back to base features
    cat_ohe = preproc.named_transformers_['cat']['onehot']
    cat_names = cat_ohe.get_feature_names_out(categorical_features)
    feature_names = numeric_features + list(cat_names)
    raw_feats = list(zip(feature_names, sample_contribs))
    grouped = defaultdict(float)
    for name, val in raw_feats:
        base = name.split('_')[0] if name in cat_names else name
        grouped[base] += float(val)
    top_feats = consolidate_features(list(grouped.items()))
    top_feats = sorted(top_feats, key=lambda x: abs(x[1]), reverse=True)[:3]

    # 4. Generate explanations and feedback using smaller Flan-T5
    from transformers import pipeline as hf_pipeline
    flan = hf_pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7
    )

    # Build explanation prompt
    prompt = (
        "You're a kind assistant helping a concerned parent understand their child's screen time habits. Below are the most important factors and how they influenced the result:\n\n"
    )
    any_increase = False
    for name, contrib in top_feats:
        pretty = feature_aliases.get(name, name)
        value = feature_values.get(name, "N/A")
        direction = "increased" if contrib > 0 else "reduced"
        if contrib > 0:
            any_increase = True
        prompt += f"- {pretty} was {value}, which {direction} the chances of exceeding healthy screen time.\n"
    prompt += (
        "\nPlease explain this in simple and caring language (2–3 sentences). Avoid repeating feature names."
    )
    if any_increase:
        prompt += " Also, suggest one gentle tip to help improve the child's screen time habits."
    explanation = flan(prompt)[0]['generated_text'].strip()

    # Follow-up coaching tips
    feedback_prompt = (
        f"Here’s an explanation of why a child's screen time prediction was made:\n\n{explanation}\n\n"
        "Now, as a friendly parenting coach, give 2–3 distinct, concise, and non-repetitive suggestions to help the parent support healthier screen time habits. Do not repeat any phrase."
    )
    raw_feedback = flan(feedback_prompt)[0]['generated_text'].strip()
    feedback = clean_feedback(raw_feedback)

    return templates.TemplateResponse("form.html", {"request": request, "result": result, "explanation": explanation, "feedback": feedback})

# Use Uvicorn when running directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
