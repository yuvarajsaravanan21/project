from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os
import uuid

PIPELINE_PATH = "house_price_pipeline.pkl"

app = Flask(__name__, static_folder="static", template_folder="templates")

# ====== Load ML Pipeline ======
if not os.path.exists(PIPELINE_PATH):
    raise FileNotFoundError(f"Pipeline not found. Run `python model.py` to create {PIPELINE_PATH}")

pipeline = joblib.load(PIPELINE_PATH)

# ====== Dropdown Options ======
def get_dropdown_options():
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        for name, trans, cols in preprocessor.transformers_:
            if name == "cat":
                return list(trans.categories_)
    except Exception:
        pass

    return [
        ["Super built-up Area", "Built-up Area", "Plot Area"],
        ["Ready To Move", "Immediate Possession", "19-Dec", "New Launch"],
        ["Electronic City Phase II", "Chikka Tirupathi", "Uttarahalli", "Lingadheeranahalli", "Kothanur"],
        ["1 BHK", "2 BHK", "3 BHK", "4 BHK"],
        ["Coomee", "Theanmp", "Soiewre", "GreenVille", "LotusPark"]
    ]

# ====== Routes ======
@app.route("/", methods=["GET", "POST"])
def index():
    options = get_dropdown_options()
    prediction = None

    if request.method == "POST":
        form = request.form

        row = {
            "area_type": form.get("area_type"),
            "availability": form.get("availability"),
            "location": form.get("location"),
            "size": form.get("size"),
            "society": form.get("society"),
            "total_sqft": float(form.get("total_sqft") or 0),
            "bath": float(form.get("bath") or 0),
            "balcony": float(form.get("balcony") or 0),
        }

        df = pd.DataFrame([row])
        pred = pipeline.predict(df)[0]
        prediction = round(float(pred), 2)

    return render_template("index.html", options=options, prediction=prediction)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()

    df = pd.DataFrame([data])
    pred = pipeline.predict(df)[0]

    return jsonify({"prediction": float(pred)})

# ====== Run App ======
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
