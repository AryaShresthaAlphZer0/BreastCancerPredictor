
import joblib
import math
from pathlib import Path

from django.conf import settings
from django.shortcuts import render

from sklearn.datasets import load_breast_cancer

# Load the trained ML pipeline
MODEL_PATH = Path(settings.BASE_DIR) / "cancer_model.pkl"
model = joblib.load(MODEL_PATH)

# Get feature names and class names, not sample records
data = load_breast_cancer()

feature_names = data.feature_names
target_names = data.target_names

# Prepare 30 fields for the HTML form
features = [
    {
        "key": f"feature_{i}",
        "label": name
    }
    for i, name in enumerate(feature_names)
]


def home(request):
    result = None
    error = None

    if request.method == "POST":
        try:
            values = []

            for feature in features:
                value = request.POST.get(feature["key"])

                if not value:
                    raise ValueError("Please fill in all 30 fields.")

                number = float(value)

                if not math.isfinite(number):
                    raise ValueError("Please enter valid finite numbers.")

                values.append(number)

            # Make prediction using the user's input
            prediction = model.predict([values])[0]

            result = target_names[prediction]

        except ValueError as e:
            error = str(e)

    return render(request, "predictor/home.html", {
        "features": features,
        "result": result,
        "error": error,
    })
