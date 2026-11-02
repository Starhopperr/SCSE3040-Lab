"Predict one delivery, from the command line."
import pandas as pd
from delivery import load_model

def main():
    # Load the trained model from the current directory
    model = load_model("model.joblib")

    # Create a one-row DataFrame matching the training feature names
    df = pd.DataFrame([{
        "distance_km": 7.0,
        "prep_time_min": 25.0,
        "traffic_level": 3,
        "rain": False
    }])

    # Generate the prediction
    prediction = model.predict(df)[0]

    # Print the result in the required format
    print(f"PREDICTION: {prediction:.1f}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
