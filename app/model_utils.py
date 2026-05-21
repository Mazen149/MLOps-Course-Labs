"""
Model loading and prediction logic.

The model must be loaded ONCE at module level, NOT inside the predict function.
"""

import joblib
import pandas as pd

# TODO 1: Load your serialized churn model from data/model.joblib
model = joblib.load(r"D:\ITI\ITI Courses\26) MLOps\Labs\Lab 1\data\model.pkl")
preprocessor = joblib.load(
    r"D:\ITI\ITI Courses\26) MLOps\Labs\Lab 1\data\preprocessor.pkl"
)


def predict_churn(features: list) -> int:
    """Takes a list of RAW feature values, converts them to a Pandas DataFrame,

    scales/encodes them via the preprocessor, and returns a churn prediction (0 or 1).
    """
    # 1. Map the list items to the exact column names the training pipeline expects
    column_names = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Geography",
        "Gender",
    ]

    # 2. Convert the single raw list into a clean 2D Pandas DataFrame (1 row)
    raw_df = pd.DataFrame([features], columns=column_names)

    # 3. Preprocessor automatically scales numbers and one-hot encodes text strings via DataFrame headers
    processed_features = preprocessor.transform(raw_df)

    # 4. Pass the processed matrix to the model
    prediction = model.predict(processed_features)

    return int(prediction[0])


if __name__ == "__main__":
    # 1. Define sample using RAW fields expected by the preprocessor pipeline.
    # Order: [CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender]
    raw_sample = [
        619,  # CreditScore
        42,  # Age
        2,  # Tenure
        0.0,  # Balance
        1,  # NumOfProducts
        1,  # HasCrCard
        1,  # IsActiveMember
        101348.88,  # EstimatedSalary
        "France",  # Geography
        "Female",  # Gender
    ]

    print(f"Raw Input Payload: {raw_sample}")

    try:
        result = predict_churn(raw_sample)
        print(f"Prediction Result: {result}")
    except Exception as e:
        print(f"\n[Execution Error]: {e}")
