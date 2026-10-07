import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# --------------------------------------------------
# MODEL METRICS
# --------------------------------------------------

def calculate_metrics(
    df,
    actual_column,
    predicted_column
):

    data = df[
        [actual_column, predicted_column]
    ].dropna()

    actual = data[actual_column]
    predicted = data[predicted_column]

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predicted
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# --------------------------------------------------
# ERROR CALCULATION
# --------------------------------------------------

def calculate_errors(
    df,
    actual_column,
    predicted_column
):

    result = df.copy()

    result["Error"] = (
        result[actual_column]
        - result[predicted_column]
    )

    result["Absolute Error"] = (
        result["Error"].abs()
    )

    result["Squared Error"] = (
        result["Error"] ** 2
    )

    return result


# --------------------------------------------------
# ERROR DRIVER ANALYSIS
# --------------------------------------------------

def find_error_drivers(
    df,
    error_df,
    actual_column,
    predicted_column
):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    feature_columns = [
        col
        for col in numeric_columns
        if col not in [
            actual_column,
            predicted_column
        ]
    ]

    results = []

    for feature in feature_columns:

        temp = pd.DataFrame({
            "feature": df[feature],
            "error": error_df["Absolute Error"]
        })

        temp = temp.dropna()

        if len(temp) < 3:
            continue

        correlation = temp["feature"].corr(
            temp["error"]
        )

        if pd.isna(correlation):
            continue

        absolute_correlation = abs(
            correlation
        )

        if absolute_correlation >= 0.7:
            strength = "Strong"

        elif absolute_correlation >= 0.4:
            strength = "Moderate"

        else:
            strength = "Weak"

        results.append({
            "Feature": feature,
            "Correlation with Error": round(
                correlation,
                4
            ),
            "Relationship": strength
        })

    result_df = pd.DataFrame(results)

    if len(result_df) > 0:

        result_df = result_df.sort_values(
            "Correlation with Error",
            key=lambda x: x.abs(),
            ascending=False
        )

    return result_df