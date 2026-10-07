import pandas as pd
import numpy as np


# --------------------------------------------------
# LOAD CSV
# --------------------------------------------------

def load_csv(uploaded_file):

    try:

        df = pd.read_csv(
            uploaded_file
        )

        return df

    except Exception as e:

        raise ValueError(
            f"Unable to read CSV file: {e}"
        )


# --------------------------------------------------
# GENERATE SAMPLE DATA
# --------------------------------------------------

def generate_sample_data(
    rows=500
):

    np.random.seed(42)

    age = np.random.randint(
        18,
        65,
        rows
    )

    experience = np.random.randint(
        0,
        30,
        rows
    )

    transactions = np.random.randint(
        10,
        500,
        rows
    )

    income = (
        20000
        + experience * 5000
        + age * 1000
        + np.random.normal(
            0,
            5000,
            rows
        )
    )

    predicted_income = (
        income
        + np.random.normal(
            0,
            10000,
            rows
        )
    )

    # Add some larger errors
    error_indices = np.random.choice(
        rows,
        size=25,
        replace=False
    )

    predicted_income[
        error_indices
    ] += np.random.normal(
        0,
        30000,
        25
    )

    df = pd.DataFrame({

        "Age": age,

        "Experience": experience,

        "Transactions": transactions,

        "Actual Income": income.round(2),

        "Predicted Income": predicted_income.round(2)

    })

    return df