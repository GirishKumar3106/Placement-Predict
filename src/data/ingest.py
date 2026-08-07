import os
import pandas as pd


def load_and_validate_data(file_path: str) -> pd.DataFrame:
    # Check whether the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Critical Error: Targeted data footprint not discovered at {file_path}"
        )

    print(f"Executing secure data extraction from: {file_path}")

    # Load the CSV file
    df = pd.read_csv(file_path)

    # Required columns
    required_columns = [
    "branch",
    "college_tier",
    "cgpa",
    "backlogs",
    "coding_skill_score",
    "communication_skill_score",
    "internships_count",
    "projects_count",
    "placement_status",
    "salary_package_lpa",
]

    # Find missing columns
    missing_cols = [col for col in required_columns if col not in df.columns]

    # Raise an error if any required columns are missing
    if missing_cols:
        raise ValueError(
            f"Schema Validation Failure: Missing essential feature targets: {missing_cols}"
        )

    print(
        f"Data ingestion resolved successfully. Dimensions captured: "
        f"{df.shape[0]} samples, {df.shape[1]} metrics."
    )

    return df


if __name__ == "__main__":
    DATA_PATH = os.path.join("src", "data", "raw_placement_data.csv")

    try:
        raw_data = load_and_validate_data(DATA_PATH)
    except Exception as e:
        print(f"Ingestion lifecycle termination: {str(e)}")