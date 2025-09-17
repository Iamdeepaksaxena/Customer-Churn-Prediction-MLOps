import os
import pickle
import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE

# ============================================
# Setup logging
# ============================================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_preprocessing')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ============================================
# Preprocessing Function
# ============================================
def preprocess_df(df: pd.DataFrame, target_column: str = 'Churn') -> pd.DataFrame:
    """
    Preprocess Customer Churn dataset:
    - Handle missing/invalid data
    - Convert TotalCharges to numeric
    - OneHotEncode categorical features
    - Balance dataset using SMOTE
    - Return a single DataFrame with features + target
    """
    try:
        # Handle missing values (empty strings → NaN)
        df = df.replace(" ", pd.NA)
        df = df.dropna()

        # Convert TotalCharges to numeric
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            df = df.dropna(subset=["TotalCharges"])

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # OneHotEncoding for categorical features
        categorical_cols = X.select_dtypes(include=["object"]).columns
        if len(categorical_cols) > 0:
            encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
            X_encoded = encoder.fit_transform(X[categorical_cols])

            # Build DataFrame with correct column names + index
            X_encoded = pd.DataFrame(
                X_encoded,
                columns=encoder.get_feature_names_out(categorical_cols),
                index=X.index
            )

            # Drop categorical columns and join encoded ones
            X = pd.concat([X.drop(columns=categorical_cols), X_encoded], axis=1)

            # Ensure models directory exists before saving encoder
            os.makedirs('./models', exist_ok=True)
            with open('./models/encoder.pkl', 'wb') as f:
                pickle.dump(encoder, f)

        # Handle class imbalance using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)


        

        # Combine features + target into one DataFrame
        df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

        return df_resampled

    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise


def main(target_column="Churn"):
    """
    Main function to load raw data, preprocess it, and save the processed data.
    """
    try:
        # Load raw datasets
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")

        # Preprocess datasets
        train_processed_data = preprocess_df(train_data, target_column)
        test_processed_data = preprocess_df(test_data, target_column)

        # Save processed datasets
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        print(f"Processed data saved in {data_path}")

    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except pd.errors.EmptyDataError as e:
        print(f"Empty data file: {e}")
    except Exception as e:
        print(f"Failed in main preprocessing pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
