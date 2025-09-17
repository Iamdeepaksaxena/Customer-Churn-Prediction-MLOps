import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
import logging
import pickle  # <-- added import for saving scaler

# ============================================
# Setup logging
# ============================================

log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addFilter(console_handler)

# =================================================
# Load data
# =================================================
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# =================================================
# Feature Scaling (StandardScaler)
# =================================================
def scale_features(df: pd.DataFrame, columns_to_scale: list) -> pd.DataFrame:
    """
    Scale selected numerical features using StandardScaler.
    
    Parameters:
    - df: input dataframe
    - columns_to_scale: list of column names to scale
    
    Returns:
    - DataFrame with scaled features (only the selected columns)
    """
    try:
        df_scaled = df.copy()

        # Keep only columns that exist in the dataframe
        columns_to_scale = [col for col in columns_to_scale if col in df.columns]

        scaler = StandardScaler()
        df_scaled[columns_to_scale] = scaler.fit_transform(df_scaled[columns_to_scale])

        # Save the scaler after fitting
        os.makedirs('./models', exist_ok=True)
        with open('./models/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        logger.debug(f"Scaled columns {columns_to_scale} using StandardScaler and saved scaler.pkl")
        return df_scaled

    except Exception as e:
        logger.error(f"Error during feature scaling: {e}")
        raise

# =================================================
# Save scaled data
# =================================================
def save_data(df: pd.DataFrame, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logger.debug(f"Scaled data saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving scaled data: {e}")
        raise

# =================================================
# Main function
# =================================================
def main():
    try:
        scaler_columns = ['tenure','MonthlyCharges','TotalCharges']

        train_df = load_data('./data/interim/train_processed.csv')
        test_df = load_data('./data/interim/test_processed.csv')

        train_scaled = scale_features(train_df, columns_to_scale=scaler_columns)
        test_scaled = scale_features(test_df, columns_to_scale=scaler_columns)

        save_data(train_scaled, './data/processed/train_scaled.csv')
        save_data(test_scaled, './data/processed/test_scaled.csv')

    except Exception as e:
        logger.error(f"Failed feature scaling process: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
