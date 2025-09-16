import os
import pickle
import json
import logging
import pandas as pd
import numpy as np
from dvclive import Live
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# ============================================
# Load model
# ============================================
def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logger.debug(f"Model loaded from {file_path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# ============================================
# Load test data
# ============================================
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Test data loaded from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# ============================================
# Evaluate model
# ============================================
def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else y_pred

        metrics_dict = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc': roc_auc_score(y_test, y_pred_proba)
        }
        logger.debug(f"Metrics calculated: {metrics_dict}")
        return metrics_dict
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

# ============================================
# Save metrics to JSON
# ============================================
def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logger.debug(f"Metrics saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving metrics: {e}")
        raise

# ============================================
# Main evaluation function
# ============================================
def main():
    try:
        # Load test data
        test_df = load_data('./data/processed/test_scaled.csv')
        X_test = test_df.iloc[:, :-1].values
        y_test = test_df.iloc[:, -1].values

        # List of saved models
        model_files = [
            'models/knn_best_model.pkl',
            'models/svc_best_model.pkl',
            'models/logistic_regression_best_model.pkl',
            'models/random_forest_best_model.pkl',
            'models/decision_tree_best_model.pkl'
        ]

        all_metrics = {}

        # Start MLflow experiment
        mlflow.set_experiment("Customer_Churn_Model_Evaluation_v2")

        with mlflow.start_run():
            for model_file in model_files:
                model_name = os.path.basename(model_file).split('_best_model')[0]
                clf = load_model(model_file)

                metrics = evaluate_model(clf, X_test, y_test)
                all_metrics[model_name] = metrics

                # DVCLive tracking
                with Live(save_dvc_exp=True) as live:
                    for metric_name, metric_value in metrics.items():
                        live.log_metric(f"{model_name}_{metric_name}", metric_value)

                # MLflow tracking
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(f"{model_name}_{metric_name}", metric_value)

            # Save all metrics to JSON
            save_metrics(all_metrics, 'reports/metrics.json')

        logger.debug("Model evaluation completed for all models.")

    except Exception as e:
        logger.error(f"Failed to complete model evaluation: {e}")
        print(f"Error: {e}")

if __name__ == '__main__':
    main()