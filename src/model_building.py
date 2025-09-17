import os
import pandas as pd
import numpy as np
import pickle
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import yaml
import dvclive

# Ensure the "logs" directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logging configuration
logger = logging.getLogger('model_building')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model_building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# rough work for YAML
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

# ============================================
# Load data
# ============================================
def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded from {file_path} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

# ============================================
# Train and GridSearchCV
# ============================================
def train_and_save_models(X_train, y_train):
    try:
        # Define models with placeholder for scaler
        
        models = {
            'knn': Pipeline([('classifier', KNeighborsClassifier())]),
            'svc': Pipeline([('classifier', SVC())]),
            'logistic_regression': Pipeline([('classifier', LogisticRegression(max_iter=1000))]),
            'random_forest': Pipeline([('classifier', RandomForestClassifier())]),
            'decision_tree': Pipeline([('classifier', DecisionTreeClassifier())]),
        }

        # Hyperparameter grids
        N_neighbors = [3, 5, 7, 9, 11, 15]
        P = [1, 2]

        param_grid = {
            'knn': [{
                'classifier__n_neighbors': N_neighbors,
                'classifier__p': P
            }],
            'svc': [{
                'classifier__kernel': ['rbf', 'linear', 'poly', 'sigmoid']
            }],
            'logistic_regression': [{}],
            'random_forest': [{
                'classifier__n_estimators': [100, 200],
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__min_samples_split': [2, 4, 6, 8],
                'classifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8]
            }],
            'decision_tree': [{
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__min_samples_split': [2, 4, 6, 8],
                'classifier__min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8]
            }],
        }

        results = {}

        # GridSearchCV for each model
        for name, pipeline in models.items():
            logger.debug(f"Running GridSearchCV for {name}...")
            grid = GridSearchCV(estimator=pipeline, param_grid=param_grid[name],
                                cv=5, scoring='accuracy', n_jobs=-1)
            grid.fit(X_train, y_train)

            # Save best model
            os.makedirs('models', exist_ok=True)
            model_path = f'models/{name}_best_model.pkl'
            with open(model_path, 'wb') as f:
                pickle.dump(grid.best_estimator_, f)

            logger.debug(f"Saved {name} best model to {model_path}")

            results[name] = {
                'best_params': grid.best_params_,
                'best_score': grid.best_score_
            }

        # Log results
        for model_name, res in results.items():
            logger.debug(f"{model_name} Best Params: {res['best_params']}")
            logger.debug(f"{model_name} Best CV Score: {res['best_score']:.4f}")

        return results

    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise

# ============================================
# Main function
# ============================================
def main():
    try:
        # Load scaled training data
        params = load_params("params.yaml")
        train_df = load_data('./data/processed/train_scaled.csv')

        X_train = train_df.iloc[:, :-1].values
        y_train = train_df.iloc[:, -1].values
        # Create param grid dynamically from YAML
        param_grid = {
            "knn": [{
                "classifier__n_neighbors": params["models"]["knn"]["n_neighbors"],
                "classifier__p": params["models"]["knn"]["p"]
            }],
            "svc": [{
                "classifier__kernel": params["models"]["svc"]["kernel"]
            }],
            "logistic_regression": [{}],
            "random_forest": [{
                "classifier__n_estimators": params["models"]["random_forest"]["n_estimators"],
                "classifier__criterion": params["models"]["random_forest"]["criterion"],
                "classifier__min_samples_split": params["models"]["random_forest"]["min_samples_split"],
                "classifier__min_samples_leaf": params["models"]["random_forest"]["min_samples_leaf"]
            }],
            "decision_tree": [{
                "classifier__criterion": params["models"]["decision_tree"]["criterion"],
                "classifier__min_samples_split": params["models"]["decision_tree"]["min_samples_split"],
                "classifier__min_samples_leaf": params["models"]["decision_tree"]["min_samples_leaf"]
            }],
        }

        results = train_and_save_models(X_train, y_train)

        # Save results to a file
        os.makedirs('models', exist_ok=True)
        with open('models/model_results.pkl', 'wb') as f:
            pickle.dump(results, f)
        logger.debug("Saved all model results to models/model_results.pkl")

    except Exception as e:
        logger.error(f"Failed to complete model building: {e}")
        print(f"Error: {e}")


if __name__ == '__main__':
    main()

