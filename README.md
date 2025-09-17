# Customer-Churn-Prediction-MLOps
📌 Customer Churn Prediction – MLOps Project
🔹 Project Overview

This project implements an end-to-end MLOps pipeline for predicting customer churn.
It covers:
Data ingestion, preprocessing, feature engineering,Model training and evaluation
Experiment tracking (MLflow, logs, reports)
Reproducible pipelines (DVC)
CI/CD automation (GitHub Actions + Docker + DockerHub)
Deployment using Flask + Docker
📂 Project Structure
Customer-Churn-Prediction-MLOps/
│── dvclive/                  # DVC live metrics
│── logs/                     # Logs
│── mlruns/                   # MLflow experiment tracking
│── models/                   # Saved models
│── myvenv/                   # Virtual environment (not pushed to git)
│── reports/                  # Reports/metrics
│── src/                      # ML pipeline scripts
│    ├── data_ingestion.py
│    ├── data_preprocessing.py
│    ├── feature_engineering.py
│    ├── model_building.py
│    └── model_evaluation.py
│── static/css/style.css      # Frontend styling
│── templates/                # HTML frontend
│    ├── index.html
│    └── result.html
│── app.py                    # Flask app
│── Dockerfile                # Docker build instructions
│── requirements.txt          # Project dependencies
│── dvc.yaml                  # DVC pipeline stages
│── dvc.lock                  # DVC lock file
│── .github/workflows/ci_cd.yml  # CI/CD pipeline
│── .gitignore
│── .dvcignore

⚙️ Setup Instructions
1️⃣ Clone the Repository
git clone https://github.com/Iamdeepaksaxena/Customer-Churn-Prediction-MLOps.git
cd Customer-Churn-Prediction-MLOps

2️⃣ Create Virtual Environment & Install Dependencies
python -m venv myvenv
./myvenv\Scripts\activate      # (Windows)

pip install -r requirements.txt
🔄 Reproducible ML Pipeline (DVC)
We use DVC (Data Version Control) to automate and reproduce the ML pipeline.
Run pipeline using below command:
dvc repro

This will execute all stages defined in dvc.yaml see below like that:
data_ingestion → data_preprocessing → feature_engineering → model_building → model_evaluation

IF you want to visualize pipeline you can do it using below command:
dvc dag

🚀 Running the Application
Local Run (Flask)
python app.py
App runs on → http://127.0.0.1:5000

🐳 Running with Docker
Build Image
docker build -t itsdeepakkumar/customer_churn .
Run Container
docker run -p 5000:5000 itsdeepakkumar/customer_churn
Now access app → http://127.0.0.1:5000

🔗 CI/CD Pipeline (GitHub Actions + DockerHub)
1️⃣ CI (Continuous Integration)
On every push/pull request:
Runs linting (flake8)
Runs tests (pytest)
Ensures dependencies install correctly

2️⃣ CD (Continuous Deployment)
GitHub Actions builds Docker image
Pushes image automatically to DockerHub
Repo: itsdeepakkumar/customer_churn
GitHub Secrets Configured:
DOCKER_USERNAME → your Docker Hub username
DOCKER_PASSWORD → your Docker Hub PAT/password
