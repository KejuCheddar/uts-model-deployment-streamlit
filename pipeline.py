import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Regression models
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              mean_squared_error, mean_absolute_error, r2_score)

import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')

# CONFIGURATION
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(DATA_DIR, "A.csv")
TARGETS_PATH = os.path.join(DATA_DIR, "A_targets.csv")
MODEL_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

NUMERICAL_FEATURES = [
    'cgpa', 'tenth_percentage', 'twelfth_percentage', 'backlogs',
    'study_hours_per_day', 'attendance_percentage', 'projects_completed',
    'internships_completed', 'coding_skill_rating', 'communication_skill_rating',
    'aptitude_skill_rating', 'hackathons_participated', 'certifications_count',
    'sleep_hours', 'stress_level', 'skill_composite', 'academic_score',
    'experience_score', 'healthy_sleep', 'high_achiever'
]

CATEGORICAL_FEATURES = [
    'gender', 'branch', 'part_time_job', 'family_income_level',
    'city_tier', 'internet_access', 'extracurricular_involvement'
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# DATA INGESTION
def load_data(features_path: str, targets_path: str) -> pd.DataFrame:
    """Muat dan gabungkan dataset fitur dan target."""
    print(f"[INFO] Memuat data dari:\n  {features_path}\n  {targets_path}")
    df_features = pd.read_csv(features_path)
    df_targets = pd.read_csv(targets_path)
    df = df_features.merge(df_targets, on='Student_ID')
    print(f"[INFO] Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Terapkan feature engineering untuk membuat fitur turunan."""
    df = df.copy()
    df['skill_composite'] = (
        df['coding_skill_rating'] +
        df['communication_skill_rating'] +
        df['aptitude_skill_rating']
    ) / 3

    df['academic_score'] = (
        df['cgpa'] * 0.4 +
        df['attendance_percentage'] / 100 * 10 * 0.3 +
        df['study_hours_per_day'] * 0.3
    )

    df['experience_score'] = (
        df['internships_completed'] * 2 +
        df['projects_completed'] +
        df['certifications_count'] +
        df['hackathons_participated']
    )

    df['healthy_sleep'] = (
        (df['sleep_hours'] >= 6) & (df['sleep_hours'] <= 9)
    ).astype(int)

    df['high_achiever'] = (
        (df['cgpa'] >= 8.0) & (df['coding_skill_rating'] >= 7)
    ).astype(int)

    return df


def prepare_data(df: pd.DataFrame):
    """
    Persiapkan fitur dan target, lakukan train-test split 80:20.
    
    Note: Untuk regresi, hanya mahasiswa yang Placed yang digunakan
    karena salary=0 untuk Not Placed bukan target prediksi yang bermakna.
    """
    X = df[ALL_FEATURES]
    y_clf = (df['placement_status'] == 'Placed').astype(int)

    # Untuk regresi: filter hanya yang Placed (salary > 0)
    df_placed = df[df['placement_status'] == 'Placed']
    X_reg = df_placed[ALL_FEATURES]
    y_reg = df_placed['salary_lpa']

    X_train, X_test, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print(f"[INFO] Train (clf): {len(X_train)} | Test: {len(X_test)}")
    print(f"[INFO] Train (reg, placed only): {len(X_reg_train)} | Test: {len(X_reg_test)}")
    return X_train, X_test, y_clf_train, y_clf_test, X_reg_train, X_reg_test, y_reg_train, y_reg_test


# PREPROCESSOR
def build_preprocessor() -> ColumnTransformer:
    """Bangun preprocessing pipeline."""
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    return ColumnTransformer([
        ('num', numeric_transformer, NUMERICAL_FEATURES),
        ('cat', categorical_transformer, CATEGORICAL_FEATURES)
    ])


# TRAINING WITH MLFLOW
def train_classification(X_train, X_test, y_train, y_test, preprocessor):
    """Latih 3 model klasifikasi dan track dengan MLflow."""
    mlflow.set_experiment("Student_Placement_Classification")

    clf_models = {
        'Logistic_Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'params': {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42}
        },
        'Random_Forest_Classifier': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'params': {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42}
        },
        'Gradient_Boosting_Classifier': {
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
        }
    }

    best_model = None
    best_f1 = 0
    best_name = ""

    print("\n" + "="*50)
    print("CLASSIFICATION EXPERIMENT")
    print("="*50)

    for name, config in clf_models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', config['model'])
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            y_prob = pipeline.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            auc = roc_auc_score(y_test, y_prob)

            # Log parameters
            mlflow.log_params(config['params'])
            mlflow.log_param("model_type", name)
            mlflow.log_param("dataset", "Dataset_A")

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score_weighted", f1)
            mlflow.log_metric("roc_auc", auc)

            # Log model artifact
            mlflow.sklearn.log_model(pipeline, f"model_{name}")

            print(f"\n  [{name}]")
            print(f"  Accuracy  : {acc:.4f}")
            print(f"  F1-Score  : {f1:.4f}")
            print(f"  ROC-AUC   : {auc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = pipeline
                best_name = name

    print(f"\n Best Classifier: {best_name} (F1={best_f1:.4f})")

    # Save best model
    pkl_path = os.path.join(MODEL_DIR, "best_classifier.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f" Saved: {pkl_path}")

    return best_model, best_name


def train_regression(X_train, X_test, y_train, y_test, preprocessor):
    """Latih 3 model regresi dan track dengan MLflow."""
    mlflow.set_experiment("Student_Salary_Regression")

    reg_models = {
        'Ridge_Regression': {
            'model': Ridge(alpha=1.0),
            'params': {'alpha': 1.0}
        },
        'Random_Forest_Regressor': {
            'model': RandomForestRegressor(n_estimators=100, random_state=42),
            'params': {'n_estimators': 100, 'random_state': 42}
        },
        'Gradient_Boosting_Regressor': {
            'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
        }
    }

    best_model = None
    best_r2 = -np.inf
    best_name = ""

    print("\n" + "="*50)
    print("  REGRESSION EXPERIMENT")
    print("="*50)

    for name, config in reg_models.items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('regressor', config['model'])
            ])

            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)

            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Log parameters
            mlflow.log_params(config['params'])
            mlflow.log_param("model_type", name)
            mlflow.log_param("dataset", "Dataset_A")

            # Log metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)

            # Log model artifact
            mlflow.sklearn.log_model(pipeline, f"model_{name}")

            print(f"\n  [{name}]")
            print(f"  RMSE   : {rmse:.4f}")
            print(f"  MAE    : {mae:.4f}")
            print(f"  R²     : {r2:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_model = pipeline
                best_name = name

    print(f"\n Best Regressor: {best_name} (R²={best_r2:.4f})")

    # Save best model
    pkl_path = os.path.join(MODEL_DIR, "best_regressor.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f" Saved: {pkl_path}")

    return best_model, best_name


# MAIN
def main():
    print("="*60)
    print("RUNNING PIPELINE FILE")
    print("Dataset A")
    print("="*60)

    # Set MLflow tracking URI
    mlruns_path = os.path.join(DATA_DIR, "mlruns")
    os.makedirs(mlruns_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")

    # 1. Data Ingestion
    df = load_data(FEATURES_PATH, TARGETS_PATH)

    # 2. Feature Engineering
    df = apply_feature_engineering(df)

    # 3. Prepare data
    X_train, X_test, y_clf_train, y_clf_test, X_reg_train, X_reg_test, y_reg_train, y_reg_test = prepare_data(df)

    # 4. Build preprocessor
    preprocessor = build_preprocessor()

    # 5. Train classification models
    best_clf, clf_name = train_classification(X_train, X_test, y_clf_train, y_clf_test, preprocessor)

    # 6. Train regression models
    best_reg, reg_name = train_regression(X_reg_train, X_reg_test, y_reg_train, y_reg_test, preprocessor)

    # 7. Save feature metadata
    meta = {
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'all_features': ALL_FEATURES,
        'best_classifier': clf_name,
        'best_regressor': reg_name
    }
    with open(os.path.join(MODEL_DIR, 'feature_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print("\n" + "="*60)
    print("PIPELINE SELESAI")
    print("="*60)
    print(f"  Best Classifier : {clf_name}")
    print(f"  Best Regressor  : {reg_name}")
    print(f"  Model disimpan di: {MODEL_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()


"""
Jalankan : mlflow ui
Jalankan: python pipeline.py
"""