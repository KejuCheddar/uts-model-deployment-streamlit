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
    print(f"[INFO] Memuat data dari:\n  {features_path}\n  {targets_path}")
    df_features = pd.read_csv(features_path)
    df_targets = pd.read_csv(targets_path)
    df = df_features.merge(df_targets, on='Student_ID')
    print(f"[INFO] Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")
    return df


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
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
    X = df[ALL_FEATURES]
    y_clf = (df['placement_status'] == 'Placed').astype(int)

    # Split klasifikasi (dengan stratify)
    X_train, X_test, y_clf_train, y_clf_test = train_test_split(
        X, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # Split regresi — hanya mahasiswa yang Placed (salary nyata)
    df_placed = df[df['placement_status'] == 'Placed']
    X_reg = df_placed[ALL_FEATURES]
    y_reg = df_placed['salary_lpa']

    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print(f"[INFO] Train size (clf): {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"[INFO] Train size (reg, placed only): {X_reg_train.shape[0]} | Test: {X_reg_test.shape[0]}")
    print(f"[INFO] Class distribution (train): Placed={y_clf_train.sum()}, Not Placed={(~y_clf_train.astype(bool)).sum()}")
    print(f"[INFO] Class distribution (test):  Placed={y_clf_test.sum()}, Not Placed={(~y_clf_test.astype(bool)).sum()}")

    return X_train, X_test, y_clf_train, y_clf_test, X_reg_train, X_reg_test, y_reg_train, y_reg_test


# PREPROCESSOR
def build_preprocessor() -> ColumnTransformer:
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
    mlflow.set_experiment("Student_Placement_Classification")

    clf_models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'params': {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42}
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'params': {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42}
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
        }
    }

    clf_results = {}
    clf_pipelines = {}

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

            clf_results[name] = {'Accuracy': acc, 'F1-Score': f1, 'ROC-AUC': auc}
            clf_pipelines[name] = pipeline

            mlflow.log_params(config['params'])
            mlflow.log_param("model_type", name)
            mlflow.log_param("dataset", "Dataset_A")
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score_weighted", f1)
            mlflow.log_metric("roc_auc", auc)
            mlflow.sklearn.log_model(pipeline, f"model_{name.replace(' ', '_')}")

            print(f"\n  {name}")
            print(f"  Accuracy : {acc:.4f}")
            print(f"  F1-Score : {f1:.4f}")
            print(f"  ROC-AUC  : {auc:.4f}")

    # Pilih best model berdasarkan F1-Score (sama seperti notebook)
    best_clf_name = max(clf_results, key=lambda x: clf_results[x]['F1-Score'])
    best_clf = clf_pipelines[best_clf_name]

    print(f"\n Best Classifier: {best_clf_name} (F1={clf_results[best_clf_name]['F1-Score']:.4f})")

    pkl_path = os.path.join(MODEL_DIR, "best_classifier.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(best_clf, f)
    print(f" Saved: {pkl_path}")

    return best_clf, best_clf_name, clf_results


def train_regression(X_train, X_test, y_train, y_test, preprocessor):
    mlflow.set_experiment("Student_Salary_Regression")

    reg_models = {
        'Ridge Regression': {
            'model': Ridge(alpha=1.0),
            'params': {'alpha': 1.0}
        },
        'Random Forest': {
            'model': RandomForestRegressor(n_estimators=100, random_state=42),
            'params': {'n_estimators': 100, 'random_state': 42}
        },
        'Gradient Boosting': {
            'model': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'params': {'n_estimators': 100, 'learning_rate': 0.1, 'random_state': 42}
        }
    }

    reg_results = {}
    reg_pipelines = {}

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

            reg_results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
            reg_pipelines[name] = pipeline

            mlflow.log_params(config['params'])
            mlflow.log_param("model_type", name)
            mlflow.log_param("dataset", "Dataset_A")
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            mlflow.sklearn.log_model(pipeline, f"model_{name.replace(' ', '_')}")

            print(f"\n  {name}")
            print(f"  RMSE : {rmse:.4f}")
            print(f"  MAE  : {mae:.4f}")
            print(f"  R²   : {r2:.4f}")

    # Pilih best model berdasarkan R² tertinggi (sama seperti notebook)
    best_reg_name = max(reg_results, key=lambda x: reg_results[x]['R²'])
    best_reg = reg_pipelines[best_reg_name]

    print(f"\n Best Regressor: {best_reg_name} (R²={reg_results[best_reg_name]['R²']:.4f})")

    pkl_path = os.path.join(MODEL_DIR, "best_regressor.pkl")
    with open(pkl_path, 'wb') as f:
        pickle.dump(best_reg, f)
    print(f" Saved: {pkl_path}")

    return best_reg, best_reg_name, reg_results


# MAIN
def main():
    mlruns_path = os.path.join(DATA_DIR, "mlruns")
    os.makedirs(mlruns_path, exist_ok=True)
    mlflow.set_tracking_uri(f"file:///{mlruns_path.replace(os.sep, '/')}")

    # 1. Load data
    df = load_data(FEATURES_PATH, TARGETS_PATH)

    # 2. Feature engineering
    df = apply_feature_engineering(df)

    # 3. Prepare data
    X_train, X_test, y_clf_train, y_clf_test, X_reg_train, X_reg_test, y_reg_train, y_reg_test = prepare_data(df)

    # 4. Build preprocessor
    preprocessor = build_preprocessor()

    # 5. Train classification
    best_clf, clf_name, clf_results = train_classification(
        X_train, X_test, y_clf_train, y_clf_test, preprocessor
    )

    # 6. Train regression (hanya data Placed)
    best_reg, reg_name, reg_results = train_regression(
        X_reg_train, X_reg_test, y_reg_train, y_reg_test, preprocessor
    )

    # 7. Save feature metadata (sama seperti notebook, tanpa best_classifier/regressor)
    meta = {
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'all_features': ALL_FEATURES
    }
    with open(os.path.join(MODEL_DIR, 'feature_metadata.json'), 'w') as f:
        json.dump(meta, f)

    # 8. Final summary (format sama persis dengan notebook)
    print("\nModel berhasil disimpan!")
    print(f"  models/best_classifier.pkl  : {clf_name}")
    print(f"  models/best_regressor.pkl   : {reg_name}")
    print(f"  models/feature_metadata.json")
    print()
    print(f"FINAL SUMMARY")
    print(f"Best Classifier : {clf_name}")
    print(f"  - Accuracy : {clf_results[clf_name]['Accuracy']:.4f}")
    print(f"  - F1-Score : {clf_results[clf_name]['F1-Score']:.4f}")
    print(f"  - ROC-AUC  : {clf_results[clf_name]['ROC-AUC']:.4f}")
    print()
    print(f"Best Regressor  : {reg_name}")
    print(f"  - RMSE : {reg_results[reg_name]['RMSE']:.4f}")
    print(f"  - MAE  : {reg_results[reg_name]['MAE']:.4f}")
    print(f"  - R²   : {reg_results[reg_name]['R²']:.4f}")


if __name__ == "__main__":
    main()