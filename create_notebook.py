import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

# Cell 1 - Title
cells.append(nbf.v4.new_markdown_cell("""# 🎓 UTS Model Deployment - DTSC6012001
## Dataset A: Student Placement & Salary Prediction

**Video Presentasi:** [Link Video](https://youtu.be/PLACEHOLDER)

---

### Deskripsi Dataset
Dataset ini berisi informasi mahasiswa mencakup performa akademik, keterampilan teknis, dan faktor gaya hidup.  
- **Features**: 22 fitur (akademik, skill, lifestyle)
- **Target Klasifikasi**: `placement_status` (Placed / Not Placed)  
- **Target Regresi**: `salary_lpa` (estimasi gaji dalam LPA)
"""))

# Cell 2 - Imports
cells.append(nbf.v4.new_code_cell("""# ========================
# 1. IMPORT LIBRARIES
# ========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
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

# Metrics
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score, roc_auc_score,
                              mean_squared_error, mean_absolute_error, r2_score)

import pickle
import os

print("✅ Libraries loaded successfully!")
print(f"Pandas version: {pd.__version__}")
"""))

# Cell 3 - Load Data
cells.append(nbf.v4.new_markdown_cell("""## 📥 2. Data Loading & Initial Inspection"""))
cells.append(nbf.v4.new_code_cell("""# Load features and targets
df_features = pd.read_csv('A.csv')
df_targets = pd.read_csv('A_targets.csv')

# Merge on Student_ID
df = df_features.merge(df_targets, on='Student_ID')

print(f"Dataset shape: {df.shape}")
print(f"Features shape: {df_features.shape}")
print(f"Targets shape: {df_targets.shape}")
df.head()
"""))

cells.append(nbf.v4.new_code_cell("""# Basic info
print("=== Dataset Info ===")
df.info()
print()
print("=== Statistical Summary ===")
df.describe().round(2)
"""))

# Cell - EDA
cells.append(nbf.v4.new_markdown_cell("""## 🔍 3. Exploratory Data Analysis (EDA)

### 3.1 Missing Values Analysis
Penanganan missing values penting untuk memastikan model tidak dilatih dengan data yang tidak lengkap.
"""))
cells.append(nbf.v4.new_code_cell("""# Check missing values
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})
missing_df = missing_df[missing_df['Missing Count'] > 0]
print("Kolom dengan missing values:")
print(missing_df)

# Visualize
fig, ax = plt.subplots(figsize=(8, 4))
missing_df['Missing %'].plot(kind='bar', ax=ax, color='coral')
ax.set_title('Persentase Missing Values per Kolom')
ax.set_ylabel('Persentase (%)')
ax.set_xlabel('Kolom')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 3.2 Target Variable Distribution
Analisis distribusi target untuk memahami class imbalance pada klasifikasi dan distribusi nilai pada regresi.
"""))
cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Classification target
placement_counts = df['placement_status'].value_counts()
colors = ['#2ecc71', '#e74c3c']
axes[0].pie(placement_counts, labels=placement_counts.index, autopct='%1.1f%%', colors=colors)
axes[0].set_title('Distribusi Placement Status\\n(Target Klasifikasi)')

# Regression target
axes[1].hist(df['salary_lpa'], bins=40, color='#3498db', edgecolor='white', alpha=0.8)
axes[1].set_title('Distribusi Salary LPA\\n(Target Regresi)')
axes[1].set_xlabel('Salary (LPA)')
axes[1].set_ylabel('Frekuensi')
axes[1].axvline(df['salary_lpa'].mean(), color='red', linestyle='--', label=f'Mean: {df["salary_lpa"].mean():.2f}')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Placement Status Distribution:\\n{placement_counts.to_string()}")
print(f"\\nSalary Stats: Mean={df['salary_lpa'].mean():.2f}, Std={df['salary_lpa'].std():.2f}")
print(f"Class Imbalance Ratio: {placement_counts[0]/placement_counts[1]:.2f}:1")
"""))

cells.append(nbf.v4.new_markdown_cell("""### 3.3 Correlation Analysis
Analisis korelasi membantu mengidentifikasi fitur yang paling berpengaruh terhadap target variabel.
"""))
cells.append(nbf.v4.new_code_cell("""# Encode target for correlation
df_corr = df.copy()
df_corr['placement_encoded'] = (df_corr['placement_status'] == 'Placed').astype(int)

# Select numeric columns
numeric_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in ['Student_ID']]

# Correlation with targets
corr_placement = df_corr[numeric_cols].corr()['placement_encoded'].sort_values(ascending=False)
corr_salary = df_corr[numeric_cols].corr()['salary_lpa'].sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Placement correlation
corr_p = corr_placement.drop(['placement_encoded']).head(12)
colors_p = ['#2ecc71' if x > 0 else '#e74c3c' for x in corr_p]
axes[0].barh(corr_p.index, corr_p.values, color=colors_p)
axes[0].set_title('Korelasi Fitur vs Placement Status')
axes[0].set_xlabel('Korelasi')
axes[0].axvline(0, color='black', linewidth=0.5)

# Salary correlation
corr_s = corr_salary.drop(['salary_lpa']).head(12)
colors_s = ['#3498db' if x > 0 else '#e74c3c' for x in corr_s]
axes[1].barh(corr_s.index, corr_s.values, color=colors_s)
axes[1].set_title('Korelasi Fitur vs Salary LPA')
axes[1].set_xlabel('Korelasi')
axes[1].axvline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_code_cell("""# Heatmap korelasi antar fitur numerik utama
key_features = ['cgpa', 'coding_skill_rating', 'aptitude_skill_rating', 
                'communication_skill_rating', 'internships_completed',
                'projects_completed', 'attendance_percentage', 
                'study_hours_per_day', 'placement_encoded', 'salary_lpa']

fig, ax = plt.subplots(figsize=(12, 8))
corr_matrix = df_corr[key_features].corr()
mask = np.triu(np.ones_like(corr_matrix), k=1)
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, ax=ax, vmin=-1, vmax=1, linewidths=0.5)
ax.set_title('Heatmap Korelasi Fitur Utama', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("""### 3.4 Feature Distribution Analysis"""))
cells.append(nbf.v4.new_code_cell("""# Distribution of key features by placement status
key_features_dist = ['cgpa', 'coding_skill_rating', 'internships_completed', 
                     'projects_completed', 'attendance_percentage']

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, feat in enumerate(key_features_dist):
    placed = df[df['placement_status']=='Placed'][feat]
    not_placed = df[df['placement_status']=='Not Placed'][feat]
    axes[i].hist(placed, alpha=0.6, bins=20, color='#2ecc71', label='Placed')
    axes[i].hist(not_placed, alpha=0.6, bins=20, color='#e74c3c', label='Not Placed')
    axes[i].set_title(feat)
    axes[i].legend(fontsize=8)
    axes[i].set_xlabel('Value')
plt.suptitle('Distribusi Fitur Utama berdasarkan Placement Status', y=1.02, fontsize=13)
plt.tight_layout()
plt.show()
"""))

# Feature Engineering
cells.append(nbf.v4.new_markdown_cell("""## ⚙️ 4. Feature Engineering

Feature engineering dilakukan untuk menciptakan fitur baru yang lebih representatif dari data yang ada.
Alasan pemilihan fitur engineering:
1. **skill_composite**: Kombinasi skill rating (coding, communication, aptitude) mencerminkan kemampuan holistik mahasiswa
2. **academic_score**: Agregasi performa akademik (CGPA, kehadiran, jam belajar) sebagai indikator kesiapan kerja
3. **experience_score**: Kombinasi internship & proyek sebagai proxy pengalaman praktis
"""))
cells.append(nbf.v4.new_code_cell("""# Feature Engineering
df_eng = df.copy()

# 1. Composite skill score (skill holistik)
df_eng['skill_composite'] = (df_eng['coding_skill_rating'] + 
                              df_eng['communication_skill_rating'] + 
                              df_eng['aptitude_skill_rating']) / 3

# 2. Academic performance score
df_eng['academic_score'] = (df_eng['cgpa'] * 0.4 + 
                             df_eng['attendance_percentage'] / 100 * 10 * 0.3 +
                             df_eng['study_hours_per_day'] * 0.3)

# 3. Experience score  
df_eng['experience_score'] = (df_eng['internships_completed'] * 2 + 
                               df_eng['projects_completed'] + 
                               df_eng['certifications_count'] + 
                               df_eng['hackathons_participated'])

# 4. Lifestyle balance (sleep quality indicator)
df_eng['healthy_sleep'] = ((df_eng['sleep_hours'] >= 6) & (df_eng['sleep_hours'] <= 9)).astype(int)

# 5. High achiever flag
df_eng['high_achiever'] = ((df_eng['cgpa'] >= 8.0) & 
                            (df_eng['coding_skill_rating'] >= 7)).astype(int)

print("✅ Feature Engineering selesai!")
print(f"Fitur baru: skill_composite, academic_score, experience_score, healthy_sleep, high_achiever")
print(f"\\nNew features stats:")
new_features = ['skill_composite', 'academic_score', 'experience_score']
df_eng[new_features].describe().round(3)
"""))

# Data Preparation
cells.append(nbf.v4.new_markdown_cell("""## 🔧 5. Data Preprocessing & Train-Test Split"""))
cells.append(nbf.v4.new_code_cell("""# Define features
CATEGORICAL_FEATURES = ['gender', 'branch', 'part_time_job', 'family_income_level', 
                         'city_tier', 'internet_access', 'extracurricular_involvement']
NUMERICAL_FEATURES = ['cgpa', 'tenth_percentage', 'twelfth_percentage', 'backlogs',
                       'study_hours_per_day', 'attendance_percentage', 'projects_completed',
                       'internships_completed', 'coding_skill_rating', 'communication_skill_rating',
                       'aptitude_skill_rating', 'hackathons_participated', 'certifications_count',
                       'sleep_hours', 'stress_level', 'skill_composite', 'academic_score', 
                       'experience_score', 'healthy_sleep', 'high_achiever']

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

X = df_eng[ALL_FEATURES]
y_clf = (df_eng['placement_status'] == 'Placed').astype(int)
y_reg = df_eng['salary_lpa']

# Train-test split 80:20
X_train, X_test, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

_, _, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Test size:  {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.0f}%)")
print(f"\\nClass distribution (train): Placed={y_clf_train.sum()}, Not Placed={(~y_clf_train.astype(bool)).sum()}")
print(f"Class distribution (test):  Placed={y_clf_test.sum()}, Not Placed={(~y_clf_test.astype(bool)).sum()}")
"""))

cells.append(nbf.v4.new_code_cell("""# Build preprocessing pipeline
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, NUMERICAL_FEATURES),
    ('cat', categorical_transformer, CATEGORICAL_FEATURES)
])

print("✅ Preprocessor pipeline siap!")
"""))

# Modeling - Classification
cells.append(nbf.v4.new_markdown_cell("""## 🤖 6. Modeling

### 6.1 Classification Task (Placement Status Prediction)

**Alasan pemilihan algoritma:**
- **Logistic Regression**: Baseline model yang sederhana, interpretable, dan cepat. Cocok untuk binary classification.
- **Random Forest Classifier**: Ensemble method yang robust terhadap outlier dan overfitting, menangani fitur heterogen dengan baik.
- **Gradient Boosting Classifier**: Boosting algorithm yang umumnya memberikan performa terbaik untuk structured data dengan kemampuan menangkap non-linear relationship.
"""))
cells.append(nbf.v4.new_code_cell("""# Classification models
clf_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

clf_results = {}
clf_pipelines = {}

for name, model in clf_models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    pipeline.fit(X_train, y_clf_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    acc = accuracy_score(y_clf_test, y_pred)
    f1 = f1_score(y_clf_test, y_pred, average='weighted')
    auc = roc_auc_score(y_clf_test, y_prob) if y_prob is not None else None
    
    clf_results[name] = {'Accuracy': acc, 'F1-Score': f1, 'ROC-AUC': auc}
    clf_pipelines[name] = pipeline
    
    print(f"\\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}" if auc else "")

clf_df = pd.DataFrame(clf_results).T
print("\\n=== CLASSIFICATION RESULTS SUMMARY ===")
print(clf_df.round(4))
"""))

cells.append(nbf.v4.new_code_cell("""# Confusion Matrix for best classifier
best_clf_name = clf_df['F1-Score'].idxmax()
best_clf = clf_pipelines[best_clf_name]
y_pred_best = best_clf.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_clf_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Not Placed', 'Placed'],
            yticklabels=['Not Placed', 'Placed'])
axes[0].set_title(f'Confusion Matrix\\n{best_clf_name}')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Model comparison
metrics = ['Accuracy', 'F1-Score', 'ROC-AUC']
x = np.arange(len(clf_df.index))
width = 0.25
bars_colors = ['#3498db', '#2ecc71', '#e74c3c']

for i, metric in enumerate(metrics):
    axes[1].bar(x + i*width, clf_df[metric], width, label=metric, color=bars_colors[i], alpha=0.8)

axes[1].set_xlabel('Model')
axes[1].set_ylabel('Score')
axes[1].set_title('Perbandingan Model Klasifikasi')
axes[1].set_xticks(x + width)
axes[1].set_xticklabels(clf_df.index, rotation=10)
axes[1].legend()
axes[1].set_ylim(0, 1.1)

plt.tight_layout()
plt.show()

print(f"\\n✅ Best Classification Model: {best_clf_name}")
print(f"\\nClassification Report:")
print(classification_report(y_clf_test, y_pred_best, target_names=['Not Placed', 'Placed']))
"""))

# Regression
cells.append(nbf.v4.new_markdown_cell("""### 6.2 Regression Task (Salary Prediction)

**Alasan pemilihan algoritma:**
- **Ridge Regression**: Regularized linear regression yang mengatasi multicollinearity, tepat sebagai baseline regresi.
- **Random Forest Regressor**: Non-parametric ensemble yang mampu menangkap hubungan non-linear antar fitur dan target.
- **Gradient Boosting Regressor**: State-of-the-art untuk tabular data, unggul dalam menangkap pola kompleks secara bertahap.
"""))
cells.append(nbf.v4.new_code_cell("""# Regression models
reg_models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

reg_results = {}
reg_pipelines = {}

for name, model in reg_models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    
    pipeline.fit(X_train, y_reg_train)
    y_pred = pipeline.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred))
    mae = mean_absolute_error(y_reg_test, y_pred)
    r2 = r2_score(y_reg_test, y_pred)
    
    reg_results[name] = {'RMSE': rmse, 'MAE': mae, 'R²': r2}
    reg_pipelines[name] = pipeline
    
    print(f"\\n{'='*40}")
    print(f"  {name}")
    print(f"{'='*40}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  R²   : {r2:.4f}")

reg_df = pd.DataFrame(reg_results).T
print("\\n=== REGRESSION RESULTS SUMMARY ===")
print(reg_df.round(4))
"""))

cells.append(nbf.v4.new_code_cell("""# Regression visualization
best_reg_name = reg_df['R²'].idxmax()
best_reg = reg_pipelines[best_reg_name]
y_pred_reg = best_reg.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Actual vs Predicted
axes[0].scatter(y_reg_test, y_pred_reg, alpha=0.3, color='#3498db', s=20)
min_val, max_val = min(y_reg_test.min(), y_pred_reg.min()), max(y_reg_test.max(), y_pred_reg.max())
axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
axes[0].set_xlabel('Actual Salary (LPA)')
axes[0].set_ylabel('Predicted Salary (LPA)')
axes[0].set_title(f'Actual vs Predicted\\n{best_reg_name}')
axes[0].legend()

# Model comparison (R²)
colors = ['#e74c3c' if r < 0.5 else '#f39c12' if r < 0.7 else '#2ecc71' for r in reg_df['R²']]
axes[1].bar(reg_df.index, reg_df['R²'], color=colors, alpha=0.8, edgecolor='white')
axes[1].set_title('Perbandingan Model Regresi (R²)')
axes[1].set_ylabel('R² Score')
axes[1].set_ylim(0, 1)
axes[1].axhline(y=0.7, color='green', linestyle='--', alpha=0.5, label='Target R²=0.7')
axes[1].legend()
for i, (name, row) in enumerate(reg_df.iterrows()):
    axes[1].text(i, row['R²'] + 0.01, f"{row['R²']:.3f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()

print(f"\\n✅ Best Regression Model: {best_reg_name}")
"""))

# Save models
cells.append(nbf.v4.new_markdown_cell("""## 💾 7. Interpretasi Hasil & Simpan Model Terbaik

**Interpretasi Metrik Evaluasi:**
- **Klasifikasi**: Model terbaik dipilih berdasarkan F1-Score (weighted) karena dataset imbalanced. ROC-AUC digunakan sebagai metrik tambahan untuk mengevaluasi kemampuan diskriminasi model.
- **Regresi**: Model terbaik dipilih berdasarkan R² tertinggi dan RMSE terendah. R² mengukur proporsi variansi yang dapat dijelaskan model.
"""))
cells.append(nbf.v4.new_code_cell("""# Simpan model terbaik
os.makedirs('models', exist_ok=True)

# Save best classification model
with open('models/best_classifier.pkl', 'wb') as f:
    pickle.dump(best_clf, f)

# Save best regression model
with open('models/best_regressor.pkl', 'wb') as f:
    pickle.dump(best_reg, f)

# Save feature metadata
feature_metadata = {
    'numerical_features': NUMERICAL_FEATURES,
    'categorical_features': CATEGORICAL_FEATURES,
    'all_features': ALL_FEATURES
}
import json
with open('models/feature_metadata.json', 'w') as f:
    json.dump(feature_metadata, f)

print("✅ Model berhasil disimpan!")
print(f"  📁 models/best_classifier.pkl  → {best_clf_name}")
print(f"  📁 models/best_regressor.pkl   → {best_reg_name}")
print(f"  📁 models/feature_metadata.json")
print()
print(f"=== FINAL SUMMARY ===")
print(f"Best Classifier : {best_clf_name}")
print(f"  - Accuracy : {clf_results[best_clf_name]['Accuracy']:.4f}")
print(f"  - F1-Score : {clf_results[best_clf_name]['F1-Score']:.4f}")
print(f"  - ROC-AUC  : {clf_results[best_clf_name]['ROC-AUC']:.4f}")
print()
print(f"Best Regressor  : {best_reg_name}")
print(f"  - RMSE : {reg_results[best_reg_name]['RMSE']:.4f}")
print(f"  - MAE  : {reg_results[best_reg_name]['MAE']:.4f}")
print(f"  - R²   : {reg_results[best_reg_name]['R²']:.4f}")
"""))

# Set cells
nb.cells = cells

# Write notebook
with open('/home/claude/UTS_ModelDeployment/EDA_Modeling.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created!")
