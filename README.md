# 🎓 UTS Model Deployment — DTSC6012001
**Dataset A** (NIM Ganjil) | Student Placement & Salary Prediction

---

## 📁 Struktur File

```
UTS_ModelDeployment/
├── A.csv                    ← Dataset fitur
├── A_targets.csv            ← Dataset target
├── EDA_Modeling.ipynb       ← Soal 1: EDA & Modeling
├── pipeline.py              ← Soal 2: Sklearn Pipeline + MLflow
├── app_streamlit.py         ← Soal 3: Monolithic Streamlit
├── api_fastapi.py           ← Soal 4a: FastAPI Backend
├── frontend_streamlit.py    ← Soal 4b: Streamlit Frontend (Decoupled)
├── requirements.txt         ← Dependencies
├── models/
│   ├── best_classifier.pkl  ← Gradient Boosting Classifier
│   ├── best_regressor.pkl   ← Gradient Boosting Regressor
│   └── feature_metadata.json
└── mlruns/                  ← MLflow experiment logs
```

---

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Soal 1 — Jupyter Notebook
```bash
jupyter notebook EDA_Modeling.ipynb
```

### 3. Soal 2 — Pipeline + MLflow
```bash
python pipeline.py

# Lihat MLflow UI:
mlflow ui
# Buka: http://localhost:5000
```

### 4. Soal 3 — Monolithic Streamlit
```bash
streamlit run app_streamlit.py
# Buka: http://localhost:8501
```

### 5. Soal 4 — Decoupled Architecture

**Terminal 1 — Backend FastAPI:**
```bash
uvicorn api_fastapi:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

**Terminal 2 — Frontend Streamlit:**
```bash
streamlit run frontend_streamlit.py
# Buka: http://localhost:8501
```

---

## 📊 Hasil Model

| Task | Algoritma | Metrik | Nilai |
|------|-----------|--------|-------|
| Klasifikasi | Gradient Boosting | F1-Score | **0.874** |
| Klasifikasi | Gradient Boosting | ROC-AUC | **~0.88** |
| Regresi | Gradient Boosting | R² | **0.774** |
| Regresi | Gradient Boosting | RMSE | **3.97 LPA** |

---

## 🔧 API Endpoints (FastAPI)

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/` | API info |
| GET | `/health` | Status model |
| POST | `/predict/placement` | Prediksi klasifikasi |
| POST | `/predict/salary` | Prediksi regresi |
| POST | `/predict/both` | Prediksi combined |

---

## 🎯 Fitur Dataset

**Original (22 fitur):** gender, branch, cgpa, tenth_percentage, twelfth_percentage,
backlogs, study_hours_per_day, attendance_percentage, projects_completed,
internships_completed, coding_skill_rating, communication_skill_rating,
aptitude_skill_rating, hackathons_participated, certifications_count, sleep_hours,
stress_level, part_time_job, family_income_level, city_tier, internet_access,
extracurricular_involvement

**Engineered (5 fitur):**
- `skill_composite` — rata-rata 3 skill rating
- `academic_score` — skor akademik tertimbang
- `experience_score` — skor pengalaman total
- `healthy_sleep` — flag tidur sehat (6-9 jam)
- `high_achiever` — flag mahasiswa berprestasi tinggi
