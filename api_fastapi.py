import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Path Setup 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# FastAPI App 
app = FastAPI(
    title="Student Placement Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Models 
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "best_classifier.pkl"), "rb") as f:
        clf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "best_regressor.pkl"), "rb") as f:
        reg = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "feature_metadata.json")) as f:
        meta = json.load(f)
    return clf, reg, meta

clf_model, reg_model, feature_meta = load_artifacts()

# Input Schema 
class StudentInput(BaseModel):
    gender: str = Field(..., example="Male", description="Jenis kelamin: Male / Female")
    branch: str = Field(..., example="Computer Science", description="Jurusan kuliah")
    cgpa: float = Field(..., ge=4.0, le=10.0, example=8.2, description="CGPA (4.0 - 10.0)")
    tenth_percentage: float = Field(..., ge=0, le=100, example=82.0)
    twelfth_percentage: float = Field(..., ge=0, le=100, example=78.5)
    backlogs: int = Field(default=0, ge=0, le=20, example=0)
    study_hours_per_day: float = Field(..., ge=0, le=16, example=5.0)
    attendance_percentage: float = Field(..., ge=0, le=100, example=88.0)
    projects_completed: int = Field(..., ge=0, le=20, example=4)
    internships_completed: int = Field(..., ge=0, le=10, example=2)
    coding_skill_rating: int = Field(..., ge=1, le=10, example=8)
    communication_skill_rating: int = Field(..., ge=1, le=10, example=7)
    aptitude_skill_rating: int = Field(..., ge=1, le=10, example=7)
    hackathons_participated: int = Field(default=0, ge=0, le=20, example=2)
    certifications_count: int = Field(default=0, ge=0, le=20, example=3)
    sleep_hours: float = Field(..., ge=2, le=14, example=7.0)
    stress_level: int = Field(..., ge=1, le=10, example=4)
    part_time_job: str = Field(default="No", example="No", description="Yes / No")
    family_income_level: str = Field(..., example="Medium", description="Low / Medium / High")
    city_tier: str = Field(..., example="Tier 1", description="Tier 1 / Tier 2 / Tier 3")
    internet_access: str = Field(default="Yes", example="Yes", description="Yes / No")
    extracurricular_involvement: str = Field(..., example="Medium",
                                             description="Low / Medium / High")

    @field_validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError("gender harus 'Male' atau 'Female'")
        return v

    @field_validator('family_income_level')
    def validate_income(cls, v):
        if v not in ['Low', 'Medium', 'High']:
            raise ValueError("family_income_level harus 'Low', 'Medium', atau 'High'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "gender": "Male",
                "branch": "Computer Science",
                "cgpa": 8.5,
                "tenth_percentage": 85.0,
                "twelfth_percentage": 82.0,
                "backlogs": 0,
                "study_hours_per_day": 6.0,
                "attendance_percentage": 90.0,
                "projects_completed": 5,
                "internships_completed": 2,
                "coding_skill_rating": 8,
                "communication_skill_rating": 7,
                "aptitude_skill_rating": 8,
                "hackathons_participated": 3,
                "certifications_count": 4,
                "sleep_hours": 7.0,
                "stress_level": 4,
                "part_time_job": "No",
                "family_income_level": "Medium",
                "city_tier": "Tier 1",
                "internet_access": "Yes",
                "extracurricular_involvement": "High"
            }
        }


# Output Schemas 
class PlacementResponse(BaseModel):
    placement_status: str
    placed: bool
    probability_placed: float
    probability_not_placed: float
    confidence: str

class SalaryResponse(BaseModel):
    predicted_salary_lpa: float
    salary_range_low: float
    salary_range_high: float
    currency: str = "INR LPA"

class CombinedResponse(BaseModel):
    placement: PlacementResponse
    salary: SalaryResponse
    recommendation: str


# Feature Engineering 
def prepare_input(data: StudentInput) -> pd.DataFrame:
    row = data.model_dump()
    df = pd.DataFrame([row])

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

    df['healthy_sleep'] = int(6 <= df['sleep_hours'].values[0] <= 9)
    df['high_achiever'] = int(
        df['cgpa'].values[0] >= 8.0 and df['coding_skill_rating'].values[0] >= 7
    )

    return df[feature_meta['all_features']]

# ENDPOINTS

@app.get("/", tags=["Health"])
def root():
    """Health check dan info API."""
    return {
        "status": "running",
        "message": "Student Placement Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "classify": "POST /predict/placement",
            "regress": "POST /predict/salary",
            "combined": "POST /predict/both",
            "docs": "GET /docs"
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Cek status model."""
    return {
        "status": "healthy",
        "classifier": type(clf_model.named_steps.get('classifier', clf_model)).__name__,
        "regressor": type(reg_model.named_steps.get('regressor', reg_model)).__name__,
        "features_count": len(feature_meta['all_features'])
    }


@app.post("/predict/placement", response_model=PlacementResponse, tags=["Prediction"])
def predict_placement(data: StudentInput):
    """
    **Prediksi Status Penempatan Kerja (Klasifikasi)**

    Menerima data profil mahasiswa dan mengembalikan:
    - Status: Placed / Not Placed
    - Probabilitas prediksi
    - Tingkat keyakinan model
    """
    try:
        X = prepare_input(data)
        pred = clf_model.predict(X)[0]
        prob = clf_model.predict_proba(X)[0]

        placed = bool(pred == 1)
        prob_placed = float(prob[1])
        prob_not_placed = float(prob[0])

        if prob_placed >= 0.8:
            confidence = "Sangat Tinggi (≥80%)"
        elif prob_placed >= 0.65:
            confidence = "Tinggi (65-80%)"
        elif prob_placed >= 0.5:
            confidence = "Sedang (50-65%)"
        else:
            confidence = "Rendah (<50%)"

        return PlacementResponse(
            placement_status="Placed" if placed else "Not Placed",
            placed=placed,
            probability_placed=round(prob_placed, 4),
            probability_not_placed=round(prob_not_placed, 4),
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/salary", response_model=SalaryResponse, tags=["Prediction"])
def predict_salary(data: StudentInput):
    """
    **Estimasi Gaji (Regresi)**

    Menerima data profil mahasiswa dan mengembalikan:
    - Prediksi gaji dalam LPA (Lakh Per Annum)
    - Range estimasi gaji (±1 RMSE)
    """
    try:
        X = prepare_input(data)
        pred = float(reg_model.predict(X)[0])
        pred = max(0.0, pred)

        rmse = 3.97  # dari evaluasi model
        low = max(0.0, pred - rmse)
        high = pred + rmse

        return SalaryResponse(
            predicted_salary_lpa=round(pred, 2),
            salary_range_low=round(low, 2),
            salary_range_high=round(high, 2),
            currency="INR LPA"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/both", response_model=CombinedResponse, tags=["Prediction"])
def predict_both(data: StudentInput):
    """
    **Prediksi Lengkap: Placement + Salary**

    Mengembalikan hasil klasifikasi dan regresi sekaligus,
    beserta rekomendasi berdasarkan profil mahasiswa.
    """
    try:
        X = prepare_input(data)

        # Classification
        clf_pred = clf_model.predict(X)[0]
        clf_prob = clf_model.predict_proba(X)[0]
        placed = bool(clf_pred == 1)
        prob_placed = float(clf_prob[1])

        if clf_prob[1] >= 0.8:
            confidence = "Sangat Tinggi (≥80%)"
        elif clf_prob[1] >= 0.65:
            confidence = "Tinggi (65-80%)"
        elif clf_prob[1] >= 0.5:
            confidence = "Sedang (50-65%)"
        else:
            confidence = "Rendah (<50%)"

        placement = PlacementResponse(
            placement_status="Placed" if placed else "Not Placed",
            placed=placed,
            probability_placed=round(float(clf_prob[1]), 4),
            probability_not_placed=round(float(clf_prob[0]), 4),
            confidence=confidence
        )

        # Regression
        reg_pred = max(0.0, float(reg_model.predict(X)[0]))
        rmse = 3.97
        salary = SalaryResponse(
            predicted_salary_lpa=round(reg_pred, 2),
            salary_range_low=round(max(0.0, reg_pred - rmse), 2),
            salary_range_high=round(reg_pred + rmse, 2),
            currency="INR LPA"
        )

        # Recommendation
        skill_avg = (data.coding_skill_rating + data.communication_skill_rating +
                     data.aptitude_skill_rating) / 3
        if placed and reg_pred >= 15:
            rec = "Profil sangat kuat! Siap bersaing di perusahaan tier-1."
        elif placed and reg_pred >= 10:
            rec = "Peluang penempatan baik. Tingkatkan skill untuk salary lebih tinggi."
        elif not placed and data.cgpa >= 7.5:
            rec = "CGPA baik, namun perlu tambah pengalaman & skill teknis."
        else:
            rec = "Fokus pada peningkatan CGPA, skill, dan portofolio proyek."

        return CombinedResponse(
            placement=placement,
            salary=salary,
            recommendation=rec
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
