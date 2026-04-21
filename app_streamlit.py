"""
LINK Streamlit = https://uts-model-deployment-app.streamlit.app/ 
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Path Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Page Config 
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS 
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 12px; color: white;
        text-align: center; margin-bottom: 2rem;
    }
    .metric-card {
        background: white; border-radius: 10px; padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea; margin-bottom: 1rem;
    }
    .placed-badge {
        background: #d4edda; color: #155724;
        padding: 0.5rem 1.5rem; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem;
        display: inline-block;
    }
    .not-placed-badge {
        background: #f8d7da; color: #721c24;
        padding: 0.5rem 1.5rem; border-radius: 20px;
        font-weight: bold; font-size: 1.1rem;
        display: inline-block;
    }
    .sidebar-section {
        background: #f8f9fc; border-radius: 8px;
        padding: 0.8rem; margin-bottom: 0.5rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: bold; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# Load Models 
@st.cache_resource
def load_models():
    clf_path = os.path.join(MODEL_DIR, "best_classifier.pkl")
    reg_path = os.path.join(MODEL_DIR, "best_regressor.pkl")
    meta_path = os.path.join(MODEL_DIR, "feature_metadata.json")
    with open(clf_path, "rb") as f:
        clf = pickle.load(f)
    with open(reg_path, "rb") as f:
        reg = pickle.load(f)
    with open(meta_path) as f:
        meta = json.load(f)
    return clf, reg, meta

clf_model, reg_model, feature_meta = load_models()

# Feature Engineering Helper 
def engineer_features(row: dict) -> pd.DataFrame:
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


# SIDEBAR INPUTS
with st.sidebar:
    st.markdown("# Data Mahasiswa")
    st.markdown("---")

    # Academic Info
    st.markdown("## Akademik")
    gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
    branch = st.selectbox("Jurusan", [
        "Computer Science", "Information Technology", "Electronics",
        "Mechanical", "Civil", "Electrical", "Other"
    ])
    cgpa = st.slider("CGPA", 4.0, 10.0, 7.5, 0.1)
    tenth_pct = st.slider("Nilai Kelas 10 (%)", 40.0, 100.0, 75.0, 0.5)
    twelfth_pct = st.slider("Nilai Kelas 12 (%)", 40.0, 100.0, 75.0, 0.5)
    backlogs = st.number_input("Jumlah Backlog", 0, 10, 0)
    attendance_pct = st.slider("Kehadiran (%)", 50.0, 100.0, 85.0, 0.5)
    study_hours = st.slider("Jam Belajar/Hari", 0.0, 12.0, 4.0, 0.5)

    st.markdown("### Technical Skills")
    coding_skill = st.slider("Coding Skill (1-10)", 1, 10, 6)
    comm_skill = st.slider("Communication Skill (1-10)", 1, 10, 6)
    aptitude_skill = st.slider("Aptitude Skill (1-10)", 1, 10, 6)

    st.markdown("### Pengalaman")
    internships = st.number_input("Internship", 0, 5, 1)
    projects = st.number_input("Proyek", 0, 15, 3)
    certifications = st.number_input("Sertifikasi", 0, 15, 2)
    hackathons = st.number_input("Hackathon", 0, 10, 1)

    st.markdown("### Lifestyle")
    sleep_hours = st.slider("Jam Tidur/Malam", 3.0, 12.0, 7.0, 0.5)
    stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 5)
    part_time_job = st.selectbox("Part-time Job", ["Yes", "No"])
    family_income = st.selectbox("Pendapatan Keluarga", ["Low", "Medium", "High"])
    city_tier = st.selectbox("Tier Kota", ["Tier 1", "Tier 2", "Tier 3"])
    internet_access = st.selectbox("Akses Internet", ["Yes", "No"])
    extracurricular = st.selectbox("Ekstrakurikuler", ["Low", "Medium", "High"])

    predict_btn = st.button("🔮 Prediksi Sekarang", use_container_width=True)


# Main Header
st.markdown("""
<div class="main-header">
    <h1>Student Placement & Salary Predictor</h1>
    <p>Prediksi status penempatan kerja dan estimasi gaji menggunakan Model Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["Prediksi", "Analisis Fitur", "Info Model"])

with tab1:
    if predict_btn:
        input_data = {
            'gender': gender, 'branch': branch, 'cgpa': cgpa,
            'tenth_percentage': tenth_pct, 'twelfth_percentage': twelfth_pct,
            'backlogs': backlogs, 'study_hours_per_day': study_hours,
            'attendance_percentage': attendance_pct,
            'projects_completed': projects, 'internships_completed': internships,
            'coding_skill_rating': coding_skill,
            'communication_skill_rating': comm_skill,
            'aptitude_skill_rating': aptitude_skill,
            'hackathons_participated': hackathons,
            'certifications_count': certifications,
            'sleep_hours': sleep_hours, 'stress_level': stress_level,
            'part_time_job': part_time_job,
            'family_income_level': family_income,
            'city_tier': city_tier, 'internet_access': internet_access,
            'extracurricular_involvement': extracurricular
        }

        X_input = engineer_features(input_data)

        # Predictions
        clf_pred = clf_model.predict(X_input)[0]
        clf_prob = clf_model.predict_proba(X_input)[0]
        placement_label = "Placed" if clf_pred == 1 else "Not Placed"
        placement_prob = clf_prob[1] if clf_pred == 1 else clf_prob[0]

        reg_pred = reg_model.predict(X_input)[0]
        reg_pred = max(0.0, reg_pred)

        # Result Header
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("###Status Penempatan Kerja")
            badge_class = "placed-badge" if clf_pred == 1 else "not-placed-badge"
            st.markdown(
                f'<div style="text-align:center; margin:1rem 0">'
                f'<span class="{badge_class}">{placement_label}</span></div>',
                unsafe_allow_html=True
            )
            st.metric(
                "Confidence",
                f"{placement_prob*100:.1f}%",
                help="Probabilitas prediksi model"
            )

            # Probability gauge
            fig, ax = plt.subplots(figsize=(5, 3))
            categories = ['Not Placed', 'Placed']
            probs = [clf_prob[0]*100, clf_prob[1]*100]
            colors = ['#e74c3c', '#2ecc71']
            bars = ax.barh(categories, probs, color=colors, height=0.5, edgecolor='white')
            for bar, prob in zip(bars, probs):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f'{prob:.1f}%', va='center', fontweight='bold')
            ax.set_xlim(0, 115)
            ax.set_xlabel('Probabilitas (%)')
            ax.set_title('Distribusi Probabilitas Prediksi')
            ax.spines[['top','right','left']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("### Estimasi Gaji")
            st.metric(
                "Prediksi Gaji",
                f"₹ {reg_pred:.2f} LPA",
                help="Estimasi gaji dalam Lakh Per Annum"
            )
            if reg_pred > 0:
                st.info(f"Setara sekitar **Rp {reg_pred * 180_000:.0f}/tahun**")

            # Salary gauge
            fig2, ax2 = plt.subplots(figsize=(5, 3))
            salary_range = np.linspace(0, 22, 300)
            # Normal-ish distribution around predicted value
            mean_salary = 13.9
            ax2.fill_between(salary_range,
                             np.exp(-0.5*((salary_range - mean_salary)/3.5)**2),
                             alpha=0.3, color='#3498db', label='Distribusi Gaji')
            ax2.axvline(reg_pred, color='#e74c3c', linewidth=2.5,
                        label=f'Prediksi: {reg_pred:.1f} LPA')
            ax2.axvline(mean_salary, color='#2ecc71', linewidth=1.5,
                        linestyle='--', label=f'Rata-rata: {mean_salary:.1f} LPA')
            ax2.set_xlabel('Salary (LPA)')
            ax2.set_title('Posisi Gaji vs Distribusi')
            ax2.legend(fontsize=8)
            ax2.set_yticks([])
            ax2.spines[['top','right','left']].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # Profile Summary
        st.markdown("---")
        st.markdown("### Ringkasan Profil Mahasiswa")
        skill_composite = (coding_skill + comm_skill + aptitude_skill) / 3
        exp_score = internships*2 + projects + certifications + hackathons

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CGPA", f"{cgpa:.1f}/10")
        c2.metric("Skill Composite", f"{skill_composite:.1f}/10")
        c3.metric("Experience Score", f"{exp_score}")
        c4.metric("Attendance", f"{attendance_pct:.0f}%")

    else:
        st.info("Isi data mahasiswa di sidebar, lalu klik **Prediksi Sekarang**")


with tab2:
    st.markdown("### Analisis Fitur Input")
    if predict_btn:
        # Radar chart of key metrics
        categories = ['CGPA', 'Coding', 'Communication', 'Aptitude',
                      'Attendance', 'Experience']
        # Normalize to 0-10 scale
        values = [
            cgpa,
            coding_skill,
            comm_skill,
            aptitude_skill,
            attendance_pct / 10,
            min(exp_score / 2, 10)
        ]
        # Repeat first to close polygon
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        values_plot = values + values[:1]

        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.plot(angles, values_plot, 'o-', linewidth=2, color='#667eea')
        ax.fill(angles, values_plot, alpha=0.25, color='#667eea')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_title('Profil Kompetensi Mahasiswa', size=14, pad=20)
        ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.7)
        st.pyplot(fig)
        plt.close()

        # Bar chart comparison
        st.markdown("#### Perbandingan dengan Rata-rata Dataset")
        metrics_data = {
            'Metrik': ['CGPA', 'Coding Skill', 'Communication', 'Aptitude',
                       'Internships', 'Projects'],
            'Input Anda': [cgpa, coding_skill, comm_skill, aptitude_skill,
                           internships, projects],
            'Rata-rata Dataset': [7.5, 6.0, 6.1, 5.9, 1.2, 3.1]
        }
        df_compare = pd.DataFrame(metrics_data)

        fig2, ax2 = plt.subplots(figsize=(9, 4))
        x = np.arange(len(df_compare['Metrik']))
        w = 0.35
        ax2.bar(x - w/2, df_compare['Input Anda'], w, label='Input Anda',
                color='#667eea', alpha=0.85)
        ax2.bar(x + w/2, df_compare['Rata-rata Dataset'], w, label='Rata-rata Dataset',
                color='#adb5bd', alpha=0.85)
        ax2.set_xticks(x)
        ax2.set_xticklabels(df_compare['Metrik'], rotation=15)
        ax2.legend()
        ax2.set_title('Perbandingan Profil vs Rata-rata Dataset')
        ax2.spines[['top', 'right']].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()
    else:
        st.info("Lakukan prediksi terlebih dahulu untuk melihat analisis fitur.")


with tab3:
    st.markdown("### Informasi Model")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        #### Model Klasifikasi
        **Algoritma**: Gradient Boosting Classifier
        - **F1-Score (Weighted)**: 0.874
        - **ROC-AUC**: ~0.88
        - **Accuracy**: ~0.89
        - **Strategi**: Class-balanced training

        #### Preprocessing
        - Numerical: Median Imputation + StandardScaler
        - Categorical: Mode Imputation + OneHotEncoder
        - Feature Engineering: 5 fitur turunan
        """)

    with col2:
        st.markdown("""
        #### Model Regresi
        **Algoritma**: Gradient Boosting Regressor
        - **R² Score**: 0.774
        - **RMSE**: ~3.97 LPA
        - **Training**: Hanya mahasiswa Placed

        #### Dataset
        - **Sumber**: Dataset A (NIM Ganjil)
        - **Ukuran**: 5,000 mahasiswa
        - **Fitur**: 22 original + 5 engineered
        - **Target**: Placement Status & Salary LPA
        """)

    st.markdown("---")
    st.markdown("#### Arsitektur Pipeline")
    st.code("""
    Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', Pipeline([MedianImputer → StandardScaler]), numerical_features),
            ('cat', Pipeline([ModeImputer  → OneHotEncoder  ]), categorical_features)
        ])),
        ('model', GradientBoostingClassifier / GradientBoostingRegressor)
    ])
    """, language='python')
