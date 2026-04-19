"""
========================================================
UTS Model Deployment - DTSC6012001
Soal 4: Decoupled Architecture — Streamlit Frontend
========================================================
Jalankan (setelah FastAPI aktif):
  streamlit run frontend_streamlit.py

Pastikan FastAPI sudah berjalan di: http://localhost:8000
"""

import requests
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Config ──────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Student Predictor — Decoupled",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-banner {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #4facfe 100%);
        padding: 1.8rem; border-radius: 14px; color: white;
        text-align: center; margin-bottom: 1.5rem;
    }
    .api-status-ok {
        background: #d4edda; color: #155724; border-radius: 8px;
        padding: 0.5rem 1rem; display: inline-block; font-weight: bold;
    }
    .api-status-err {
        background: #f8d7da; color: #721c24; border-radius: 8px;
        padding: 0.5rem 1rem; display: inline-block; font-weight: bold;
    }
    .result-box {
        background: #f8f9fc; border-radius: 12px;
        padding: 1.5rem; margin: 1rem 0;
        border: 1px solid #e0e0e0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white; border: none; border-radius: 8px;
        font-weight: bold; padding: 0.6rem 1.5rem; width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ── API Health Check ────────────────────────────────────────
def check_api_health():
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            return True, r.json()
    except Exception:
        pass
    return False, {}


# ── Helper: POST request ────────────────────────────────────
def call_api(endpoint: str, payload: dict):
    try:
        r = requests.post(f"{API_URL}{endpoint}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "❌ Tidak dapat terhubung ke FastAPI. Pastikan server berjalan di port 8000."
    except requests.exceptions.HTTPError as e:
        detail = r.json().get("detail", str(e)) if r else str(e)
        return None, f"❌ Error dari API: {detail}"
    except Exception as e:
        return None, f"❌ Error: {str(e)}"


# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.markdown("""
<div class="main-banner">
    <h1>🚀 Student Placement Predictor</h1>
    <p>Decoupled Architecture — Streamlit Frontend × FastAPI Backend</p>
    <small>UTS DTSC6012001 | Dataset A</small>
</div>
""", unsafe_allow_html=True)

# API Status bar
is_healthy, health_data = check_api_health()
col_s1, col_s2, col_s3 = st.columns([1, 2, 1])
with col_s2:
    if is_healthy:
        st.markdown(
            f'<div style="text-align:center"><span class="api-status-ok">'
            f'✅ FastAPI Connected — '
            f'Classifier: {health_data.get("classifier","?")} | '
            f'Regressor: {health_data.get("regressor","?")}'
            f'</span></div>', unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div style="text-align:center"><span class="api-status-err">'
            '❌ FastAPI Offline — jalankan: uvicorn api_fastapi:app --reload --port 8000'
            '</span></div>', unsafe_allow_html=True
        )

st.markdown("")

# ═══════════════════════════════════════════════════════════
# SIDEBAR INPUT
# ═══════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧑‍🎓 Data Mahasiswa")
    st.caption("Isi profil mahasiswa untuk prediksi")
    st.markdown("---")

    with st.expander("🎓 Data Akademik", expanded=True):
        gender = st.selectbox("Jenis Kelamin", ["Male", "Female"])
        branch = st.selectbox("Jurusan", [
            "Computer Science", "Information Technology",
            "Electronics", "Mechanical", "Civil", "Electrical", "Other"
        ])
        cgpa = st.slider("CGPA", 4.0, 10.0, 7.8, 0.1)
        tenth_pct = st.slider("Nilai Kelas 10 (%)", 40.0, 100.0, 78.0, 0.5)
        twelfth_pct = st.slider("Nilai Kelas 12 (%)", 40.0, 100.0, 76.0, 0.5)
        backlogs = st.number_input("Jumlah Backlog", 0, 10, 0)
        attendance_pct = st.slider("Kehadiran (%)", 50.0, 100.0, 87.0, 0.5)
        study_hours = st.slider("Jam Belajar/Hari", 0.0, 12.0, 5.0, 0.5)

    with st.expander("💻 Technical Skills"):
        coding_skill = st.slider("Coding Skill (1-10)", 1, 10, 7)
        comm_skill = st.slider("Communication Skill (1-10)", 1, 10, 7)
        aptitude_skill = st.slider("Aptitude Skill (1-10)", 1, 10, 7)

    with st.expander("🏆 Pengalaman & Aktivitas"):
        internships = st.number_input("Internship", 0, 5, 1)
        projects = st.number_input("Proyek", 0, 15, 4)
        certifications = st.number_input("Sertifikasi", 0, 15, 3)
        hackathons = st.number_input("Hackathon", 0, 10, 2)

    with st.expander("🌿 Gaya Hidup"):
        sleep_hours = st.slider("Jam Tidur/Malam", 3.0, 12.0, 7.0, 0.5)
        stress_level = st.slider("Tingkat Stres (1-10)", 1, 10, 4)
        part_time_job = st.selectbox("Part-time Job", ["No", "Yes"])
        family_income = st.selectbox("Pendapatan Keluarga", ["Low", "Medium", "High"])
        city_tier = st.selectbox("Tier Kota", ["Tier 1", "Tier 2", "Tier 3"])
        internet_access = st.selectbox("Akses Internet", ["Yes", "No"])
        extracurricular = st.selectbox("Ekstrakurikuler", ["Low", "Medium", "High"])

    st.markdown("---")
    st.markdown("### 🎯 Pilih Prediksi")
    scenario = st.radio("Skenario", [
        "🏢 Placement saja (Klasifikasi)",
        "💰 Salary saja (Regresi)",
        "🔗 Keduanya (Combined)"
    ])

    predict_btn = st.button("🚀 Kirim ke API & Prediksi", use_container_width=True)


# ── Build Payload ────────────────────────────────────────────
payload = {
    "gender": gender, "branch": branch, "cgpa": cgpa,
    "tenth_percentage": tenth_pct, "twelfth_percentage": twelfth_pct,
    "backlogs": backlogs, "study_hours_per_day": study_hours,
    "attendance_percentage": attendance_pct,
    "projects_completed": projects, "internships_completed": internships,
    "coding_skill_rating": coding_skill,
    "communication_skill_rating": comm_skill,
    "aptitude_skill_rating": aptitude_skill,
    "hackathons_participated": hackathons,
    "certifications_count": certifications,
    "sleep_hours": sleep_hours, "stress_level": stress_level,
    "part_time_job": part_time_job,
    "family_income_level": family_income,
    "city_tier": city_tier, "internet_access": internet_access,
    "extracurricular_involvement": extracurricular
}

# ═══════════════════════════════════════════════════════════
# MAIN TABS
# ═══════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs(["🔮 Hasil Prediksi", "📦 Request & Response JSON", "🔬 Skenario Batch"])

with tab1:
    if predict_btn:
        if not is_healthy:
            st.error("FastAPI tidak aktif! Jalankan server terlebih dahulu.")
        else:
            # ── Skenario 1: Placement Only
            if scenario == "🏢 Placement saja (Klasifikasi)":
                with st.spinner("Mengirim request ke API /predict/placement ..."):
                    result, err = call_api("/predict/placement", payload)
                if err:
                    st.error(err)
                else:
                    st.success("✅ Response diterima dari FastAPI!")
                    col1, col2 = st.columns(2)
                    with col1:
                        placed = result['placed']
                        badge = "🟢 **PLACED**" if placed else "🔴 **NOT PLACED**"
                        st.markdown(f"### {badge}")
                        st.metric("Probabilitas Placed",
                                  f"{result['probability_placed']*100:.1f}%")
                        st.metric("Keyakinan Model", result['confidence'])

                        # Prob bar
                        fig, ax = plt.subplots(figsize=(5, 2.5))
                        labels = ['Not Placed', 'Placed']
                        vals = [result['probability_not_placed']*100,
                                result['probability_placed']*100]
                        ax.barh(labels, vals, color=['#e74c3c', '#2ecc71'], height=0.5)
                        for i, v in enumerate(vals):
                            ax.text(v+1, i, f'{v:.1f}%', va='center', fontweight='bold')
                        ax.set_xlim(0, 115)
                        ax.set_xlabel('Probabilitas (%)')
                        ax.set_title('Distribusi Probabilitas')
                        ax.spines[['top','right','left']].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    with col2:
                        st.markdown("#### 📊 Detail Hasil")
                        st.json(result)

            # ── Skenario 2: Salary Only
            elif scenario == "💰 Salary saja (Regresi)":
                with st.spinner("Mengirim request ke API /predict/salary ..."):
                    result, err = call_api("/predict/salary", payload)
                if err:
                    st.error(err)
                else:
                    st.success("✅ Response diterima dari FastAPI!")
                    col1, col2 = st.columns(2)
                    with col1:
                        pred_sal = result['predicted_salary_lpa']
                        st.markdown(f"### 💰 Estimasi Gaji: **₹ {pred_sal:.2f} LPA**")
                        st.metric("Range Bawah", f"₹ {result['salary_range_low']:.2f} LPA")
                        st.metric("Range Atas", f"₹ {result['salary_range_high']:.2f} LPA")

                        # Salary gauge
                        fig, ax = plt.subplots(figsize=(5, 3))
                        x = np.linspace(0, 22, 300)
                        y = np.exp(-0.5*((x - 13.9)/3.5)**2)
                        ax.fill_between(x, y, alpha=0.2, color='#3498db')
                        ax.axvline(pred_sal, color='#e74c3c', lw=2.5,
                                   label=f'Prediksi: {pred_sal:.1f} LPA')
                        ax.axvspan(result['salary_range_low'],
                                   result['salary_range_high'],
                                   alpha=0.15, color='#e74c3c', label='Range ±RMSE')
                        ax.axvline(13.9, color='#2ecc71', lw=1.5, ls='--',
                                   label='Mean: 13.9 LPA')
                        ax.set_xlabel('Salary (LPA)')
                        ax.set_title('Estimasi Gaji')
                        ax.legend(fontsize=8)
                        ax.set_yticks([])
                        ax.spines[['top','right','left']].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
                    with col2:
                        st.markdown("#### 📊 Detail Hasil")
                        st.json(result)

            # ── Skenario 3: Combined
            else:
                with st.spinner("Mengirim request ke API /predict/both ..."):
                    result, err = call_api("/predict/both", payload)
                if err:
                    st.error(err)
                else:
                    st.success("✅ Response diterima dari FastAPI!")
                    col1, col2 = st.columns(2)

                    with col1:
                        pl = result['placement']
                        placed = pl['placed']
                        badge = "🟢 PLACED" if placed else "🔴 NOT PLACED"
                        st.markdown(f"#### {badge}")
                        st.metric("Prob. Placed",
                                  f"{pl['probability_placed']*100:.1f}%")
                        st.metric("Confidence", pl['confidence'])

                    with col2:
                        sal = result['salary']
                        st.markdown(f"#### 💰 ₹ {sal['predicted_salary_lpa']} LPA")
                        st.metric("Range",
                                  f"₹{sal['salary_range_low']} – {sal['salary_range_high']} LPA")

                    st.info(f"**Rekomendasi:** {result['recommendation']}")

                    with st.expander("📋 Raw JSON Response"):
                        st.json(result)
    else:
        st.info("👈 Isi form di sidebar dan klik **Kirim ke API & Prediksi**")
        st.markdown("""
        #### Arsitektur Decoupled
        ```
        [Streamlit Frontend]  ──POST──▶  [FastAPI Backend]
        (frontend_streamlit.py)           (api_fastapi.py)
              Port: 8501                     Port: 8000
                        ↓
                  [ML Models .pkl]
        ```
        **Keunggulan arsitektur decoupled:**
        - Backend & Frontend dapat di-deploy & scale secara independen
        - API dapat digunakan oleh banyak client (web, mobile, dll)
        - Pemisahan concern yang jelas (UI vs Business Logic)
        """)


with tab2:
    st.markdown("### 📦 Request & Response Detail")
    st.markdown("#### Request Payload (dikirim ke API):")
    st.json(payload)

    if predict_btn and is_healthy:
        endpoint_map = {
            "🏢 Placement saja (Klasifikasi)": "/predict/placement",
            "💰 Salary saja (Regresi)": "/predict/salary",
            "🔗 Keduanya (Combined)": "/predict/both"
        }
        ep = endpoint_map[scenario]
        st.markdown(f"#### Endpoint: `POST {API_URL}{ep}`")
        result, err = call_api(ep, payload)
        if result:
            st.markdown("#### Response JSON:")
            st.json(result)
            st.code(
                f"curl -X POST '{API_URL}{ep}' \\\n"
                f"  -H 'Content-Type: application/json' \\\n"
                f"  -d '{{}}'",
                language="bash"
            )
    else:
        st.info("Lakukan prediksi untuk melihat detail request/response.")


with tab3:
    st.markdown("### 🔬 Skenario Batch Testing")
    st.markdown("Uji beberapa profil mahasiswa sekaligus untuk perbandingan.")

    test_cases = {
        "Mahasiswa Berprestasi": {
            "cgpa": 9.2, "coding_skill_rating": 9, "internships_completed": 3,
            "projects_completed": 8, "attendance_percentage": 95.0
        },
        "Mahasiswa Rata-rata": {
            "cgpa": 7.0, "coding_skill_rating": 6, "internships_completed": 1,
            "projects_completed": 3, "attendance_percentage": 80.0
        },
        "Mahasiswa Berisiko": {
            "cgpa": 5.5, "coding_skill_rating": 4, "internships_completed": 0,
            "projects_completed": 1, "attendance_percentage": 65.0
        }
    }

    if st.button("▶️ Jalankan Batch Testing", use_container_width=True):
        if not is_healthy:
            st.error("FastAPI tidak aktif!")
        else:
            results_list = []
            base = {
                "gender": "Male", "branch": "Computer Science",
                "tenth_percentage": 75.0, "twelfth_percentage": 75.0,
                "backlogs": 0, "study_hours_per_day": 5.0,
                "communication_skill_rating": 7, "aptitude_skill_rating": 7,
                "hackathons_participated": 1, "certifications_count": 2,
                "sleep_hours": 7.0, "stress_level": 5, "part_time_job": "No",
                "family_income_level": "Medium", "city_tier": "Tier 1",
                "internet_access": "Yes", "extracurricular_involvement": "Medium"
            }

            for name, overrides in test_cases.items():
                test_payload = {**base, **overrides}
                res, err = call_api("/predict/both", test_payload)
                if res:
                    results_list.append({
                        "Profil": name,
                        "CGPA": overrides['cgpa'],
                        "Coding Skill": overrides['coding_skill_rating'],
                        "Status": res['placement']['placement_status'],
                        "Prob Placed (%)": f"{res['placement']['probability_placed']*100:.1f}",
                        "Salary (LPA)": res['salary']['predicted_salary_lpa'],
                        "Rekomendasi": res['recommendation'][:50] + "..."
                    })

            if results_list:
                df_res = pd.DataFrame(results_list)
                st.dataframe(df_res, use_container_width=True)

                # Visualization
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                profiles = [r["Profil"] for r in results_list]
                probs = [float(r["Prob Placed (%)"]) for r in results_list]
                salaries = [float(r["Salary (LPA)"]) for r in results_list]
                colors = ['#2ecc71' if p >= 50 else '#e74c3c' for p in probs]

                axes[0].bar(profiles, probs, color=colors, alpha=0.85, edgecolor='white')
                axes[0].set_title('Probabilitas Placed (%)')
                axes[0].set_ylabel('%')
                axes[0].axhline(50, color='gray', ls='--', lw=1)
                for i, v in enumerate(probs):
                    axes[0].text(i, v+1, f'{v:.0f}%', ha='center', fontweight='bold')

                axes[1].bar(profiles, salaries, color='#3498db', alpha=0.85, edgecolor='white')
                axes[1].set_title('Estimasi Salary (LPA)')
                axes[1].set_ylabel('LPA')
                for i, v in enumerate(salaries):
                    axes[1].text(i, v+0.2, f'{v:.1f}', ha='center', fontweight='bold')

                for ax in axes:
                    ax.spines[['top','right']].set_visible(False)
                    ax.tick_params(axis='x', rotation=10)

                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                st.success("✅ Batch testing selesai!")
