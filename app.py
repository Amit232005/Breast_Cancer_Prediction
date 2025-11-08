# app.py — AI Breast Cancer Diagnostic Dashboard (Pro Edition by Amit Barik)
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from streamlit_lottie import st_lottie
from fpdf import FPDF
import io
import base64
import time

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="🎗️ AI Breast Cancer Diagnostic Dashboard",
    layout="wide",
    page_icon="🎀"
)

# ---------------------------------------------------------
# DATA & MODEL
# ---------------------------------------------------------
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = joblib.load("breast_cancer_model.joblib")
scaler = joblib.load("scaler.joblib")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().round(2)

# ---------------------------------------------------------
# LOTTIE LOADER
# ---------------------------------------------------------
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

lottie_awareness = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_hl6wlhqz.json")
lottie_doctor = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_tutvdkg0.json")
lottie_success = load_lottie_url("https://assets2.lottiefiles.com/packages/lf20_jbrw3hcz.json")
lottie_ribbons = load_lottie_url("https://lottie.host/85e69c38-f1ed-4f14-bd4e-3a0022b48ed1/nKLEFz0NEu.json")

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
with st.sidebar:
    st.image("breast_cancer_banner.png", use_container_width=True)
    st.markdown("### 🌗 Theme Settings")
    dark_mode = st.toggle("Enable Dark Mode")

    if lottie_awareness:
        st_lottie(lottie_awareness, height=150)

    st.markdown("### 📊 Dataset Summary")
    st.info(f"**Samples:** {X.shape[0]}\n**Features:** {X.shape[1]}")
    st.info(f"**Benign (1):** {int(sum(y==1))} | **Malignant (0):** {int(sum(y==0))}")

    st.markdown("### 🧠 Tech Stack")
    st.markdown("• Python 🐍\n• scikit-learn 🤖\n• Streamlit 🌐\n• Matplotlib & Seaborn 📊\n• FPDF 📄\n• Lottie 🎞️")

    st.caption("💖 Early Detection Saves Lives")
    st.caption("Made with ❤️ by Amit Barik")

# ---------------------------------------------------------
# THEME STYLES
# ---------------------------------------------------------
if dark_mode:
    bg = "#0f172a"
    text = "#f9a8d4"
    box = "#1e293b"
else:
    bg = "linear-gradient(120deg, #fff0f4, #ffe4ec, #fff5f9)"
    text = "#db2777"
    box = "linear-gradient(90deg,#fdf2f8,#fce7f3,#fbcfe8,#f9a8d4)"

st.markdown(f"""
<style>
body {{
    background: {bg};
    font-family: 'Poppins', sans-serif;
}}
.header {{
    border-radius: 20px;
    background: {box};
    padding: 40px;
    text-align: center;
    color: {text};
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
}}
footer {{
    text-align: center;
    font-size: 14px;
    margin-top: 60px;
    color: {'#e2e8f0' if dark_mode else '#444'};
}}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# HEADER SECTION
# ---------------------------------------------------------
if lottie_ribbons:
    st_lottie(lottie_ribbons, height=220)

st.markdown(f"""
<div class="header">
    <h1 style='font-size:60px;'>🎗️ AI Breast Cancer Predictor</h1>
    <h4>Empowering Diagnosis • Awareness • AI-Driven Insight</h4>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# USER INPUTS
# ---------------------------------------------------------
st.markdown("---")
st.subheader("🧬 Enter Tumor Feature Values:")

ui_features = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness']
inputs = {}
cols = st.columns(2)
for i, feat in enumerate(ui_features):
    with cols[i % 2]:
        col = X[feat]
        inputs[feat] = st.slider(f"{feat.title()}", float(col.min()), float(col.max()), float(col.mean()))

# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------
if st.button("🔍 Predict Result"):
    with st.spinner("Analyzing features..."):
        full_input = X.mean().to_frame().T.iloc[0].copy()
        for k, v in inputs.items():
            full_input[k] = v
        scaled = scaler.transform([full_input.values])
        pred = model.predict(scaled)[0]
        prob = model.predict_proba(scaled)[0]
        time.sleep(1.5)

    st.markdown("---")
    if pred == 1:
        st.success("🎉 Prediction: **Benign (Non-Cancerous)** ✅")
        st.info(f"🧪 Probability → Benign: `{prob[1]:.3f}` | Malignant: `{prob[0]:.3f}`")
        if lottie_success:
            st_lottie(lottie_success, height=180)
    else:
        st.error("⚠️ Prediction: **Malignant (Cancerous)** ❗")
        st.info(f"🧪 Probability → Malignant: `{prob[0]:.3f}` | Benign: `{prob[1]:.3f}`")
        if lottie_doctor:
            st_lottie(lottie_doctor, height=180)

# ---------------------------------------------------------
# PERFORMANCE METRICS
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📈 Model Performance Overview")

col1, col2 = st.columns([1, 2])
with col1:
    st.metric("Model Accuracy", f"{acc * 100:.2f}%")

with col2:
    st.markdown("#### Confusion Matrix")
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap="RdPu", xticklabels=data.target_names, yticklabels=data.target_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

st.markdown("#### Classification Report")
st.dataframe(report_df)

# ---------------------------------------------------------
# AUTO SUMMARY INSIGHT
# ---------------------------------------------------------
st.markdown("#### 🧠 AI Insight Summary")
if acc > 0.9:
    st.success("✅ The model shows excellent reliability with over 90% accuracy — ideal for assisting medical screening systems.")
else:
    st.warning("⚠️ Model accuracy below 90%. Consider retraining with more balanced data or additional features.")

if report_df.loc["0", "recall"] > 0.9:
    st.info("💡 The model is highly sensitive to malignant cases — effective for early detection.")
else:
    st.info("📊 The model has moderate recall for malignant detection. Consider hyperparameter tuning.")

# ---------------------------------------------------------
# DOWNLOAD REPORTS
# ---------------------------------------------------------
csv = report_df.to_csv().encode("utf-8")
st.download_button("📥 Download Report (CSV)", csv, "breast_cancer_report.csv", "text/csv")

# PDF EXPORT WITH CHART
buf = io.BytesIO()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap="RdPu", xticklabels=data.target_names, yticklabels=data.target_names, ax=ax)
plt.title("Confusion Matrix")
plt.tight_layout()
fig.savefig(buf, format="png")
buf.seek(0)
img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

pdf = FPDF()
pdf.add_page()
pdf.set_fill_color(255, 240, 245)
pdf.rect(5, 5, 200, 287, 'D')
pdf.set_font("Arial", "B", 18)
pdf.set_text_color(219, 39, 119)
pdf.cell(0, 12, "AI Breast Cancer Diagnostic Report", ln=True, align="C")
pdf.set_font("Arial", "", 12)
pdf.multi_cell(0, 10, f"Developer: Amit Barik\nAccuracy: {acc * 100:.2f}%\n")
# Save confusion matrix image temporarily
temp_img_path = "confusion_matrix_temp.png"
fig.savefig(temp_img_path)

# Add image to PDF from file
pdf.image(temp_img_path, x=30, y=60, w=140)

pdf.set_y(130)
pdf.set_font("Arial", "B", 13)
pdf.cell(0, 10, "Classification Summary:", ln=True)
pdf.set_font("Arial", "", 11)
for i, row in report_df.iterrows():
    pdf.cell(0, 8, f"{i} -> Precision: {row['precision']} | Recall: {row['recall']} | F1: {row['f1-score']}", ln=True)
pdf.ln(10)
pdf.set_font("Arial", "I", 10)
pdf.set_text_color(120, 120, 120)
pdf.cell(0, 8, "Generated by AI Breast Cancer Predictor | © Amit Barik", ln=True, align="C")
# For fpdf2, output(dest="S") already returns bytes
pdf_bytes = pdf.output(dest="S")
pdf_output = io.BytesIO(pdf_bytes)


st.download_button(
    label="📄 Download PDF Report",
    data=pdf_output,
    file_name="AI_Breast_Cancer_Report.pdf",
    mime="application/pdf"
)

# ---------------------------------------------------------
# ABOUT FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.subheader("💖 About This Project")
st.markdown("""
### 🎯 Objective:
This dashboard leverages **AI & Machine Learning** to classify breast tumors as **malignant** or **benign**,
empowering early detection and awareness.

### 🧠 Built With:
- Python, scikit-learn, Streamlit, Seaborn, FPDF, Lottie  
- Real-world Breast Cancer Dataset (UCI)

### 👩‍💻 Developer:
**Amit Barik** — BCA Student, Netaji Subhas University  
Dedicated to building impactful AI tools in healthcare 💡
""")

st.markdown("""
<footer>
Made with ❤️ and AI for Breast Cancer Awareness Month 🎗️ <br>
<small>© 2025 - Project by <b>Amit Barik</b></small>
</footer>
""", unsafe_allow_html=True)
