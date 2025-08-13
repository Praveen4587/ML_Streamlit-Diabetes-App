# 🩺 Diabetes Prediction App (Pima Indians Dataset)

**Index:** ITBIN-2211-0280  
streamlit live demo: https://mlapp-diabetes-app-ak6jpru9c5bpetcv39frrn.streamlit.app/

A Machine Learning model deployment project using **Streamlit** to predict diabetes based on medical attributes from the **Pima Indians Diabetes Dataset**.  
This project covers the complete ML workflow — from data preprocessing & model training in Jupyter Notebook to deployment as an interactive web app.

---

## 📂 Project Structure

your-project/
├── app.py # Streamlit app for prediction & visualization
├── requirements.txt # Python dependencies
├── model.pkl # Saved trained model pipeline & metadata
├── data/
│ └── dataset.csv # Diabetes dataset (Pima Indians)
├── notebooks/
│ └── model_training.ipynb # EDA, preprocessing, training
└── README.md # Project documentation


---

## 📊 Dataset Information

- **Source:** [Kaggle - Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Records:** 768 patients
- **Features:**
  - Pregnancies
  - Glucose
  - BloodPressure
  - SkinThickness
  - Insulin
  - BMI
  - DiabetesPedigreeFunction
  - Age
- **Target:** `Outcome` (0 = non-diabetic, 1 = diabetic)

---

## ⚙️ How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
