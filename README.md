# W26-AIGC5005-Final
### Diabetes 30-Day Hospital Readmission Risk Predictor

**Course:** AIGC 5005 — AI Capstone Project Preparation  
**Institution:** Humber Polytechnic  
**Team:** Fadi Kash Kannaiti · Ogbeide Iria · Oguzhan Tekin · Sara Yenigun  
**Instructor:** Hossein Pourmodheji  

## Project Overview

Hospital readmissions within 30 days are a major challenge in healthcare, especially for diabetic patients. Traditional scoring systems often rely on a limited number of variables and may miss complex patterns hidden in EHR data.  

In this project, we developed and compared multiple predictive models to identify patients at higher risk of readmission after discharge. The goal was not only to build accurate models, but also to make the solution more interpretable and practically usable through deployment in a **Streamlit web app**. :contentReference[oaicite:3]{index=3}

---

## Objective

The main objectives of this project were to:

- Build a **Logistic Regression baseline**
- Train and compare advanced models including:
  - Random Forest
  - XGBoost
  - PyTorch MLP
  - TensorFlow/Keras MLP
- Handle severe class imbalance using **SMOTE** and class-weighting strategies
- Evaluate models using metrics suited to imbalanced healthcare datasets
- Perform **SHAP explainability analysis**
- Analyze **threshold sensitivity**
- Deploy the best-performing model as a public application 

---

## Dataset

- **Source:** UCI Machine Learning Repository / Kaggle
- **Dataset:** Diabetes 130-US Hospitals for Years 1999–2008
- **Size:** 101,766 patient encounters
- **Hospitals:** 130 US hospitals
- **Target:** Readmitted within 30 days (`<30`) = 1, otherwise = 0 :contentReference[oaicite:5]{index=5}

### Features included
The dataset contains variables related to:

- Demographics
- Admission and discharge details
- Diagnoses
- Lab procedures
- Medication changes
- Prior hospital utilization history :contentReference[oaicite:6]{index=6}

---

## Preprocessing

The preprocessing pipeline included:

- Dropping highly missing columns such as:
  - `weight`
  - `payer_code`
  - `medical_specialty`
- Removing identifiers and zero-variance columns:
  - `encounter_id`
  - `patient_nbr`
  - `examide`
  - `citoglipton`
- Encoding categorical variables
- Mapping ICD-9 diagnosis codes into broader disease categories
- Stratified train/test split (80/20)
- Applying **SMOTE only on the training set**
- Applying **StandardScaler** only for neural network models :contentReference[oaicite:7]{index=7}

---

## Models Used

This project compared five models:

1. **Logistic Regression** – baseline model  
2. **Random Forest (Tuned)**  
3. **XGBoost**  
4. **PyTorch Multi-Layer Perceptron (MLP)**  
5. **TensorFlow/Keras Multi-Layer Perceptron (MLP)** :contentReference[oaicite:8]{index=8}

---

## Why These Metrics?

Because the dataset is highly imbalanced, **accuracy alone is misleading**. A model could predict mostly non-readmissions and still get a high accuracy score.

So this project focused on:

- **F1-score**
- **Recall**
- **Precision**
- **AUC-ROC**
- **AUC-PR** 

---

## Results

### Full Model Comparison

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | AUC-PR |
|------|---------:|----------:|-------:|---:|--------:|-------:|
| Logistic Regression | 0.8680 | 0.2312 | 0.0756 | 0.1140 | 0.5578 | 0.1440 |
| Random Forest (Tuned) | 0.8759 | 0.2094 | 0.0380 | 0.0644 | 0.5974 | 0.1560 |
| XGBoost | 0.6852 | 0.1826 | 0.5192 | 0.2702 | 0.6603 | 0.2141 |
| PyTorch MLP | 0.8851 | 0.2797 | 0.0148 | 0.0281 | 0.6186 | 0.1690 |
| TensorFlow/Keras MLP | 0.8757 | 0.2183 | 0.0416 | 0.0699 | 0.6170 | 0.1623 | :contentReference[oaicite:10]{index=10}

### Best Model
**XGBoost** delivered the best overall balance of performance:

- **F1-score:** 0.2702
- **AUC-ROC:** 0.6603
- **Recall:** 0.5192 :contentReference[oaicite:11]{index=11}

### Deep Learning Insight
Although the default threshold made the neural networks appear weak on F1-score, threshold calibration showed that the **PyTorch MLP** could achieve:

- **F1-score:** 0.2406
- **Recall:** 0.5577  
at a threshold of **0.15**, which was the **highest recall among all models**. :contentReference[oaicite:12]{index=12}

---

## Threshold Sensitivity Analysis

For imbalanced healthcare prediction, the default threshold of `0.50` is often not ideal.

For the **PyTorch MLP**:

| Threshold | Precision | Recall | F1 |
|----------|----------:|-------:|---:|
| 0.50 | 0.2797 | 0.0148 | 0.0281 |
| 0.25 | 0.2001 | 0.2010 | 0.2005 |
| 0.15 | 0.1534 | 0.5577 | 0.2406 | :contentReference[oaicite:13]{index=13}

This showed that lowering the threshold significantly improved the model’s usefulness for **screening high-risk patients**.

---

## Explainability

To improve transparency, the project included **SHAP explainability analysis** and feature importance evaluation.

### Top predictors identified
- `number_inpatient`
- `num_medications`
- `time_in_hospital`
- `number_diagnoses`
- `number_emergency` :contentReference[oaicite:14]{index=14}

These features are clinically meaningful because they reflect prior utilization, disease complexity, and instability.

---

## Robustness Testing

To test model stability, Gaussian noise was injected into numeric features at varying levels. The results showed that:

- **Random Forest** was the most stable under moderate noise
- All models degraded at very high noise levels, as expected :contentReference[oaicite:15]{index=15}

---

## Deployment

The best-performing model was deployed as a **public Streamlit app**.

### App features
- Input form for patient discharge details
- Predicted readmission risk
- Risk category: **Low / Moderate / High**
- Clinical action recommendation
- Probability visualization
- Risk factor chart 

---

## Tech Stack

- **Python**
- **Pandas / NumPy**
- **Scikit-learn**
- **XGBoost**
- **PyTorch**
- **TensorFlow / Keras**
- **SHAP**
- **Matplotlib / Seaborn**
- **Streamlit** :contentReference[oaicite:17]{index=17}

---

## Project Structure

```bash
├── data/
├── notebooks/
├── models/
├── app/
│   └── streamlit_app.py
├── images/
├── requirements.txt
└── README.md

## How to Run the Notebooks (Google Colab)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `diabetic_data.csv` to `/content/` using the file panel
   - Download from: https://www.kaggle.com/datasets/brandao/diabetes
3. Open each notebook in order and click **Runtime → Run all**

**Run order:** `04_advanced_models.ipynb` is self-contained — it runs the full
preprocessing pipeline automatically, so you do not need to run Notebooks 01–03
first. However, running all four in order gives you the complete set of output
charts for the report.

---

## How to Run the App Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/W26-AIGC5005-Final.git
cd W26-AIGC5005-Final

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate the model files by running Notebook 04 in Colab first,
#    then download models/best_model.pkl and models/feature_names.json
#    and place them in the models/ folder

# 4. Run the app
streamlit run app.py
```

The app will open at `http://localhost:8501` in your browser.

---

## How to Deploy on Streamlit Community Cloud (Free)

1. Push this repository to GitHub (must be **public**)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Sign in with your GitHub account
4. Click **New app**
5. Select repository: `W26-AIGC5005-Final`
6. Set main file path: `app.py`
7. Click **Deploy**

Your app will be live at a public URL within 2–3 minutes.
Share this URL in your report and video presentation.

> **Important:** The `models/` folder containing `best_model.pkl` must be
> committed to GitHub for Streamlit Cloud to find it. The model file may be
> large — if over 100 MB, use [Git LFS](https://git-lfs.com/) or host the
> model on Google Drive and load it programmatically in `app.py`.

---

## How to Access on Mobile

Once deployed on Streamlit Community Cloud, simply open the public URL on
any smartphone or tablet browser. No app installation is required.
The app is fully responsive and works on iOS and Android.

---


## References

- Strack, B., et al. (2014). Impact of HbA1c measurement on hospital readmission rates. *BioMed Research International.*
- Emi-Johnson, O. G., & Nkrumah, K. J. (2025). Predicting 30-day hospital readmission in patients with diabetes. *Cureus, 17*(4).
- Gandra, A. (2024). Predicting hospital readmissions in diabetes patients. *International Journal of Health Sciences, 8*(3).
- Chawla, N. V., et al. (2002). SMOTE: Synthetic minority over-sampling technique. *JAIR, 16*, 321–357.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS.*
