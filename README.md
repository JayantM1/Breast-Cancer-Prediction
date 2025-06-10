# Breast-Cancer-Prediction
# Breast Cancer Prediction 🧬🎯

This project demonstrates a machine learning approach to predict whether a tumor is malignant or benign based on features extracted from breast cancer biopsies. The model uses the famous Breast Cancer Wisconsin dataset available in Scikit-learn.

## 📁 Project Structure

- `breast_cancer_prediction.ipynb` — Jupyter notebook containing the full data preprocessing, model training, and evaluation pipeline.

## 🧰 Technologies Used

- Python 🐍
- Jupyter Notebook / Google Colab
- Scikit-learn
- Pandas
- Matplotlib
- Seaborn (if used)

## 📊 Dataset

- **Source:** Scikit-learn's `load_breast_cancer()` utility.
- **Features:** 30 numerical features computed from digitized images of breast mass.
- **Target:** Diagnosis (0 = malignant, 1 = benign)

## 🔍 Workflow Overview

1. **Data Loading & Exploration**  
   Understand the structure and distribution of the dataset.

2. **Data Preprocessing**  
   Handle missing values, scale features, and encode labels if necessary.

3. **Model Training**  
   Train classification models like:
   - Logistic Regression
   - Random Forest
   - Support Vector Machine
   - K-Nearest Neighbors

4. **Evaluation**  
   Evaluate performance using metrics such as accuracy, confusion matrix, and classification report.

5. **Visualization**  
   Use plots to interpret model performance and feature importance.

## 📈 Model Performance

The notebook includes comparison across multiple models. Final accuracy and evaluation metrics are displayed for each.

## 🚀 How to Run

1. Clone the repo or download the `.ipynb` file.
2. Open in Jupyter Notebook or upload to [Google Colab](https://colab.research.google.com/).
3. Run all cells in sequence.

## ✅ Future Improvements

- Hyperparameter tuning
- Use of cross-validation
- Deployment using Streamlit or Flask (optional)
- Model explainability using SHAP or LIME

## 📃 License

This project is open-source and available under the [MIT License](LICENSE).

---

