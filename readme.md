# 💳 Credit Card Fraud Detection System using Machine Learning

This project demonstrates a **Machine Learning–based Credit Card Fraud Detection System** built with **Python**. It applies data preprocessing, feature scaling, and model training techniques to accurately classify fraudulent transactions in a highly imbalanced dataset.  

---

## 🚀 Features

* Data preprocessing and cleaning for high-dimensional transaction data  
* Handling of imbalanced data using **Random Under/Over Sampling** or **SMOTE**  
* Exploratory Data Analysis (EDA) with visual insights  
* Model training using multiple algorithms (e.g., Logistic Regression, Random Forest, Decision Tree)  
* Evaluation with metrics such as **Accuracy**, **Precision**, **Recall**, **F1-score**, and **ROC-AUC Curve**  
* Real-time prediction capability for new transactions  
* Well-documented code in Jupyter Notebook for reproducibility  

---

## 🛠️ Technologies Used

* Python 🐍  
* NumPy  
* Pandas  
* Scikit-learn  
* Matplotlib & Seaborn  
* Jupyter Notebook  

---

## 📂 Project Structure

```plaintext
Credit-Card-Fraud-Detection/
│
├── main.ipynb                     # Main notebook with full implementation
├── requirements.txt               # Dependencies (if added)
├── .gitignore                     # To ignore unnecessary files (like dataset)
├── README.md                      # Project overview and details
└── creditcard.csv                 # Dataset (ignored in repo for size/privacy)
```

---

## ⚙️ Workflow Overview

1. **Data Loading:**  
   Load the transaction dataset (`creditcard.csv`) containing features and labels.

2. **Exploratory Data Analysis (EDA):**  
   Understand distribution, correlation, and detect imbalance between classes.

3. **Data Preprocessing:**  
   * Scale numerical features  
   * Handle missing values (if any)  
   * Apply class balancing techniques (e.g., SMOTE)

4. **Model Training:**  
   Train ML models such as:
   * Logistic Regression  
   * Decision Tree Classifier  
   * Random Forest Classifier  

5. **Evaluation:**  
   Use metrics like confusion matrix, classification report, and ROC curve to evaluate model performance.

6. **Prediction:**  
   Test model on unseen data and evaluate its fraud detection efficiency.

---

## 📊 Example Results

| Metric              | Logistic Regression | Random Forest | Decision Tree |
|----------------------|--------------------|----------------|----------------|
| Accuracy (%)         | 99.2               | 99.8           | 98.7           |
| Precision (%)        | 94.5               | 98.9           | 92.1           |
| Recall (%)           | 90.2               | 96.7           | 89.4           |
| F1-score (%)         | 92.3               | 97.7           | 90.7           |

*(Values are approximate and depend on the dataset and train-test split.)*

---

## ▶️ Getting Started

### Prerequisites

Ensure you have Python 3.x installed.  
Install the required libraries:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### Run the Project

Open the Jupyter Notebook and run all cells:

```bash
jupyter notebook main.ipynb
```

---

## 🧠 Applications

* Banking and financial fraud detection  
* Payment gateway transaction monitoring  
* Risk assessment and anomaly detection systems  
* Cybersecurity and fintech analytics  

---

## 📌 Notes

* The dataset is **highly imbalanced**, meaning fraudulent transactions are very rare compared to legitimate ones.  
* Techniques like **SMOTE**, **Random Oversampling**, or **Under-sampling** can improve model balance.  
* Avoid overfitting by cross-validation and proper parameter tuning.  

---

## 💡 Future Enhancements

* Integration with a real-time API or dashboard for live fraud alerts  
* Implementation of deep learning models (e.g., LSTM, Autoencoders)  
* Deployment using Flask or FastAPI for web-based prediction  
* Feature importance visualization and interpretability (e.g., SHAP, LIME)  

---

👤 **Author**

Made with ❤️ by **Vanshaj P Mohan**, a Data Science Enthusiast.  
Exploring how machine learning can make digital transactions safer and smarter.
