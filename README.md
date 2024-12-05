# 📂 Predictive Maintenance 🚀  

Welcome to the **Predictive Maintenance** project! This repository demonstrates a machine learning-powered solution to predict equipment failures before they happen. By analyzing historical data and leveraging advanced models, we aim to minimize downtime, reduce costs, and optimize operational efficiency.  

---

## 🛠️ **Tech Stack**  

- **Programming Language**: Python 🐍  
- **Tools & Frameworks**:  
  - **Visual Studio Code** 🖥️  
  - **Scikit-learn (sklearn)** 🤖  
  - **FastAPI** ⚡  

---

## 📜 **Table of Contents**  

1. [Project Overview](#project-overview)  
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)  
3. [Model Selection & Training](#model-selection--training)  
4. [API Deployment](#api-deployment)  
5. [How to Run](#how-to-run)  
6. [Future Enhancements](#future-enhancements)  

---

## 🔍 **Project Overview**  

Predictive maintenance is the practice of using machine learning models to analyze equipment sensor data and predict failures. This project includes:  

1. **Data Preprocessing**: Cleaning, normalizing, and preparing sensor data.  
2. **EDA**: Understanding the data distribution, trends, and feature relationships.  
3. **Model Training**: Experimenting with machine learning algorithms for predictive performance.  
4. **API Integration**: Building a FastAPI-based interface for real-time predictions.  

---

## 📊 **Exploratory Data Analysis (EDA)**  

### Key Steps:  

- **Data Cleaning** 🧹:  
  Removing null values, duplicates, and outliers.  

- **Feature Engineering** 🛠️:  
  - Creating new features like moving averages and sensor interaction terms.  
  - Encoding categorical variables.  

- **Visualization** 📈:  
  - Heatmaps for correlation analysis.  
  - Line plots to observe sensor trends over time.  
  - Distribution plots for feature analysis.  

### Key Insights:  
- Found strong correlations between specific sensor metrics and failure events.  
- Certain time intervals showed higher failure rates.  

---

## 🧠 **Model Selection & Training**  

### Algorithms Tested:  

1. **Logistic Regression** 📐  
2. **Random Forest** 🌲  
3. **Gradient Boosting (XGBoost)** 🚀  

### Process:  

- Split the data into **training** (80%) and **testing** (20%) datasets.  
- Applied **cross-validation** to ensure generalization.  
- Tuned hyperparameters using **GridSearchCV** for the best performance.  

### Metrics:  
- **Accuracy**: Overall prediction correctness.  
- **Precision**: How well failures are detected.  
- **Recall**: How many true failures are captured.  
- **F1-Score**: Balancing precision and recall.  

### Best Model:  
- **Random Forest** achieved the highest accuracy and interpretability for this dataset.  

---

## 🌐 **API Deployment**  

### FastAPI Integration  

We’ve integrated the trained model into a **FastAPI** application to provide real-time predictions.  

**Endpoints**:  
1. **`/predict`**: Accepts sensor data and returns a failure prediction (0 or 1).  
2. **`/health`**: API health check.  

### Example Request:  

```json
POST /predict
{
  "OperatingHours": 0,
  "Temperature": 0,
  "VibrationLevel": 0,
  "LoadPercentage": 0,
  "LastMaintanenceDate": "2024-12-05"
}
```

### Example Response:  

```json
{
  "Next Maintenance Date": "2025/06/28"
}
```

---

## 🛠️ **How to Run**  

### Prerequisites:  

- Python 3.10+  
- FastAPI & Uvicorn  

### Installation:  

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/predictive-maintenance.git
   cd predictive-maintenance
   ```  

2. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```  

3. Run the API:  
   ```bash
   uvicorn main:app --reload
   ```  

4. Open your browser at **`http://127.0.0.1:8000/docs`** to access the interactive API documentation.  

---

## 🚀 **Future Enhancements**  

1. Integrate deep learning models for more complex data patterns.  
2. Add support for streaming real-time sensor data.  
3. Visualize failure predictions on a dashboard.  

---

## ❤️ **Contributing**  

We welcome contributions! Feel free to:  

- Fork this repo.  
- Submit pull requests with your improvements.  
- Report bugs or suggest features in the Issues section.  

---

## 📄 **License**  

This project is licensed under the MIT License.  


