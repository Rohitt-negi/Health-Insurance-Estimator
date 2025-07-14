 # 🏥 Health Insurance Cost Estimator

A comprehensive **machine learning pipeline** to predict a person’s annual health insurance premium based on demographic, lifestyle, and medical history data. Covers end-to-end workflow—from data analysis and model building in Jupyter to saving a production-ready XGBoost pipeline and providing a prediction demo function.

---

## ✨ Features

- **Exploratory Data Analysis (EDA)** with insightful visualizations  
- **Feature engineering**, including medical condition count and income numerical mapping  
- **Robust preprocessing**: scaling numerical features and one-hot encoding categoricals  
- **Model comparison** across:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - Linear & Ridge Regression
  - SVR  
- **Hyperparameter tuning** using `GridSearchCV` on XGBoost  
- **Performance metrics**: RMSE & R² for model evaluation  
- **Feature importance analysis** for interpretability  
- **Pipeline serialization** with `joblib`  
- **Pre-built prediction function** to generate premium estimates from user inputs

---

## 📋 Project Structure

```
health-insurance-predictor/
├── data/
│   ├── notebook/
│   ├── exploration.ipynb
│   ├── model_comparison.ipynb
│   ├── xgboost_tuning.ipynb
│
├── src/               
│   ├── estimator.py 
│   ├── predictor.py                   
│
├── health_insurance_cost_estimator.pkl                   # Serialized XGBoost pipeline
├── prediction_helper.py      # Model/scaler selection and prediction logic
├── requirements.txt          # Project dependencies
├── README.md                 # Project overview
└── .gitignore                # # Specifies files/folders to ignore in yaml

```

---

## 📊 Dataset

⚠️ The dataset `insurance_data_6000.xlsx` (Sheet1)  contains **~6,000** records with:

#### Age 
#### Gender  
#### Region  
#### Marital_status  
#### Number_Of_Dependents  
#### BMI_Category
#### Smoking_Status,  
#### Employment_Status  
#### Income_Level  
#### Income_Lakhs
#### Medical_History   
#### Insurance_Plan  
#### Annual_Premium_Amount
---

## 📓 Model Development

Key steps performed in notebook scripts (`notebooks/`):

1. **EDA**: Data exploration & visualization  
2. **Feature Engineering**: 
   - `Num_Medical_Conditions`
   - Income-level to numeric mapping  
3. **Preprocessing pipeline**: `StandardScaler` + `OneHotEncoder`  
4. **Model training** & evaluation across 6 algorithms  
5. **Hyperparameter tuning** for XGBoost using `GridSearchCV`  
6. **Feature importance** visualization (top 20)  
7. **Pipeline export** using `joblib`

---

## 🖥️ Usage & Demo

### Local setup


1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/health-insurance-predictor.git
   cd health-insurance-predictor
   ```

2. **Create a virtual environment and activate it**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train & evaluate models:**
   ```bash
   python src/estimator.py
   ```
5. **Test premium prediction:**
```python
from src.predictor import predict_insurance_cost
est = predict_insurance_cost(
   age=45, gender='Male', region='Northeast',
   marital_status='Married', num_dependants=2,
   bmi_category='Overweight', smoking_status='No Smoking',
   employment_status='Salaried', income_level='25L-40L',
   medical_history='High blood pressure', insurance_plan='Gold'
)
print(f"Estimated premium: ₹{est:.2f}")
```


# 📈 Results Snapshot

<img width="1710" height="1112" alt="Screenshot 2025-07-14 at 2 27 34 PM" src="https://github.com/user-attachments/assets/ef09ed5e-cfca-416a-a10c-cccdbdd4e5bd" />
<img width="1710" height="1112" alt="Screenshot 2025-07-14 at 2 27 53 PM" src="https://github.com/user-attachments/assets/87a4b0f8-be00-497e-af87-fd29d4ddda77" />
<img width="1710" height="1112" alt="Screenshot 2025-07-14 at 2 27 59 PM" src="https://github.com/user-attachments/assets/a87b41bf-71fc-45c2-af3a-ea854208269a" />
<img width="1710" height="1112" alt="Screenshot 2025-07-14 at 2 28 10 PM" src="https://github.com/user-attachments/assets/63e317ca-1b76-4016-91a0-f5497661fd95" /><br>



  
🧠 Best Practices & References
This README follows industry guidelines like including project purpose, setup, usage, and reproducibility. For more detail on documentation best practices, see:

Tips for clear READMEs and reproducibility 
deepdatascience.wordpress.com
medium.datadriveninvestor.com
thejunkland.com
arxiv.org
reddit.com
github.com

Emphasizing structure, clarity, and examples

🙋‍♂️ Author  

Rohit Singh Negi  

Github--> https://github.com/Rohitt-negi  

LinkedIn--> https://www.linkedin.com/in/rohit-negi-688425299/


🚀 Contributions
Contributions are welcome! Feel free to:
Add support for additional regression models


