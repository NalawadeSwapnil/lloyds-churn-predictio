# 🏦 Customer Churn Prediction — Lloyds Banking Group

## 📋 Project Overview
A complete machine learning pipeline to predict 
customer churn for a banking dataset, built as part 
of the Lloyds Banking Group Data Science Virtual 
Experience on Forage.

## 📊 Dataset
- 1,000 banking customers across 5 sheets
- Customer_Demographics
- Transaction_History
- Customer_Service
- Online_Activity
- Churn_Status (target variable)

## 🛠️ Tools & Libraries
- Python
- pandas
- scikit-learn
- matplotlib
- imbalanced-learn (SMOTE)

## 📁 Project Structure
```
churn_project/
├── load_data.py
├── step_2_explore.py
├── step3_merge.py
├── step4_charts.py
├── step5_cleaning.py
├── t2_step1_compare.py
├── t2_step1b_improve.py
├── t2_step2_crossval.py
├── t2_step3_tuning.py
├── t2_step4_evaluation.py
├── Customer_Churn_Data_Large.xlsx
├── merged_data.csv
├── cleaned_data.csv
├── crossval_scores.csv
├── best_params.csv
├── final_metrics.csv
└── predictions.csv
```
## 🔍 Task 1 — EDA & Data Preprocessing
- Merged 5 Excel sheets into one master dataset
- Explored data with statistics and visualisations
- Handled missing values by filling with zero
- Detected and capped outliers using IQR method
- Encoded categorical variables using one-hot encoding
- Normalised numerical features using StandardScaler

## 🤖 Task 2 — Machine Learning Model
- Compared 4 algorithms:
  Logistic Regression, Decision Tree,
  Random Forest and Gradient Boosting
- Applied SMOTETomek to handle class imbalance
- Used 5-fold stratified cross validation
- Tuned hyperparameters using GridSearchCV
- Evaluated using Precision, Recall, F1 and ROC-AUC
- Applied decision threshold tuning (0.3)

## 📈 Results

### Task 1 — Dataset Summary
| Property         | Value                        |
|------------------|------------------------------|
| Total customers  | 1,000                        |
| Raw features     | 14                           |
| Final features   | 19 + 1 target                |
| Churn rate       | 20.4%                        |
| Missing values   | 0 after cleaning             |

### Task 2 — Model Performance
| Metric    | Score  |
|-----------|--------|
| Accuracy  | 66.5%  |
| Precision | 20.5%  |
| Recall    | 22.0%  |
| F1 Score  | 21.2%  |
| ROC-AUC   | 52.7%  |

## 💡 Key Findings
- Churn rate is 20.4% — 1 in 5 customers left
- LoginFrequency is the strongest churn predictor
- Low income customers churn at a slightly higher rate
- No single feature strongly predicts churn alone
- Richer behavioural data would improve model performance

## 📚 What I Learned
- Loading and merging real Excel datasets in Python
- Exploratory data analysis and visualisation
- Data cleaning and preprocessing techniques
- Comparing and evaluating machine learning models
- Handling imbalanced datasets using SMOTE
- Cross validation and hyperparameter tuning
