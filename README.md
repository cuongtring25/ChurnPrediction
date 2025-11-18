# ChurnPrediction

## Problem:
SkilioMall is an e-commerce platform popular among Gen
Z in Southeast Asia. While the platform is fast-growing, the
team has noticed a worrying trend — a significant number
of users stop engaging after a period of time. Hence,
this repo will **build a production-lean churn prediction
model based** on the provided anonymized user data.
## Model used in this solution:
### Logistic Regression:
🎯 Logistic Regression is one of the few models that gives direct interpretability:

+ Stakeholders can understand the reasoning (e.g., “If days_since_last_order increases, churn likelihood increases.”)

+ Helps identify key churn drivers, which is essential for retention strategy.

🎯 Baseline Model — Fast & Reliable

+ Very fast to train.

+ Works well on large datasets.

+ A solid baseline to compare other models against.

🎯 Works Well on Linear Relationships

If churn is influenced by additive effects: fewer sessions, lower frequency, increasing inactivity
Logistic Regression captures these signals very well.
### XGBOOST classification:
🚀 Best-in-Class Accuracy for Tabular Data:
XGBoost is one of the strongest algorithms for structured business data (tables, user metrics, transactions).

🚀 Captures Nonlinear Patterns:

+ User behavior is rarely linear.
+ Churn often depends on interactions between features:

“High refund rate + high complaints + low CSAT”

“High GMV but very low engagement recently”

+ XGBoost learns:

    nonlinear relationships

    feature interactions

    thresholds

    diminishing returns

…which Logistic Regression cannot capture.

🚀 Handles Outliers, Missing Data, Skewed Distributions

E-commerce data is messy:

    some users make 50 orders
    
    some 0
    
    some never complain
    
    some have extremely high search activity

Tree-based models like XGBoost handle this naturally.

🚀 Feature Importance for Business Insight

You can extract:

    which variables contribute the most,
    
    which behaviors predict churn strongly,
    
    which signals matter in combination.
## How to run

First, you need to contain all the libraries in the requirement.txt:


        pip install -r requirements.txt
Second, you need a dataset, in the repo i have give you the example dataset: new_data.xlsx
Lastly, you will run the predict.py file to execute the evualation:

        python predict.py
this will create a result.xlsx file as a result.
