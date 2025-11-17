import joblib
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
"""
   supposing      
        
"""
df = pd.read_excel("new_data.xlsx")

#this will handle all preprocessing step, return X_test and y_
def DatePreprocessing(dataframe,scaling=False):
    #hashing encoding columns city
    he = ce.HashingEncoder(cols='city')
    df_hash = he.fit_transform(df)
    #encoding other category feature using dummies
    df_hash_dummies = pd.get_dummies(df_hash,columns=['country','marketing_source','app_version_major'],dtype=int).drop(columns=['user_id'])
    #dropping high_correlation_cols
    high_correlation_cols = ['sessions_90d', 'orders_2024', 'gmv_2024', 'refund_rate_2024', 'emails_click_rate_90d', 'rfm_recency', 'rfm_frequency', 'rfm_monetary']
    df_cleaned = df_hash_dummies.drop(columns=high_correlation_cols)

    X = df_cleaned.drop(columns=['churn_label'],axis=1)
    y = df_cleaned['churn_label']
    _, X_test = np.split(X, [int(0.7 * len(df))])
    _, Y_test = np.split(y, [int(0.7 * len(df))])
    if scaling:
        # listing the columns scaled
        columns_scaled = ['age', 'reg_days', 'sessions_30d', 'avg_session_duration_90d',
                          'median_pages_viewed_30d', 'search_queries_30d',
                          'orders_30d', 'orders_90d', 'aov_2024',
                          'category_diversity_2024', 'days_since_last_order',
                          'refunds_count_2024', 'support_tickets_2024', 'avg_csat_2024',
                          'review_count_2024', 'avg_review_stars_2024']

        scaler = RobustScaler()
        X_test_scaled = X_test.copy()
        X_test_scaled[columns_scaled] = scaler.fit_transform(X_test[columns_scaled])
        return X_test_scaled,Y_test
    return X_test,Y_test

X,y = DatePreprocessing(df,scaling=True)


#XGBOOST
# model = XGBClassifier()
# model.load_model('src/xgb_model.json')
#logistic regression

model = joblib.load('src/model_log.pkl')
y_pred = model.predict(X)

print(classification_report(y,y_pred))