import joblib
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
"""
   This prediction is tested on the new_data ( which just a dataset containing last 12000 users
   in Skilio) 
   
   The new_data still has churn_label columns because i want to evaluate how accurate the model is
   but in real case, there might won't include that columns
"""
df = pd.read_excel("new_data.xlsx")

#this will handle all preprocessing step, returning the dataset cleaned
def DataPreprocessing(dataframe,scaling=False):
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
    """
        in real dataset, this y representing churn_label might not appear
    """

    if scaling:
        # listing the columns scaled
        columns_scaled = ['age', 'reg_days', 'sessions_30d', 'avg_session_duration_90d',
                          'median_pages_viewed_30d', 'search_queries_30d',
                          'orders_30d', 'orders_90d', 'aov_2024',
                          'category_diversity_2024', 'days_since_last_order',
                          'refunds_count_2024', 'support_tickets_2024', 'avg_csat_2024',
                          'review_count_2024', 'avg_review_stars_2024']

        scaler = RobustScaler()
        X_scaled = X.copy()
        X_scaled[columns_scaled] = scaler.fit_transform(X[columns_scaled])
        return X_scaled,y
    return X,y

X,y = DataPreprocessing(df,scaling=True)


#XGBOOST
# model = XGBClassifier()
# model.load_model('src/xgb_model.json')
#logistic regression

model = joblib.load('src/model_log.pkl')
y_pred = model.predict(X)

"""saving the result"""
results_df = pd.DataFrame({'user_id':df['user_id'],
                           'churn_label': y_pred})
results_df.to_csv('result.csv',index=False)