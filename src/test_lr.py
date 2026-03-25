from data_loader import get_dataloaders, merge_and_create_features
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler

import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

matchups, feature_cols = merge_and_create_features()
train_df = matchups[matchups['YEAR'] <= 2024]
test_df = matchups[matchups['YEAR'] == 2025]

X_train = train_df[feature_cols].values
y_train = train_df['TeamAwins'].values
X_test = test_df[feature_cols].values
y_test = test_df['TeamAwins'].values

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(C=0.1)
lr.fit(X_train, y_train)
p_test = lr.predict_proba(X_test)
print("LR LOG LOSS:", log_loss(y_test, p_test))
print("LR ACCURACY:", lr.score(X_test, y_test))
