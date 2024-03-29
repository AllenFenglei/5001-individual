#!/usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import math

# read data and translate date formate
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
train_data = train_data.fillna(value=-1)
test_data = test_data.fillna(value=-1)
# infer the right date format
train_data['purchase_date'] = pd.to_datetime(train_data['purchase_date'], infer_datetime_format=True)
test_data['purchase_date'] = pd.to_datetime(test_data['purchase_date'], infer_datetime_format=True)
train_data['release_date'] = pd.to_datetime(train_data['release_date'], infer_datetime_format=True)
test_data['release_date'] = pd.to_datetime(test_data['release_date'], infer_datetime_format=True)
train_data["purchase_date"] = pd.to_datetime(train_data["purchase_date"], '%Y-%m-%d').dt.strftime('%Y%m').astype(float)
test_data["purchase_date"] = pd.to_datetime(test_data["purchase_date"], '%Y-%m-%d').dt.strftime('%Y%m').astype(float)
train_data["release_date"] = pd.to_datetime(train_data["release_date"], '%Y-%m-%d').dt.strftime('%Y%m').astype(float)
test_data["release_date"] = pd.to_datetime(test_data["release_date"], '%Y-%m-%d').dt.strftime('%Y%m').astype(float)

# hot codeing, combining the infoof reviews and prices
X = train_data["genres"].str.lower().str.get_dummies(',')
X = X.join(train_data["categories"].str.lower().str.get_dummies(','))
X = X.join(train_data["tags"].str.get_dummies(','))
X = X.join(train_data["price"]).join(train_data['total_positive_reviews']).join(train_data['total_negative_reviews'])
X['diff'] = train_data['total_positive_reviews']-train_data['total_negative_reviews']
y = train_data["playtime_forever"]

# do the totally same thing to the test data
X_ = test_data["genres"].str.lower().str.get_dummies(',')
X_= X_.join(test_data["categories"].str.lower().str.get_dummies(','))
X_ = X_.join(test_data["tags"].str.get_dummies(','))
X_ = X_.join(test_data["price"]).join(test_data['total_positive_reviews']).join(test_data['total_negative_reviews'])
X_['diff'] = test_data['total_positive_reviews']-test_data['total_negative_reviews']

# for the encoders do not occur in the test set, delete them
for col in X.columns.values:
    if col not in X_.columns.values:
        X = X.drop([col],axis=1)

X = X.join(train_data["is_free"])
X = X.join(train_data["purchase_date"])
X = X.join(train_data["release_date"])
X['date_diff'] = train_data['purchase_date']-train_data['release_date']

X_ = X_.join(test_data["is_free"])
X_ = X_.join(test_data["purchase_date"])
X_ = X_.join(test_data["release_date"])
X_['date_diff'] = test_data['purchase_date']-test_data['release_date']
# now make the test data same as train data
for col in X_.columns.values:
    if col not in X.columns.values:
        X_ = X_.drop([col],axis=1)

# try to do feature engineering
from sklearn.feature_selection import SelectKBest, f_regression
print(X.shape)
selector = SelectKBest(f_regression, k=80)
X = selector.fit_transform(X, y)
print(X.shape)
X_ = selector.transform(X_)
print(X_.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=1)


from sklearn.metrics import mean_squared_error
# Cross validation
def cross_validation(reg, X, y, transfer_flag):
    mse = []
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=i)
        reg.fit(X_train, y_train)
        if transfer_flag == True:
            y_pre = y_transfer_back(reg.predict(X_test))
            y_test = y_transfer_back(np.array(y_test))
        else:
            y_pre = reg.predict(X_test)
        mse.append(mean_squared_error(y_test, y_pre)**0.5)
    return np.array(mse)

# Random forest
regr = RandomForestRegressor(max_depth=12, random_state=5, n_estimators=200, min_samples_split=5)
mse = cross_validation(regr, X, y, False)
print("RMSE: %0.2f (+/- %0.2f)" % (mse.mean(), mse.std() * 2))
regr.fit(X, y)

# KNN
from sklearn.neighbors import KNeighborsRegressor
neigh = KNeighborsRegressor(n_neighbors=18, weights='uniform', algorithm='auto')
mse = cross_validation(neigh, X, y, False)
print("MSE: %0.2f (+/- %0.2f)" % (mse.mean(), mse.std() * 2))
neigh.fit(X_train, y_train)
print(neigh.score(X_train, y_train))
print(neigh.score(X_test, y_test))
print(neigh.score(X, y))
neigh.fit(X, y)

# vote for result
from sklearn.ensemble import VotingRegressor
vr = VotingRegressor([('knn', neigh), ('rf', regr)])
mse = cross_validation(vr, X, y, False)
print("MSE: %0.2f (+/- %0.2f)" % (mse.mean(), mse.std() * 2))
vr.fit(X, y)

# generate output
output = pd.DataFrame(vr.predict(X_).reshape(-1,1), columns=['playtime_forever'])
output['id'] = [i for i in range(len(output['playtime_forever']))]
output = output[['id', 'playtime_forever']]
output.to_csv('Result.csv', index=False)
