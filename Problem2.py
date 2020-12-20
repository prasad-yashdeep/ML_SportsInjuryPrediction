from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import plot_confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
import itertools
import random
from sklearn.linear_model import LinearRegression
import math

df = pd.read_csv('Dataset/data.csv')

df.loc[(df.Temperature == -999), 'Temperature'] = 55

# changing the playdate values
df["PlayerDay"] = df["PlayerDay"].abs()
features_df1 = df.set_index('GameID')


y = features_df1['Injury']
X = features_df1.drop(columns=['Injury'])
# print(X.shape)
res = RandomOverSampler(random_state=0, sampling_strategy={
                        1: 2900, 2: 3900, 3: 800, 4: 2900})
X_resampled, y_resampled = res.fit_resample(X, y)
dt_yresam = pd.DataFrame(y_resampled)
dt_yresam.columns = ['T']
# print(dt_yresam['T'].value_counts())
#sns.countplot(x='T',data=dt_yresam, palette='hls')
dt_yresam.sample(frac=1)
# print(X_resampled.shape)

lis1 = []
for i in y_resampled:
    if(i == 0):
        lis1.append(0)
    elif(i == 1):
        lis1.append(random.randint(1, 6))
    elif(i == 2):
        lis1.append(random.randint(7, 14))
    elif(i == 3):
        lis1.append(random.randint(15, 28))
    elif(i == 4):
        lis1.append(random.randint(29, 42))

lis1 = np.array(lis1)
# print(lis1)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, lis1, test_size=0.2, random_state=42, shuffle=True)
reg = LinearRegression(normalize=True).fit(X_train, y_train)
y_pred = reg.predict(X_test)
# print(len(y_pred))
for i in range(len(y_pred)):
    y_pred[i] = int(round(y_pred[i]))
# y_pred

print("linear Regression")
print(mean_squared_error(y_test, y_pred))

print("--------------------")

regr = RandomForestRegressor(max_depth=20, random_state=0, n_estimators=100)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
for i in range(len(y_pred)):
    y_pred[i] = int(round(y_pred[i]))


print("RandomForestRegressor")
print(mean_squared_error(y_test, y_pred))
print("--------------------")


regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
for i in range(len(y_pred)):
    y_pred[i] = int(round(y_pred[i]))


print("SVR")
print(mean_squared_error(y_test, y_pred))

print("---------------------")
