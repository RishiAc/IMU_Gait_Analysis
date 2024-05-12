import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # statistical data visualization
#%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

df = pd.read_csv('/Users/local_user/Downloads/TestForProject/out.csv')
col_names = ['index','time_series','x_accel','y_accel', 'z_accel', 'pitch','roll','labels'] 
predictors = ["x_accel","y_accel","z_accel", "pitch","roll"]
#includes label array
df.columns = col_names
#print(df.head)
#print(df['labels'].value_counts())
time_series = df['time_series']
#remove things not relecant to predicting
X=df.drop(df.index[0:1575])
X = X.drop(df.index[20700:])
X = X.drop(['labels', 'index','time_series'], axis = 1)
print(X)

#y = preidction, setting as Labels
y = df['labels']
y = y.drop(df.index[0:1575])
y = y.drop(df.index[20700:])
dataTotal = X.assign(labels = y)
print(dataTotal)

from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 42)

def predict(train, test, predictors, model):
    model.fit(train[predictors], train["labels"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["labels"], preds], axis=1)
    return combined

def backtest (data, model, predictors, start =6000 , step = 600):
    all_predictions = []
    for i in range (start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict (train, test, predictors, model)
        all_predictions.append(predictions)
    return pd.concat (all_predictions)


splitAmt = .2
train = dataTotal.iloc[:int(20699*splitAmt)]
test = dataTotal.iloc[int(20699*splitAmt):]

from sklearn.ensemble import RandomForestClassifier
# instantiate the classifier 
rfc = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
predictions = backtest(dataTotal, rfc, predictors)
testPredict = predict(train, test, predictors, rfc)
print(testPredict["labels"].value_counts())
print(testPredict["Predictions"].value_counts())
"""print(predictions["Predictions"].value_counts())

print(predictions["labels"].value_counts())
print(predictions["Predictions"])"""
from sklearn.metrics import accuracy_score

#print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(predictions[], predictions["Predictions"])))

"""
# fit the model
rfc.fit(X_train, y_train)
print(X_train.shape, X_test.shape)
y_pred = rfc.predict(X_test)
print(y_pred)
from sklearn.metrics import accuracy_score

print('Model accuracy score with 100 decision-trees : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))




clf = RandomForestClassifier(n_estimators=10, random_state=0)

# fit the model to the training set


clf.fit(X_train, y_train)
feature_scores = pd.Series(clf.feature_importances_, index=X_train.columns).sort_values(ascending=False)

print(feature_scores)


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)
timestamps = []
lenPredict = len(y_pred)-1

for x in range (lenPredict, 0, -1):
    if y_pred[x] ==1 and y_pred[x-1]!=1:
        timestamps.append(time_series[x+(len(time_series)-len(y_pred))])
timedf = pd.DataFrame(timestamps)
timedf.to_csv('timeseries.csv')   



        
"""