# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# function that extract year and month to create columns in dataframe
def extract_date(df,column):
    df[column+'_year'] = df[column].apply(lambda x: x.year)
    df[column+'_month'] = df[column].apply(lambda x: x.month)

def normalization(df,columns):
    for column in columns:
        df[column] = (df[column]-df[column].min())/(df[column].max()-df[column].min())

train_data = pd.read_csv('/kaggle/input/msbd5001-fall2019/train.csv',parse_dates=['purchase_date','release_date'])
test_data = pd.read_csv('/kaggle/input/msbd5001-fall2019/test.csv',parse_dates=['purchase_date', 'release_date'])

num_train = train_data.shape[0]   # 357
num_test = test_data.shape[0]   # 90

# replace 2 NaT purchase_date with release_date
temp = train_data[train_data['purchase_date'].isnull()]
indexes = list(temp['id'])
#print(indexes)
for idx in indexes:
    train_data.loc[idx,'purchase_date'] = train_data.loc[idx,'release_date']
    train_data.loc[idx,'purchase_date'] = train_data.loc[idx,'release_date']
    
#train_data[train_data['purchase_date'].isnull()]

extract_date(train_data,'purchase_date')
extract_date(train_data,'release_date')

# replace 2 NaT purchase_date with release_date
temp = test_data[test_data['purchase_date'].isnull()]
indexes = list(temp['id'])
#print(indexes)
for idx in indexes:
    test_data.loc[idx,'purchase_date'] = test_data.loc[idx,'release_date']
    test_data.loc[idx,'purchase_date'] = test_data.loc[idx,'release_date']

extract_date(test_data,'purchase_date')
extract_date(test_data,'release_date')
#test_data[test_data['purchase_date'].isnull()]

test_data['id'] += len(train_data['id'])
data = pd.concat([train_data.drop(['playtime_forever'], axis=1), test_data],axis=0,ignore_index=True)

# replace NaN with mean value
meanval1 = data['total_positive_reviews'].mean()
data['total_positive_reviews'].fillna(meanval1, inplace=True)
meanval2 = data['total_negative_reviews'].mean()
data['total_negative_reviews'].fillna(meanval2, inplace=True)
data.describe()

# one-hot encoding for category features
genres_one_hot = data['genres'].str.get_dummies(',')
#categories_one_hot = data['categories'].str.get_dummies(',')
#tags_one_hot = data['tags'].str.get_dummies(',')
#data = pd.concat([data, genres_one_hot, categories_one_hot, tags_one_hot], axis=1)
data = pd.concat([data, genres_one_hot], axis=1)
data.drop(['genres','categories','tags'], axis=1, inplace=True)
data.drop(['purchase_date', 'release_date'], axis=1,inplace=True)

data = data.loc[:,~data.columns.duplicated()]
normalization(data,['total_negative_reviews','price','total_positive_reviews','purchase_date_month','release_date_month'])

data["release_latest"] = (data["release_date_year"]- data["release_date_year"].min())
data["release_passed"] = (data["release_date_year"].max()- data["release_date_year"])
data["purchase_latest"] = (data["purchase_date_year"]- data["purchase_date_year"].min())
data["purchase_passed"] = (data["purchase_date_year"].max()- data["purchase_date_year"])

normalization(data, ["release_latest","release_passed","purchase_latest","purchase_passed","purchase_date_year","release_date_year"])

train_x = data.iloc[:num_train].drop(['id'], axis=1)
train_y = train_data['playtime_forever']

###
train = pd.concat([train_x,train_y],axis=1)
group1 = train[train['playtime_forever']<=20]   #341
group3 = train[train['playtime_forever']>60]   #4
group2 = train[train['playtime_forever']>20]   #12
group2 = group2[group2['playtime_forever']<=60]

new_train = pd.concat([group1,group3],axis=0)
for i in range(50):
    new_train = pd.concat([new_train,group3],axis=0)
    if i < 20:
         new_train = pd.concat([new_train,group2],axis=0)

new_train = new_train.sample(frac=1)
train_x = new_train.drop(['playtime_forever'],axis=1)
train_y = new_train['playtime_forever']
print(train_x.shape)
###

test_x = data.iloc[num_train:].drop(['id'], axis=1)

bst = xgb.XGBRegressor(Seed=1850,n_estimators=500)
bst.fit(train_x,train_y)
test_y = bst.predict(test_x)
test_y[test_y<0]=0

rf = RandomForestRegressor(n_estimators=500)
rf.fit(train_x, train_y)
test_y_2 = rf.predict(test_x)
test_y_2[test_y<0]=0

sample_submission = pd.read_csv('/kaggle/input/msbd5001-fall2019/samplesubmission.csv')
sample_submission

indexes = [str(n) for n in range(num_test)]
indexes = np.asarray(indexes)
prediction = np.vstack((indexes,(test_y+test_y_2)/2)).T
submission = pd.DataFrame(data=prediction,columns=['id','playtime_forever'])

submission.to_csv('submission.csv', index = False)