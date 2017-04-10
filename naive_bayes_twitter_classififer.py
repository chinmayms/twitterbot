from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.cross_validation import cross_val_score, KFold
from sklearn import metrics
from ggplot import *



# create data frame containing your data, each column can be accessed # by df['column   name']
df_bot = pd.read_csv('bots_data.csv',encoding='latin-1')
df_non_bot = pd.read_csv('nonbots_data.csv',encoding='latin-1')
df = pd.concat([df_bot,df_non_bot])

def verified(df):
    if df['verified'] == 'TRUE':
        return 1
    else:
        return 0

def default_profile(df):
    if df['default_profile'] == 'TRUE':
        return 1
    else:
        return 0

def default_profile_image(df):
    if df['default_profile_image'] == 'TRUE':
        return 1
    else:
        return 0


df['verified'] = df.apply(verified,axis=1)
df['default_profile_image'] = df.apply(default_profile_image,axis=1)
df['default_profile'] = df.apply(default_profile,axis=1)

df = df[['followers_count','friends_count','listedcount','verified','statuses_count','default_profile','default_profile_image','bot']]

# print(df.head())



# add columns to your data frame
df['is_train'] = np.random.uniform(0, 1, len(df)) <= 0.75


# define training and test sets
train = df[df['is_train'] == True]
test = df[df['is_train'] == False]

trainTargets = np.array(train['bot']).astype(int)
testTargets = np.array(test['bot']).astype(int)



# columns you want to model
features = df.columns[0:7]

# call Gaussian Naive Bayesian class with default parameters
gnb = GaussianNB()


# train model


gnb.fit(train[features], trainTargets).predict(train[features])

y_pred = gnb.predict(test[features])
fpr,tpr , _ = metrics.roc_curve(testTargets,y_pred)

df_roc = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
g = ggplot(df_roc, aes(x='fpr', y='tpr')) +\
    geom_line() +\
    geom_abline(linetype='dashed')
print(g)




# print(gnb.score(test[features],testTargets))

K = 10  # folds
R2 = cross_val_score(gnb, test, y=testTargets, cv=KFold(testTargets.size, K), n_jobs=1).mean()
print("The %d-Folds estimate of the coefficient of determination is R2 = %s"
      % (K, R2))


