
# coding: utf-8

# In[40]:

from sklearn import svm
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score
import numpy as np
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from ggplot import *


bots = pd.read_csv("bots_data.csv",encoding = "latin")
bots = bots[['screen_name','description','followers_count','friends_count','favourites_count','verified','statuses_count','default_profile','default_profile_image','has_extended_profile','name','bot']]

non_bots = pd.read_csv("nonbots_data.csv",encoding = "latin")
non_bots = non_bots[['screen_name','description','followers_count','friends_count','favourites_count','verified','statuses_count','default_profile','default_profile_image','has_extended_profile','name','bot']]


a= pd.concat([bots,non_bots])
a = a.dropna(how = 'any')
le = preprocessing.LabelEncoder()
le.fit(a.verified)
a.verified = le.transform(a.verified)
a.default_profile = le.transform(a.default_profile)
a.default_profile_image = le.transform(a.default_profile_image)
a.has_extended_profile = le.transform(a.has_extended_profile)

train_features = a[['followers_count','verified','friends_count','favourites_count','statuses_count','default_profile','default_profile_image','has_extended_profile']]
train_labels = a['bot'].values

X_train, X_test, y_train, y_test = train_test_split(train_features,train_labels, test_size=.75,
                                                    random_state=0)

clf= svm.SVC()
clf.fit(X_train,y_train)

clf_Mnb = MultinomialNB()
clf_Mnb.fit(X_train,y_train)


# In[41]:

y_pred = clf.predict(X_test)
fpr,tpr , _ = metrics.roc_curve(y_test,y_pred)

df_roc = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
g = ggplot(df_roc, aes(x='fpr', y='tpr')) +    geom_line() +    geom_abline(linetype='dashed')
print(g)


# In[42]:

y_pred = clf_Mnb.predict(X_test)
fpr,tpr , _ = metrics.roc_curve(y_test,y_pred)

df_roc = pd.DataFrame(dict(fpr=fpr, tpr=tpr))
g = ggplot(df_roc, aes(x='fpr', y='tpr')) +    geom_line() +    geom_abline(linetype='dashed')
print(g)

