
# coding: utf-8

# In[117]:

from sklearn import svm
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import cross_val_score


bots = pd.read_csv("bots_data.csv",encoding = "latin")
bots = bots[['screen_name','description','followers_count','friends_count','favourites_count','verified','statuses_count','default_profile','default_profile_image','has_extended_profile','name','bot']]

non_bots = pd.read_csv("nonbots_data.csv",encoding = "latin")
non_bots = non_bots[['screen_name','description','followers_count','friends_count','favourites_count','verified','statuses_count','default_profile','default_profile_image','has_extended_profile','name','bot']]




# In[94]:




# In[118]:

a= pd.concat([bots,non_bots])
a = a.dropna(how = 'any')
le = preprocessing.LabelEncoder()
le.fit(a.verified)
a.verified = le.transform(a.verified)
a.default_profile = le.transform(a.default_profile)
a.default_profile_image = le.transform(a.default_profile_image)
a.has_extended_profile = le.transform(a.has_extended_profile)



# In[119]:

train_features = a[['followers_count','verified','friends_count','favourites_count','statuses_count','default_profile','default_profile_image','has_extended_profile']]
train_labels = a['bot'].values
clf= svm.SVC()
clf.fit(train_features,train_labels)


# In[120]:

scores_svm = cross_val_score(clf,train_features,train_labels)
print(scores_svm)


# In[ ]:




# In[ ]:




# In[ ]:



