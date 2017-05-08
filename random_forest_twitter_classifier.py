from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from nltk.tokenize import WordPunctTokenizer,word_tokenize
import numpy as np
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.feature_extraction.text import *
from random import choice
from string import ascii_uppercase
import timestring
from datetime import datetime

#create data frame containing your data, each column can be accessed # by df['column   name']
df_bot = pd.read_csv('bots_data.csv',encoding='latin-1')
df_non_bot = pd.read_csv('nonbots_data.csv',encoding='latin-1')
# df = pd.concat([df_bot,df_non_bot])

df = pd.read_csv('training_data_2_csv_UTF.csv')


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
    if pd.notnull(df['default_profile_image']) and df['default_profile_image'] == 'TRUE':
        return 1
    else:
        return 0

def extended_profile(df):
    if df['has_extended_profile'] == 'TRUE':
        return 1
    else:
        return 0

def name_bot(df):
    if pd.notnull(df['screen_name']) and 'bot' in df['screen_name'].lower():
        return 1
    else:
        return 0

def name_only_bot(df):
    if pd.notnull(df['name']) and 'bot' in str(df['name']).lower():
        return 1
    else:
        return 0

def desc_bot(df):
    if pd.notnull(df['description']) and 'bot' in (df['description'].lower()):
        return 1
    else:
        return 0

def listed_count(df):
    if df['listed_count'] == 'None':
        return 0
    else:
        return df['listed_count']

def friends_count(df):
    if df['friends_count'] == 'None':
        return 0
    else:
        return df['friends_count']

def followers(df):
    if df['followers_count'] == 'None':
        return 0
    else:
        return df['followers_count']

def length_desc(df):
    if pd.notnull(df['description']):
        return len(df['description'])
    else:
        return 0
def length_location(df):
    if pd.notnull(df['location']):
        return len(df['location'])
    else:
        return 0

def days(df):
    try:
        if pd.notnull(df['created_at']):
            dt = datetime.strptime(str(timestring.Date(df['created_at'])), "%Y-%m-%d %H:%M:%S")
            return abs(datetime.now() - dt).days
    except:
        return 365

    return 365

#dt_1 = parser.parse(df['created_at'][0])

print(type(df['created_at'][0]))
dt_2 = datetime.strptime(str(timestring.Date(df['created_at'][0])),"%Y-%m-%d %H:%M:%S")
dt_1 = datetime.strptime(str(timestring.Date(df['created_at'][1])),"%Y-%m-%d %H:%M:%S")

print(abs(datetime.now()-dt_2).days)


df['verified'] = df.apply(verified,axis=1)
df['default_profile_image'] = df.apply(default_profile_image,axis=1)
df['default_profile'] = df.apply(default_profile,axis=1)
df['name'] = df.apply(name_bot,axis=1)
df['name_only'] = df.apply(name_only_bot,axis=1)
df['desc'] = df.apply(desc_bot,axis=1)
df['location'] = df.apply(length_location,axis=1)
df['len_desc'] = df.apply(length_desc,axis=1)
df['ep'] = df.apply(extended_profile,axis=1)
df['date'] = df.apply(days,axis=1)
df = df[['followers_count','friends_count','listedcount','verified','statuses_count','default_profile','default_profile_image','name','name_only','desc','location','len_desc','favourites_count','ep','date','bot']]

print(df.shape)

df.to_csv('intermediate.csv',index=False)

X_train,X_test,y_train,y_test = train_test_split(df.drop('bot',axis=1),df['bot'],test_size=0.1)


# call Gaussian Naive Bayesian class with default parameters

rf = RandomForestClassifier(n_estimators=100,criterion='entropy')

# train model
rf.fit(X_train,y_train)

preds = rf.predict(X_test)

print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))

# df_test = pd.read_csv('test_data_4_students.csv')
#
# print(df_test.shape)
#
# df_test['verified'] = df_test.apply(verified,axis=1)
# df_test['default_profile_image'] = df_test.apply(default_profile_image,axis=1)
# df_test['default_profile'] = df_test.apply(default_profile,axis=1)
# df_test['name'] = df_test.apply(name_bot,axis=1)
# df_test['name_only'] = df_test.apply(name_only_bot,axis=1)
# df_test['desc'] = df_test.apply(desc_bot,axis=1)
# df_test['location'] = df_test.apply(length_location,axis=1)
# df_test['len_desc'] = df_test.apply(length_desc,axis=1)
# df_test['ep'] = df_test.apply(extended_profile,axis=1)
# df_test['listed_count'] = df_test.apply(listed_count,axis=1)
# df_test['friends_count'] = df_test.apply(friends_count,axis=1)
# df_test['followers_count'] = df_test.apply(followers,axis=1)
# df_test['date'] = df_test.apply(days,axis=1)
# df_test = df_test[['followers_count','friends_count','listed_count','verified','statuses_count','default_profile','default_profile_image','name','name_only','desc','location','len_desc','favorites_count','ep','date']]
#
# df_test.to_csv('test_clean.csv')

df_test = pd.read_csv('test_clean.csv')


kaggle_preds = rf.predict(df_test)

print(kaggle_preds)

df_test['bot'] = kaggle_preds

df_test.to_csv('answers.csv')






