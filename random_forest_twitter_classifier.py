#Python imports
import timestring
from datetime import datetime
import pandas as pd

#Sklearn Imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


#Load training data into a dataframe
df = pd.read_csv('training_data_2_csv_UTF.csv')

#Methods to apply on dataframe for data cleaning
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

#Apply functions to clean columns of dataframe
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

#Feature selection, drop unwanted columns from dataframe
df = df[['followers_count','friends_count','listedcount','verified','statuses_count','default_profile','default_profile_image','name','name_only','desc','location','len_desc','favourites_count','ep','date','bot']]

#USe train test split to split data into training and testing sets
X_train,X_test,y_train,y_test = train_test_split(df.drop('bot',axis=1),df['bot'],test_size=0.1)

# construct parameter grid
param_grid = {'max_depth': [1, 3, 6, 9, 12, 15, None],
              'max_features': ['auto','log2',None],
              'min_samples_split': [2,4,6,8,10],
              'min_samples_leaf': [1, 3, 6, 9, 12, 15],
              'bootstrap': [True, False],
              'criterion': ['gini', 'entropy']}


#Declare Random Forest classifier Object
rf = GridSearchCV(RandomForestClassifier(), param_grid = param_grid).fit(X_train, y_train)

# assess predictive accuracy
predict = grid_search.predict(X_test)

# train model by calling fit method
rf.fit(X_train,y_train)

#Store predictions on test set (Cross Validation)
preds = rf.predict(X_test)


#Take a look at the classification report and confusion matrix to evaluate model accuracy
print(classification_report(y_test,preds))
print(confusion_matrix(y_test,preds))


#Load Test data into dataframe
df_test = pd.read_csv('test_data_4_students.csv')

#Apply functions to clean test dataset
df_test['verified'] = df_test.apply(verified,axis=1)
df_test['default_profile_image'] = df_test.apply(default_profile_image,axis=1)
df_test['default_profile'] = df_test.apply(default_profile,axis=1)
df_test['name'] = df_test.apply(name_bot,axis=1)
df_test['name_only'] = df_test.apply(name_only_bot,axis=1)
df_test['desc'] = df_test.apply(desc_bot,axis=1)
df_test['location'] = df_test.apply(length_location,axis=1)
df_test['len_desc'] = df_test.apply(length_desc,axis=1)
df_test['ep'] = df_test.apply(extended_profile,axis=1)
df_test['listed_count'] = df_test.apply(listed_count,axis=1)
df_test['friends_count'] = df_test.apply(friends_count,axis=1)
df_test['followers_count'] = df_test.apply(followers,axis=1)
df_test['date'] = df_test.apply(days,axis=1)

#Remove unwanted columns (Feature Selection)
df_test = df_test[['followers_count','friends_count','listed_count','verified','statuses_count','default_profile','default_profile_image','name','name_only','desc','location','len_desc','favorites_count','ep','date']]

#Store cleaned test data to disk for future use
df_test.to_csv('test_clean.csv')

#Load cleaned test dataset
df_test = pd.read_csv('test_clean.csv')

#Run model onto test data and get predictions
kaggle_preds = rf.predict(df_test)

#Add predictions to test dataframe
df_test['bot'] = kaggle_preds

#Save dataframe with predictions to disk
df_test.to_csv('answers.csv')






