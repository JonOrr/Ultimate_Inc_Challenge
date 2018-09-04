# -*- coding: utf-8 -*-
"""
Created on Tue Sep  4 01:11:22 2018

Ultimate Part 3 

@author: Jon
"""

import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

ulti_filename   = 'ultimate_data_challenge.json'

with open(ulti_filename) as f:
    ulti_data = json.load(f)

ulti_df = pd.DataFrame(ulti_data)

zero_user_df     = ulti_df.loc[ulti_df['trips_in_first_30_days'] == 0]
retained_user_df = ulti_df.loc[ulti_df['trips_in_first_30_days'] != 0]

zero_user_df.describe()

retained_user_df.describe()

retained_user_percent = len(retained_user_df)/ len(ulti_df) 
print('Ultimate retains', '%.2f' % (retained_user_percent*100), '%', 'of all users \n\n')


#################################
# Retention By Black Car Service
#################################

# Does the ultimate black service contribute to the difference
zero_user_percent_ub = zero_user_df['ultimate_black_user'].sum() / len(zero_user_df)
retained_user_percent_ub =  retained_user_df['ultimate_black_user'].sum() / len(retained_user_df)

# Non-retained users used ultimate black 34%
print("%.2f" % (zero_user_percent_ub*100), '%', 'of Non-retained users used ultimate black for their first ride')
# Reatined users used ultimate black 39%
print("%.2f" % (retained_user_percent_ub*100), '%', 'of Non-retained users used ultimate black for their first ride')

# What percentage of people that used ultimate black retained?
# 37 percent of all users used ultimate black their first ride
# 69.22% of users retained

black_car_user_df = ulti_df.loc[ulti_df['ultimate_black_user'] == True]
# 18854 users are black_car_users
zero_black = black_car_user_df.loc[black_car_user_df['trips_in_first_30_days'] == 0]
retain_black = black_car_user_df.loc[black_car_user_df['trips_in_first_30_days'] != 0]
black_retention_rate = len(retain_black) / len(black_car_user_df)

print('Ultimate Black car retains', "%.2f" % (black_retention_rate*100), '%', 'of users.')
# Black car retains 71.47% of users. So slgihtly higher. 

######################
# Retention By City
######################
# Cites are Astapor, King's Landing, and Winterfell

# Astapor
zero_Astapor     = zero_user_df.loc[zero_user_df['city'] == 'Astapor']
retained_Astapor = retained_user_df.loc[retained_user_df['city'] == 'Astapor']

retention_rate_Astapor = len(retained_Astapor) / (len(retained_Astapor) + len(zero_Astapor))
# 67.745%
print('Ultimate Black car retained', "%.2f" % (retention_rate_Astapor*100), '%', 'of users whose first ride originated from Astapor.')


# King's Landing
zero_Kings     = zero_user_df.loc[zero_user_df['city'] == 'King\'s Landing']
retained_Kings = retained_user_df.loc[retained_user_df['city'] == 'King\'s Landing']

retention_rate_Kings = len(retained_Kings) / (len(retained_Kings) + len(zero_Kings))
# 65.13%
print('Ultimate Black car retained', "%.2f" % (retention_rate_Kings*100), '%', 'of users whose first ride originated from King\'s Landing.')

# Winterfell
zero_Winter     = zero_user_df.loc[zero_user_df['city'] == 'Winterfell']
retained_Winter =retained_user_df.loc[retained_user_df['city'] == 'Winterfell']
 
retention_rate_Winter = len(retained_Winter) / (len(retained_Winter) + len(zero_Winter))
# 72.04 %
print('Ultimate Black car retained', "%.2f" % (retention_rate_Winter*100), '%', 'of users whose first ride originated from Winterfell.')


######################
# Retention By phone
######################
# iPhone
zero_iPhone     = zero_user_df.loc[zero_user_df['phone'] == 'iPhone']
retained_iPhone = retained_user_df.loc[retained_user_df['phone'] == 'iPhone']

retention_rate_iPhone = len(retained_iPhone) / (len(retained_iPhone) + len(zero_iPhone))
# 69.07%
print('Ultimate Black car retained', "%.2f" % (retention_rate_iPhone*100), '%', 'of users who signed up via iPhone.')


# Android
zero_Android    = zero_user_df.loc[zero_user_df['phone'] == 'Android']
retained_Android = retained_user_df.loc[retained_user_df['phone'] == 'Android']

retention_rate_Android = len(retained_Android) / (len(retained_Android) + len(zero_Android))
# 69.265%
print('Ultimate Black car retained', "%.2f" % (retention_rate_Android*100), '%', 'of users who signed up via Android.')



##### Modeling
new_ulti_df = ulti_df.copy(deep = True)

retained_classification = []
for i in range(len(new_ulti_df)):
    if new_ulti_df['trips_in_first_30_days'][i] == 0:
        retained_classification.append(0)
    else:
        retained_classification.append(1)
        
new_ulti_df['retained_classification'] = retained_classification    

black_car_classification = []
for i in range(len(new_ulti_df)):
    if new_ulti_df['ultimate_black_user'][i] == 0:
        black_car_classification.append(0)
    else:
        black_car_classification.append(1)   
        
new_ulti_df['black_car_classification'] = black_car_classification


city_classification = []
for i in range(len(new_ulti_df)):
    if new_ulti_df['city'][i] == 'Astapor':
        city_classification.append(1)
        
    elif new_ulti_df['city'][i] == 'King\'s Landing':
         city_classification.append(2)
    else: 
        city_classification.append(3)

new_ulti_df['city_classification'] = city_classification


from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
        
df = new_ulti_df._get_numeric_data()
df.dropna(inplace = True)
df.drop('ultimate_black_user', axis = 1, inplace = True)

# Drop NA Values. They only appeared in the avg_rating of driver. In a longer time frame study I would recommend looking into
# the cases with NA values as a separate dataframe. Maybe those customers were so dissatisfied that they did not want to leave a rating. 

ml_cols = [col for col in df.columns if col not in ['retained_classification', 'trips_in_first_30_days']]

X = df[ml_cols]
y = df['retained_classification'] # The label column
# from sklearn import preprocessing
# X = preprocessing.scale(X) 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)  # , random_state = 42

clf = RandomForestClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
from sklearn.cross_validation import cross_val_score
print('Ten fold Cross Validated Accuracy score:', 
      '%.4f' % np.mean(cross_val_score(clf, X_train, y_train, cv=10)))



from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV

# The following is revised from the documentation for randomized search cv.

# Utility function to report best scores
def report(results, n_top=1):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# specify parameters and distributions to sample from
param_dist = {"max_depth": [5, 8, 15, 25, 30],
              "max_features": sp_randint(1, 8),
              "min_samples_split": sp_randint(2, 100),
              "min_samples_leaf": sp_randint(1, 10),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

# Run randomized search
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                   n_iter=n_iter_search)

random_search.fit(X_train, y_train)
report(random_search.cv_results_)


clf_rcv = RandomForestClassifier(bootstrap = False, criterion = 'entropy', max_depth =  8, max_features =  3, min_samples_leaf = 2, min_samples_split = 46)
clf_rcv.fit(X_train, y_train)

y_pred = clf_rcv.predict(X_test)
from sklearn.cross_validation import cross_val_score
print('Ten fold Cross Validated Accuracy score:', 
      '%.4f' % np.mean(cross_val_score(clf, X_train, y_train, cv=10)))

# Get numerical feature importances
importances = list(clf_rcv.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, 
                       importance in zip(ml_cols, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, 
                             key = lambda x: x[1], 
                             reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



# Set the style
plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'orange')

# Tick labels for x axis
plt.xticks(x_values, ml_cols, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Variable Importances: All Numeric Variables'); 
plt.show()
plt.clf()


ml_cols.remove('weekday_pct')


X = df[ml_cols]
y = df['retained_classification'] # The label column
# from sklearn import preprocessing
# X = preprocessing.scale(X) 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)  # , random_state = 42

clf_rcv = RandomForestClassifier(bootstrap = False, criterion = 'entropy', max_depth =  8, max_features =  3, min_samples_leaf = 2, min_samples_split = 46)
clf_rcv.fit(X_train, y_train)

y_pred = clf_rcv.predict(X_test)
from sklearn.cross_validation import cross_val_score
print('Ten fold Cross Validated Accuracy score:', 
      '%.4f' % np.mean(cross_val_score(clf, X_train, y_train, cv=10)))

# Get numerical feature importances
importances = list(clf_rcv.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, 
                       importance in zip(ml_cols, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, 
                             key = lambda x: x[1], 
                             reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



# Set the style
plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'teal')

# Tick labels for x axis
plt.xticks(x_values, ml_cols, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Variable Importances: All numeric variables Sans Weekday'); 
plt.show()
plt.clf()


ml_cols.remove('avg_rating_by_driver')


X = df[ml_cols]
y = df['retained_classification'] # The label column
# from sklearn import preprocessing
# X = preprocessing.scale(X) 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)  # , random_state = 42

clf_rcv = RandomForestClassifier(bootstrap = False, criterion = 'entropy', max_depth =  8, max_features =  3, min_samples_leaf = 2, min_samples_split = 46)
clf_rcv.fit(X_train, y_train)

y_pred = clf_rcv.predict(X_test)
from sklearn.cross_validation import cross_val_score
print('Ten fold Cross Validated Accuracy score:', 
      '%.4f' % np.mean(cross_val_score(clf, X_train, y_train, cv=10)))

# Get numerical feature importances
importances = list(clf_rcv.feature_importances_)

# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, 
                       importance in zip(ml_cols, importances)]

# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, 
                             key = lambda x: x[1], 
                             reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];



# Set the style
plt.style.use('fivethirtyeight')

# list of x locations for plotting
x_values = list(range(len(importances)))

# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical', color = 'purple')

# Tick labels for x axis
plt.xticks(x_values, ml_cols, rotation='vertical')

# Axis labels and title
plt.ylabel('Importance'); 
plt.xlabel('Variable');
plt.title('Variable Importances: All Numeric Variables sans Weekday and Driver Rating for User'); 
plt.show()
plt.clf()


retain_probs = clf_rcv.predict_proba(X_test)[:,1]
six_month_indep_probs = np.power(retain_probs, 6)
avg_six_month_prob = np.average(six_month_indep_probs)
print('Average six month probability:', '%.2f%%' %(100*avg_six_month_prob))