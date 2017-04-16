#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
sys.path.append("../data/")
import pandas
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','deferral_payments', 'total_payments', 'loan_advances', 'bonus',
                 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees',
                 'to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',
                 'shared_receipt_with_poi'] # You will need to use more features

### Load the dictionary containing the dataset
with open("../data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pandas.DataFrame.from_records(list(data_dict.values()))
#df.replace('NaN', 0, inplace = True)
#employees = pandas.Series(list(data_dict.keys()))

# set the index of data frame to be the employees series
#df.set_index(employees, inplace=True)

## 1.1 Data Exploration

print "Number of data points", len(df.keys())

#Identify person of interest
poi_count =0
#List for storing person of interest names
for value in df["poi"]:
    if value==True:
        poi_count=poi_count+1

print "Number of person of interest count:",poi_count
print "Number of non Person of Interest", len(df)-poi_count

print "Number of features per person", len(df.keys())

#Calculating features with missing values
missing_values={}

for col in df:
    missing_values[col]=0
    for index in range(len(df[col])):
        if df[col][index]=="NaN":
            missing_values[col]=missing_values[col]+1

print "Missing values in dataset:"
for item in missing_values.keys():
    print item, missing_values[item]


### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)