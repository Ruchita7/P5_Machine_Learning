#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
sys.path.append("../data/")
import pandas
import matplotlib.pyplot
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest, f_classif
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

financial_features = ['salary', 'deferral_payments', 'total_payments', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock']

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',"shared_receipt_with_poi"]


features_list = ['poi'] + financial_features + email_features  # You will need to use more features

### Load the dictionary containing the dataset
with open("../data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))

# set the index of data frame to be the employees series
df.set_index(employees, inplace=True)

## 1.1 Data Exploration

print "Number of data points", len(df)

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

df.replace('NaN', 0, inplace = True)
df.to_csv('../data/enron_data.csv')

## 1.2 Outlier identification

def identify_outlier(data_dict,point1Label, point2Label):
    features=[]
    features.append(point1Label)
    features.append(point2Label)
    data = featureFormat(data_dict,features)

    for point in data:
        point1=point[0]
        point2=point[1]
        matplotlib.pyplot.scatter(point1,point2)

    matplotlib.pyplot.xlabel(point1Label)
    matplotlib.pyplot.ylabel(point2Label)
    matplotlib.pyplot.show()

#to be uncommented
#identify_outlier(data_dict,"salary","bonus")
#identify_outlier(data_dict,"total_payments","total_stock_value")
#identify_outlier(data_dict,"exercised_stock_options","expenses")

### Task 2: Remove outliers

def remove_outlier(data_dict,df):
    outliers=["TOTAL","THE TRAVEL AGENCY IN THE PARK","LOCKHART EUGENE E"]
    for outlier in outliers:
        data_dict.pop(outlier,0)

    df.drop(outliers, inplace=True)
    return df


df=remove_outlier(data_dict,df)
print "Number of data points after removing outliers:", len(data_dict)

### Task 3: Create new feature(s)

def compute_fraction_from_poi_emails(df):
    fraction = 0.
    poi_messages=df["from_poi_to_this_person"]
    all_messages=df["to_messages"]
    if poi_messages != 0 and all_messages != 0:
        fraction = float(poi_messages) / all_messages
    return fraction

df["fraction_from_poi"]=df.apply(compute_fraction_from_poi_emails,axis=1)

def compute_fraction_to_poi_emails(df):
    fraction = 0.
    poi_messages=df["from_this_person_to_poi"]
    all_messages=df["from_messages"]
    if poi_messages != 0 and all_messages != 0:
        fraction = float(poi_messages) / all_messages
    return fraction

df["fraction_to_poi"]=df.apply(compute_fraction_to_poi_emails,axis=1)

new_features_list=["fraction_from_poi","fraction_to_poi"]
features_list = features_list+new_features_list

### Store to my_dataset for easy export below.
data_dict = df.to_dict('index')

my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

k_best = SelectKBest(k=5)
k_transform = k_best.fit_transform(features,labels)
print k_best.get_support()
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