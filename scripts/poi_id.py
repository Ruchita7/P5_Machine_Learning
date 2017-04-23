#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
sys.path.append("../data/")
import pandas
import matplotlib.pyplot
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from sklearn import cross_validation
from sklearn.feature_selection import SelectKBest, f_classif
from tester import dump_classifier_and_data
from sklearn import preprocessing
from copy import deepcopy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

financial_features = ['salary', 'deferral_payments', 'total_payments', 'bonus',
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
                      'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock']

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi',"shared_receipt_with_poi"]

target_label = ['poi']
features_list =  target_label + financial_features + email_features  # You will need to use more features

### Load the dictionary containing the dataset
with open("../data/final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

df = pandas.DataFrame.from_records(list(data_dict.values()))
employees = pandas.Series(list(data_dict.keys()))

# set the index of data frame to be the employees series
df.set_index(employees, inplace=True)

## 1.1 Data Exploration

print "Number of data points:", len(df)

#Identify person of interest
poi_count =0
#List for storing person of interest names
for value in df["poi"]:
    if value==True:
        poi_count=poi_count+1

print "Number of person of interest:",poi_count
print "Number of non Person of Interest:", len(df)-poi_count
print "Number of features per person:", len(df.keys())

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

def compute_messages_exchanged_poi(df):
    return df["from_this_person_to_poi"]+df["from_poi_to_this_person"]+df["shared_receipt_with_poi"]

df["messages_with_poi"]=df.apply(compute_messages_exchanged_poi,axis=1)

new_features_list=["fraction_from_poi","fraction_to_poi","messages_with_poi"]
features_list = features_list+new_features_list


### Store to my_dataset for easy export below.
data_dict = df.to_dict('index')
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#Selecting top k=5-10 features

def selectKfeatures(k,features_list):
    k_best = SelectKBest(k=k)
    k_best.fit_transform(features, labels)
    scores = k_best.scores_
    features_selected = {}
    for i in k_best.get_support(indices=True):
        features_selected[features_list[i]] = scores[i]
        print "%s: %f" % (features_list[i], scores[i])
    return features_selected.keys()

# make a copy of features_list
features_list_new = deepcopy(features_list)
# remove 'poi' from features_list_new
features_list_new.remove('poi')

features_selected=[]
for i in range(5,11):
    print "\n%d Features selected"%i
    features_selected=selectKfeatures(i,features_list_new)


my_features_list=target_label+features_selected
print "Selected features: %s"%my_features_list

#extract features selected from my features list
data = featureFormat(my_dataset, my_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# feature scaling through min max scaling
min_max_scaler= preprocessing.MinMaxScaler()
features = min_max_scaler.fit_transform(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM",
         "Decision Tree", "Random Forest", "AdaBoost",
         "Naive Bayes","Extra Trees"]

classifiers = [
    KNeighborsClassifier(2),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(max_depth=5, n_estimators=10),
    AdaBoostClassifier(),
    GaussianNB(),
    ExtraTreesClassifier()]

for name, clf in zip(names, classifiers):
    clf.fit(features_train, labels_train)
    predictor = clf.predict(features_test)
    print clf
    print "name: %s"%name
    accuracy = accuracy_score(labels_test,predictor)
    precision = precision_score(labels_test,predictor)
    recall = recall_score(labels_test,predictor)
    print "accuracy:%f precision:%f recall:%f"%(accuracy,precision,recall)


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

parameters = {'max_depth': [1,2,3,4,5,6,8,9,10],
              'min_samples_split':[2,3,4,5],
              'min_samples_leaf':[1,2,3,4,5,6,7,8,9,10],
              'criterion':('gini', 'entropy')}
dt_clf = DecisionTreeClassifier(random_state = 42)
cv = cross_validation.StratifiedShuffleSplit(labels, n_iter=10)
clf = GridSearchCV(dt_clf, parameters,cv=cv, scoring = 'f1')
clf.fit(features,labels)

predictor = clf.predict(features_test)
precision = precision_score(labels_test,predictor)
print precision
recall = recall_score(labels_test,predictor)
print recall
print f1_score(labels_test,predictor)
dt_best_estimator=clf.best_estimator_
print dt_best_estimator
print clf.best_score_

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(dt_best_estimator, my_dataset, my_features_list)