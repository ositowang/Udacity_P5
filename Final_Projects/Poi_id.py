#!/usr/bin/python

import sys
import pickle
import pprint
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary','bonus','total_payments','deferral_payments','exercised_stock_options',
                     'restricted_stock','restricted_stock_deferred','total_stock_value','expenses',
                     'other','director_fees','loan_advances','deferred_income','long_term_incentive',
                     'from_poi_to_this_person','from_this_person_to_poi','to_messages','from_messages',
                     'shared_receipt_with_poi','fraction_from_poi','fraction_to_poi']
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

#Take a quick look at the dataset
print "\n","number of data point:"
print len(data_dict)


print "\n","number of poi:"
pois = [a for a, b in data_dict.items() if b['poi']]
print len(pois)
### Task 2: Remove outliers
#Find the NaN values
for a,b in data_dict.items():
	nan_num = 0
	for i in b.values():
		if i == "NaN":
			nan_num = nan_num + 1

	if nan_num == 20:
		print a,"has",nan_num,"Nan Values"

# Like the quiz, let's find the outliers
salary_list = []
bonus_list = []

for features in data_dict.values():
	if features["salary"] == "NaN" or features["bonus"] == "NaN":
		continue
	salary_list.append(features["salary"])
	bonus_list.append(features["bonus"])


# plot the salary and bonus to find out the outliers
plt.scatter(salary_list, bonus_list)
plt.show()


#Find the most outstanding points
for key,value in data_dict.items():
	if value["salary"] == max(salary_list):
		print ""
		print key,"'s salary : ",value["salary"]

	if value["bonus"] == max(bonus_list):
		print ""
		print key,"'s bonus : ",value["bonus"]

	if key == "THE TRAVEL AGENCY IN THE PARK" or key == "LOCKHART EUGENE E":
		print ""
		print key,":"
		print value
# Remove Outliers
data_dict.pop("TOTAL")
data_dict.pop("THE TRAVEL AGENCY IN THE PARK")
data_dict.pop("LOCKHART EUGENE E")

print "Number of data points after cleanning:", len(data_dict)

### Task 3: Create new feature(s)
def compute_Fraction(poi_messages, all_messages):
	fraction = 0.
	if poi_messages == "NaN" or all_messages == "NaN":
		return 0
	fraction = float(poi_messages) / all_messages
	return fraction


for i in data_dict:
	data_dict[i]['fraction_from_poi'] = compute_Fraction(data_dict[i]['from_poi_to_this_person'],
														data_dict[i]['to_messages'])
	data_dict[i]['fraction_to_poi'] = compute_Fraction(data_dict[i]['from_this_person_to_poi'],
													  data_dict[i]['from_messages'])
# Select the 5 most valuable features for further use
my_array = featureFormat(data_dict,features_list)
labels, features =targetFeatureSplit(my_array)
k_best_feature = SelectKBest()
k_best_feature.fit(features,labels)
feature_scores = k_best_feature.scores_
feature_scoring =  zip(features_list[1:],feature_scores)
six_best = sorted(feature_scoring, key = lambda x:x[1],reverse=True)[:6]
features_list = ["poi"] + [x[0] for x in six_best]
print features_list
print six_best



### Store to my_dataset for easy export below.
my_dataset = data_dict


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

#as the different features has huge difference from 1 to 10*7
#scaling the data into [0.1]
scaler = MinMaxScaler()
features_new = scaler.fit_transform(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.

#Create train and test dataset
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels,
																			test_size=0.3, random_state=42)
#Naive Bayes Classifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from time import time

starting_time = time()
parameter = {}
Naive_CLF = GaussianNB()
Naive_CLF = Pipeline([('sc',scaler),('gnc',Naive_CLF)])
GS_CLF = GridSearchCV(Naive_CLF,parameter)
GS_CLF.fit(features_train,labels_train)
Naive_CLF = GS_CLF.best_estimator_

print "Gaussian NB scores:",Naive_CLF.score(features_train,labels_train)
print "Gaussian NB time elapsed:", time()-starting_time,"seconds"

#Test the Naive Bayses Classifier
from tester import dump_classifier_and_data,test_classifier

print "GaussianNB Test Results:",test_classifier(Naive_CLF,my_dataset,features_list)

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

Starting_Time1 = time()

parameter = {'criterion': ['gini', 'entropy'],
			'min_samples_split': [2, 5, 10, 20],
			'max_depth': [None, 2, 5, 10],
			'splitter': ['random', 'best'],
			'max_leaf_nodes': [None, 5, 10, 20]   }

Decision_Tree = tree.DecisionTreeClassifier()

DT_CLF = GridSearchCV(Decision_Tree,parameter)
DT_CLF.fit(features_train,labels_train)

clf_DT = DT_CLF.best_estimator_

print "ecision Tree Classifier",clf_DT.score(features_train,labels_train)
print "Decision Tree Classifier:", round(time()-Starting_Time1, 3), "s"

##  Test Point
print "Decision Tree Classifier Test Results:",test_classifier(clf_DT,my_dataset,features_list)


#RandomForest
from sklearn.ensemble import RandomForestClassifier
Starting_Time2 = time()

print '\nRandomForest\n'
Random_CLF = RandomForestClassifier()
parameters = {'criterion': ['gini', 'entropy'],
			'max_depth': [None, 2, 4, 6],
			'max_leaf_nodes': [None,5, 10],
			'n_estimators': [1, 5, 10, 20, 40],
			'min_samples_split':[4, 6],
			}
Ran_CLF = GridSearchCV(Random_CLF, parameters)
Ran_CLF.fit(features_train,labels_train)

clf_RF = Ran_CLF.best_estimator_

print "\nRandomForest:",clf_RF.score(features_train,labels_train)
print "Random Forest Classifier:", round(time()-Starting_Time2, 3), "s"

##  Test Point
print "\nRandomForest Test Results:",test_classifier(clf_RF,my_dataset,features_list)
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
clf = Naive_CLF

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
