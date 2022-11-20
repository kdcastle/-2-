# PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/kdcastle/2ndOSSAssignment

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sys

def load_dataset(dataset_path):
	data_frame = pd.read_csv(dataset_path)
	return data_frame


def dataset_stat(dataset_df):
	# To-Do: Implement this function
    
    
	NumOfClass_0 = len(dataset_df.loc[dataset_df["target"] == 0])
    
	NumOfClass_1 = len(dataset_df.loc[dataset_df["target"] == 1])
    
	NumOfFeat = dataset_df.shape[1] - 1
    
	return NumOfFeat, NumOfClass_0, NumOfClass_1

def split_dataset(dataset_df, testset_size):
	# To-Do: Implement this function

	AData = dataset_df.iloc[:, :-1].values
    
	BTarget = dataset_df.iloc[:,-1].values
    
	a_train, a_test, b_train, b_test = train_test_split(AData, BTarget, test_size=testset_size)
    
	return a_train, a_test, b_train, b_test


def decision_tree_train_test(x_train, x_test, y_train, y_test):
	# To-Do: Implement this function

	DecTree = DecisionTreeClassifier()
    
	DecTree.fit(x_train, y_train)

	Acc = accuracy_score(y_test, DecTree.predict(x_test))    
    
	Recall = recall_score(y_test, DecTree.predict(x_test))
    
	Prec = precision_score(y_test, DecTree.predict(x_test))

	return Acc, Prec, Recall


def random_forest_train_test(x_train, x_test, y_train, y_test):
	# To-Do: Implement this function

	RandForest = RandomForestClassifier()
    
	RandForest.fit(x_train, y_train)

	Acc = accuracy_score(y_test, RandForest.predict(x_test))
    
	Recall = recall_score(y_test, RandForest.predict(x_test))
    
	Prec = precision_score(y_test, RandForest.predict(x_test))

	return Acc, Prec, Recall


def svm_train_test(x_train, x_test, y_train, y_test):
	# To-Do: Implement this function

	pipe = make_pipeline(
		StandardScaler(),
		SVC()
	)
    
	pipe.fit(x_train, y_train)

	Acc = accuracy_score(y_test, pipe.predict(x_test))
    
	Recall = recall_score(y_test, pipe.predict(x_test))
    
	Prec = precision_score(y_test, pipe.predict(x_test))

	return Acc, Prec, Recall


def print_performances(acc, prec, recall):
	#Do not modify this function!
	print("Accuracy: ", acc)
	print("Precision: ", prec)
	print("Recall: ", recall)


if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print("Number of features: ", n_feats)
	print("Number of class 0 data entries: ", n_class0)
	print("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)