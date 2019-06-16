# The data repository
#!git clone https://github.com/zae-bayern/elpv-dataset.git

# Importing necessary libraries
import sys
from sklearn import preprocessing
from sklearn import utils
from sklearn.decomposition import PCA
import numpy as np
from elpv_reader import load_dataset
from sklearn.model_selection import train_test_split
from sklearn import datasets,svm, metrics
from sklearn.svm import LinearSVC
import datetime as dt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler

#Getting the data
images, proba, types = load_dataset()
