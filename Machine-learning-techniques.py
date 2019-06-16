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
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import datetime as dt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier

#Getting the data
images, proba, types = load_dataset()

images_flat = []
for i in images:
  images_flat.append(i.flatten())

labels = []
for i in proba:
  if i <0.5 and i>0:
    labels.append(0.0)
  elif i<1 and i> 0.5:
    labels.append(1.0)
  else:
    labels.append(i)
    
lab_enc = preprocessing.LabelEncoder()
Y = lab_enc.fit_transform(labels)

# Normalization
scaler = MinMaxScaler()

pca = PCA(n_components=256)
pca.fit(scaled_data)
X = pca.transform(scaled_data)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=42)

def results(labels, pred):
  print(confusion_matrix(labels,pred))  
  print(classification_report(labels,pred))  
  print(accuracy_score(labels, pred))
scaled_data = scaler.fit_transform(images_flat)

for c in [0.01]:
  #changing the parameter C to get the optimal classification
  
  lr = LogisticRegression(C=c, solver='liblinear',max_iter=1000)
  start_time = dt.datetime.now()
  #print('Start learning at {}'.format(str(start_time)))
  #multi_class='multinomial')
  lr.fit(X_train, y_train)
  #print('Stop learning {}'.format(str(end_time)))
  end_time = dt.datetime.now() 
  elapsed_time= end_time - start_time
  print('Elapsed learning {}'.format(str(elapsed_time)))
  start_time_test = dt.datetime.now()
  print ("Accuracy of logistic regression for C=%s: %s" 
          % (c, accuracy_score(y_test, lr.predict(X_test))))
  end_time_test = dt.datetime.now() 
  elapsed_time_test = end_time_test - start_time_test
  print('Elapsed testing {}'.format(str(elapsed_time_test)))
  results(y_test, lr.predict(X_test))
  print(accuracy_score(y_train,lr.predict(X_train)))
  
for c in [5]:
  #changing the parameter C to get the optimal classification
  model = LinearSVC(C=c, max_iter = 50000)
  start_time = dt.datetime.now()
  model.fit(X_train,y_train)
  end_time = dt.datetime.now() 
  elapsed_time= end_time - start_time
  print('Elapsed learning {}'.format(str(elapsed_time)))
  start_time_test = dt.datetime.now()
  print ("Accuracy of SVM for C=%s: %s" 
          % (c, accuracy_score(y_test, model.predict(X_test))))
  end_time_test = dt.datetime.now() 
  elapsed_time_test = end_time_test - start_time_test
  print('Elapsed testing {}'.format(str(elapsed_time_test)))
  results(y_test, model.predict(X_test))
  print(accuracy_score(y_train,model.predict(X_train)))

for c in [5]:
  #changing the parameter C to get the optimal classification
  model = SVC(C=c, gamma = 'auto', kernel = 'sigmoid')
  start_time = dt.datetime.now()
  model.fit(X_train,y_train)
  end_time = dt.datetime.now() 
  elapsed_time= end_time - start_time
  print('Elapsed learning {}'.format(str(elapsed_time)))
  start_time_test = dt.datetime.now()
  print ("Accuracy of rbf for C=%s: %s" 
          % (c, accuracy_score(y_test, model.predict(X_test))))
  end_time_test = dt.datetime.now() 
  elapsed_time_test = end_time_test - start_time_test
  print('Elapsed testing {}'.format(str(elapsed_time_test)))
  results(y_test, model.predict(X_test))
  print(accuracy_score(y_train,model.predict(X_train)))

model = BernoulliNB()
start_time = dt.datetime.now()
#model.fit(X_train,y_train)
model.fit(X_train, y_train)
end_time = dt.datetime.now() 
elapsed_time= end_time - start_time
print('Elapsed learning {}'.format(str(elapsed_time)))
start_time_test = dt.datetime.now()
print ("Accuracy of CNB for C=%s: %s" 
          % (c, accuracy_score(y_test, model.predict(X_test))))
end_time_test = dt.datetime.now() 
elapsed_time_test = end_time_test - start_time_test
print('Elapsed testing {}'.format(str(elapsed_time_test)))
results(y_test, model.predict(X_test))
print(accuracy_score(y_train,model.predict(X_train)))

scores = {}
#scores_list = []
for k in range(5,6):
  model = KNeighborsClassifier(n_neighbors=k)
  start_time = dt.datetime.now()
  model.fit(X_train,y_train)
  end_time = dt.datetime.now() 
  elapsed_time= end_time - start_time
  print('Elapsed learning {}'.format(str(elapsed_time)))
  start_time_test = dt.datetime.now()
  print ("Accuracy of rbf for K=%s: %s" 
          % (k, accuracy_score(y_test, model.predict(X_test))))
  end_time_test = dt.datetime.now() 
  elapsed_time_test = end_time_test - start_time_test
  print('Elapsed testing {}'.format(str(elapsed_time_test)))
  results(y_test, model.predict(X_test))
  print(accuracy_score(y_train,model.predict(X_train)))
