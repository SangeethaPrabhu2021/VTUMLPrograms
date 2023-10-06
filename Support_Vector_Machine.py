#10.Demonstrate the working of SVM classifier for a suitable data set

import pandas as pd
dataset = pd.read_csv('10iris.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
decision_function_shape='ovr', degree=3, gamma='auto_deprecated',
kernel='linear', max_iter=-1, probability=False, random_state=0,
shrinking=True, tol=0.001, verbose=False)

from sklearn.metrics import confusion_matrix, accuracy_score

y_pred = classifier.predict(X_test)
from sklearn import metrics
print('Accuracy metrics')
print('Accuracy of the SVM is',metrics.accuracy_score(y_test,y_pred))
print('Confusion matrix')
print(metrics.confusion_matrix(y_test,y_pred))
print('Recall and Precison ')
print(metrics.recall_score(y_test,y_pred,average='weighted'))
print(metrics.precision_score(y_test,y_pred,average='weighted'))