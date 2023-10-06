from sklearn import svm

# Sample data
X = [[0, 0], [1, 1]]
y = [0, 1]

# Create an SVM classifier
clf = svm.SVC()

# Fit the model to the data
clf.fit(X, y)

# Make predictions
predictions = clf.predict([[2, 2]])

print("SVM Predicts:", predictions[0])
