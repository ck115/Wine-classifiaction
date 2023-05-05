import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Load the wine dataset from a CSV file
wine_df = pd.read_csv('winequality-red.csv')

# Separate the features (X) and the target variable (y)
X = wine_df.drop(['fixed acidity', 'chlorides', 'density', 'total sulfur dioxide'], axis=1)
y = wine_df['quality']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a KNN classifier with k=2 on the training data
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Compute the accuracy of the predictions
prediction_accuracy = accuracy_score(y_test, y_pred)
print("Prediction Accuracy: {:.2f}%".format(prediction_accuracy*100))

# Compute cross-validation scores for different values of k
k_values = [i for i in range(1, 31)]
scores = []

scaler = StandardScaler()
X = scaler.fit_transform(X)

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    score = cross_val_score(knn, X, y, cv=5)
    scores.append(np.mean(score))

# Plot the cross-validation scores vs. k
plt.plot(k_values, scores, marker='o')
plt.xlabel("K values")
plt.ylabel("Accuracy score")

# Choose the value of k that gave the highest cross-validation score
best_index = np.argmax(scores)
best_k = k_values[best_index]

# Train a KNN classifier with the best value of k on the training data
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Compute scores
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=1)

# Print
print("Accuracy: {:.2f}%".format(accuracy*100))
print("Precision: {:.2f}%".format(precision*100))
print("Recall: {:.2f}%".format(recall*100))
print("F1 score: {:.2f}%".format(f1*100))

plt.show()
