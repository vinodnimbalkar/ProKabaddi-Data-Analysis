import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

# define column names
distance_columns = ['TotalMatches', 'TotalPoints', 'TotalRaidPoints', 'TotalDefencePoints', 'TotalRaids', 'SuccRaids',
                    'UnSuccRaids', 'EmptyRaids', 'Tackles', 'SuccTackles', 'UnSuccTackles', 'GreenCards', 'RedCards', 'YellowCards', 'PlayerType']


with open("PlayerStats.csv", "r") as csvfile:
    pkl = pd.read_csv(csvfile)


# create design matrix X and target vector y
X = np.array(pkl.ix[:, 4:17]) 	# end index is exclusive
y = np.array(pkl['PlayerType']) 	# another way of indexing a pandas pkl

# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
pred = knn.predict(X_test)

# evaluate accuracy
print(accuracy_score(y_test, pred))

# creating odd list of K for KNN
myList = list(range(1,50))

# subsetting just the odd ones
neighbors = filter(lambda x: x % 2 != 0, myList)

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=2, scoring='accuracy')
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()


def train(X_train, y_train):
	# do nothing 
	return

def predict(X_train, y_train, x_test, k):
	# create list for distances and targets
	distances = []
	targets = []

	for i in range(len(X_train)):
		# first we compute the euclidean distance
		distance = np.sqrt(np.sum(np.square(x_test - X_train[i, :])))
		# add it to list of distances
		distances.append([distance, i])

	# sort the list
	distances = sorted(distances)

	# make a list of the k neighbors' targets
	for i in range(k):
		index = distances[i][1]
		targets.append(y_train[index])

	# return most common target
	return Counter(targets).most_common(1)[0][0]

def kNearestNeighbor(X_train, y_train, X_test, predictions, k):
	# train on the input data
	train(X_train, y_train)

	# loop over all observations
	for i in range(len(X_test)):
		predictions.append(predict(X_train, y_train, X_test[i, :], k))

# making our predictions 
predictions = []

try:
	kNearestNeighbor(X_train, y_train, X_test, predictions, 7)

	# transform the list into an array
	predictions = np.asarray(predictions)

	# evaluating accuracy
	accuracy = accuracy_score(y_test, predictions)
	print('\nThe accuracy of our classifier is %d%%' % accuracy*100)

except ValueError:
	print('Can not be more neighbors than training samples!!')