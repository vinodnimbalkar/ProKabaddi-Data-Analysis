import pandas as pd
import math
from scipy.spatial import distance
import random
from numpy.random import permutation
from sklearn.neighbors import KNeighborsClassifier

with open("data/PlayerStats.csv", "r") as csvfile:
    pkl = pd.read_csv(csvfile)

# Select Anup Kumar from our dataset
selected_player = pkl[pkl["Player"] == "Anup Kumar"].iloc[0]

# Choose only the numeric columns (we'll use these to compute euclidean
# distance)
distance_columns = ['TotalMatches', 'TotalPoints', 'TotalRaidPoints', 'TotalDefencePoints', 'TotalRaids', 'SuccRaids',
                    'UnSuccRaids', 'EmptyRaids', 'Tackles', 'SuccTackles', 'UnSuccTackles', 'GreenCards', 'RedCards', 'YellowCards']

def euclidean_distance(row):
    """
    A simple euclidean distance function
    """
    inner_value = 0
    for k in distance_columns:
        inner_value += (row[k] - selected_player[k]) ** 2
    return math.sqrt(inner_value)

# Find the distance from each player in the dataset to Anup Kumar.
anup_kumar_distance = pkl.apply(euclidean_distance, axis=1)
# print(anup_kumar_distance)

# Select only the numeric columns from the ProKabaddi dataset
pkl_numeric = pkl[distance_columns]

# Normalize all of the numeric columns
pkl_normalized = (pkl_numeric - pkl_numeric.mean()) / pkl_numeric.std()

# Fill in NA values in pkl_normalized
pkl_normalized.fillna(0, inplace=True)

# Find the normalized vector for Anup Kumar.
anup_normalized = pkl_normalized[pkl["Player"] == "Anup Kumar"]

# Find the distance between Anup Kumar and everyone else.
euclidean_distances = pkl_normalized.apply(
    lambda row: distance.euclidean(row, anup_normalized), axis=1)

# Create a new dataframe with distances.
df = pd.DataFrame(data={"dist": euclidean_distances,
                        "idx": euclidean_distances.index})
df.sort_values("dist", inplace=True)
# Find the most similar player to Anup Kumar (the lowest distance to Anup
# is Anup, the second smallest is the most similar player)
second_smallest = df.iloc[1]["idx"]
most_similar_to_anup = pkl.loc[int(second_smallest)]["PlayerType"]
print(most_similar_to_anup)
             

# Randomly shuffle the index of ProKabaddi.
random_indices = permutation(pkl.index)
# Set a cutoff for how many items we want in the test set (in this case
# 1/3 of the items)
test_cutoff = math.floor(len(pkl) / 3)
# Generate the test set by taking the first 1/3 of the randomly shuffled
# indices.
test = pkl.loc[random_indices[1:test_cutoff]]
# Generate the train set with the rest of the data.
train = pkl.loc[random_indices[test_cutoff:]]
# Get the actual values for the test set.
actual = test[distance_columns]

# Accuracy calculation function
def accuracy(test_data):
    correct = 0
    for i in (test_data):
        if i[2] == i[3]:
            correct += 1
        accuracy = float(correct) / len(test_data) * 100  # accuracy
    return accuracy

K = 5                                          # Assumed K value
#knn_predict(test, train, K)
print("Accuracy : ",accuracy(test))

