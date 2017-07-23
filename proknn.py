import pandas as pd
import math

with open("PlayerStats.csv", "r") as csvfile:
	pkl = pd.read_csv(csvfile)

# Select Anup Kumar from our dataset
selected_player = pkl[pkl["Player"] == "Anup Kumar"].iloc[0]

# Choose only the numeric columns (we'll use these to compute euclidean distance)
distance_columns = ['TotalMatches', 'TotalPoints', 'TotalRaidPoints', 'TotalDefencePoints', 'TotalRaids', 'SuccRaids', 'UnSuccRaids', 'EmptyRaids', 'Tackles', 'SuccTackles', 'UnSuccTackles', 'GreenCards', 'RedCards', 'YellowCards']

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
print(anup_kumar_distance)

'''
# Select only the numeric columns from the ProKabaddi dataset
pkl_numeric = pkl[distance_columns]

# Normalize all of the numeric columns
pkl_normalized = (pkl_numeric - pkl_numeric.mean()) / pkl_numeric.std()

print(pkl_normalized)
'''
