# Importing pandas
import pandas as pd
# Importing training data set
X_train=pd.read_csv('X_train.csv')
Y_train=pd.read_csv('Y_train.csv')
# Importing testing data set
X_test=pd.read_csv('X_test.csv')
Y_test=pd.read_csv('Y_test.csv')

print (X_train.head())

import matplotlib.pyplot as plt
X_train[X_train.dtypes[(X_train.dtypes=="float64")|(X_train.dtypes=="int64")]
                        .index.values].hist(figsize=[11,11])
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train[['EmptyRaids', 'SuccRaids','SuccTackles', 
                   'Tackles', 'TotalDefencePoints','TotalMatches','TotalPoints',
		   'TotalRaids','UnSuccRaids','UnSuccTackles']],Y_train)
# Checking the performance of our model on the testing data set
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,knn.predict(X_test[['EmptyRaids', 'SuccRaids','SuccTackles', 
                   'Tackles', 'TotalDefencePoints','TotalMatches','TotalPoints',
		   'TotalRaids','UnSuccRaids','UnSuccTackles']]))

Y_train.PlayerType.value_counts()/Y_train.PlayerType.count()

########################################################
from sklearn.preprocessing import MinMaxScaler
min_max=MinMaxScaler()
# Scaling down both train and test data set
X_train_minmax=min_max.fit_transform(X_train[['EmptyRaids', 'SuccRaids','SuccTackles', 
                   'Tackles', 'TotalDefencePoints','TotalMatches','TotalPoints',
		   'TotalRaids','UnSuccRaids','UnSuccTackles']])
X_test_minmax=min_max.fit_transform(X_test[['EmptyRaids', 'SuccRaids','SuccTackles', 
                   'Tackles', 'TotalDefencePoints','TotalMatches','TotalPoints',
		   'TotalRaids','UnSuccRaids','UnSuccTackles']])

# Fitting k-NN on our scaled data set
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_minmax,Y_train)
# Checking the model's accuracy
accuracy_score(Y_test,knn.predict(X_test_minmax))

#########################################################
# Standardizing the train and test data
from sklearn.preprocessing import scale
X_train_scale=scale(X_train[['EmptyRaids', 'SuccRaids','SuccTackles', 
                   'Tackles', 'TotalDefencePoints','TotalMatches','TotalPoints',
		   'TotalRaids','UnSuccRaids','UnSuccTackles']])
X_test_scale=scale(X_test[['EmptyRaids', 'SuccRaids','SuccTackles', 
                   'Tackles', 'TotalDefencePoints','TotalMatches','TotalPoints',
		   'TotalRaids','UnSuccRaids','UnSuccTackles']])
# Fitting logistic regression on our standardized data set
from sklearn.linear_model import LogisticRegression
log=LogisticRegression(penalty='l2',C=.01)
log.fit(X_train_scale,Y_train)
# Checking the model's accuracy
accuracy_score(Y_test,log.predict(X_test_scale))
#####################################################