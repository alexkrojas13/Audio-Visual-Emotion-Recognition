'''
The data has 286 rows and 146 columns.
The classes:  {1, 2, 3, 4, 5, 6}
The accuracy on the test dataset is 0.724.
The average cross-validation accuracy over 4 folds is 0.779 with standard deviation of 0.043.
'''

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score 

# load the data
df = pd.read_csv('C:\data\emotion_data.csv')
# the data shape
print('The data has %d rows and %d columns.' % df.shape)

# the features
x = df.iloc[:,:-1].values
# the class labels
y = df.iloc[:,-1].values

# the classes
classes=set(list(y))
print('The classes: ', classes)

# split the data into training and test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# standardize the features
std_scaler = StandardScaler()
x_train_std = std_scaler.fit_transform(x_train)
x_test_std = std_scaler.transform(x_test)

# the classifier
svm = SVC(C=10.0, kernel='linear', random_state=0)

# train the model
svm = svm.fit(x_train_std, y_train)

# prediction on the test set
y_pred = svm.predict(x_test_std)

# evaluation
accuracy = accuracy_score(y_test, y_pred)
print('The accuracy on the test dataset is %.3f.' % accuracy)

# cross validation accuracy
x_tot_std = np.vstack((x_train_std, x_test_std))
y_tot = np.vstack((y_train.reshape((y_train.shape[0],1)), y_test.reshape((y_test.shape[0],1))))
scores = cross_val_score(svm, x_train_std, y_train, cv=4)
print('The average cross-validation accuracies over %d folds is %.3f with standard deviation of %.3f.' 
      % (4, scores.mean(), scores.std()))


