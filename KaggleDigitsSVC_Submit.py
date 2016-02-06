# This program is for in the Kaggle Digit Recognizer competition. This program
# uses the test set for prediction, building predictions.csv as the output 
# file containing the predictions for the test data. The output file is 
# formatted for submission to Kaggle.
#
# The basic algorithm is as follows:
#    1. Read in and normalize our data, setting some aside for testing.
#    2. In order to reduce the number of features, Principal Component Analysis
#        Decomposition is performed on the training data.
#    3. With a reduced number of features, a Support Vector Classification
#        model is trained on the training data.
#    4. The model is then used to predict labels for the test set of data.
#
#
# Information on the Kaggle competition can be found here:
#    https://www.kaggle.com/c/digit-recognizer
#
# Walter Schwenger - wjs018@gmail.com

import numpy as np
import pandas
import csv

from sklearn import svm, decomposition

print('Reading in training data...')

# Read in our training data

train_data_df = pandas.read_csv('train.csv', header = 0)

# Separate labels and data

train_labels = train_data_df.ix[:,0:1].copy()
train_data = train_data_df.ix[:,1:].copy()

# Normalize images so that there is a consistent range in values for pixels

print('Normalizing training data...')

# Normalize by dividing each row by the maximum in that row

maxs = train_data.max(axis=1)
train_data = train_data.div(maxs, axis=0)

# Next, we want to do some PCA decomposition to reduce the dimensionality

print('Decomposing...')

# The number of principal components to reduce the feature set to. The larger
# the number of components, the more accurate prediction will be. However, 
# execution will take much longer. I found 50 was a decent sweet spot with
# diminishing returns beyond this point.

num_components = 50

# Create our PCA operator and calculate the components

pca = decomposition.PCA(n_components=num_components, whiten=True)
pca.fit(train_data.as_matrix())

# Transform our data into the features calculated via PCA

transformed = pca.transform(train_data.as_matrix())

# Now we are prepared for the Support Vector Classifier

print('Training the SVC...')

# Create our SVC classifier

classifier = svm.SVC()

# Train our model

classifier.fit(transformed, np.ravel(train_labels.as_matrix()))

# Now that our model is trained, we are ready to move on to testing

print('Reading in test data...')

# Read in our test data

test_data_df = pandas.read_csv('test.csv', header = 0)

# Normalize the test data

print('Normalizing test data...')

# Normalization is the same procedure as the training data

maxs_test = test_data_df.max(axis=1)
test_data_df = test_data_df.div(maxs_test, axis=0)

# The test data is now normalized and ready for prediction

print('Predicting with the SVC...')

# First, we need to reduce the dimensionality of the test data with the PCA we
# calculated from the training data.

test_transformed = pca.transform(test_data_df.as_matrix()[:,:])

# Now, build our list of predictions

predicted = classifier.predict(test_transformed)

# Write our predictions to predictions.csv

print('Writing to predictions.csv...')

# Need a header for our columns

header = ['ImageID','Label']

# Write the predictions.csv file

with open('predictions.csv', 'wb') as f:
    writer = csv.writer(f)
    
    for i in range(0,len(predicted.tolist())+1):
        if i == 0:
            writer.writerow(header)
        else:
            writer.writerow([i, predicted.tolist()[i-1]])