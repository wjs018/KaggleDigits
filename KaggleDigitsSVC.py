# This program is for being able to measure model accuracy in the Kaggle Digit
# Recognizer competition. This program only uses the training set, subsetting
# part of the data to be used as a test set. This lets the user get instant
# feedback as to the accuracy of the algorithm and more quickly tweak it for
# improvement.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas

from sklearn import svm, metrics, decomposition


print('Reading in data...')

# Fraction of data to train on, 1-FRACTION will be tested

FRACTION = 9.0 / 10.0

# Read in our training data

train_data_df = pandas.read_csv('train.csv', header = 0)

# Build our labels and data separately

train_labels = train_data_df.ix[:,0:1].copy()
train_data = train_data_df.ix[:,1:].copy()

# Display the first few images with labels

display_images = []
display_labels = []

for i in range(0,4):
    
    display_images.append(train_data.as_matrix()[i,:].reshape(28,28))
    display_labels.append(train_labels.as_matrix()[i,:])
    
    # Plot images and labels
    
    plt.subplot(2, 4, i + 1)
    plt.axis('off')
    plt.imshow(display_images[i], cmap=plt.get_cmap('gray'), interpolation='nearest')
    plt.title('Training: %i' % display_labels[i])
    
# Need to get a total number of images

num_images = train_labels.shape[0]

# Normalize images so that there is a consistent range in values for pixels

print('Normalizing...')

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

pca_components = decomposition.PCA(n_components=num_components, whiten=True)
pca_components.fit(train_data.as_matrix()[:num_images*FRACTION,:])

# Transform our data into the features calculated via PCA

transformed = pca_components.transform(train_data.as_matrix()[:num_images*FRACTION,:])

# Now we are prepared for the Support Vector Classifier

print('Training the SVC...')

# Create our SVC classifier

classifier = svm.SVC()

# Train our model

classifier.fit(transformed, np.ravel(train_labels.as_matrix()[:num_images*FRACTION,:]))

# After training our model, we are ready to test its accuracy with the test data

print('Predicting with the SVC...')

# Now, predict the rest of the train data so that we can score our model making
# sure that the test data is decomposed via PCA like before

expected = train_labels.as_matrix()[num_images*FRACTION:,:]
test_transformed = pca_components.transform(train_data.as_matrix()[num_images*FRACTION:,:])
predicted = classifier.predict(test_transformed)

# Print report

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
    
# Plot and label some predicted results

predicted_images = []
predicted_labels = []

for i in range(0,4):
    
    predicted_images.append(train_data.as_matrix()[num_images*FRACTION + i,:].reshape(28,28))
    predicted_labels.append(predicted[i])
    
    # Plot the predictions
    
    plt.subplot(2, 4, i + 5)
    plt.axis('off')
    plt.imshow(predicted_images[i], cmap=plt.get_cmap('gray'), interpolation='nearest')
    plt.title('Prediction: %i' % predicted_labels[i])

# Show the plot finally

plt.show()