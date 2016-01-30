# Kaggle Digit Recognizer Submission

These programs utilize machine learning algorithms to classify hand written numbers by digit. If you want more information about the data and purpose, as well as where the data can be downloaded, head to the [Kaggle competition page](https://www.kaggle.com/c/digit-recognizer).

## Files

* `KaggleDigitsSVC.py`
  * This file only requires training data as it partitions the training data into a training set and test set so that prediction performance can be scored.
  * The algorithm implemented here consists of:
    1. Training data is read in from file.
    2. Images are normalized in intensity.
    3. Images are decomposed into principal components to reduce feature size.
    4. An SVC model is trained on the training data.
    5. Model is used to predict the test set.
    6. Predictions are scored for accuracy and results printed to terminal.
* `KaggleDigitsSVC_Submit.py`
  * This file requires both the training and test data.
  * The algorithm implemented here is the same as `KaggleDigitsSVC.py` with the exception of model scoring.
  * The predictions are output in a .csv file that is formatted correctly to be submitted to Kaggle.

## Performance

* `KaggleDigitsSVC-Submit.py` - **0.98229** out of possible 1.0
  * using 50 principal components

## Contact

* Walter Schwenger, wjs018@gmail.com
