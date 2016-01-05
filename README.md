# genre-classification
Machine learning genre classification of musical tracks

This library contains a simple algorithm that uses a dimension-reduced Gaussian Naive Bayes classifier to discern the genre of a given musical track. This is an 'agnostic' classifier, in the sense that it does not use any particular musical feature recognition. It only looks at the .wav file, mapped into frequency space via the [MFCC transform](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/). 

Please note that the [python_speech_features](https://github.com/jameslyons/python_speech_features) library must be installed.

The folder that contains the training data should also contain a .csv file that contains filenames in its first column and associated genres in its second.

To use this classifier,

```python
from genre_class import *

# Define paths to folders containing tracks to train on and tracks to be classified
query_path = '/path/to/queries'
training_path = '/path/to/training/data'

# Build collection of Track objects with genres given
training_collection,genre_set = train(training_path)

# Build collection of Track objects with genres guessed by classifier
query_collection = classify(query_path,training_collection,genre_set)

# Print results
for track in query_collection:
    print track.path +', '+ track.genre
```
