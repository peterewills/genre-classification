# genre-classification
Machine learning genre classification of musical tracks

This library contains a simple algorithm that uses a dimension-reduced Gaussian Naive Bayes classifier to discern the genre of a given musical track. This is an 'agnostic' classifier, in the sense that it does not use any particular musical feature recognition. It only looks at the .wav file, mapped into frequency space via the [MFCC transform](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/). 

To use this classifier,

```python
from features import mfcc
from features import logfbank
import scipy.io.wavfile as wav

(rate,sig) = wav.read("file.wav")
mfcc_feat = mfcc(sig,rate)
fbank_feat = logfbank(sig,rate)

print fbank_feat[1:3,:]
```

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


