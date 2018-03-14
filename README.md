# genre-classification

This library contains a simple algorithm that uses a dimension-reduced Gaussian Naive Bayes classifier to discern the genre of a given musical track. This is an 'agnostic' classifier, in the sense that it does not use any particular musical feature recognition. It only looks at the .wav file, mapped into frequency space via the [MFCC transform](http://www.practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/). The machine-learning core of this algorithm uses [scikit-learn](http://scikit-learn.org/stable/).

Please note that the [python_speech_features](https://github.com/jameslyons/python_speech_features) library must be installed.

The folder that contains the training data should also contain a .csv file that contains filenames in its first column and associated genres in its second.

 
## Usage


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

This is meant to function as an exploration of dimension reduction and machine learning rather than a maximally effective genre classification scheme. A truly effective scheme would strategically employ musical features in classification (see, for example, [Barbedo 07](http://www.asp.eurasipjournals.com/content/pdf/1687-6180-2007-064960.pdf)). A report that discusses the methods used in the algorithm (as well as some alternatives) can be found [here](https://drive.google.com/file/d/0B_uQ2Fw3yuKWZzNoNndLUzRoRVE/view?usp=sharing).

## Credits
 
Author: Peter Wills (peter@pwills.com)
 
## License
 
The MIT License (MIT)

Copyright (c) 2017 Peter Wills

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
