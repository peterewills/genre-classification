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
