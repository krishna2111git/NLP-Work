#Run the python script Corpus_Preprocessing to preprocess the articles and create a vocabulary
python Corpus_Preprocessing.py

#Read the preprocessed corpus, vocabulary perform the pre SGD tasks like sub-sampling corpus,
#negative sampling, creating sets of training words and finaly performing the NN SGD
python app.py