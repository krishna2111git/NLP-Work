# Overview of what we are going to do:
We will create word embeddings for all unique words in a wikipedia article. Mind you, no web-scraping in this tutorial.

# Code Files in our project:
Corpus_Preprocessing.py : This is a script that applies pre-processing on the corpus text and creates vocabulary
Method_Definitions.py : It’s a module that contains method definitions for tasks other than performing gradient descent.
SGD_nn.py : This contains method that will perform gradient descent using PyTorch neural network
app.py : Main execution script for skip-gram process, where all the methods are called
