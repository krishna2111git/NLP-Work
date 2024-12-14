import json
import os
import pandas as pd
from definitions import *
from gradient_descent import *




vocab = pd.Series("Vocabulary/vocab.csv", header=0)
vocab_size = len(vocab)
vec_size = 50

with open(r"Data_Preprocessing/corpus_voc_indices_list.json", 'r') as file:
    index_corpus = json.load(file)


#Subsampling the corpus
word_prob, orig_corpus_word_counts, orig_corpus_len = words_probs(index_corpus, vocab_size)
index_corpus = subsample_corpus(word_prob, index_corpus)



#Initializing word Vectors
center_vectors = initialize_word_vectors(vocab_size, vec_size)
context_vectors = initialize_word_vectors(vocab_size, vec_size)

#Calculate word probability for negative sampling
smoothed_word_prob = (orig_corpus_word_counts**0.75)/orig_corpus_len

#Window size of the context words
window_radius = 3

#List of set of words for each center word
training_word_sets = []
#Iterating over each word in the corpus
for article in index_corpus:
    for sentence in index_corpus[article]:
        for index, center_index in enumerate(sentence):
            #Selecting the context words for the current word
            context_words = select_context_words(index, sentence, window_radius)

            #Selecting Negative Samples for the current word
            negative_samples = get_negative_samples(context_words, smoothed_word_prob, num_samples=3)

            training_set = {"center_idx" : center_index,
                            "context_words" : context_words,
                            "neg_samples" : negative_samples}
            
            training_word_sets.append(training_set)


#Iterate over the training set
for set in training_word_sets :
    #Get the indices of the set of words
    center_idx = set["center_idx"]
    context_idxs = set["context_words"]
    neg_sample_idxs = set["neg_samples"]

    #Fetch the word vectors from their indices
    center_vec = center_vectors[center_idx]
    context_vecs = context_vectors[context_idxs]
    neg_sample_vecs = context_vectors[neg_sample_idxs]

    #Iterate over each positive context word and optimize it and rest of the words in the set 
    for context_vec, context_idx in zip(context_vecs, context_idxs):
        #Optimize the set of center, ps context word, neg context words
        new_center_vec, new_context_vec, new_neg_sample_vecs = train_skip_gram(center_vec, context_vecs, neg_sample_vecs, learning_rate=0.01, 
                                                    max_iterations=1000, tolerance=1e-6)
        
        #Assigning the new optimized vectors to the embedding matrices
        center_vectors[center_idx] = new_center_vec
        context_vectors[context_idx] = new_context_vec
        context_vectors[neg_sample_idxs] = new_neg_sample_vecs


#Save the optimize embeddings
os.makedirs("Embeddings", exist_ok=True)

pd.DataFrame(center_vectors, index=vocab.to_list()).to_json("Embeddings/Center_Vectors.json", orient="split", indent=1)
pd.DataFrame(context_vectors, index=vocab.to_list()).to_json("Embeddings/Context_Vectors.json", orient="split", indent=1)