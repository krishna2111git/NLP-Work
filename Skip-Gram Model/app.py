import json
import os
import pandas as pd
from Method_Definitions import *
from SGD_nn import *



vocab = pd.read_csv("Vocabulary/vocab.csv", header=0)['Words']
vocab_size = len(vocab)
vec_size = 50

with open(r"Data_Preprocessing/corpus_voc_indices_list.json", 'r') as file:
    index_corpus = json.load(file)


#Subsampling the corpus
word_prob, orig_corpus_word_counts, orig_corpus_len = words_probs(index_corpus, vocab_size)
index_corpus = subsample_corpus(word_prob, index_corpus)


#Calculate word probability for negative sampling
smooth_word_counts = (orig_corpus_word_counts**0.75)
smooth_word_probs = smooth_word_counts/smooth_word_counts.sum()

#Window size of the context words
window_radius = 10

#List of set of words for each center word
training_word_sets = []
#Iterating over each word in the corpus
for article in index_corpus:
    for sentence in index_corpus[article]:
        for index, center_index in enumerate(sentence):
            #Selecting the context words for the current word
            context_words = select_context_words(index, sentence, window_radius)

            #Selecting Negative Samples for the current word
            negative_samples = get_negative_samples(context_words, smooth_word_probs, num_samples=10)

            for context_word in context_words:
                a_training_set = (center_index, context_word, negative_samples)
                training_word_sets.append(a_training_set)
            
            

#Create the instance of the Embedding Model
embedding_model = EmbeddingModel(vocab_size, vec_size)

#Call the gradient descent method
train_skip_gram(training_word_sets, embedding_model, learning_rate=0.01, epochs=10, batch_size=256)

# Save the model's embeddings
os.makedirs("Embeddings", exist_ok=True)

torch.save(embedding_model.center_embeddings.weight, "Embeddings/center_embeddings.pt")
torch.save(embedding_model.context_embeddings.weight, "Embeddings/context_embeddings.pt")

