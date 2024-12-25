import json
import re
import os

import pandas as pd


#Read the articles from text files
directory_path = "Articles"
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
articles_dict = {}
for file in files:
    with open(os.path.join(directory_path,file), 'r', encoding="utf-8") as f:
        articles_dict[file[:-4]]=f.read()
        
#break down each article into a list of sentences
sentence_lists = {}
for page_title in articles_dict:
    article_text = articles_dict[page_title]
    list_of_article_sentences = list(article_text.split(". "))
    sentence_lists[page_title] = list_of_article_sentences

#preprocssing the texts in the article
processed_sentence_lists = {}
for article, sentences in zip(sentence_lists.keys(), sentence_lists.values()):
    processed_sentences = []
    for sentence in sentences:
        processed_sentence = re.sub(r"\b\w*\d\w*\b", "", sentence) #removes all words containing numbers
        processed_sentence = re.sub(r'[^a-zA-Z\s]', " ", processed_sentence) #keeps only words composed of pure english letters
        processed_sentence = re.sub(r'\s+', " ", processed_sentence)#remove the extra spaces included by the last step
        processed_sentence = processed_sentence.lower().strip() # convert all text to lower case
        processed_sentences.append(processed_sentence)
    processed_sentence_lists[article] = processed_sentences

#Converting the sentences of articles into list of words
corpus_words_list = {}
for article, sentences in zip(processed_sentence_lists.keys(), processed_sentence_lists.values()):
    list_of_sentences = []
    for sentence in sentences:
        list_of_words = sentence.split()
        list_of_words = [word.strip() for word in list_of_words]
        list_of_sentences.append(list_of_words)
    corpus_words_list[article] = list_of_sentences


#Creating vocabulary from the words in corpus

list_of_sets_in_an_article = []
list_of_sets_in_entire_corpus = []
for article in corpus_words_list:
    for sentence in corpus_words_list[article]:
        set_of_unique_words_in_a_sentence = set(sentence)
        list_of_sets_in_an_article.append(set_of_unique_words_in_a_sentence)
    set_of_words_in_an_article = set.union(*list_of_sets_in_an_article)
    list_of_sets_in_entire_corpus.append(set_of_words_in_an_article)
vocab = list(set.union(*list_of_sets_in_entire_corpus))

vocab = sorted(vocab)


#Saving the vocabulary to CSV file
os.makedirs("Vocabulary", exist_ok=True)
vocab_series = pd.Series(vocab, name="Words")
vocab_series.to_csv("Vocabulary/vocab.csv", index=False)

#Create the corpus of the word indices
corpus_voc_indices_list = {}
for article in corpus_words_list:
    list_of_sentences = []
    for sentence in corpus_words_list[article]:
        words_in_sentence = []
        for word in sentence:
            word_index = vocab.index(word)
            words_in_sentence.append(word_index)
        list_of_sentences.append(words_in_sentence)
    corpus_voc_indices_list[article] = list_of_sentences


#saving the indices corpus
os.makedirs("Data_Preprocessing", exist_ok=True)
with open(r"Data_Preprocessing/corpus_voc_indices_list.json", 'w') as file:
    json.dump(corpus_voc_indices_list, file, indent=4)



        
