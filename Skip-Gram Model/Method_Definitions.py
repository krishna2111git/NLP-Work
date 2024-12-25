import random
import numpy as np





def words_probs(index_corpus, vocab_size):
    list_of_words = []
    for article in index_corpus:
        for sentence in index_corpus[article]:
            for word_index in sentence:
                    list_of_words.append(word_index)

    count_of_words = [list_of_words.count(i) for i in range(vocab_size)]
    total_words_in_corpus = sum(count_of_words)
    prob_of_words = [word_count/total_words_in_corpus for word_count in count_of_words]
    return np.array(prob_of_words), np.array(count_of_words), total_words_in_corpus





def subsample_corpus(words_probs : list, index_corpus: dict, t=1e-3):
    words_retain_probs = []
    for word_prob in words_probs:
        retain_prob = min(1, (t / word_prob)**0.5 + (t / word_prob))
        words_retain_probs.append(retain_prob)
    

    new_index_corpus = {}
    for article in index_corpus:
        list_of_sentences = []
        list_of_sentences_set = []
        for sentence in index_corpus[article]:
            list_of_words = []
            for word_index in sentence:
                if random.random()<words_retain_probs[word_index]:
                    list_of_words.append(word_index)
                sentence_words = set(list_of_words)

            list_of_sentences.append(list_of_words)
            list_of_sentences_set.append(sentence_words)

        new_index_corpus[article] = list_of_sentences
    
    return new_index_corpus




def select_context_words(center_idx, sentence, window_radius):
    #calculating the starting index of the window
    start_index = max(0, center_idx - window_radius)
    end_index = min(len(sentence)-1, center_idx+window_radius)
    context_words = sentence[start_index : center_idx] + sentence[center_idx+1 : end_index+1]
    return context_words




def get_negative_samples(context_idxs, word_prob, num_samples=5):
    """
    Selects negative samples for a given context word.
    
    Parameters:
        context_word (str): The current context word for which negative samples are selected.
        num_samples (int): Number of negative samples to select.
        
    Returns:
        list: List of negative sample word indices.
    """
    vocab_size = len(word_prob)
    negative_samples = []
    
    while len(negative_samples) < num_samples:
        # Randomly sample a word index using the unigram distribution
        neg_idx = np.random.choice(vocab_size, p=word_prob)
        if neg_idx not in context_idxs:  # Avoid selecting the context words
            negative_samples.append(neg_idx)
    
    return negative_samples