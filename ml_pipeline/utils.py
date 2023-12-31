from collections import Counter

# Function to convert text to a list of integers based on a word-to-int mapping
def text_to_int(text: str, word_to_int: dict):
    return [word_to_int[word] for word in text.split()]

# Function to convert a list of integers back to text
def int_to_text(int_arr, int_to_word: dict):
    return ' '.join([int_to_word[index] for index in int_arr if index != 0])

import numpy as np
from keras.utils import to_categorical, pad_sequences
from keras.models import load_model

import numpy as np

# Function to get sentiment for a given text using a trained model
def get_sentiment(model, text, word_to_int, int_to_word, sequence_length):
    text_int_embedding = text_to_int(text, word_to_int)
    text_int_embedding = pad_sequences(maxlen=sequence_length, sequences=[text_int_embedding], padding="post", value=0)
    sentiment_index = np.argmax(model.predict(text_int_embedding))
    return sentiment_index

# Function to get predicted sentiments for a list of text embeddings
def get_predicted_sentiments(model, X_test, int_to_word):
    result = np.argmax(model.predict(X_test), axis=1)
    positive_sentences = [int_to_text(embedding, int_to_word) for i, embedding in enumerate(X_test) if result[i] == 1]
    negative_sentences = [int_to_text(embedding, int_to_word) for i, embedding in enumerate(X_test) if result[i] == 0]
    return positive_sentences, negative_sentences

# Function to load text data from a file (default filename is 'data/alice.txt')
def load_data(filename: str = 'data/alice.txt'):
    with open(filename, encoding='utf-8-sig') as fin:
        lines = []
        for line in fin:
            line = line.strip().lower()
            if len(line) == 0:
                continue
            lines.append(line)
        fin.close()
        text = " ".join(lines)
    return text
