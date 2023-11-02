import nltk
import re
from collections import Counter
from nltk.corpus import stopwords
from keras.utils import pad_sequences
import numpy as np

# Download the stopwords data from NLTK
nltk.download('stopwords')

# Get a list of English stopwords
stop = stopwords.words('english')

# Function to preprocess text data
def pre_process_text_data(text):
    # Normalize and remove special characters
    text = text.lower()
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    
    # Remove stopwords
    words = text.split()
    words = [w for w in words if w not in stop]
    words = ' '.join(words)
    
    return words

# Function to preprocess and tokenize text data in a DataFrame
def process_data(df):
    # Preprocess text data in the DataFrame
    df['text'] = df['text'].apply(pre_process_text_data)
    
    # Count words in the preprocessed text
    counts = Counter()
    for i, review in enumerate(df['text']):
        counts.update(review.split())

    words = sorted(counts, key=counts.get, reverse=True)
    
    return df, words

# Function to convert text to a list of integers based on a word-to-int mapping
def text_to_int(text, word_to_int):
    return [word_to_int[word] for word in text.split()]

# Function to convert a list of integers back to text
def int_to_text(int_arr, int_to_word):
    return ' '.join([int_to_word[index] for index in int_arr if index != 0])

# Function to map reviews to sequences of integers
def map_reviews(df, word_to_int):
    mapped_reviews = []
    for review in df['text']:
        mapped_reviews.append(text_to_int(review, word_to_int))
    
    return mapped_reviews

# Function to get the maximum sequence length among mapped reviews
def get_sequence_length(mapped_reviews):
    length_sent = [len(review) for review in mapped_reviews]
    sequence_length = max(length_sent)
    
    return sequence_length

# Function to pad and encode mapped reviews
def pad_and_encode(mapped_reviews, sequence_length):
    X = pad_sequences(maxlen=sequence_length, sequences=mapped_reviews, padding="post", value=0)
    
    return X

# Function to process sentiment data, returning X (padded and encoded reviews), y (sentiment labels),
# word-to-int and int-to-word mappings
def process_sentiment_data(df):
    df, words = process_data(df)
    
    # Create word-to-int and int-to-word mappings
    word_to_int = {word: i for i, word in enumerate(words, start=1)}
    int_to_word = {i: word for i, word in enumerate(words, start=1)}
    
    # Map reviews to sequences of integers
    mapped_reviews = map_reviews(df, word_to_int)
    
    # Get the maximum sequence length among mapped reviews
    sequence_length = get_sequence_length(mapped_reviews)
    
    # Pad and encode the mapped reviews
    X = pad_and_encode(mapped_reviews, sequence_length)
    
    # Get sentiment labels
    y = df['airline_sentiment'].values
    
    return X, y, word_to_int, int_to_word

# Additional functions for text generation

# Function to preprocess text for text generation
def pre_process(text: str) -> str:
    text = text.lower()
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text

# Function to get input and label words for text generation
def get_input_and_labels(text: str, seq_length: int = 10, step: int = 1):
    input_words = []
    label_words = []

    text_arr = text.split()

    for i in range(0, len(text_arr) - seq_length, step):
        x = text_arr[i : (i + seq_length)]
        y = text_arr[i + seq_length]
        input_words.append(x)
        label_words.append(y)

    return input_words, label_words

# Function to process text data for text generation, returning X (input words) and y (label words)
def process_text_generation_data(text: str, seq_length: int = 10):
    processed_text = pre_process(text)
    input_words, label_words = get_input_and_labels(processed_text, seq_length=seq_length)
    
    counts = Counter()
    counts.update(processed_text.split())
    words = sorted(counts, key=counts.get, reverse=True)
    nb_words = len(processed_text.split())

    word2index = {word: i for i, word in enumerate(words)}
    index2word = {i: word for i, word in enumerate(words)}

    total_words = len(set(words))

    # Create input and label arrays for text generation
    X = np.zeros((len(input_words), seq_length, total_words), dtype=bool)
    y = np.zeros((len(input_words), total_words), dtype=bool)

    for i, input_word in enumerate(input_words):
        for j, word in enumerate(input_word):
            X[i, j, word2index[word]] = 1
        y[i, word2index[label_words[i]]] = 1

    return X, y, words, nb_words, total_words, word2index, index2word, input_words
