import os
from ml_pipeline import train, process, utils
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from projectpro import save_point, checkpoint

# Define your main engine code here
def main():
    # Code to call the relevant functions from the ml_pipeline modules based on input

    # Example: Set input_type to determine the operation you want to perform
    input_type = 2  # Change this based on your input

    if input_type == 1:
        # Load the sentiment data from a CSV file
        df = pd.read_csv('data/airline_sentiment.csv')

        # Process sentiment data and convert it to a format suitable for training
        X, y, word_to_int, int_to_word = process.process_sentiment_data(df)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)

        # Convert the labels to one-hot encoded format
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)

        # Train a sentiment analysis LSTM model
        model = train.train_sentiment_lstm(X_train, y_train, X_test, y_test, word_to_int)

        # Get predicted positive and negative sentences
        positive_sentences, negative_sentences = utils.get_predicted_sentiments(model, X_test, int_to_word)

        # Save a checkpoint
        checkpoint('5b420a')

        # Print the top 5 positive sentences
        print("Positive Sentences:")
        for sentence in positive_sentences[:5]:
            print(sentence)

        # Print the top 5 negative sentences
        print("\nNegative Sentences:")
        for sentence in negative_sentences[:5]:
            print(sentence)

    elif input_type == 2:
        # Additional code for text generation

        # Load the text data
        text = utils.load_data()

        # Process text data for text generation
        X, y, words, nb_words, total_words, word2index, index2word, input_words = process.process_text_generation_data(text)
        print(f'Input of X: {X.shape}\nInput of y: {y.shape}')

        # Define constants for text generation
        SEQLEN = 10
        HIDDEN_SIZE = 128

        # Create a text generation model
        model = train.create_text_generation_model(HIDDEN_SIZE, SEQLEN, total_words)

        # Save a checkpoint
        save_point('5b420a')

        # Train the text generation model
        model = train.train_text_generation_model(X, y, input_words, SEQLEN, total_words, model, word2index, index2word)

        # Define the starting words for text generation
        test_words = input_words[-28701]

        # Generate and print text paragraphs
        for _ in range(2):
            print(' '.join(train.generate_paragraph(model, test_words, 12, 10))

if __name__ == "__main__":
    main()
