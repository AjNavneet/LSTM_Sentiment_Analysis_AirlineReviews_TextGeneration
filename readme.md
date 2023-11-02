# Sentiment Detection and Text Generation with Many-to-One LSTMs on Airline Reviews

### Overview
This project explores sentiment analysis and text generation using Long Short-Term Memory (LSTM) neural networks.

---

#### Sentiment Detection
We analyze airline sentiments using LSTMs. The goal is to predict sentiment labels (0 or 1) based on customer reviews. We convert text reviews into numbers and use LSTMs for prediction.

---

#### Text Generation
We also generate text inspired by "Alice's Adventures in Wonderland" using LSTMs. The model learns to predict the next word in a sequence, creating coherent and contextually relevant sentences.

We address challenges in text generation, such as language variability and context dependence, using techniques like entropy scaling and softmax temperature.

---

## Aim

This project aims to develop a sentiment detection model using many-to-one LSTMs to accurately predict sentiment labels (0 or 1) based on airline text reviews. 

Additionally, we aim to utilize many-to-one LSTMs to generate contextually relevant text by training on "Alice's Adventures in Wonderland" and predicting the next word in a sequence.

---

## Data Description

- **airline_sentiment.csv**: This dataset contains information related to airline sentiments. It is provided in a CSV format with two columns: "airline_sentiment" and "text". The "airline_sentiment" column includes sentiment labels (0 or 1) indicating the sentiment associated with each text review. The "text" column contains the actual text reviews provided by airline customers.

- **alice.txt**: The "alice.txt" dataset is the Project Gutenberg eBook of "Alice’s Adventures in Wonderland" by Lewis Carroll. It is a classic literary work in the form of a text file. This dataset serves as the training text for text generation using many-to-one LSTMs. It provides a rich and diverse collection of sentences and phrases to learn from, allowing the model to generate contextually relevant text based on the training data.


---

## Tech Stack

- **Language**: `Python`
- **Libraries**: `pandas`, `numpy`, `keras`, `tensorflow`, `collections`, `nltk`

---

# Approach

## Sentiment Analysis

### Dataset
- Obtain the airline sentiment dataset consisting of sentiment labels (0 or 1) and corresponding text reviews.

### Preprocessing
- Perform data preprocessing tasks, including text cleaning, tokenization, and removing stop words.
- Convert the text reviews into a bag-of-words representation.

### Many-to-One LSTM
- Utilize many-to-one LSTM architecture to train the sentiment detection model.
- Feed the bag-of-words representation of the text reviews as input to the LSTM.

### Training
- Split the dataset into training and testing sets.
- Train the LSTM model using the training set.
- Evaluate the model's performance on the testing set.

## Text Generation

### Dataset
- Obtain the "Alice's Adventures in Wonderland" text dataset.

### Preparing and Structuring
- Preprocess the text data by cleaning, tokenizing, and structuring the sentences and phrases.
- Create sequences.

### Many-to-One LSTM
- Implement many-to-one LSTM architecture for text generation.
- Train the LSTM model using the prepared dataset.

### The Problem with Text Generation
- Understand the challenges associated with text generation, such as the variability of language and the dependence on context, style, and word choice.
- Recognize that natural language utilizes a wide variety of words, which may have similar meanings and require careful consideration during generation.

### Randomness through Entropy Scaling
- Explore the concept of entropy scaling to introduce controlled randomness into text generation.

### Softmax Temperature
- Introduce the concept of softmax temperature, a hyperparameter used to control the randomness of predictions in LSTMs and neural networks.
- Predicting using the Temperature


---

## Modular Code Overview

- **data**: Contains input data files.
- **lib**: Includes a Jupyter Notebook with code and documentation.
- **output**: Stores generated files and results.
- **Readme.md**: Provides project information.
- **requirements.txt**: Lists required packages.
- **Engine.py**: The main engine file.
- **ML_Pipeline**: Contains modules for different steps in the machine learning pipeline.

---

## Key Concepts Explored

1. Gain insights into sentiment analysis and its importance in analyzing textual data.
2. Learn essential text preprocessing techniques.
3. Understand how to convert text into numbers.
4. Implement LSTM models for sentiment detection.
5. Train and evaluate LSTM models using labeled sentiment data.
6. Explore text generation using LSTMs.
7. Understand challenges in text generation and techniques for better results.

---