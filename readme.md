# Sentiment Analysis and Text Generation using Many-to-One LSTMs on Airline Reviews

### Overview
This project explores sentiment analysis and text generation utilizing Long Short-Term Memory (LSTM) neural networks.

---

#### Sentiment Analysis
We use LSTMs to analyze airline sentiments, aiming to predict sentiment labels (0 or 1) based on customer reviews. Converting text reviews into numerical data, we utilize many-to-one LSTMs for accurate predictions.

---

#### Text Generation
Text generation is also explored by training LSTMs on "Alice's Adventures in Wonderland." The model learns to predict the next word in a sequence, creating coherent and contextually relevant sentences. Challenges like language variability and context dependence are addressed through entropy scaling and softmax temperature techniques.

---

## Aim

This project's objectives are twofold:
1. Develop a sentiment detection model using many-to-one LSTMs to predict sentiment labels (0 or 1) based on airline text reviews.
2. Utilize many-to-one LSTMs for text generation, training on "Alice's Adventures in Wonderland" and predicting the next word in a sequence.

---

## Data Description

- **airline_sentiment.csv**: This dataset includes information on airline sentiments in CSV format, containing "airline_sentiment" (sentiment labels: 0 or 1) and "text" (customer reviews).

- **alice.txt**: Project Gutenberg's eBook of "Alice’s Adventures in Wonderland" by Lewis Carroll, provided as a text file. This dataset serves as training data for text generation using many-to-one LSTMs.

---

## Tech Stack

- **Language**: `Python`
- **Libraries**: `pandas`, `numpy`, `keras`, `tensorflow`, `collections`, `nltk`

---

# Approach

## Sentiment Analysis

### Dataset
- Obtain the airline sentiment dataset with sentiment labels and text reviews.

### Preprocessing
- Perform data preprocessing, including text cleaning, tokenization, and stop word removal.
- Convert text reviews into a bag-of-words representation.

### Many-to-One LSTM
- Utilize many-to-one LSTM architecture for sentiment detection.
- Feed the bag-of-words representation as input to the LSTM.

### Training
- Split the dataset into training and testing sets.
- Train the LSTM model using the training set.
- Evaluate model performance on the testing set.

## Text Generation

### Dataset
- Obtain "Alice's Adventures in Wonderland" text dataset.

### Preparing and Structuring
- Preprocess the text data by cleaning, tokenizing, and structuring sentences and phrases.
- Create sequences.

### Many-to-One LSTM
- Implement many-to-one LSTM architecture for text generation.
- Train the LSTM model using the prepared dataset.

### Challenges in Text Generation
- Recognize challenges such as language variability and context dependence in text generation.
- Understand that natural language uses diverse words with similar meanings, requiring careful consideration during generation.

### Entropy Scaling
- Explore entropy scaling to introduce controlled randomness into text generation.

### Softmax Temperature
- Introduce softmax temperature as a hyperparameter to control prediction randomness in LSTMs and neural networks.
- Predict using temperature.

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
