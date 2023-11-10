import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np

# Load the CSV file using pandas
data = pd.read_csv('tweets_emotions/text_emotion.csv')

# keep only these emotions as there are too few samples of the rest

data = data[data['sentiment'].isin(['worry', 'neutral', 'happiness', 'sadness', 'love'])]

# Encode the sentiment labels
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

# Save the label encoder
with open('tweets_emotions/label_encoder_csv.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# Split the dataset into content and sentiment
X, y = data['content'], data['sentiment']

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Pad sequences
max_seq_length = max([len(x) for x in X_seq])  # Get max sequence length
X_pad = pad_sequences(X_seq, maxlen=max_seq_length)

# Split the data into training, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Reset indexes
y_train, y_val, y_test = np.array(y_train), np.array(y_val), np.array(y_test)

# Save the processed data
with open('tweets_emotions/processed_data_csv.pkl', 'wb') as file:
    pickle.dump((X_train, X_val, X_test, y_train, y_val, y_test, max_seq_length), file)

# Save the tokenizer
with open('tweets_emotions/tokenizer_csv.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)