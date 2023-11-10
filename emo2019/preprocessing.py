import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import torch
from torch.utils.data import Dataset, DataLoader
import json

# Load the dataset from JSON files
with open('/Users/advait/Desktop/gitpositories/SC4001-Assignment-2/Text-Emotion-Recognition/emo2019/emo-train.json') as file:
    train_data = pd.DataFrame(json.load(file))
    
with open('/Users/advait/Desktop/gitpositories/SC4001-Assignment-2/Text-Emotion-Recognition/emo2019/emo-test.json') as file:
    test_data = pd.DataFrame(json.load(file))

# Rename 'Label' to 'sentiment' for consistency with the original code
train_data.rename(columns={'Label': 'sentiment', 'text': 'content'}, inplace=True)
test_data.rename(columns={'Label': 'sentiment', 'text': 'content'}, inplace=True)

# drop all rows whose sentiment is others
train_data = train_data[train_data.sentiment != 'others']
test_data = test_data[test_data.sentiment != 'others']

# Encode the sentiment labels
label_encoder = LabelEncoder()
train_data['sentiment'] = label_encoder.fit_transform(train_data['sentiment'])
test_data['sentiment'] = label_encoder.transform(test_data['sentiment'])  # Use the same encoder


# Save the label encoder
with open('emo2019/label_encoder_pytorch.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# No need to split the dataset as we already have train and test sets
X_train, y_train = train_data['content'], train_data['sentiment']
X_test, y_test = test_data['content'], test_data['sentiment']

# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_seq_length = max([len(x) for x in X_train_seq])  # Get max sequence length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length)

# Split the training data into training and validation sets
X_train_pad, X_val_pad, y_train, y_val = train_test_split(X_train_pad, y_train, test_size=0.2, random_state=42)

# Save the processed data including the validation set
with open('emo2019/processed_data_pytorch.pkl', 'wb') as file:
    pickle.dump((X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test, max_seq_length), file)

# Save the tokenizer
with open('emo2019/tokenizer_pytorch.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
