import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the dataset
data = pd.read_csv('text_emotion.csv')

# Keep only the necessary columns
data = data[['content', 'sentiment']]

# Encode the sentiment labels
label_encoder = LabelEncoder()
data['sentiment'] = label_encoder.fit_transform(data['sentiment'])

# Save the label encoder
with open('label_encoder_pytorch.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

# After splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(data['content'], data['sentiment'], test_size=0.2, random_state=42)

# Reset the indices
X_train = X_train.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)


# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_seq_length = max([len(x) for x in X_train_seq])  # Get max sequence length
X_train_pad = pad_sequences(X_train_seq, maxlen=max_seq_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_seq_length)

# Save the processed data
with open('processed_data_pytorch.pkl', 'wb') as file:
    pickle.dump((X_train_pad, X_test_pad, y_train, y_test, max_seq_length), file)

# Save the tokenizer
with open('tokenizer_pytorch.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
