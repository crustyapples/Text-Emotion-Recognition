import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import classification_report
import pickle

# Assuming the dataset.py contains the TextEmotionDataset class
from dataset import TextEmotionDataset

# Load the tokenizer, processed data, and label encoder from your saved files
with open('tokenizer_pytorch.pkl', 'rb') as file:
    tokenizer = pickle.load(file)
with open('processed_data_pytorch.pkl', 'rb') as file:
    X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test, max_seq_length = pickle.load(file)
with open('label_encoder_pytorch.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Define Dataloaders for training, validation, and test sets
train_dataset = TextEmotionDataset(X_train_pad, y_train)
val_dataset = TextEmotionDataset(X_val_pad, y_val)
test_dataset = TextEmotionDataset(X_test_pad, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the RNN model
class RNNModel(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(RNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1]  # get last time step only
        out = self.dropout(lstm_out)
        out = self.fc(out)
        soft_out = self.softmax(out)
        return soft_out

# Parameters
vocab_size = len(tokenizer.word_index) + 1
output_size = len(label_encoder.classes_)
embedding_dim = 100
hidden_dim = 256
n_layers = 2

# Instantiate the model
model = RNNModel(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Function to train the model with early stopping
def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, epochs=10, patience=3):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation phase
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                output = model(inputs)
                loss = criterion(output, labels)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'saved_models/rnn_model_pytorch.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered")
            break

# Function to evaluate the model
def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            output = model(inputs)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.numpy())
            true_labels.extend(labels.numpy())
    return classification_report(true_labels, all_preds, target_names=label_encoder.classes_)

# Train the model with early stopping
train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer)

# Load the best model and evaluate on the test set
model.load_state_dict(torch.load('saved_models/rnn_model_pytorch.pth'))
report = evaluate_model(model, test_loader)
print(report)
