import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from sklearn.metrics import classification_report
import pickle
from dataset import TextEmotionDataset

# Load the tokenizer to get the vocabulary size
with open('tokenizer_pytorch.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Define the vocabulary size (needed for the Embedding layer)
vocab_size = len(tokenizer.word_index) + 1

# Load the processed data
with open('processed_data_pytorch.pkl', 'rb') as file:
    X_train_pad, X_test_pad, y_train, y_test, max_seq_length = pickle.load(file)

# Load the label encoder
with open('label_encoder_pytorch.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Create datasets and dataloaders
train_dataset = TextEmotionDataset(X_train_pad, y_train)
test_dataset = TextEmotionDataset(X_test_pad, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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

# Function to train the model
def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch + 1}/{epochs} - Loss: {loss.item()}')

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

# Train the model
train_model(model, train_loader, criterion, optimizer)

# Evaluate the model
report = evaluate_model(model, test_loader)
print(report)

# Save the model
torch.save(model.state_dict(), 'saved_models/rnn_model_pytorch.pth')
