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

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, n_filters, filter_sizes, drop_prob=0.5):
        super(CNNModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, n_filters, (fs, embedding_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(drop_prob)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_size)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # add channel (1, L, D)
        x = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]  # [(N, Co, W), ...]*len(Ks)
        x = [torch.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)
        logit = self.fc(x)
        return logit

# Parameters
vocab_size = len(tokenizer.word_index) + 1
output_size = len(label_encoder.classes_)
embedding_dim = 100
n_filters = 100
filter_sizes = [3, 4, 5]

# Instantiate the model
model = CNNModel(vocab_size, output_size, embedding_dim, n_filters, filter_sizes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
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
torch.save(model.state_dict(), 'saved_models/cnn_model_pytorch.pth')
