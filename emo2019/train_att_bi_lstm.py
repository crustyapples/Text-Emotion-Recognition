import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from sklearn.metrics import classification_report
import pickle
from utils import TextEmotionDataset

# Load the tokenizer to get the vocabulary
with open('tokenizer_pytorch.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Define the vocabulary size and create GloVe vectors only for the words in our vocabulary
vocab_size = len(tokenizer.word_index) + 1
glove = GloVe(name='6B', dim=100)
embedding_matrix = torch.zeros((vocab_size, glove.dim))
for word, index in tokenizer.word_index.items():
    embedding_matrix[index] = glove.vectors[glove.stoi[word]] if word in glove.stoi else torch.randn(glove.dim)

# Load the processed data
with open('processed_data_pytorch.pkl', 'rb') as file:
    X_train_pad, X_val_pad, X_test_pad, y_train, y_val, y_test, max_seq_length = pickle.load(file)

# Load the label encoder
with open('label_encoder_pytorch.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Create datasets and dataloaders
train_dataset = TextEmotionDataset(X_train_pad, y_train)
val_dataset = TextEmotionDataset(X_val_pad, y_val)
test_dataset = TextEmotionDataset(X_test_pad, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.main = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_dim * num_directions)
        attn_weights = self.main(lstm_output)
        attn_weights = torch.softmax(attn_weights.squeeze(2), dim=1)
        return attn_weights


class EnhancedRNNModel(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, pretrained_embeddings, drop_prob=0.5):
        super(EnhancedRNNModel, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=False)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                            dropout=drop_prob if n_layers > 1 else 0,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(drop_prob)
        self.attention = Attention(hidden_dim * 2)  # * 2 because of bidirectional
        self.fc = nn.Linear(hidden_dim * 2, output_size)  # * 2 because of bidirectional
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        # lstm_out shape: (batch_size, seq_len, hidden_dim * num_directions)
        attn_weights = self.attention(lstm_out)
        # attn_weights shape: (batch_size, seq_len)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
        # attn_applied shape: (batch_size, hidden_dim * num_directions)
        out = self.dropout(attn_applied)
        out = self.fc(out)
        soft_out = self.softmax(out)
        return soft_out

# Instantiate the model with the embedding matrix
model = EnhancedRNNModel(
    vocab_size, 
    len(label_encoder.classes_), 
    100,  # embedding dimension
    256,  # hidden dimension
    2,    # number of LSTM layers
    embedding_matrix  # pretrained GloVe embeddings
)

# Loss and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Function to train the model with early stopping
def train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, grad_clip=5.0, patience=3):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        # Validation phase for early stopping
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                output = model(inputs)
                val_loss += criterion(output, labels).item()
        val_loss /= len(val_loader)

        print(f'Epoch {epoch + 1}/{epochs} - Train Loss: {loss.item()} - Val Loss: {val_loss}')

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'saved_models/att_bi_lstm_emo.pth')  # Save the best model
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

# Train the model with early stopping
train_model_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, grad_clip=5.0, patience=3)

# Load the best model and evaluate on the validation and test sets
model.load_state_dict(torch.load('saved_models/att_bi_lstm_emo.pth'))

def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in loader:
            output = model(inputs)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return classification_report(true_labels, all_preds, target_names=label_encoder.classes_)

# Evaluate the model on the validation set
val_report = evaluate_model(model, val_loader)
print("Validation Report:")
print(val_report)

# Evaluate the model on the test set
test_report = evaluate_model(model, test_loader)
print("Test Report:")
print(test_report)