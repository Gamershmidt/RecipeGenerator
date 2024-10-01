# Import Necessary Libraries
import string
from collections import Counter
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torchinfo import summary
from torchinfo import summary
from torchmetrics import Accuracy
from torchvision import datasets
import torch
import torch.nn as nn
import torchtext
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# Load Data
train = pd.read_csv("/home/sofia/Документы/Symptom2Disease/data/processed/train.csv")
X_train = train['text']
y_train = train['label']
test = pd.read_csv("/home/sofia/Документы/Symptom2Disease/data/processed/test.csv")
X_test = test['text']
y_test = test['label']

# Create Vocabulary using torchtext vocab class
counter = Counter()
for text in X_train:
    counter.update(text.split())

vocab = torchtext.vocab.vocab(counter, specials=['<unk>', '<pad>'])
vocab.set_default_index(vocab['<unk>'])

max_words = X_train.apply(lambda x: x.split()).apply(len).max()

# Define Dataset Class
class DiseaseDataset(torch.utils.data.Dataset):
    def __init__(self, symptoms, labels):
        self.symptoms = symptoms
        self.labels = torch.tensor(labels.to_numpy())

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.symptoms[idx]
        label = self.labels[idx]

        # Convert the text to a sequence of word indices
        text_indices = [vocab[word] for word in text.split()]

        # Padding for same length sequence
        if len(text_indices) < max_words:
            text_indices = text_indices + [1] * (max_words - len(text_indices))

        return torch.tensor(text_indices), label

# Prepare DataLoader
train_dataset = DiseaseDataset(X_train, y_train)
val_dataset = DiseaseDataset(X_test, y_test)

batch_size = 8

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Define the RNN Model
class RNNModel(torch.nn.Module):
    def __init__(self, vocab_size=1409, embedding_dim=256, hidden_dim=128, num_classes=24, drop_prob=0.4, num_layers=1,
                 bidir=False, seq="lstm"):
        super(RNNModel, self).__init__()
        self.seq = seq
        self.bidir_f = 2 if bidir else 0
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        if seq == "lstm":
            self.rnn = torch.nn.LSTM(embedding_dim, hidden_dim,
                                     num_layers=num_layers,
                                     batch_first=True,
                                     bidirectional=bidir)
        else:
            self.rnn = torch.nn.GRU(embedding_dim, hidden_dim,
                                    num_layers=num_layers,
                                    batch_first=True,
                                    bidirectional=bidir)

        self.dropout = torch.nn.Dropout(drop_prob)  # dropout layer
        self.fc = torch.nn.Linear(hidden_dim * self.bidir_f, num_classes)  # fully connected layer

    def forward(self, text_indices):
        # Embed the text indices
        embedded_text = self.embedding(text_indices)

        # Pass the embedded text through the RNN
        rnn_output, hidden_states = self.rnn(embedded_text)
        # Take the last output of the RNN
        last_rnn_output = rnn_output[:, -1, :]
        x = self.dropout(last_rnn_output)
        # Pass the last output of the RNN through the fully connected layer
        x = self.fc(x)

        # Return the final output
        return x

# Training Function with MLflow Integration
def train(model, num_epochs=10):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        device = "cpu"
        model = model.to(device)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_accuracy = 0.0
            for data in train_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                acc = (labels == outputs.argmax(dim=-1)).float().mean().item()
                train_loss += loss.item()
                train_accuracy += acc
                loss.backward()
                optimizer.step()

            avg_train_loss = train_loss / len(train_loader)
            avg_train_accuracy = train_accuracy / len(train_loader)

            # Validation
            model.eval()
            val_loss = 0.0
            val_accuracy = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    val_accuracy += (labels == outputs.argmax(dim=-1)).float().mean().item()

            avg_val_loss = val_loss / len(val_loader)
            avg_val_accuracy = val_accuracy / len(val_loader)


            print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_accuracy:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {avg_val_accuracy:.4f}')


# Model Parameters
num_classes = len(np.unique(y_train))
vocab_size = len(vocab)
emb_dim = 256
hidden_dim = 128
drop_prob = 0.4
epochs = 13
model_lstm = RNNModel(vocab_size, emb_dim, hidden_dim, num_classes, drop_prob, num_layers=3, bidir=True, seq="lstm")

train(model_lstm,epochs)

# Save the Model
model_save_path = "/home/sofia/Документы/Symptom2Disease/models/model_lstm.h5"
torch.save(model_lstm.state_dict(), model_save_path)

print(f"Model saved at {model_save_path}")
