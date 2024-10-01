# Import Necessary Libraries
import string
from collections import Counter
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import torch
import torch.nn as nn
import torchtext
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


# Define the RNN model
class RNNModel(torch.nn.Module):
    def __init__(self, vocab_size=1409, embedding_dim=256, hidden_dim=128, num_classes=24, drop_prob=0.4, num_layers=1,
                 bidir=False,
                 seq="lstm"):
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
        #         print("EMB SHAPE: ",embedded_text.shape)

        # Pass the embedded text through the RNN
        rnn_output, hidden_states = self.rnn(embedded_text)
        # Take the last output of the RNN
        last_rnn_output = rnn_output[:, -1, :]
        x = self.dropout(last_rnn_output)
        # Pass the last output of the RNN through the fully connected layer
        x = self.fc(x)

        # Return the final output
        return x


import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
df = pd.read_csv("code/datasets/Symptom2Disease.csv")
df.drop("Unnamed: 0", inplace=True, axis=1)
# set of English stopwords we will remove from our text data
stop_words = set(stopwords.words('english'))


def clean_text(sent):
    # remove punctuations
    sent = sent.translate(str.maketrans('', '', string.punctuation)).strip()

    # remove stopwords
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(sent)
    words = [word for word in words if word not in stop_words]

    return " ".join(words).lower()


# clean text rows in dataframe
df["text"] = df["text"].apply(clean_text)
# get list of diseases in our dataset
diseases = df["label"].unique()

# helper dictionaries to convert diseases to index and vice versa
idx2dis = {k: v for k, v in enumerate(diseases)}
dis2idx = {v: k for k, v in idx2dis.items()}

# convert disease name to index (label encoding)
df["label"] = df["label"].apply(lambda x: dis2idx[x])
# Split the data into train,test set
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["label"], test_size=0.2, random_state=42)
# pytorch dataset object use index to return item, so need to reset non-continuoues index of divided dataset
X_train.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
counter = Counter()
for text in X_train:
    counter.update(text.split())

vocab = torchtext.vocab.vocab(counter, specials=['<unk>', '<pad>'])


def make_pred(model, text):
    max_words = 64
    text = clean_text(text)
    # Convert the text to a sequence of word indices
    text_indices = [vocab[word] for word in text.split() if word in vocab]

    # padding for same length sequence
    if len(text_indices) < max_words:
        text_indices = text_indices + [1] * (max_words - len(text_indices))
    text_indices = torch.tensor(text_indices).cpu()
    pred = model(text_indices.unsqueeze(0))

    return idx2dis[pred.argmax(1).item()]
