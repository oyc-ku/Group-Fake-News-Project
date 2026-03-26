# Classes used for neural network model

import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


# Class for LSTM model
class WordRNN(nn.Module):
    def __init__(self, embedding_model, embedding_dim, hidden_dim, tagset_size=1):
        super(WordRNN, self).__init__()

        self.word_embeddings = nn.Embedding.from_pretrained(torch.tensor(embedding_model.wv.vectors))
        self.word_embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, article_data, article_lengths):
        article_embeddings_tensor = self.word_embeddings(article_data)
        
        packed_data = pack_padded_sequence(article_embeddings_tensor, article_lengths, batch_first=True, enforce_sorted=False)
        output, (h_n, c_n) = self.lstm(article_embeddings_tensor)
        last_hidden = h_n[-1]

        logit = self.fc(last_hidden)

        return logit


# PyTorch dataset to load articles and labels
class ArticleDataset(Dataset):
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        articles = self.data["content"].iloc[idx]
        padded_articles = pad_sequence(articles.tolist(), batch_first=True)
        lengths = torch.tensor([a.size(0) for a in articles], dtype=torch.long).cpu()
        
        labels = self.data["type"].iloc[idx]
        labels_tensor = torch.from_numpy(labels.to_numpy(dtype=np.float32))

        return padded_articles, lengths, labels_tensor


# Custom sampler to prepare batches of articles from the dataset with roughly same amount of tokens (to increase training speed)
class MySampler(Sampler):
    def __init__(self, groups, bucket_to_batch_size):
        self.groups = groups
        self.bucket_to_batch_size = bucket_to_batch_size

    def __iter__(self):
        batches = []
        
        for bucket_size, group_indices in self.groups.indices.items():    
            indices = list(group_indices)
            random.shuffle(indices)

            batch_size = self.bucket_to_batch_size[bucket_size]
            for i in range(0, len(indices), batch_size):
                batch = indices[i:i+batch_size]
                if batch:
                    batches.append(batch)

        random.shuffle(batches)
        
        yield from batches

    def __len__(self):
        return len(self.batches)
