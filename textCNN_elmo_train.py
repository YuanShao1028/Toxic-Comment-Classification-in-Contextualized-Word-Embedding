#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 20:10:49 2018

@author: Yuanshao
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn, optim
from allennlp.modules.elmo import Elmo, batch_to_ids
import nltk
import re
import time

options_file = "data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
weight_file = "data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"


batch_size = 32
max_features = 100000
maxlen = 200
embed_size = 1024
epochs = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clean(comment):
    comment = re.sub("\\n"," ",comment)
    # remove leaky elements like ip,user
    comment = re.sub("\d{1,3}.\d{1,3}.\d{1,3}.\d{1,3}"," ",comment)
    # removing usernames
    comment = re.sub("\[\[.*\]","",comment)
    comment = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(comment))
    comment = re.sub(r"[ ]+", " ", comment)
    comment = re.sub(r"\!+", "!", comment)
    comment = re.sub(r"\,+", ",", comment)
    comment = re.sub(r"\?+", "?", comment)
    return(comment.lower())
    
    
def tokenize(sent, maxLen = 200):
    words = nltk.word_tokenize(sent)
    if maxLen != -1:
        words = words[0:maxlen]
        words += [' '] * max(maxlen - len(words), 0)
    return words

def data_to_ids(dataset, bsize = 2048):
    tensor_list = []
    for idx in range(0, len(dataset), bsize):
        #print(idx)
        data = dataset[idx : idx + bsize]
        ids = batch_to_ids(data)
        tensor_list.append(ids)
    result = torch.cat(tensor_list)
    return result

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
#X_test = test["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_train = list(map(lambda x: clean(x), X_train))
#X_test = list(map(lambda x: clean(x), X_test))
print("start token")
X_train = list(map(lambda x: tokenize(x), X_train))
#X_test = list(map(lambda x: tokenize(x), X_test))
print("end token")
start_token = time.time()
train_token_id = data_to_ids(X_train)
end_token = time.time()
print('data2ids time:')
print(end_token - start_token)
print(train_token_id.size())

train_set = torch.utils.data.TensorDataset(train_token_id.long(), torch.from_numpy(y_train).float())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        p = .1
        self.embedding_dim = 300
        self.elmo_embedding = Elmo(options_file, weight_file, 1, dropout=0)
        self.lin_0 = nn.Linear(1024, 300)
        self.Ci = 1
        self.Co = 128
        self.kernel_list = [1,2,3,5]
        self.convs = nn.ModuleList([nn.Conv2d(self.Ci, self.Co, (k, self.embedding_dim)) for k in self.kernel_list])
        self.dropout = nn.Dropout(p=p)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(128*4, 6)
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        x = self.elmo_embedding(x)['elmo_representations'][0]
        x = self.lin_0(x)
        x = self.relu(x)
        x = x.unsqueeze(1)
        x = [self.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        print(x.size())
        x = self.dropout(x)
        x = self.lin(x)
        return self.sig(x)

def train(model, epoch = 4):

    learnin1g_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learnin1g_rate)
    for e in range(epoch):
        running_loss = 0.0
        start = time.time()
        for i, (data, target) in enumerate(train_loader):
            model.zero_grad()
            data, target = data.to(device), target.to(device)
            y_pred = model(data)
            loss = F.binary_cross_entropy(y_pred, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
        end = time.time()
        last_time = end - start
        print('last time is %d' %last_time)
    print("train complete")
model = Net().to(device)
print(model)

MODEL_SAVE_PATH = "elmo_batch_32.pth"
train(model)
torch.save(model.state_dict(), MODEL_SAVE_PATH)

