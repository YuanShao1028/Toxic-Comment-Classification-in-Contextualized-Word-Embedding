#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 20:17:16 2018

@author: Yuanshao
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from allennlp.modules.elmo import Elmo, batch_to_ids
import nltk
import re


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
        print(idx)
        data = dataset[idx : idx + bsize]
        ids = batch_to_ids(data)
        tensor_list.append(ids)
    result = torch.cat(tensor_list)
    return result

test = pd.read_csv("data/test.csv")
X_test = test["comment_text"].fillna("fillna").values
X_test = list(map(lambda x: clean(x), X_test))
X_test = list(map(lambda x: tokenize(x), X_test))
test_token_id = data_to_ids(X_test)
test_loader = torch.utils.data.DataLoader(test_token_id.long(), batch_size=128)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        p = .1
        self.elmo_embedding = Elmo(options_file, weight_file, 1, dropout=0)
        self.embedding_dim = 300
        self.lin_0 = nn.Linear(1024, 300)
        self.Ci = 1
        self.Co = 32
        self.kernel_list = [1,2,3,5]
        self.convs = nn.ModuleList([nn.Conv2d(self.Ci, self.Co, (k, self.embedding_dim)) for k in self.kernel_list])
        self.dropout = nn.Dropout(p=p)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(128, 6)
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

model = Net().to(device)
model.load_state_dict(torch.load("elmo_batch_32.pth"))
model.eval()
print("start to eval")
preds = []
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
with torch.no_grad():
    count = 0
    for data in test_loader:
        count += 1
        data = data.to(device)
        output = model(data)
        pred = output.data.cpu()
        preds.append(pred.numpy())
        if count % 10 == 9:
            print(count)
y_test = np.concatenate(preds, axis=0)
sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test
sample_submission.to_csv("submission2.csv", index=False)
