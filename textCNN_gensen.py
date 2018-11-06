#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:49:34 2018

@author: Yuanshao
"""

from __future__ import absolute_import, division, unicode_literals
import sys
sys.path.append('.')
import torch
from gensen import GenSen, GenSenSingle
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data
from torch import nn, optim
import nltk
import re
import time
_model_folder = './data/models'
filename_prefix1 = 'nli_large_bothskip'
filename_prefix2 = 'nli_large_bothskip_parse'
_pretrained_emb = './data/embedding/glove.6B.300d.h5'

batch_size = 32
max_features = 100000
maxlen = 200

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

gensen_1 = GenSenSingle(
    model_folder=_model_folder,
    filename_prefix=filename_prefix1,
    pretrained_emb=_pretrained_emb,
    cuda=True
)

gensen_2 = GenSenSingle(
    model_folder=_model_folder,
    filename_prefix=filename_prefix2,
    pretrained_emb=_pretrained_emb,
    cuda=True
)

gensen = GenSen(gensen_1, gensen_2)

print("finish init gensen")

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

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_train = list(map(lambda x: clean(x), X_train))
X_test = list(map(lambda x: clean(x), X_test))
print("start token")
X_train = list(map(lambda x: tokenize(x), X_train))
X_test = list(map(lambda x: tokenize(x), X_test))
print("end token")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        p = .1
        self.gensen_embedding = gensen
        self.lin_0 = nn.Linear(4096, 300)
        self.embedding_dim = 300
        self.Ci = 1
        self.Co = 128
        self.kernel_list = [1,2,3,5]
        self.convs = nn.ModuleList([nn.Conv2d(self.Ci, self.Co, (k, self.embedding_dim)) for k in self.kernel_list])
        self.dropout = nn.Dropout(p=p)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(128 * 4, 6)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.gensen_embedding.get_representation(x, pool='last', return_numpy=False)
        x = self.lin_0(x)
        x = self.relu(x)
        x = x.unsqueeze(1)
        x = [self.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = self.dropout(x)
        x = self.lin(x)
        return self.sig(x)

def train(model, X_train, y_train, epoch = 4):

    learnin1g_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learnin1g_rate)
    for e in range(epoch):
        count = 0
        start = time.time()
        running_loss = 0.0
        for idx in range(0, len(X_train), batch_size):
            model.zero_grad()
            batch = X_train[idx : idx + batch_size]
            sent = [' '.join(s) for s in batch]
            #code, _ = gensen.get_representation(sent, pool='last', return_numpy=False)
            target = y_train[idx : idx + batch_size]
            target = torch.from_numpy(target).float().to(device)
            y_pred = model(sent)
            loss = F.binary_cross_entropy(y_pred, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            #print(loss.item())
            count += 1
            if count % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                  (e + 1, count + 1, running_loss / 50))
                running_loss = 0.0
        end = time.time()
        last_time = end - start
        print('last time is %d' %last_time)
    print("train complete")
    
model = Net().to(device)
print(model)
train(model, X_train, y_train)
MODEL_SAVE_PATH = "gensen_batch_32.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
#model.eval()
preds = []
test_batch_size = 128
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
with torch.no_grad():
    for idx in range(0, len(X_test), test_batch_size):
        data = X_test[idx : idx + test_batch_size]
        data = [' '.join(s) for s in data]
        output = model(data)
        pred = output.data.cpu()
        preds.append(pred.numpy())
y_test = np.concatenate(preds, axis=0)

sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test
sample_submission.to_csv("submission2.csv", index=False)

