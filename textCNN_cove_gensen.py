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
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data
from torch import nn, optim
import nltk
import re

_model_folder = './data/models'
filename_prefix1 = 'nli_large_bothskip'
filename_prefix2 = 'nli_large_bothskip_parse'
_pretrained_emb = './data/embedding/glove.6B.300d.h5'

from cove import CoVe
PATH_TO_W2V = 'data/embedding/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2
MODEL_PATH = 'data/models/wmtlstm.pth'

batch_size = 32
max_features = 100000
maxlen = 200
V = 1 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

params_model = {'bsize': 32, 'word_emb_dim': 300, 'enc_lstm_dim': 300, 'nlayers': 2,
                'pool_type': 'none', 'dpout_model': 0.0, 'version': V}

cove_model = CoVe(params_model)
cove_model.load_state(MODEL_PATH)
cove_model.set_w2v_path(PATH_TO_W2V)
cove_model = cove_model.to(device)

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
print("start build vocab")
cove_model.build_vocab([' '.join(s) for s in X_train], tokenize=False)
cove_model.build_vocab([' '.join(s) for s in X_test], tokenize=False)
print("end build vocab")



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        p = .1
        self.gensen_embedding = gensen
        self.cove_embedding = cove_model
        self.lin_gensen_0 = nn.Linear(4096, 300)
        self.lin_cove_0 = nn.Linear(600, 300)
        self.embedding_dim = 300
        self.Ci = 2
        self.Co = 32
        self.kernel_list = [1,2,3,5]
        self.convs = nn.ModuleList([nn.Conv2d(self.Ci, self.Co, (k, self.embedding_dim)) for k in self.kernel_list])
        self.dropout = nn.Dropout(p=p)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(128, 6)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x1, _ = self.gensen_embedding.get_representation(x, pool='last', return_numpy=False)
        x1 = self.lin_gensen_0(x1)
        x1 = self.relu(x1)
        x2 = self.cove_embedding.encode(x, batch_size, tokenize = False)
        x2 = torch.from_numpy(x2).float().to(device)
        x2 = self.lin_cove_0(x2)
        x2 = self.relu(x2)
        x = torch.stack([x1,x2], dim = 1)
        x = [self.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        print(x.size())
        x = self.dropout(x)
        x = self.lin(x)
        return self.sig(x)

def train(model, X_train, y_train, epoch = 4):

    learnin1g_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learnin1g_rate)
    for e in range(epoch):
        count = 0
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
    print("train complete")
    
model = Net().to(device)
print(model)
train(model, X_train, y_train)
MODEL_SAVE_PATH = "gensen_cove_batch_32.pth"
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

