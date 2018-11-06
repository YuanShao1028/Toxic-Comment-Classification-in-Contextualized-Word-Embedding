from __future__ import absolute_import, division, unicode_literals
import sys
import os
import torch
import logging
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.utils.data
from torch import optim
import nltk
import re
# get models.py from InferSent repo
from sent2vec import Sent2VecSingle, Sent2Vec


PATH_TO_DATA = 'data'
PATH_TO_W2V = 'data/embedding/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2

#MODEL_PATH = ['data/models/sent2vec/mtask/adv_shared_private/shared.pth',
#              'data/models/sent2vec/mtask/adv_shared_private/quora.pth',
#              'data/models/sent2vec/mtask/adv_shared_private/snli.pth',
#              'data/models/sent2vec/mtask/adv_shared_private/multinli.pth']

#MODEL_PATH = ['data/models/sent2vec/mtask/adv_shared_private/shared.pth']

MODEL_PATH = ['data/models/sent2vec/mtask/shared_private/shared.pth',
               'data/models/sent2vec/mtask/shared_private/quora.pth',
               'data/models/sent2vec/mtask/shared_private/snli.pth',
               'data/models/sent2vec/mtask/shared_private/multinli.pth']

#MODEL_PATH = ['data/models/sent2vec/mtask/shared_private/shared.pth']

V = 1  # version of InferSent
batch_size = 32
max_features = 100000
maxlen = 200

assert all([os.path.isfile(path) for path in MODEL_PATH]) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

params_model = {'bsize': 32, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'none', 'dpout_model': 0.0, 'version': V}
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

models = [Sent2VecSingle(params_model) for _ in range(len(MODEL_PATH))]

for i in range(len(MODEL_PATH)):
    models[i].load_state(MODEL_PATH[i])
    models[i].set_w2v_path(PATH_TO_W2V)
    models[i] = models[i].to(device)

sent2vec = Sent2Vec(models, 'concat')

print("start build vocab")
sent2vec.build_vocab([' '.join(s) for s in X_train], tokenize=False)
sent2vec.build_vocab([' '.join(s) for s in X_test], tokenize=False)
print("end build vocab")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        p = .1
        self.sent2vec_embedding = sent2vec
        self.lin_0 = nn.Linear(4096 * 4, 300)
        #self.dropout_0 = nn.Dropout(p=0.5) 
        self.embedding_dim = 300 #300
        self.lstm = nn.LSTM(self.embedding_dim, 512, 1, batch_first=True, bidirectional=True)#512
        self.hidden = (
            Variable(torch.zeros(2, 1, 1024)),
            Variable(torch.zeros(2, 1, 1024)))

        self.dropout = nn.Dropout(p=p)
        self.lin_1 = nn.Linear(1024, 200)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(p=p)
        self.lin_2 = nn.Linear(200, 6)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.sent2vec_embedding.get_representation(x, batch_size, tokenize=False)
        x = torch.from_numpy(x).float().to(device)
        x = self.lin_0(x)
        x = self.relu(x)
        #x = self.dropout_0(x)
        x, self.hidden = self.lstm(x)
        x, _ = torch.max(x, dim = 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.lin_1(x)
        x = self.relu(x)
        x = self.dropout_2(x)
        x = self.lin_2(x)
        return self.sig(x)

def train(model, X_train, y_train, epoch = 2):
    learnin1g_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learnin1g_rate)
    for e in range(epoch):
        count = 0
        running_loss = 0.0
        for idx in range(0, len(X_train), batch_size):
            model.zero_grad()
            batch = X_train[idx : idx + batch_size]
            sent = [' '.join(s) for s in batch]
            target = y_train[idx : idx + batch_size]
            target = torch.from_numpy(target).float().to(device)
            y_pred = model(sent)
            loss = F.binary_cross_entropy(y_pred, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1
            if count % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                  (e + 1, count + 1, running_loss / 50))
                running_loss = 0.0
    print("train complete")

model = Net().to(device)
print(model)
train(model, X_train, y_train)
MODEL_SAVE_PATH = "cove_batch_32.pth"
torch.save(model.state_dict(), MODEL_SAVE_PATH)
#model.eval()
preds = []
test_batch_size = 32
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
