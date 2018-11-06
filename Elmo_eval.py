# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
InferSent models. See https://github.com/facebookresearch/InferSent.
"""

from __future__ import absolute_import, division, unicode_literals

import sys
import os
import logging
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pandas as pd
from torch import nn, optim
import torch.nn.functional as F
import torch.utils.data
from elmo import ELMo
import nltk
import re
import gc

MODEL_PATH = 'data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
OPT_PATH = 'data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'
params_model = {'bsize': 8, 'pool_type': 'no', 'which_layer': 'all',
                    'optfile': OPT_PATH,
                    'wgtfile': MODEL_PATH}
batch_size = 8
max_features = 100000
maxlen = 200
embed_size = 1024
epochs = 2
EMBEDDING_FILE_FASTTEXT = "glove.6B.300d.txt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def clean(data_arr):
    for data in data_arr:
        data = data.lower()
        data=re.sub("\\n","",data)
        data=re.sub("\[\[.*\]","",data)
        data = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', data)
    return data_arr

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_train = clean(X_train)
X_test = clean(X_test)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        p = .1
        self.elmo_embedding = ELMo(params_model)
        self.lin_0 = nn.Linear(3072, 300)
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
        x = self.elmo_embedding.encode(x, params_model['bsize'], tokenize=True)
        x = torch.from_numpy(x).float().to(device)
        x = self.lin_0(x)
        x = self.relu(x)
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
            sent = X_train[idx : idx + batch_size]
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
train(model, X_train, y_train)

preds = []
test_batch_size = 8
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
with torch.no_grad():
    for idx in range(0, len(X_test), test_batch_size):
        data = X_test[idx : idx + test_batch_size]
        output = model(data)
        pred = output.data.cpu()
        preds.append(pred.numpy())
y_test = np.concatenate(preds, axis=0)

sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test
sample_submission.to_csv("submission2.csv", index=False)
 

'''
# Set PATHs
PATH_SENTEVAL = '../'
PATH_TO_DATA = '../data'
PATH_TO_W2V = '../../glove/glove.840B.300d.txt'  # or crawl-300d-2M.vec for V2

MODEL_PATH = '../data/models/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5'
OPT_PATH = '../data/models/elmo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json'

# MODEL_PATH = '../data/models/elmo/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5'
# OPT_PATH = '../data/models/elmo/elmo_2x4096_512_2048cnn_2xhighway_options.json'

V = 1  # version of InferSent

assert os.path.isfile(MODEL_PATH) and os.path.isfile(PATH_TO_W2V), \
    'Set MODEL and GloVe PATHs'

# import senteval
sys.path.insert(0, PATH_SENTEVAL)
import senteval


def prepare(params, samples):
    # params['elmo'].build_vocab([' '.join(s) for s in samples], tokenize=False)
    pass


def batcher(params, batch):
    sentences = [' '.join(s) for s in batch]
    embeddings = params['elmo'].encode(
        sentences, params.batch_size, tokenize=False
    )
    return embeddings


"""
Evaluation of trained model on Transfer Tasks (SentEval)
"""

# define senteval params
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    # Load InferSent model
    params_model = {'bsize': 64, 'pool_type': 'mean', 'which_layer': 'all',
                    'optfile': OPT_PATH,
                    'wgtfile': MODEL_PATH}

    model = ELMo(params_model)
    params_senteval['elmo'] = model.cuda()

    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark',
                      'Length', 'WordContent', 'Depth', 'TopConstituents',
                      'BigramShift', 'Tense', 'SubjNumber', 'ObjNumber',
                      'OddManOut', 'CoordinationInversion']
    results_transfer = se.eval(transfer_tasks)
'''
