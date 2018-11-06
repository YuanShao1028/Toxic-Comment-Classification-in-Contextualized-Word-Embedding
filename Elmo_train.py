import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.autograd import Variable
import gc
from data_process import clean, tokenize
from allennlp.modules.elmo import Elmo, batch_to_ids
import nltk
import re

#options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
#weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

options_file = "capstone/data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
weight_file = "capstone/data/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"


batch_size = 32
max_features = 100000
maxlen = 200
embed_size = 1024
epochs = 2
EMBEDDING_FILE_FASTTEXT = "glove.6B.300d.txt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def data_to_ids(dataset, bsize = 2048):
    tensor_list = []
    for idx in range(0, len(dataset), bsize):
        print(idx)
        data = dataset[idx : idx + bsize]
        ids = batch_to_ids(data)
        tensor_list.append(ids)
    result = torch.cat(tensor_list)
    return result

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_train = clean(X_train)
X_test = clean(X_test)

X_train = list(map(lambda x: tokenize(x), X_train))
X_test = list(map(lambda x: tokenize(x), X_test))

train_token_id = data_to_ids(X_train)


print(train_token_id.size())

train_set = torch.utils.data.TensorDataset(train_token_id.long(), torch.from_numpy(y_train).float())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        p = .1
        self.elmo_embedding = Elmo(options_file, weight_file, 1, dropout=0)
        self.lin_0 = nn.Linear(1024, 300)
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
        x = self.elmo_embedding(x)['elmo_representations'][0]
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

def train(model, epoch = 5):

    learnin1g_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=learnin1g_rate)
    for e in range(epoch):
        running_loss = 0.0
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
    print("train complete")
model = Net().to(device)
print(model)

MODEL_SAVE_PATH = "elmo_batch_32.pth"
train(model)
torch.save(model.state_dict(), MODEL_SAVE_PATH)

