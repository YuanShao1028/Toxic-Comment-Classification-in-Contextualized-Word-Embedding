import numpy as np
import pandas as pd
import torch
from itertools import islice  
import torch.nn.functional as F
import torch.utils.data
from keras.preprocessing import text, sequence
from torch import nn, optim
import gc
import time

batch_size = 32
max_features = 100000
maxlen = 200
embed_size = 300
epochs = 2
EMBEDDING_FILE_FASTTEXT = "data/embedding/glove.6B.300d.txt"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
X_train = train["comment_text"].fillna("fillna").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
X_test = test["comment_text"].fillna("fillna").values

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train = sequence.pad_sequences(X_train_seq, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test_seq, maxlen=maxlen)
gc.collect()
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

train_set = torch.utils.data.TensorDataset(torch.from_numpy(X_train).long(), torch.from_numpy(y_train).float())

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(torch.from_numpy(X_test).long(), batch_size=1024)
word_index = tokenizer.word_index


def generateEmbeddingMatrix(input_file, word_index, dimension, max_feature, skip_header=False):
    embedding_map = {}
    for line in islice(input_file, int(skip_header), None):
        line = line.rstrip().split(' ')
        word = line[0]
        embedding = np.array(line[1:],dtype='float32')
        embedding_map[word] = embedding
    print(len(embedding_map))
    embedding_matrix = np.zeros((min(len(word_index) + 1, max_feature), dimension))
    for word, index in word_index.items():
        if index > max_feature:
            continue
        embedding_vector = embedding_map.get(word)
        if embedding_vector is not None:
            embedding_matrix[index] = embedding_vector
    gc.collect()
    return embedding_matrix    

input_file = open(EMBEDDING_FILE_FASTTEXT, encoding="utf-8")
embedding_matrix = generateEmbeddingMatrix(input_file, word_index, embed_size, max_features)
input_file.close()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        p = .1
        self.embeddings = nn.Embedding(num_embeddings=max_features, embedding_dim=embed_size)
        self.embeddings.weight.data = torch.Tensor(embedding_matrix)
        self.embedding_dim = 300
        self.Ci = 1
        self.Co = 128
        self.kernel_list = [1,2,3,5]
        self.convs = nn.ModuleList([nn.Conv2d(self.Ci, self.Co, (k, self.embedding_dim)) for k in self.kernel_list])
        #self.conv1 = nn.Conv2d(1, 32, (1, self.embedding_dim))
        #self.conv2 = nn.Conv2d(1, 32, (2, self.embedding_dim))
        #self.conv3 = nn.Conv2d(1, 32, (3, self.embedding_dim))
        #self.conv5 = nn.Conv2d(1, 32, (5, self.embedding_dim))
        self.dropout = nn.Dropout(p=p)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(128 * 4, 6)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.embeddings(x)
        #print(x.size())   #(bsize, seqlen, dim)
        x = x.unsqueeze(1)
        x = [self.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        #print(x.size())
        x = self.dropout(x)
        x = self.lin(x)
        return self.sig(x)
    
def train(model, epoch = 2):
    
    learnin1g_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learnin1g_rate)
    #model.train()
    for e in range(epoch):
        running_loss = 0.0
        start = time.time()
        for i, (data, target) in enumerate(train_loader):
            model.zero_grad()
            #data, target = Variable(data), Variable(target)
            data, target = data.to(device), target.to(device)
            y_pred = model(data)
            loss = F.binary_cross_entropy(y_pred, target)
            #print(loss.data[0])
            #model.zero_grad()
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


train(model)

preds = []
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        output = model(data)
        pred = output.data.cpu()
        preds.append(pred.numpy())
y_test = np.concatenate(preds, axis=0)

sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_test
sample_submission.to_csv("submission1.csv", index=False)
