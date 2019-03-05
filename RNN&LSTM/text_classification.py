import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
import numpy as np
from torch.optim import lr_scheduler
import TextClassData 

from collections import OrderedDict
BASE_DIR = 'data'

class Bag_of_words(nn.Module):
    # Using bag of word to embedding the word 
    def __init__(self):
        super(Bag_of_words, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(7639,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    
    
class Average_pooling(nn.Module):
    # Using bag of word to embedding the word 
    def __init__(self):
        super(Average_pooling, self).__init__()
        self.embed = nn.EmbeddingBag(7800, 64, mode = 'sum')
        self.linear = nn.Linear(64,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, length):
        x = self.embed(x)
#        print(x.shape)
#        print(x)
        N = x.shape[0]
        x = x/length.reshape(N,-1)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x
    
    
class Average_pooling_glove(nn.Module):
    # Using bag of word to embedding the word 
    def __init__(self):
        super(Average_pooling_glove, self).__init__()
        self.linear = nn.Linear(300,1)
#        self.linear = nn.Linear(300,20)
#        self.relu = nn.ReLU()
#        self.linear2 = nn.Linear(20,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.linear(x)
#        x = self.relu(x)
#        x = self.linear2(x)
        x = self.sigmoid(x)
        return x


class RNN(nn.Module):
    # Using bag of word to embedding the word 
    def __init__(self, hidden_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(300, hidden_size, 1)
        self.linear = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, length):
        # the shape of x should be (seq_len, batch, input_size)
        h_n, _ = self.rnn(x)
        index = torch.arange(x.shape[1])
        h = h_n[(length-1).squeeze(),index,:]
        x = self.linear(h)
        x = self.sigmoid(x)
        return x
    
class LSTM(nn.Module):
    # Using bag of word to embedding the word 
    def __init__(self, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(300, hidden_size, 1)
        self.linear = nn.Linear(hidden_size,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, length):
        # the shape of x should be (seq_len, batch, input_size)
        h_n, _ = self.lstm(x)
        index = torch.arange(x.shape[1])
        h = h_n[(length-1).squeeze(),index,:]
        x = self.linear(h)
        x = self.sigmoid(x)
        return x
    

    
def load_glove(root_dic, name_dic):
    filenames_dic = os.path.join(root_dic, name_dic)
    dic = {}
    with open(filenames_dic, 'r', encoding='utf-8') as f:
        for l in f:
            line = l.split()
            vec = np.asarray(line[1:], dtype = 'float32')
            dic[line[0]] = vec
    return dic


def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
    

def train(trainloader, valloader, net, criterion, optimizer, scheduler, device, mode):
    for epoch in range(30):  # loop over the dataset multiple times
        start = time.time()
        scheduler.step()
#        adjust_learning_rate(optimizer, 0.9)
        running_loss = 0.0
        for i, data in enumerate(trainloader):
            # TODO: zero the parameter gradients
            # TODO: forward pass
            # TODO: backward pass
            # TODO: optimize the network
            optimizer.zero_grad()
            if (mode == 1 or mode == 3):
                texts, labels = data
                texts = texts.type('torch.FloatTensor').to(device)
                labels = labels.type('torch.FloatTensor').to(device)
                outputs = net(texts)
            elif(mode == 2):
                texts, labels, length = data
                texts = texts.type('torch.LongTensor').to(device)
                labels = labels.type('torch.FloatTensor').to(device)
                length = length.type('torch.FloatTensor').to(device)
                outputs = net(texts, length)
            elif(mode == 4 or mode == 5):
                texts, labels, length = data
                texts = torch.transpose(texts, 0, 1)
                texts = texts.type('torch.FloatTensor').to(device)
                labels = labels.type('torch.FloatTensor').to(device)
                length = length.type('torch.LongTensor').to(device)
                outputs = net(texts, length)
            else:
                pass
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            #print statistics
            running_loss += loss.item()
            if i % 100 == 99:    # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end-start))
                start = time.time()
                running_loss = 0.0
        if epoch % 2 == 0:
            test(trainloader, net, device, mode, True)
            test(valloader, net, device, mode, False)
            pass
    print('Finished Training')
        
        

def test(testloader, net, device, mode, train_mode = True):
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for data in testloader:
            if (mode == 1 or mode == 3):
                texts, labels = data
                texts = texts.type('torch.FloatTensor').to(device)
                labels = labels.type('torch.FloatTensor').to(device)
                outputs = net(texts)
            elif(mode == 2):
                texts, labels, length = data
                texts = texts.type('torch.LongTensor').to(device)
                labels = labels.type('torch.FloatTensor').to(device)
                length = length.type('torch.FloatTensor').to(device)
                outputs = net(texts, length)
            elif(mode == 4 or mode == 5):
                texts, labels, length = data
                texts = torch.transpose(texts, 0, 1)
                texts = texts.type('torch.FloatTensor').to(device)
                labels = labels.type('torch.FloatTensor').to(device)
                length = length.type('torch.LongTensor').to(device)
                outputs = net(texts, length)
            else:
                pass
            predicted = torch.zeros_like(outputs)
            predicted[outputs.data>0.5] = 1
            total += float(labels.size(0))
            correct += float((predicted.squeeze() == labels).sum().item())
    if train_mode:
        print('Train acc: %s %%' % (
                100 * float(correct) / float(total)))
    else:
        print('Val acc: %s %%' % (
                100 * float(correct) / float(total)))


def predict(unlabelledset, net, device, mode, train_mode = True):
    with torch.no_grad():
        for data in unlabelledset:
            if (mode == 1 or mode == 3):
                texts, labels = data
                texts = texts.type('torch.FloatTensor').to(device)
                labels = labels.type('torch.FloatTensor').to(device)
                outputs = net(texts)
            elif(mode == 2):
                texts, labels, length = data
                texts = texts.type('torch.LongTensor').to(device)
                labels = labels.type('torch.FloatTensor').to(device)
                length = length.type('torch.FloatTensor').to(device)
                outputs = net(texts, length)
            elif(mode == 4 or mode == 5):
                texts, labels, length = data
                texts = torch.transpose(texts, 0, 1)
                texts = texts.type('torch.FloatTensor').to(device)
                labels = labels.type('torch.FloatTensor').to(device)
                length = length.type('torch.LongTensor').to(device)
                outputs = net(texts, length)
            else:
                pass
            predicted = torch.zeros_like(outputs)
            predicted[outputs.data>0.5] = 1
            filename = 'predictions_q' + str(mode)
            predicted_write = predicted.cpu().numpy()
            np.savetxt(filename, predicted_write, fmt="%d")
                      


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set the mode for different method
    # mode = 1 for 6.1 bag of word
    # mode = 2 for 6.2 average pooling
    # mode = 3 for 6.3 
    # mode = 4 for 6.4 RNN with Glove embedding
    # mode = 5 for 6.5 LSTM with Glove embedding
    
    mode = 2
    #######################################################
    ##   6.1 bag of words
    #######################################################
    if (mode == 1):
        trainset = TextClassData.Bag_of_words(root = 'data', name = 'train.txt', mode = 'train',
                                 preload = True, transform = transforms.ToTensor())
        vocab = trainset._getVocal()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True)
        
        valset = TextClassData.Bag_of_words(root = 'data', name = 'dev.txt', mode = 'val',
                                 preload = True, dic = vocab, transform = transforms.ToTensor())
        valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                              shuffle=True)
        
        testset = TextClassData.Bag_of_words(root = 'data', name = 'test.txt', mode = 'test',
                                 preload = True, dic = vocab, transform = transforms.ToTensor())
    
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                              shuffle = False)
    
        unlabelledset = TextClassData.Bag_of_words(root = 'data', name = 'unlabelled.txt', mode = 'unlabelled',
                                 preload = True, dic = vocab, transform = transforms.ToTensor())
        unlabelledloader = torch.utils.data.DataLoader(unlabelledset, batch_size=unlabelledset.len,
                                              shuffle = False)
        
        net = Bag_of_words().to(device)
        optimizer = optim.Adam(net.parameters(), lr=3e-2, weight_decay = 0)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.6)

    ########################################################
    ###  6.2 average pooling
    ########################################################
    elif(mode == 2):
        trainset = TextClassData.Words_embedding(root = 'data', name = 'train.txt', mode = 'train',
                                 preload = True, transform = transforms.ToTensor())
        vocab = trainset._getVocal()
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                              shuffle=True)
        
        valset = TextClassData.Words_embedding(root = 'data', name = 'dev.txt', mode = 'val',
                                 preload = True, dic = vocab, transform = transforms.ToTensor())
        valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                              shuffle=True)
        
        
        testset = TextClassData.Words_embedding(root = 'data', name = 'test.txt', mode = 'test',
                                 preload = True, dic = vocab, transform = transforms.ToTensor())
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                              shuffle = False)
        
        
        unlabelledset = TextClassData.Words_embedding(root = 'data', name = 'unlabelled.txt', mode = 'unlabelled',
                                 preload = True, dic = vocab, transform = transforms.ToTensor())
        unlabelledloader = torch.utils.data.DataLoader(unlabelledset,batch_size=unlabelledset.len, shuffle = False)
        
    
        net = Average_pooling().to(device)
        optimizer = optim.Adam(net.parameters(), lr=9e-3, weight_decay = 0)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.6)
        
    ########################################################
    ###  6.3 average pooling with glove embedding 
    ########################################################
    elif(mode == 3):
        dic = load_glove(root_dic = 'glove.6B', name_dic = 'glove.6B.300d.txt')
        trainset = TextClassData.Words_embedding_glove(root = 'data', name = 'train.txt',  mode = 'train',
                                                       transform = True, dic = dic)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=200,
                                              shuffle=True)
        
        valset = TextClassData.Words_embedding_glove(root = 'data', name = 'dev.txt',  mode = 'val',
                                                     transform = True, dic = dic)
        valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                              shuffle=True)
        
        
        testset = TextClassData.Words_embedding_glove(root = 'data', name = 'test.txt',  mode = 'test',
                                                      transform = True, dic = dic)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                              shuffle = False)
    
        
        unlabelledset = TextClassData.Words_embedding_glove(root = 'data', name = 'unlabelled.txt',  mode = 'unlabelled',
                                                      transform = True, dic = dic)
        unlabelledloader = torch.utils.data.DataLoader(unlabelledset, batch_size=unlabelledset.len,
                                              shuffle = False)
        
        net = Average_pooling_glove().to(device)
#        optimizer = optim.Adam(net.parameters(), lr=1e-2, weight_decay = 0)
        optimizer = optim.SGD(net.parameters(), lr=3e-3, momentum = 0.6, weight_decay = 0)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
        
        
    ########################################################
    ###  6.4 RNN with glove embedding 
    ########################################################
    elif(mode == 4 or mode==5):
        dic = load_glove(root_dic = 'glove.6B', name_dic = 'glove.6B.300d.txt')
        trainset = TextClassData.Words_embedding_glove_pad(root = 'data', name = 'train.txt', mode = 'train', 
                                                       transform = True, dic = dic)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True)
        
        valset = TextClassData.Words_embedding_glove_pad(root = 'data', name = 'dev.txt', mode = 'val',
                                                     transform = True, dic = dic)
        valloader = torch.utils.data.DataLoader(valset, batch_size=100,
                                              shuffle=True)
        
        
        testset = TextClassData.Words_embedding_glove_pad(root = 'data', name = 'test.txt', mode = 'test',
                                                      transform = True, dic = dic)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                              shuffle = False)
        
        unlabelledset = TextClassData.Words_embedding_glove_pad(root = 'data', name = 'unlabelled.txt', mode = 'unlabelled',
                                                      transform = True, dic = dic)
        unlabelledloader = torch.utils.data.DataLoader(unlabelledset, batch_size=unlabelledset.len,
                                              shuffle = False)
        
        
        if mode == 4:
            net = RNN(30).to(device)
#            optimizer = optim.SGD(net.parameters(), lr=1.5e-3, weight_decay = 0)
            optimizer = optim.Adam(net.parameters(), lr=2e-2, weight_decay = 0.5)
#            optimizer = optim.SGD(net.parameters(), lr=3e-2, momentum = 0.6, weight_decay = 0)
#            optimizer = optim.ASGD(net.parameters(), lr=2e-2, weight_decay = 0.1)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
        elif mode==5:
            net = LSTM(15).to(device)
            
            optimizer = optim.Adam(net.parameters(), lr=2e-2, weight_decay = 0.5)
            exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        else: pass
    
    else:
        pass
    
#    criterion = nn.CrossEntropyLoss()
#    criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()

    
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay = 0)

    train(trainloader, valloader, net, criterion, optimizer, exp_lr_scheduler, device, mode)
    test(testloader, net, device, mode)
    
    predict(unlabelledloader, net, device, mode)

if __name__== "__main__":
    main()