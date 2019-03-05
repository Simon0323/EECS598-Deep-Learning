# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:24:18 2019

@author: sunhu
"""
import torch
from torch.utils.data import Dataset, DataLoader
import os.path as osp
import numpy as np
from copy import deepcopy
import re
      
def build_dic(path, vocab = None, mode = 'train', maxlen = 0):
    count_vocab, vocab_temp, lengthlist = 0, {}, []
    if vocab is not None:
        vocab_temp = deepcopy(vocab)
    labels, arraylist = [], []
    with open(path, 'r') as f:
        for line in f:
            if mode == 'unlabelled':
                temp = line.split()
            else:
                labels.append(int(line[0]))
                temp = line.split()[1:]
            listofnumber = []
            for word in temp:
                if (word not in vocab_temp):
                    if (vocab != None): continue
                    vocab_temp[word] = count_vocab
                    count_vocab += 1
                listofnumber.append(vocab_temp[word])
            if maxlen != 0:
                temp = np.pad(np.asarray(listofnumber), (0, maxlen-len(listofnumber)), 
                              'constant' , constant_values = 0)
            else:
                temp = np.asarray(listofnumber)
            arraylist.append(temp)
            lengthlist.append(len(listofnumber))
    return vocab_temp, arraylist, labels, lengthlist


def max_length(filenames):
    maxlen = 0
    with open(filenames, 'r') as f:
        for line in f:
            temp = line[2:].split()
            if (len(temp)> maxlen):
                maxlen=len(temp) 
    return maxlen



class Bag_of_words(Dataset):
    """
    A customized data loader for MNIST.
    """
    def __init__(self, root, name, mode='train', dic = None, transform=None,  preload=False):
        self.data, self.labels, self.filenames = [], None, None
        self.mode = mode
        self.vocab = dic
        self.transform = transform
        self.filenames = osp.join(root, name)
        # if preload dataset into memory
        if preload:
            self._preload()
        
    def _preload(self):
        """
        Preload dataset to memory
        """
        self.vocab, arraylist, self.labels, _ = build_dic(self.filenames, self.vocab, self.mode)
        count_vocab = len(self.vocab)
        for index in range(len(arraylist)):
            temp = np.zeros(count_vocab)
            if(len(arraylist[index])!=0):
                temp[arraylist[index]] = 1
            self.data.append(temp)
        self.len = len(arraylist)                
        
    def _getVocal(self):
        return self.vocab
    
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.data is not None:
            # If dataset is preloaded
            data = self.data[index]
            label = 0
            if self.mode != 'unlabelled':
                label = self.labels[index]
        
        if self.transform is not None:
            data = torch.tensor(torch.from_numpy(data))
        return data, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    
    
    
    
class Words_embedding(Dataset):
    """
    A customized data loader for MNIST.
    """
    def __init__(self, root, name, mode='train', dic = None, transform=None, preload=False):
        self.data, self.labels, self.filenames = [], None, None
        self.mode = mode
        self.vocab = dic
        self.transform = transform
        self.lengthlist = None
        self.filenames = osp.join(root, name)
        # if preload dataset into memory
        if preload:
            self._preload()
        
    def _preload(self):
        """
        Preload dataset to memory
        """
        maxlen = max_length(self.filenames)
        self.vocab, self.data, self.labels, self.lengthlist = build_dic(self.filenames, self.vocab, self.mode, maxlen)
        self.len = len(self.data)                
        
    def _getVocal(self):
        return self.vocab
    
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.data is not None:
            data = self.data[index]
            label = 0
            if self.mode != 'unlabelled':
                label = self.labels[index]
            length = self.lengthlist[index]

        if self.transform is not None:
            data = torch.tensor(torch.from_numpy(data))
        return data, label, length

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len


class Words_embedding_glove(Dataset):
    """
    A customized data loader for MNIST.
    """
    def __init__(self, root, name, mode, transform, dic = None):
        self.data, self.labels, self.filenames = [], [], None
        self.vocab = dic
        self.mode = mode
        self.transform = transform
        self.filenames = osp.join(root, name)
        self._preload()
        
            
    def _preload(self):
        """
        Preload dataset to memory
        """
#        count_unexist, dic_unexist = 0, {}
        with open(self.filenames, 'r') as f:
            for line in f:
                temp = line.split()
                if self.mode == 'unlabelled':
                    temp = line.split()
                else:
                    self.labels.append(int(temp[0]))
                    temp = line.split()[1:]
                words = temp[1:]
                words_avg = np.zeros_like(self.vocab['hello'])
                words_number = 0
                for word in words:
                    if word in self.vocab:
                        words_avg += self.vocab[word]
                        words_number += 1
                    else:
                        temp_word = re.sub("[^a-zA-Z]+","",word)
                        if temp_word in self.vocab:
                            words_avg += self.vocab[temp_word]
                            words_number += 1                            
                    """    
                    else:
                        if(word in dic_unexist): dic_unexist[word] += 1
                        else: dic_unexist[word] = 1
                        count_unexist += 1
                    """
                self.data.append(words_avg/words_number)
        self.len = len(self.data)                
        
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.data is not None:
            # If dataset is preloaded
            data = self.data[index]
            label = 0
            if self.mode != 'unlabelled':
                label = self.labels[index]
 
        if self.transform is not None:
            data = torch.tensor(torch.from_numpy(data))
        return data, label

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len
    
    
class Words_embedding_glove_pad(Dataset):
    """
    A customized data loader for MNIST.
    """
    def __init__(self, root, name, mode='train', transform= None, dic = None):
        self.data, self.labels, self.lengthOfSen, self.filenames = [], [], [], None
        self.vocab = dic
        self.mode = mode
        self.transform = transform
        self.filenames = osp.join(root, name)
        self._preload()
        
    def _preload(self):
        """
        Preload dataset to memory
        """
        maxlen = max_length(self.filenames)
        with open(self.filenames, 'r') as f:
            for line in f:
                temp = line.split()
                if self.mode == 'unlabelled':
                    words = temp
                else:
                    self.labels.append(int(temp[0]))
                    words = temp[1:-1]
                words_embedding = []
                for word in words:
                    if word in self.vocab:
                        words_embedding.append(self.vocab[word])
                temp = np.pad(np.array(words_embedding), ((0, maxlen-len(words_embedding)),(0,0)), 
                              'constant' , constant_values = 0)
                self.lengthOfSen.append(len(words_embedding))
                self.data.append(temp)
        self.len = len(self.data)   
        print(self.len)             
        
        
    def __getitem__(self, index):
        """ Get a sample from the dataset
        """
        if self.data is not None:
            # If dataset is preloaded
            data = self.data[index]
            label = 0
            if self.mode != 'unlabelled':
                label = self.labels[index]
            length = self.lengthOfSen[index]
 
        if self.transform is not None:
            data = torch.tensor(torch.from_numpy(data))
        return data, label, length

    def __len__(self):
        """
        Total number of samples in the dataset
        """
        return self.len