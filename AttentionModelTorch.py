import os, sys
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd


from notebook.pytorch.util import basic
print("Done!")

# Loading the data
original_pos_data, original_pos_label = basic.preprocess_data("/mnt/scratch7/hirak/Attention4DNASeqRepresentation/dataset/gene_range_start_codon.txt", 1)
original_neg_data, original_neg_label = basic.preprocess_data("/mnt/scratch7/hirak/Attention4DNASeqRepresentation/dataset/intragenic_start_codon.txt", 0)

data_size = 500000
batch_size = 1000
data_content = original_pos_data[:data_size] + original_neg_data[:data_size]
pos_data = None
neg_data = None
data_label = original_pos_label[:data_size] + original_neg_label[:data_size] 
pos_label = None
neg_label = None
print(len(data_content), np.sum(data_label))

total_datasize = len(data_content)-len(data_content)%batch_size
print(total_datasize, batch_size)
rand_index = np.random.permutation(total_datasize)
data_content = [data_content[i] for i in rand_index]
data_label = [data_label[i] for i in rand_index]
print(len(data_content), np.sum(data_label))

X = torch.from_numpy(np.array(data_content).astype(int))
Y = torch.from_numpy(np.array(data_label).reshape(len(data_label),1).astype(np.int))

class AttnDecoderRNN(nn.Module):#corrected batch faster
    #(self, time_steps, embedding_dim, hidden_dim, vocab_size, tagset_size, mini_batch)
    def __init__(self, vocab_size, embedding_dim, \
                 hidden_dim, batch_size=100, debug=1, \
                 tagset_size=1, time_steps=101):
        super(AttnDecoderRNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.time_steps = time_steps
        self.hidden_dim = hidden_dim
        self.minibatch_size = batch_size
        self.dropout_p = 0.25
        self.tagset_size = tagset_size
        self.hidden = self.init_hidden()
        self.debug = debug

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_one = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout_one = nn.Dropout(0.25)
        self.lstm_two = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout_two = nn.Dropout(0.25)

        self.attn_array = [nn.Linear(hidden_dim, hidden_dim) for i in range(time_steps)]
        """
        self.attn_combine = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        self.linear = nn.Linear(hidden_dim, 1)
        #embedding_dim*time_steps
        """

        self.hidden2tag_one = nn.Linear(hidden_dim*time_steps, 512)
        self.dropout_three = nn.Dropout(0.25)
        self.hidden2tag_two = nn.Linear(512, 128)
        self.dropout_four = nn.Dropout(0.25)
        self.hidden2tag_three = nn.Linear(128, 64)
        self.dropout_five = nn.Dropout(0.25)

        self.output = nn.Linear(64, tagset_size)

    def forward(self, input):
        init_embed = self.embedding(input)
        #embedded = init_embed.permute(1, 0, 2)
        if self.debug == 1:
            print("Embedding Shape: ", init_embed.shape)

        lstm_out, self.hidden_one = self.lstm_one(init_embed, self.hidden)
        lstm_out = self.dropout_one(lstm_out)
        lstm_out, self.hidden_two = self.lstm_two(lstm_out, self.hidden)
        lstm_out = self.dropout_two(lstm_out)
        #"""
        lstm_permute = lstm_out.permute(1, 0, 2)
        if self.debug == 1:
            print("LSTM Out Shape: ", lstm_permute.shape)

        attention = [self.attn_array[i](lstm_permute[i][:]) \
                     for i in range(self.time_steps)]
        attention = torch.stack(attention)
        attention = attention.permute(1, 0, 2)
        if self.debug == 1:
            print("Attention Shape: ", attention.shape)

        attn_weights = F.softmax(attention, dim=2)
        #attn_weights = attn_weights.view(self.minibatch_size, self.time_steps, 1)
        if self.debug == 1:
            print("Softmax Shape: ", attn_weights.shape)
        """
        attn_weights = torch.stack(
            [attn_weights]*self.embedding_dim, 2).view(
            self.minibatch_size, self.time_steps, -1)
        if self.debug == 1:
            print("Softmax ReShape: ", attn_weights.shape)
        """
        #attn_applied = init_embed
        attn_applied = attn_weights * init_embed
        #attn_applied = attn_applied.view(self.minibatch_size, self.time_steps, -1)
        #attn_applied = torch.sum(attn_applied, dim=1)
        if self.debug == 1:
            print("Embedding*Attention Shape: ", attn_applied.shape)

        #output = F.relu(attn_applied)
        #"""

        lstm_out = attn_applied.contiguous().view(self.minibatch_size, -1)
        #lstm_output = lstm_out.contiguous().view(self.minibatch_size, -1)
        if self.debug == 1:
            print("LSTM Output Shape: ", lstm_out.shape)


        dense_out = self.hidden2tag_one(lstm_out[:])
        dense_out = F.relu(dense_out[:])
        dense_out = self.dropout_three(dense_out[:])

        dense_out = self.hidden2tag_two(dense_out[:])
        dense_out = F.relu(dense_out[:])
        dense_out = self.dropout_four(dense_out[:])

        dense_out = self.hidden2tag_three(dense_out[:])
        dense_out = F.relu(dense_out[:])
        dense_out = self.dropout_five(dense_out[:])

        tag_space = self.output(dense_out[:])
        #print(tag_space.shape)
        #tag_scores = F.sigmoid(tag_space)
        #tag_scores = F.softmax(tag_space, dim=1)
        #print(tag_scores.shape)
        return tag_space, attn_applied

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (torch.zeros(1, self.minibatch_size, self.hidden_dim),
                torch.zeros(1, self.minibatch_size, self.hidden_dim))

model = AttnDecoderRNN(5, 16, 16)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

def weighted_binary_cross_entropy(output, target, weights=None):
        
    if weights is not None:
        assert len(weights) == 2
        
        output = torch.clamp(output,min=1e-8,max=1-1e-8) 
        #loss =  pos_weight * (target * torch.log(output)) + \
        #        neg_weight * ((1 - target) * torch.log(1 - output))
        loss = weights[1] * (target * torch.log(output)) + \
               weights[0] * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))

import torch.nn.functional as F

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
    """

    #round predictions to the closest integer
    rounded_preds = torch.round(F.sigmoid(preds))
    correct = (rounded_preds == y).float() #convert into float for division 
    acc = correct.sum()/len(correct)
    
    return acc

from tqdm import tqdm_notebook as tqdm
losses = []
accuracies = []
#batch_size = 10
model = AttnDecoderRNN(5, 16, 16, batch_size=batch_size, debug=0)
print(model)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

num_epochs = 100
with open("progress.txt", "w") as fp:
    for epoch in tqdm(range(num_epochs)):  # again, normally you would NOT do 300 epochs, it is toy data
        total_loss = 0
        total_acc = 0

        for index in range(0, len(X), batch_size):
            sentence = X[index : index+batch_size]#.reshape(len(X[0]))
            tags = Y[index : index+batch_size]#.reshape(len(Y[0]))
            #print(sentence.shape, tags.shape)
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Also, we need to clear out the hidden state of the LSTM,
            # detaching it from its history on the last instance.
            model.hidden = model.init_hidden()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.
            # sentence_in = prepare_sequence(sentence, word_to_ix)
            #targets = prepare_sequence(tags, tag_to_ix)
            targets = tags.float().flatten()

            # Step 3. Run our forward pass.
            tag_scores, attn_weight = model(sentence)
            tag_scores = tag_scores.flatten()
            #print(targets.shape, tag_scores.shape)

            #neg_weight = batch_size / (batch_size-np.sum(data_label[index : index+batch_size]))
            #pos_weight = batch_size / np.sum(data_label[index : index+batch_size])
            #weights = torch.FloatTensor([neg_weight, pos_weight])

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = criterion(tag_scores, targets)
            #loss = weighted_binary_cross_entropy(tag_scores, targets, weights=weights)
            total_loss += loss.data.numpy()

            acc = binary_accuracy(tag_scores, targets)
            total_acc += acc

            loss.backward()
            optimizer.step()

            #print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,num_epochs, loss.data[0], correct/x.shape[0]))

        losses.append(total_loss)
        accuracies.append(total_acc/(len(X)/batch_size))

        #total_loss.backward()
        #opt.step()

        #print(epoch, total_loss)#, total_acc)
        print("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(epoch+1,num_epochs, losses[-1], accuracies[-1]))
        fp.write("Epoch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}\n".format(epoch+1,num_epochs, losses[-1], accuracies[-1]))


with open("attention_loss.tsv", "w") as fp:
    for l in losses:
        fp.write("{}\n".format(l))

with open("attention_accuracy.tsv", "w") as fp:
    with a in accuracies:
        fp.write("{}\n".format(a)) 
